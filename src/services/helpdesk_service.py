import logging
import re
import threading
from sqlalchemy.orm import Session
from src.models.helpdesk import HelpdeskCategory, intent_to_display_label
from src.services.llm_service import LLMService, _log_tokens
from src.prompts import HELPDESK_CLASSIFY, HELPDESK_ORIENTATION

logger = logging.getLogger(__name__)

# Intenciones reservadas: nunca se toman de la BD para el clasificador
_RESERVED_INTENTS = {"config", "fallback"}

# Siempre presentes aunque no haya filas en la BD
_ALWAYS_VALID = {"saludo", "despedida", "desconocida"}

_ALWAYS_VALID_DESCRIPTIONS: dict[str, str] = {
    "saludo": (
        "el usuario saluda o inicia la conversación "
        "(hola, buenos días, buenas tardes, buenas noches, ¿cómo estás?)"
    ),
    "despedida": (
        "el usuario agradece, confirma o se despide de forma corta "
        "(gracias, ok, entendido, listo, perfecto, excelente, hasta luego, de nada, "
        "sí, no, claro, muy bien, bye, chao)"
    ),
    "desconocida": (
        "la consulta no encaja en ninguna categoría universitaria, es texto sin sentido "
        "o es ajena a los trámites de la institución"
    ),
}

# ── Detección de ruido pre-LLM ────────────────────────────────────────────────

_VOWELS = set("aeiouáéíóúü")
_ONLY_NON_ALPHA = re.compile(r'^[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+$')


def _is_noise(text: str) -> bool:
    """True si el texto es muy probablemente entrada sin sentido (no llama al LLM)."""
    t = text.strip()
    if not t or len(t) == 1:
        return True
    # Solo dígitos y/o símbolos — sin ninguna letra
    if _ONLY_NON_ALPHA.match(t):
        return True
    # Cualquier palabra de 5+ letras sin ninguna vocal → teclas aleatorias
    for word in re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]', ' ', t).split():
        if len(word) >= 5 and not any(c.lower() in _VOWELS for c in word):
            return True
    return False

_CLASSIFY_PROMPT_TEMPLATE = HELPDESK_CLASSIFY
_ORIENTATION_PROMPT = HELPDESK_ORIENTATION


class HelpdeskService:

    _CLASSIFY_TIMEOUT = float(60.0)

    def __init__(self, db: Session, llm: LLMService):
        self.db = db
        self.llm = llm

    # ── Intenciones dinámicas ─────────────────────────────────────────────────

    def _load_valid_intents(self) -> dict[str, str]:
        """Carga las intenciones activas desde la BD, excluyendo las reservadas."""
        rows = self.db.query(
            HelpdeskCategory.intent,
            HelpdeskCategory.description,
        ).all()
        db_intents = {
            row.intent: row.description or ""
            for row in rows
            if row.intent not in _RESERVED_INTENTS
        }
        always = {intent: _ALWAYS_VALID_DESCRIPTIONS.get(intent, "") for intent in _ALWAYS_VALID}
        return {**always, **db_intents}

    # ── Clasificador de intención ─────────────────────────────────────────────

    def classify_intent(self, question: str) -> str:
        """
        Clasifica la consulta en una de las intenciones activas en la BD.
        Las intenciones se cargan dinámicamente — agregar una fila nueva a
        helpdesk_categories con un intent nuevo es suficiente para que el
        clasificador lo detecte sin tocar código.
        Fallback (fail-closed): si el LLM falla → 'desconocida'.
        """
        question = (question or "").strip()
        if not question or _is_noise(question):
            return "desconocida"

        valid_intents = self._load_valid_intents()
        intents_list = "\n".join(
            f"{intent} → {desc}" if desc else intent
            for intent, desc in sorted(valid_intents.items())
        )
        prompt_text = _CLASSIFY_PROMPT_TEMPLATE.format(
            intents_list=intents_list,
            question=question[:220],
        )

        llm_bound = self.llm.llm.bind(
            options={
                "num_predict": 10,
                "top_k":       1,
                "top_p":       1.0,
                "temperature": 0.0,
            },
            stop=["\n", "---", " "],
        )

        try:
            result_holder: list = []
            error_holder:  list = []

            def _call():
                try:
                    r = llm_bound.invoke(prompt_text)
                    _log_tokens("helpdesk_classify", r)
                    result_holder.append(
                        getattr(r, "content", str(r)).strip().lower()
                    )
                except Exception as exc:
                    error_holder.append(exc)

            t = threading.Thread(target=_call, daemon=True)
            t.start()
            t.join(timeout=self._CLASSIFY_TIMEOUT)

            if t.is_alive():
                logger.warning(
                    "[HelpdeskService.classify_intent] Timeout (%.0fs). Fallback desconocida.",
                    self._CLASSIFY_TIMEOUT,
                )
                return "desconocida"

            if error_holder:
                logger.warning(
                    "[HelpdeskService.classify_intent] LLM error: %s. Fallback desconocida.",
                    error_holder[0],
                )
                return "desconocida"

            raw = result_holder[0] if result_holder else ""
            intent = raw.split()[0] if raw else "desconocida"
            intent = intent if intent in valid_intents.keys() else "desconocida"

            logger.debug(
                "[HelpdeskService.classify_intent] q=%r llm_answer=%r intent=%s",
                question[:60], raw, intent,
            )
            return intent

        except Exception as e:
            logger.warning(
                "[HelpdeskService.classify_intent] Error inesperado: %s. Fallback desconocida.", e
            )
            return "desconocida"

    # ── Mensaje de orientación para consultas no clasificadas ─────────────────

    def generate_orientation_message(self, question: str) -> str:
        """
        Genera un mensaje orientativo cuando la consulta no pudo clasificarse.
        El LLM explica al usuario qué tipos de consultas puede manejar.
        Fallback: mensaje estático si el LLM falla o supera el timeout.
        """
        rows = self.db.query(HelpdeskCategory.intent).filter(
            HelpdeskCategory.intent.notin_(_RESERVED_INTENTS | _ALWAYS_VALID)
        ).all()
        categories = ", ".join(intent_to_display_label(r.intent) for r in rows) if rows else "trámites universitarios"

        prompt = _ORIENTATION_PROMPT.format(
            question=(question or "")[:200],
            categories=categories,
        )
        llm_bound = self.llm.llm.bind(
            options={
                "num_predict": 80,
                "top_k":       40,
                "top_p":       0.9,
                "temperature": 0.3,
            },
            stop=["---"],
        )
        fallback = (
            f"No entendí tu consulta. Puedo orientarte con: {categories}. ¿En qué puedo ayudarte?"
            if categories != "trámites universitarios"
            else "No entendí tu consulta. ¿En qué puedo ayudarte?"
        )
        def _invoke():
            r = llm_bound.invoke(prompt)
            _log_tokens("helpdesk_orientation", r)
            return getattr(r, "content", "").strip().strip('"\'')

        return self.llm._call_with_timeout(
            _invoke,
            timeout=30.0,
            context="helpdesk.orientation",
            fallback=fallback,
        )
