import os as _os
import logging
import re
import threading
import time
from typing import List, Dict, Optional, Any

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from src.prompts import (
    RAG_SYSTEM_RULES,
    RAG_GENERAL_CONTROLLED_RULES,
    CLASSIFY_ESCALATION,
    CLASSIFY_OVERVIEW,
    FILTER_PROGRAMS,
    EXTRACT_LIST_INTENT,
)

logger = logging.getLogger(__name__)
Message = Dict[str, str]


# ──────────────────────────────────────────────────────────────
# Clasificadores de tipo de pregunta (para format hints)
# ──────────────────────────────────────────────────────────────

def _looks_like_list_request(question: str) -> bool:
    q = (question or "").lower().strip()
    list_intents = [
        "lista", "listar", "cuales", "cuáles", "qué programas", "que programas",
        "oferta", "programas hay", "muéstrame", "muestrame",
        "malla", "plan de estudios", "pensum", "materias", "asignaturas",
        "electivas", "opciones de grado",
    ]
    return any(k in q for k in list_intents)


def _looks_like_exact_fact(question: str) -> bool:
    q = (question or "").lower().strip()
    exact_intents = [
        "cuánto dura", "duración", "duracion", "semestres",
        "créditos", "creditos",
        "costo", "cuesta", "precio", "valor", "inversión", "inversion",
        "matrícula", "matricula",
        "modalidad", "ubicación", "ubicacion", "horario", "horarios",
        "título", "titulo", "registro calificado",
        "año", "ano", "actualización", "actualizacion",
    ]
    return any(k in q for k in exact_intents)


def topics_cleanup(text: str) -> str:
    """Limpia artefactos del prompt que el LLM pudo haber copiado en su respuesta."""
    t = (text or "").strip()
    t = re.sub(r"^(CONTEXTO|PREGUNTA|CONTEXTO VERIFICADO)\s*:\s*",
               "", t, flags=re.IGNORECASE).strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines and all(ln.startswith("- ") for ln in lines):
        return "\n".join(f"{i}. {ln[2:].strip()}" for i, ln in enumerate(lines, 1))
    return t


def _build_llm(base_url: str, model: str, temperature: float, **extra_kwargs) -> ChatOllama:
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        **extra_kwargs,
    )


# ──────────────────────────────────────────────────────────────
# Store de historial LangChain con TTL
# ──────────────────────────────────────────────────────────────

_HISTORY_TTL = int(_os.getenv("SESSION_TTL_SECONDS", 7200))
_HISTORY_MAXSIZE = int(_os.getenv("SESSION_MAX_SIZE", 500))


class _HistoryStore:
    """InMemoryChatMessageHistory con TTL y tamaño máximo."""

    def __init__(self, ttl: int = _HISTORY_TTL, maxsize: int = _HISTORY_MAXSIZE):
        self._ttl = ttl
        self._maxsize = maxsize
        self._data: Dict[str, InMemoryChatMessageHistory] = {}
        self._ts: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _evict(self) -> None:
        # Llamado solo desde get_or_create(), que ya sostiene el lock.
        now = time.monotonic()
        expired = [k for k, t in self._ts.items() if now - t > self._ttl]
        for k in expired:
            self._data.pop(k, None)
            self._ts.pop(k, None)
        if len(self._data) >= self._maxsize:
            oldest = sorted(self._ts, key=lambda k: self._ts[k])
            for k in oldest[: max(1, self._maxsize // 10)]:
                self._data.pop(k, None)
                self._ts.pop(k, None)

    def get_or_create(self, session_id: str) -> InMemoryChatMessageHistory:
        with self._lock:
            now = time.monotonic()
            ts = self._ts.get(session_id)
            if ts is not None and (now - ts > self._ttl):
                self._data.pop(session_id, None)
                self._ts.pop(session_id, None)
            if session_id not in self._data:
                self._evict()
                self._data[session_id] = InMemoryChatMessageHistory()
            self._ts[session_id] = time.monotonic()
            return self._data[session_id]

    def get(self, session_id: str) -> Optional[InMemoryChatMessageHistory]:
        with self._lock:
            now = time.monotonic()
            ts = self._ts.get(session_id)
            if ts is None or (now - ts > self._ttl):
                self._data.pop(session_id, None)
                self._ts.pop(session_id, None)
                return None
            return self._data.get(session_id)

    def pop(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)
            self._ts.pop(session_id, None)


# ──────────────────────────────────────────────────────────────
# LLMService
# ──────────────────────────────────────────────────────────────

class LLMService:
    """
    RAG con memoria por sesión (LangChain RunnableWithMessageHistory).
    Historial con TTL/maxsize alineado al TTLStore del pipeline.

    Timeouts configurables por variables de entorno:
      GENERATE_TIMEOUT   — generación RAG principal       (default: 120s CPU / 20s GPU)
      GENERAL_TIMEOUT    — generate_general_controlled     (default: 120s CPU / 20s GPU)
      CLASSIFIER_TIMEOUT — classify_escalation             (default:  60s CPU / 10s GPU)

    Para producción con GPU, bajar a 20/20/10.
    Para desarrollo local con CPU, los defaults de 120/120/60 son seguros.
    """

    _STOP_TOKENS = ["---", "###", "TU RESPUESTA:"]

    # Timeouts: defaults altos para CPU local.
    # En producción con GPU setear en .env: GENERATE_TIMEOUT=20, GENERAL_TIMEOUT=20, CLASSIFIER_TIMEOUT=10
    _GENERATE_TIMEOUT_SECONDS = float(_os.getenv("GENERATE_TIMEOUT", "120.0"))
    _GENERAL_TIMEOUT_SECONDS = float(_os.getenv("GENERAL_TIMEOUT", "120.0"))
    _CLASSIFIER_TIMEOUT_SECONDS = float(
        _os.getenv("CLASSIFIER_TIMEOUT", "60.0"))

    def __init__(
        self,
        base_url: str,
        model: str = "llama3.1:8b-instruct-q8_0",
        temperature: float = 0.1,
    ):
        self.base_url = (base_url or "").strip().strip("/")
        self.model = model
        self.temperature = temperature

        self.llm = _build_llm(
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
        )
        self._store = _HistoryStore()

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_rules()),
            MessagesPlaceholder(variable_name="history"),
            ("user",
             "CONTEXTO:\n{context}\n\n"
             "{style_hint}"
             "PREGUNTA:\n{question}\n\n"
             "Responde solo la pregunta. No copies el prompt."),
        ])

        self.prompt_general = ChatPromptTemplate.from_messages([
            ("system", self._system_rules_general_controlled()),
            MessagesPlaceholder(variable_name="history"),
            ("user",
             "CONTEXTO VERIFICADO:\n{context}\n\n"
             "PREGUNTA:\n{question}\n\n"
             "Responde solo la pregunta. No copies el prompt."),
        ])

        self._get_history = lambda sid: self._store.get_or_create(sid)
        logger.info(
            "✅ LLMService listo: model=%s | timeouts: generate=%.0fs general=%.0fs classifier=%.0fs",
            self.model,
            self._GENERATE_TIMEOUT_SECONDS,
            self._GENERAL_TIMEOUT_SECONDS,
            self._CLASSIFIER_TIMEOUT_SECONDS,
        )

    # ──────────────────────────────────────────────────────────────
    # Construcción de chains
    # ──────────────────────────────────────────────────────────────

    def _make_rag_chain(self, num_predict: int, top_k: int = 35, top_p: float = 0.85):
        llm_bound = self.llm.bind(
            options={
                "num_predict": num_predict,
                "top_k": top_k,
                "top_p": top_p,
                "repeat_penalty": 1.1,
                "repeat_last_n": 64,
            },
            stop=self._STOP_TOKENS,
        )
        return RunnableWithMessageHistory(
            self.prompt | llm_bound,
            self._get_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    def _make_general_chain(self, num_predict: int = 220):
        llm_bound = self.llm.bind(
            options={
                "num_predict": num_predict,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.05,
            },
            stop=self._STOP_TOKENS,
        )
        return RunnableWithMessageHistory(
            self.prompt_general | llm_bound,
            self._get_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    # ──────────────────────────────────────────────────────────────
    # Sincronización de historial (rutas sin LLM)
    # ──────────────────────────────────────────────────────────────

    def append_to_history(self, chat_session_id: str, role: str, content: str) -> None:
        if not content:
            return
        role = (role or "").lower().strip()
        hist = self._store.get_or_create(chat_session_id)
        if role in ("assistant", "ai"):
            hist.add_ai_message(content)
        else:
            hist.add_user_message(content)

    def clear_memory(self, chat_session_id: str) -> None:
        self._store.pop(chat_session_id)

    # ──────────────────────────────────────────────────────────────
    # Prompts del sistema
    # ──────────────────────────────────────────────────────────────

    def _system_rules(self) -> str:
        return RAG_SYSTEM_RULES

    def _system_rules_general_controlled(self) -> str:
        return RAG_GENERAL_CONTROLLED_RULES

    # ──────────────────────────────────────────────────────────────
    # Format hint
    # ──────────────────────────────────────────────────────────────

    def _build_style_hint(self, question: str) -> tuple[str, bool, bool]:
        is_list = _looks_like_list_request(question)
        is_exact = _looks_like_exact_fact(question)
        hint = ""
        if is_list:
            hint = "INSTRUCCIÓN: Responde con lista numerada ordenada.\n"
        elif is_exact:
            hint = "INSTRUCCIÓN: Responde con el dato exacto en 1-2 líneas.\n"
        return hint, is_list, is_exact

    # ──────────────────────────────────────────────────────────────
    # Precarga de historial externo
    # ──────────────────────────────────────────────────────────────

    def _maybe_preload_history(
        self, chat_session_id: str, history: Optional[List[Message]]
    ) -> None:
        if not history:
            return
        hist_obj = self._store.get(chat_session_id)
        if hist_obj and len(hist_obj.messages) > 0:
            return
        hist_obj = self._store.get_or_create(chat_session_id)
        for msg in history[-10:]:
            role = (msg.get("role") or "").lower().strip()
            content = msg.get("content", "")
            if not content:
                continue
            if role in ("assistant", "ai"):
                hist_obj.add_ai_message(content)
            else:
                hist_obj.add_user_message(content)

    # ──────────────────────────────────────────────────────────────
    # Utilidad interna: llamada LLM con timeout
    # ──────────────────────────────────────────────────────────────

    def _call_with_timeout(self, fn, timeout: float, context: str, fallback, raise_on_error: bool = False):
        """Ejecuta fn() en un hilo con timeout. Retorna fallback en timeout/error."""
        result_holder: list = []
        error_holder: list = []

        def _call():
            try:
                result_holder.append(fn())
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            logger.warning("[%s] Timeout (%.0fs). Fallback.", context, timeout)
            return fallback

        if error_holder:
            if raise_on_error:
                raise error_holder[0]
            logger.warning("[%s] Error: %s. Fallback.", context, error_holder[0])
            return fallback

        return result_holder[0] if result_holder else fallback

    # ──────────────────────────────────────────────────────────────
    # Clasificador semántico de dominio
    # ──────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────
    # Clasificador de escalación (contacto humano)
    # ──────────────────────────────────────────────────────────────

    _ESCALATION_PROMPT = CLASSIFY_ESCALATION

    def classify_escalation(self, question: str) -> bool:
        """
        Clasifica si la pregunta es un intento de contactar soporte humano.

        Fallback (fail-closed): si el LLM falla o supera el timeout → False.
        Es preferible no detectar escalación que bloquear preguntas legítimas.
        """
        question = (question or "").strip()
        if not question:
            return False

        prompt_text = self._ESCALATION_PROMPT.format(question=question[:220])

        llm_bound = self.llm.bind(
            options={
                "num_predict": 8,
                "top_k": 1,
                "top_p": 1.0,
                "temperature": 0.0,
            },
            stop=["\n", "---", " "],
        )

        try:
            def _invoke():
                r = llm_bound.invoke(prompt_text)
                return getattr(r, "content", str(r)).strip().upper()

            answer = self._call_with_timeout(
                _invoke, self._CLASSIFIER_TIMEOUT_SECONDS, "classify_escalation", fallback="",
            )
            is_escalation = answer.startswith("ESCALATION") and not answer.startswith("NOT_")
            logger.debug("[classify_escalation] q=%r llm_answer=%r is_escalation=%s",
                         question[:60], answer, is_escalation)
            return is_escalation
        except Exception as e:
            logger.warning("[classify_escalation] Unexpected error: %s. Fail-closed.", e)
            return False

    # ──────────────────────────────────────────────────────────────
    # Clasificador de intención de overview de programa
    # ──────────────────────────────────────────────────────────────

    _OVERVIEW_PROMPT = CLASSIFY_OVERVIEW

    def classify_overview(self, question: str) -> bool:
        """
        Clasifica si la consulta pide información general/resumen de un programa.
        Se usa como fallback cuando el fast-path (keywords) no detecta intención.

        Fallback (fail-safe): si el LLM falla o supera el timeout → False.
        Es preferible no detectar overview que forzar la ruta incorrecta.
        """
        question = (question or "").strip()
        if not question:
            return False

        prompt_text = self._OVERVIEW_PROMPT.format(question=question[:300])

        llm_bound = self.llm.bind(
            options={
                "num_predict": 8,
                "top_k": 1,
                "top_p": 1.0,
                "temperature": 0.0,
            },
            stop=["\n", "---", " "],
        )

        try:
            def _invoke():
                r = llm_bound.invoke(prompt_text)
                return getattr(r, "content", str(r)).strip().upper()

            answer = self._call_with_timeout(
                _invoke, self._CLASSIFIER_TIMEOUT_SECONDS, "classify_overview", fallback="",
            )
            is_overview = answer.startswith("OVERVIEW") and not answer.startswith("NOT_")
            logger.debug("[classify_overview] q=%r llm_answer=%r is_overview=%s",
                         question[:60], answer, is_overview)
            return is_overview
        except Exception as e:
            logger.warning("[classify_overview] Unexpected error: %s. Fail-safe.", e)
            return False

    # ──────────────────────────────────────────────────────────────
    # Generate — RAG principal
    # ──────────────────────────────────────────────────────────────

    def generate(
        self,
        context: str,
        question: str,
        chat_session_id: str,
        history: Optional[List[Message]] = None,
    ) -> str:
        self._maybe_preload_history(chat_session_id, history)
        style_hint, is_list, is_exact = self._build_style_hint(question)

        if is_list:
            num_predict = 320
        elif is_exact:
            num_predict = 140
        else:
            num_predict = 240

        chain = self._make_rag_chain(num_predict=num_predict)
        config = {"configurable": {"session_id": chat_session_id}}

        _fallback = (
            "Lo siento, tuve un problema procesando tu consulta. "
            "Por favor intenta de nuevo en unos momentos."
        )
        try:
            def _invoke():
                r = chain.invoke(
                    {"context": context, "question": question, "style_hint": style_hint},
                    config=config,
                )
                return topics_cleanup(getattr(r, "content", str(r)))

            return self._call_with_timeout(
                _invoke, self._GENERATE_TIMEOUT_SECONDS, "LLMService.generate",
                fallback=_fallback, raise_on_error=True,
            )
        except Exception as e:
            logger.error(f"[LLMService.generate] Error: {e}", exc_info=True)
            return _fallback

    # ──────────────────────────────────────────────────────────────
    # Generate — GENERAL_CONTROLLED
    # ──────────────────────────────────────────────────────────────

    def generate_general_controlled(
        self,
        context: str,
        question: str,
        chat_session_id: str,
    ) -> str:
        chain = self._make_general_chain(num_predict=220)
        config = {"configurable": {"session_id": chat_session_id}}

        _fallback = (
            "Con la información disponible no puedo confirmarlo con certeza. "
            "Si me indicas el programa exacto (o su SNIES), puedo darte una respuesta más precisa."
        )
        try:
            def _invoke():
                r = chain.invoke(
                    {"context": context, "question": question},
                    config=config,
                )
                return topics_cleanup(getattr(r, "content", str(r)))

            return self._call_with_timeout(
                _invoke, self._GENERAL_TIMEOUT_SECONDS, "LLMService.generate_general_controlled",
                fallback=_fallback, raise_on_error=True,
            )
        except Exception as e:
            logger.error(f"[LLMService.generate_general_controlled] Error: {e}", exc_info=True)
            return _fallback

    # ──────────────────────────────────────────────────────────────
    # Filtrador de programas por tema / perfil
    # ──────────────────────────────────────────────────────────────

    _FILTER_TIMEOUT_SECONDS = float(_os.getenv("FILTER_TIMEOUT", "30.0"))

    _FILTER_PROMPT = FILTER_PROGRAMS

    def filter_programs_by_topic(
        self,
        programs: List[Dict],
        topic: str,
    ) -> List[Dict]:
        """
        Usa el LLM para filtrar los programas relevantes dado un tema o perfil.

        El LLM recibe la lista de programas (SNIES + nombre) y el topic, y devuelve
        solo los SNIES que considera relevantes. Los SNIES se validan contra la lista
        original para evitar alucinaciones.

        Fallback: si el LLM falla o supera el timeout, retorna lista vacía.
        """
        if not programs or not topic:
            return []

        programs_list = "\n".join(
            f"SNIES {p['snies']}: {p['programName']}"
            + (f" [{p['division']}]" if p.get("division") else "")
            for p in programs
        )
        prompt_text = self._FILTER_PROMPT.format(
            programs_list=programs_list,
            topic=topic.strip()[:300],
        )

        llm_bound = self.llm.bind(
            options={
                "num_predict": 80,
                "top_k": 1,
                "top_p": 1.0,
                "temperature": 0.0,
            },
            stop=["\n", "---"],
        )

        try:
            def _invoke():
                r = llm_bound.invoke(prompt_text)
                return getattr(r, "content", str(r)).strip()

            answer = self._call_with_timeout(
                _invoke, self._FILTER_TIMEOUT_SECONDS, "filter_programs_by_topic", fallback="",
            )

            if not answer or answer.upper().startswith("NING"):
                return []

            # Validar SNIES contra la lista original para evitar alucinaciones
            valid_snies = {p["snies"] for p in programs}
            returned_snies = {
                s.strip() for s in re.split(r"[,\s]+", answer)
                if s.strip().isdigit()
            }
            matched_snies = returned_snies & valid_snies

            result = [p for p in programs if p["snies"] in matched_snies]

            logger.debug(
                "[filter_programs_by_topic] topic=%r llm_answer=%r matched=%d/%d",
                topic[:60], answer[:80], len(result), len(programs),
            )
            return result

        except Exception as e:
            logger.error(
                "[filter_programs_by_topic] Error inesperado: %s", e, exc_info=True)
            return []

    # ──────────────────────────────────────────────────────────────
    # Extractor de intención y filtros
    # ──────────────────────────────────────────────────────────────

    def extract_list_intent_and_filters(
        self, question: str, chat_session_id: str
    ) -> Dict[str, Any]:
        q = (question or "").strip()
        ql = q.lower()

        type_like = ""
        if re.search(r"\bdoctorad", ql):
            type_like = "doctorado"
        elif re.search(r"\bmaestr", ql):
            type_like = "maestria"
        elif re.search(r"\bespecializ", ql):
            type_like = "especializacion"

        extractor_prompt = EXTRACT_LIST_INTENT.format(question=q)

        llm_bound = self.llm.bind(
            options={"num_predict": 120, "top_k": 30, "top_p": 0.9},
            stop=["---", "###"],
        )
        try:
            result = llm_bound.invoke(extractor_prompt)
            text = getattr(result, "content", str(result)).strip()
        except Exception as e:
            logger.error(
                f"[extract_list_intent_and_filters] Error: {e}", exc_info=True)
            return {"intent": "OTHER", "filters": {}, "confidence": 0.0, "raw": ""}

        def pick(tag: str) -> str:
            m = re.search(rf"^{tag}\s*:\s*(.*)$", text,
                          flags=re.IGNORECASE | re.MULTILINE)
            return m.group(1).strip() if m else ""

        intent = pick("INTENT").upper()
        filters_line = pick("FILTERS")
        conf_s = pick("CONFIDENCE")

        if intent not in {"LIST_PROGRAMS", "RECOMMEND", "OTHER"}:
            intent = "OTHER"

        filters: Dict[str, str] = {}
        empty_values = {"(vacío)", "(vacio)", "vacio", "vacío", "none", ""}
        if filters_line and filters_line.lower() not in empty_values:
            for part in filters_line.split(";"):
                part = part.strip()
                if "=" in part:
                    k, v = part.split("=", 1)
                    k, v = k.strip().lower(), v.strip()
                    if k in {"division_like", "modality_like", "location_like", "name_like"} and v:
                        filters[k] = v
        if type_like:
            filters["type_like"] = type_like

        try:
            confidence = max(0.0, min(1.0, float(conf_s)))
        except Exception:
            confidence = 0.5 if intent != "OTHER" else 0.0

        return {"intent": intent, "filters": filters, "confidence": confidence, "raw": text}
