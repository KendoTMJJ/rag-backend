import logging
import os
import re
import time
from typing import Optional, Dict, Any, Tuple, Generic, TypeVar
from dataclasses import dataclass

from src.quality.request_gate import validate_request_gate
from src.services.retrieval_service import RetrievalService
from src.services.sql_retrieval_service import SQLRetrievalService
from src.services.llm_service import LLMService

from src.nlp.text_normalizer import normalize_and_fix
from src.nlp.intent_utils import (
    looks_like_programs_listing,
    looks_like_program_overview,
    looks_like_general_question,
    is_false_listing,
    is_reasoning_question,
    is_global_comparison,
    resolve_minmax_mode,
    extract_snies,
    extract_semester,
    extract_program_candidate,
    extract_topic_for_listing,
    detect_field,
)

from src.nlp.input_sanitizer import sanitize
from src.nlp.domain_guardrail import check_domain
from src.quality.context_reranker import rerank_top
from src.quality.output_validator import validate_output, CONFIDENCE_SHOW, CONFIDENCE_BLOCK
from src.services.retrieval_service import PROGRAM_RESOLVE_MIN_SIM

logger = logging.getLogger(__name__)


@dataclass
class RequestAnalysis:
    q_norm: str
    semester: Optional[int]
    field: Optional[str]
    narrative_field: Optional[str]
    is_listing: bool
    is_overview: bool
    is_general: bool
    is_reasoning: bool
    is_false_listing: bool
    is_global_comparison: bool
    comparison_mode: str
    asks_curriculum: bool
    asks_electives: bool
    asks_degree_options: bool
    asks_inscription: bool
    topic_for_listing: Optional[str]

    guard_allowed: bool
    guard_reason: str
    guard_detail: str

    has_structured_academic_intent: bool
    should_run_classifier: bool

    has_program_reference: bool
    can_use_memory_for_program_resolution: bool

    gate_is_grounded: bool = False
    gate_can_use_memory: bool = False
    gate_can_resolve_program: bool = False
    gate_can_run_vector_search: bool = True
    gate_should_ask_program: bool = False
    gate_should_block: bool = False
    gate_reason: str = ""

    guard_confidence: float = 0.0
    guard_category: str = "unknown"


_SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", 7200))
_MAX_SESSIONS = int(os.getenv("SESSION_MAX_SIZE", 500))

_DEFAULT_RAG_SECTIONS = [
    "program_name", "info_general", "division",
    "perfil_ingreso", "perfil_egresado", "perfil_ocupacional",
    "diferencial", "requisitos", "descripcion",
]

_OUT_OF_DOMAIN_MSG = (
    "Solo puedo responder preguntas relacionadas con los programas de posgrado "
    "de la Universidad Santo Tomás Seccional Tunja. "
    "¿Hay algo sobre nuestros programas en lo que pueda ayudarte?"
)

# FIX D15/G3: keywords que bypasean el clasificador LLM porque son
# claramente relacionadas con posgrados aunque la frase sea ambigua
_ACADEMIC_BYPASS_KEYWORDS = frozenset({
    "posgrado", "posgrados", "maestria", "maestrias", "especializacion",
    "especializaciones", "doctorado", "doctorados", "especializarme",
    "postgrado", "postgrados",
})

# Sustantivos claramente fuera del dominio académico.
# Si aparecen en la pregunta, el clasificador SIEMPRE corre —
# incluso con sesión activa — para evitar que la memoria del programa
# responda preguntas como "y cuánto cuesta un pollo".
_OFFDOMAIN_NOUNS = frozenset({
    # Comida / animales
    "pollo", "pizza", "hamburguesa", "empanada", "arepa", "perro", "gato",
    "cerdo", "res", "vaca", "pez", "pescado", "arroz", "sopa", "carne",
    "fruta", "verdura", "bebida", "cerveza", "vino", "cafe", "jugo",
    # Personas / familia
    "hijo", "hija", "bebe", "nino", "nina", "esposa", "esposo",
    "mama", "papa", "abuelo", "abuela", "embarazo", "parto",
    # Entretenimiento / tecnología cotidiana
    "pelicula", "serie", "cancion", "album", "concierto", "videojuego",
    "netflix", "spotify", "youtube", "instagram", "tiktok",
    "celular", "telefono", "carro", "moto", "bicicleta", "bus",
    # Finanzas personales no académicas
    "bitcoin", "crypto", "dolar", "euro", "forex", "accion", "nft",
    # Otros claramente OOD
    "vacuna", "medicamento", "enfermedad", "receta", "restaurante",
    "hotel", "viaje", "vuelo", "pasaporte", "visa",
})

V = TypeVar("V")


class TTLStore(Generic[V]):
    def __init__(self, ttl: int = _SESSION_TTL, maxsize: int = _MAX_SESSIONS):
        self._ttl = ttl
        self._maxsize = maxsize
        self._data: Dict[str, V] = {}
        self._ts: Dict[str, float] = {}

    def _evict(self) -> None:
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

    def get(self, key: str, default: Any = None) -> Any:
        now = time.monotonic()
        ts = self._ts.get(key)
        if ts is None or (now - ts > self._ttl):
            self._data.pop(key, None)
            self._ts.pop(key, None)
            return default
        return self._data.get(key, default)

    def set(self, key: str, value: V) -> None:
        self._evict()
        self._data[key] = value
        self._ts[key] = time.monotonic()

    def pop(self, key: str, default: Any = None) -> Any:
        self._ts.pop(key, None)
        return self._data.pop(key, default)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


class RAGPipeline:
    def __init__(self):
        self.vector = RetrievalService()
        self.sql = SQLRetrievalService()

        self._snies_store: TTLStore[str] = TTLStore()
        self._pending_field_store: TTLStore[str] = TTLStore()
        self._pending_narrative_store: TTLStore[str] = TTLStore()
        self._pending_overview_store: TTLStore[bool] = TTLStore()
        self._pending_tabular_store: TTLStore[str] = TTLStore()

        self.llm = LLMService(
            base_url=(os.getenv("OLLAMA_BASE_URL") or "").strip(),
            model=(os.getenv("OLLAMA_MODEL")
                   or "llama3.1:8b-instruct-q8_0").strip(),
        )
        self.catalog_url = os.getenv(
            "CATALOG_POSGRADOS_URL",
            "https://santototunja.edu.co/programas-academicos/programas/posgrados-presenciales",
        )

    # ─────────────────────────────────────────
    # Historial
    # ─────────────────────────────────────────

    def _assistant_text_from_payload(self, payload: Dict[str, Any]) -> str:
        data = payload.get("data") or {}
        route = payload.get("route", "")
        for key in ("answer", "message", "text"):
            if isinstance(data.get(key), str):
                return data[key]
        if isinstance(data.get("items"), list):
            n = len(data["items"])
            topic = ((data.get("filters") or {}).get("topic") or "").strip()
            return f"Mostré {n} programas{(' sobre ' + topic) if topic else ''}."
        if isinstance(data.get("catalogUrl"), str):
            return f"Compartí el catálogo oficial: {data['catalogUrl']}"
        return f"(Respuesta enviada por ruta {route})."

    def _return_no_llm(
        self, chat_session_id: str, user_text: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        self.llm.append_to_history(chat_session_id, "user", user_text)
        self.llm.append_to_history(
            chat_session_id,
            "assistant",
            self._assistant_text_from_payload(payload),
        )
        return payload

    # ─────────────────────────────────────────
    # Memoria SNIES
    # ─────────────────────────────────────────

    def _get_active_snies(self, sid: str) -> Optional[str]:
        return self._snies_store.get(sid)

    def _set_active_snies(self, sid: str, snies: str) -> None:
        self._snies_store.set(sid, str(snies).strip())

    # ─────────────────────────────────────────
    # Pendientes
    # ─────────────────────────────────────────

    def _set_pending_field(self, sid: str, v: str) -> None:
        self._pending_field_store.set(sid, v)

    def _get_pending_field(self, sid: str) -> Optional[str]:
        return self._pending_field_store.get(sid)

    def _clear_pending_field(self, sid: str) -> None:
        self._pending_field_store.pop(sid)

    def _set_pending_narrative(self, sid: str, v: str) -> None:
        self._pending_narrative_store.set(sid, v)

    def _get_pending_narrative(self, sid: str) -> Optional[str]:
        return self._pending_narrative_store.get(sid)

    def _clear_pending_narrative(self, sid: str) -> None:
        self._pending_narrative_store.pop(sid)

    def _set_pending_overview(self, sid: str) -> None:
        self._pending_overview_store.set(sid, True)

    def _get_pending_overview(self, sid: str) -> bool:
        return bool(self._pending_overview_store.get(sid, False))

    def _clear_pending_overview(self, sid: str) -> None:
        self._pending_overview_store.pop(sid)

    def _set_pending_tabular(self, sid: str, v: str) -> None:
        self._pending_tabular_store.set(sid, v)

    def _get_pending_tabular(self, sid: str) -> Optional[str]:
        return self._pending_tabular_store.get(sid)

    def _clear_pending_tabular(self, sid: str) -> None:
        self._pending_tabular_store.pop(sid)

    def _clear_all_pending(self, sid: str) -> None:
        self._clear_pending_field(sid)
        self._clear_pending_narrative(sid)
        self._clear_pending_overview(sid)
        self._clear_pending_tabular(sid)

    # ─────────────────────────────────────────
    # Detectores auxiliares
    # ─────────────────────────────────────────

    def _detect_narrative_field(self, q_norm: str) -> Optional[str]:
        q = q_norm or ""
        if (
            ("perfil" in q and ("ingreso" in q or "admis" in q))
            or ("que perfil" in q and ("entrar" in q or "ingresar" in q))
            or any(k in q for k in [
                "tipo de estudiante", "buscan", "busca el programa",
                "quien puede entrar", "quien puede ingresar",  # FIX C16
                "que tipo de profesional",
                "requisito para ser admitido",
                "que necesito para entrar", "que se necesita para ingresar",  # FIX D10
                "necesito para entrar", "necesito para ingresar",
            ])
        ):
            return "admission_profile"

        if "perfil" in q and "egres" in q:
            return "graduated_profile"

        if any(k in q for k in [
            "competencias", "habilidades al salir", "que aprendo", "que aprende",
            "enfoque del programa", "que se estudia", "perfil de egreso", "perfil egreso",
            "habilidades adquiero", "habilidades que adquiero",  # FIX C17
            "que habilidades", "habilidades obtengo", "habilidades desarrollo",
        ]):
            return "graduated_profile"

        if any(k in q for k in [
            "ocupacional", "campo laboral", "salidas",
            "perfil profesional", "salida profesional",
            "salida laboral", "campo profesional",
            "perfil ocupacional", "ocupacion", "laboral",
            "en que puedo trabajar", "donde puedo trabajar",
            "en que empresas puedo trabajar", "en que empresa puedo trabajar",
            "que trabajo", "que empleo", "campo de trabajo",
            "mercado laboral", "oportunidades laborales",
            "al salir en que puedo trabajar", "al salir donde puedo trabajar",
            "al egresar en que puedo trabajar", "salidas laborales",
            "conseguir trabajo", "para conseguir trabajo",  # FIX D15
            "sirve para trabajar", "sirve para conseguir",
        ]):
            return "occupational_profile"

        if "diferencial" in q or "por que estudiar" in q:
            return "differential"

        if any(k in q for k in [
            "que lo hace diferente", "que hace diferente", "hace diferente",
            "que tiene de especial", "por que elegir", "ventajas", "valor agregado",
        ]):
            return "differential"

        if "requisitos especificos" in q:
            return "specific_requirements"

        if any(k in q for k in [
            "que piden adicional", "documentos adicionales",
            "que piden para entrar", "requisitos de admision",
        ]):
            return "specific_requirements"

        return None

    def has_program_mention(self, q_original: str) -> bool:
        ql = (q_original or "").lower()
        if re.search(r"\b\d{5,10}\b", ql):
            return True
        if re.search(r"\b(maestr\w*|especializ\w*|doctorad\w*|posgrado\w*)\s+(en|de|sobre)\s+\S", ql):
            return True
        if re.search(r"\bprograma\s+(de|en|para)\s+\S", ql):
            return True
        return False

    def _looks_like_program_title(self, q_norm: str) -> bool:
        if not q_norm:
            return False
        if not re.match(r"^(maestria|especializacion|doctorado)\b", q_norm):
            return False
        if re.match(
            r"^(cuanto|cuantos|que\s|cual|como\s|donde\s|hay\s|tienen|ofrecen|puedes|puede|explica|habla\s)",
            q_norm,
        ):
            return False
        if re.search(r"\b(costo|precio|valor|duracion|creditos|modalidad|sede|horario|titulo)\b", q_norm):
            return False
        return bool(re.fullmatch(r"(maestria|especializacion|doctorado)\s+.{4,}", q_norm))

    def _detect_tabular_intent(self, q_norm: str) -> Dict[str, bool]:
        asks_electives = any(w in q_norm for w in [
            "electiv", "hay electivas", "electivas de", "optativas", "materias opcionales",
        ])

        _semester_detected = extract_semester(q_norm) is not None
        asks_curriculum = (
            not asks_electives
            and (
                _semester_detected
                or any(w in q_norm for w in [
                    "malla", "plan de estudios", "materias", "asignaturas", "pensum",
                    "que asignaturas", "que cursos", "que ven en", "que estudian en", "que se ve en semestre",
                ])
            )
        )

        asks_degree_options = (
            "opcion de grado" in q_norm
            or "opciones de grado" in q_norm
            or "opciones para graduarme" in q_norm
            or "formas de graduarme" in q_norm
            or "formas de grado" in q_norm
            or "alternativas de grado" in q_norm
            or "como me puedo graduar" in q_norm
            or "como puedo graduarme" in q_norm
            or "como graduarse" in q_norm
            or ("graduar" in q_norm and "como" in q_norm)
        )

        return {
            "asks_curriculum": asks_curriculum,
            "asks_electives": asks_electives,
            "asks_degree_options": asks_degree_options,
        }

    def _detect_inscription_intent(self, q_norm: str, chat_session_id: str) -> bool:
        temporal_words = ["cuando", "fecha", "plazo",
                          "apertura", "abren", "cierran", "calendario"]
        active_snies_now = self._get_active_snies(chat_session_id)

        return (
            any(w in q_norm for w in [
                "inscrip", "inscripcion", "inscrib", "proceso", "requisitos", "requisit",
                "document", "papel", "matricul", "como me inscribo", "pasos de inscripcion",
            ])
            and not any(w in q_norm for w in temporal_words)
            and not (
                active_snies_now and len(q_norm.split()) <= 6
                and not any(w in q_norm for w in ["inscrip", "inscrib", "matricul", "pasos de inscripcion"])
            )
        )

    def _should_run_classifier(
        self,
        q_norm: str,
        chat_session_id: str,
        has_structured_academic_intent: bool,
        field: Optional[str],
    ) -> bool:
        active_for_clf = self._get_active_snies(chat_session_id)
        q_tokens = q_norm.split()

        # Si la pregunta contiene keywords académicos claros → nunca clasificar
        has_academic_bypass = any(
            w in q_norm for w in _ACADEMIC_BYPASS_KEYWORDS)

        # FIX "y cuánto cuesta un pollo":
        # Si la pregunta contiene un sustantivo claramente off-domain,
        # SIEMPRE correr el clasificador — incluso con sesión activa.
        # Evita que la memoria del programa responda preguntas trampa.
        has_offdomain_noun = any(noun in q_norm for noun in _OFFDOMAIN_NOUNS)
        if has_offdomain_noun and not has_academic_bypass:
            return True  # forzar clasificador sin importar sesión

        # Si hay campo detectado pero sin sesión activa y sin contexto académico,
        # también correr clasificador (fix I1/I2/I6)
        field_without_academic_context = (
            field is not None
            and not active_for_clf
            and not has_academic_bypass
            and not self.has_program_mention(q_norm)
        )

        skip_clf = (
            len(q_tokens) <= 1
            or (active_for_clf and len(q_tokens) <= 5)
            or has_academic_bypass
            or (has_structured_academic_intent and not field_without_academic_context)
        )
        return not skip_clf

    def _can_use_memory_for_program_resolution(
        self,
        q_norm: str,
        field: Optional[str],
        narrative_field: Optional[str],
        is_overview: bool,
        asks_curriculum: bool,
        asks_electives: bool,
        asks_degree_options: bool,
        asks_inscription: bool,
    ) -> bool:
        q = (q_norm or "").strip().lower()

        if self.has_program_mention(q):
            return False

        short_followup_patterns = [
            r"^(y\s+)?cuanto\s+cuesta\??$",
            r"^(y\s+)?cuanto\s+dura\??$",
            r"^(y\s+)?cuantos?\s+creditos\s+tiene\??$",
            r"^(y\s+)?es\s+(virtual|presencial|semipresencial)\??$",
            r"^(y\s+)?que\s+horario\s+tiene\??$",
            r"^(y\s+)?donde\s+se\s+dicta\??$",
            r"^(y\s+)?en\s+que\s+sede\s+se\s+dicta\??$",
            r"^(y\s+)?que\s+titulo\s+otorga\??$",
            r"^(y\s+)?que\s+registro\s+tiene\??$",
            r"^(y\s+)?dame\s+un\s+resumen(\s+general)?\??$",
            r"^(y\s+)?de\s+que\s+trata\??$",
            r"^(y\s+)?malla(\s+curricular)?\??$",
            r"^(y\s+)?pensum\??$",
            r"^(y\s+)?electivas\??$",
            r"^(y\s+)?opciones\s+de\s+grado\??$",
            r"^(y\s+)?como\s+me\s+puedo\s+graduar\??$",
            r"^(y\s+)?como\s+puedo\s+graduarme\??$",
            r"^(y\s+)?requisitos\??$",
            r"^(y\s+)?como\s+me\s+inscribo\??$",
        ]

        if any(re.fullmatch(p, q) for p in short_followup_patterns):
            return True

        tokens = q.split()
        if len(tokens) <= 7 and any([
            field is not None,
            narrative_field is not None,
            is_overview,
            asks_curriculum,
            asks_electives,
            asks_degree_options,
            asks_inscription,
        ]):
            return True

        return False

    # ─────────────────────────────────────────
    # Resolución de programa
    # ─────────────────────────────────────────

    def _ensure_program(
        self, question: str, chat_session_id: str, *,
        allow_embedding: bool,
        allow_memory_fallback: bool = True,
        min_sim_embedding: float = PROGRAM_RESOLVE_MIN_SIM["default"],
    ) -> Tuple[Optional[str], str]:
        snies = extract_snies(question)
        if snies:
            self._set_active_snies(chat_session_id, snies)
            return str(snies).strip(), "explicit"

        full_text = question.strip()
        if len(full_text) >= 6:
            found = self.sql.resolve_program_by_name(full_text)
            sn = found.get("snies") if found else None
            if sn:
                self._set_active_snies(chat_session_id, sn)
                return str(sn).strip(), "sql_full"

        cand = extract_program_candidate(question)
        if cand:
            found = self.sql.resolve_program_by_name(cand)
            sn = found.get("snies") if found else None
            if sn:
                self._set_active_snies(chat_session_id, sn)
                return str(sn).strip(), "sql_name"

        if allow_embedding:
            for q in filter(None, [full_text, cand]):
                resolved = self.vector.resolve_program(
                    q, min_similarity=min_sim_embedding)
                sn = resolved.snies if resolved else None
                if sn:
                    self._set_active_snies(chat_session_id, sn)
                    return sn, "embedding"

        if allow_memory_fallback:
            active = self._get_active_snies(chat_session_id)
            if active:
                self._set_active_snies(chat_session_id, active)
                return str(active).strip(), "memory"

        return None, "none"

    def _try_set_active_program_from_title(
        self, question: str, chat_session_id: str
    ) -> Optional[Dict[str, Any]]:
        found = self.sql.resolve_program_by_name(question.strip())
        sn = found.get("snies") if found else None
        if not sn:
            cand = extract_program_candidate(question)
            if cand:
                found = self.sql.resolve_program_by_name(cand)
                sn = found.get("snies") if found else None
        if not sn:
            resolved = self.vector.resolve_program(
                question.strip(),
                min_similarity=PROGRAM_RESOLVE_MIN_SIM["title_select"],
            )
            sn = resolved.snies if resolved else None
        if not sn:
            return None

        snies = str(sn).strip()
        self._set_active_snies(chat_session_id, snies)

        pending_field = self._get_pending_field(chat_session_id)
        if pending_field:
            ans = self.sql.get_program_field(snies, pending_field)
            self._clear_pending_field(chat_session_id)
            if ans:
                return {"route": "PROGRAM_FIELD", "data": {"snies": snies, "source": "title_select", "field": pending_field, "text": ans, "resolved": True}}
            return {"route": "NOT_FOUND", "data": {"message": "Identifiqué el programa, pero no tengo ese dato cargado.", "resolved": False}}

        pending_narr = self._get_pending_narrative(chat_session_id)
        if pending_narr:
            text = self.sql.get_program_narrative_field(snies, pending_narr)
            self._clear_pending_narrative(chat_session_id)
            if text:
                return {"route": "NARRATIVE_SQL", "data": {"snies": snies, "source": "title_select", "field": pending_narr, "text": text, "resolved": True}}
            return {"route": "NARRATIVE_NOT_FOUND", "data": {"snies": snies, "source": "title_select", "field": pending_narr, "message": "No tengo ese campo narrativo cargado.", "resolved": False}}

        if self._get_pending_overview(chat_session_id):
            self._clear_pending_overview(chat_session_id)
            return {"route": "PROGRAM_SELECTED", "data": {"snies": snies, "message": "Listo. Ahora pregúntame: ¿De qué trata el programa?", "resolved": True}}

        pending_tab = self._get_pending_tabular(chat_session_id)
        if pending_tab:
            self._clear_pending_tabular(chat_session_id)
            if pending_tab == "curriculum":
                return {"route": "CURRICULUM", "data": {"snies": snies, "source": "title_select", "text": self.sql.get_curriculum(snies), "resolved": True}}
            if pending_tab == "electives":
                return {"route": "ELECTIVES", "data": {"snies": snies, "source": "title_select", "text": self.sql.get_electives(snies), "resolved": True}}
            if pending_tab == "degree_options":
                return {"route": "DEGREE_OPTIONS", "data": {"snies": snies, "source": "title_select", "text": self.sql.get_degree_options(snies), "resolved": True}}

        return {"route": "PROGRAM_SELECTED", "data": {"snies": snies, "message": "Listo. ¿Qué información necesitas (duración, costo, créditos, malla, etc.)?", "resolved": True}}

    # ─────────────────────────────────────────
    # Análisis centralizado
    # ─────────────────────────────────────────

    def _analyze_request(
        self,
        question: str,
        q_norm: str,
        chat_session_id: str,
    ) -> RequestAnalysis:
        semester = extract_semester(q_norm)
        field = detect_field(q_norm)
        narrative_field = self._detect_narrative_field(q_norm)

        is_reasoning = is_reasoning_question(q_norm)
        is_false_list = is_false_listing(q_norm)
        is_listing = looks_like_programs_listing(q_norm)
        is_overview = looks_like_program_overview(q_norm)
        is_general = looks_like_general_question(q_norm)
        is_global_cmp = is_global_comparison(q_norm)
        cmp_mode = resolve_minmax_mode(q_norm)

        tabular = self._detect_tabular_intent(q_norm)
        asks_inscription = self._detect_inscription_intent(
            q_norm, chat_session_id)

        topic = extract_topic_for_listing(question) if (
            is_listing and field is None) else None

        guard = check_domain(q_norm)

        has_structured_academic_intent = any([
            field is not None,
            narrative_field is not None,
            is_overview,
            tabular["asks_curriculum"],
            tabular["asks_electives"],
            tabular["asks_degree_options"],
            asks_inscription,
            is_listing,
            is_general,
        ])

        # FIX I1/I2/I4/I5/I6: pasar field a _should_run_classifier
        should_run_classifier = self._should_run_classifier(
            q_norm=q_norm,
            chat_session_id=chat_session_id,
            has_structured_academic_intent=has_structured_academic_intent,
            field=field,
        )

        has_program_reference = self.has_program_mention(question)

        can_use_memory_for_program_resolution = self._can_use_memory_for_program_resolution(
            q_norm=q_norm,
            field=field,
            narrative_field=narrative_field,
            is_overview=is_overview,
            asks_curriculum=tabular["asks_curriculum"],
            asks_electives=tabular["asks_electives"],
            asks_degree_options=tabular["asks_degree_options"],
            asks_inscription=asks_inscription,
        )

        return RequestAnalysis(
            q_norm=q_norm,
            semester=semester,
            field=field,
            narrative_field=narrative_field,
            is_listing=is_listing,
            is_overview=is_overview,
            is_general=is_general,
            is_reasoning=is_reasoning,
            is_false_listing=is_false_list,
            is_global_comparison=is_global_cmp,
            comparison_mode=cmp_mode,
            asks_curriculum=tabular["asks_curriculum"],
            asks_electives=tabular["asks_electives"],
            asks_degree_options=tabular["asks_degree_options"],
            asks_inscription=asks_inscription,
            topic_for_listing=topic,
            guard_allowed=guard.allowed,
            guard_reason=guard.reason,
            guard_detail=guard.detail,
            has_structured_academic_intent=has_structured_academic_intent,
            should_run_classifier=should_run_classifier,
            has_program_reference=has_program_reference,
            can_use_memory_for_program_resolution=can_use_memory_for_program_resolution,
        )

    # ─────────────────────────────────────────
    # MAIN
    # ─────────────────────────────────────────

    def ask(self, question: str, chat_session_id: str) -> Dict[str, Any]:
        sanitized = sanitize(question)
        question = sanitized.text

        if sanitized.has_sql_injection:
            logger.warning(
                "[SANITIZER] SQL injection pattern detected. session=%s snippet=%s",
                chat_session_id, question[:60],
            )

        q_norm = normalize_and_fix(question)
        if not q_norm:
            return self._return_no_llm(
                chat_session_id,
                question,
                {"route": "EMPTY", "data": {
                    "answer": "Por favor escribe tu pregunta.", "resolved": False}},
            )

        # Atajos triviales — antes de cualquier análisis semántico
        greetings = {"hola", "buenas", "hey", "hi", "buenos dias",
                     "buenas tardes", "buenas noches", "que tal"}
        if q_norm in greetings:
            return self._return_no_llm(chat_session_id, question,
                                       {"route": "GREETING", "data": {"resolved": True}})

        # FIX H3: "gracias por la información" → THANKS (match por prefijo)
        if q_norm in {"gracias", "muchas gracias", "thanks"} or q_norm.startswith("gracias"):
            return self._return_no_llm(chat_session_id, question,
                                       {"route": "THANKS", "data": {"resolved": True}})

        analysis = self._analyze_request(question, q_norm, chat_session_id)
        semester = analysis.semester

        if not analysis.guard_allowed:
            route = "INJECTION_BLOCKED" if analysis.guard_reason == "injection" else "OUT_OF_DOMAIN"
            logger.info(
                "[GUARDRAIL] blocked. reason=%s detail=%s session=%s",
                analysis.guard_reason, analysis.guard_detail, chat_session_id,
            )
            return self._return_no_llm(
                chat_session_id,
                question,
                {"route": route, "data": {
                    "answer": _OUT_OF_DOMAIN_MSG,
                    "reason": analysis.guard_reason,
                    "resolved": False,
                }},
            )

        bare_snies = re.fullmatch(r"snies\s+(\d{5,10})", q_norm)
        if bare_snies:
            snies_num = bare_snies.group(1)
            if self.sql._get_program_id_by_snies(snies_num):
                self._set_active_snies(chat_session_id, snies_num)

                pending_field = self._get_pending_field(chat_session_id)
                if pending_field:
                    ans = self.sql.get_program_field(snies_num, pending_field)
                    self._clear_pending_field(chat_session_id)
                    if ans:
                        return self._return_no_llm(chat_session_id, question,
                                                   {"route": "PROGRAM_FIELD", "data": {"snies": snies_num, "source": "explicit", "field": pending_field, "text": ans, "resolved": True}})
                    return self._return_no_llm(chat_session_id, question,
                                               {"route": "NOT_FOUND", "data": {"message": "Identifiqué el programa, pero no tengo ese dato cargado.", "resolved": False}})

                pending_narr = self._get_pending_narrative(chat_session_id)
                if pending_narr:
                    text = self.sql.get_program_narrative_field(
                        snies_num, pending_narr)
                    self._clear_pending_narrative(chat_session_id)
                    if text:
                        return self._return_no_llm(chat_session_id, question,
                                                   {"route": "NARRATIVE_SQL", "data": {"snies": snies_num, "source": "explicit", "field": pending_narr, "text": text, "resolved": True}})
                    return self._return_no_llm(chat_session_id, question,
                                               {"route": "NARRATIVE_NOT_FOUND", "data": {"snies": snies_num, "source": "explicit", "field": pending_narr, "message": "No tengo ese campo narrativo cargado.", "resolved": False}})

                pending_tab = self._get_pending_tabular(chat_session_id)
                if pending_tab:
                    self._clear_pending_tabular(chat_session_id)
                    if pending_tab == "curriculum":
                        text = self.sql.get_curriculum(snies_num)
                        return self._return_no_llm(chat_session_id, question,
                                                   {"route": "CURRICULUM", "data": {"snies": snies_num, "source": "explicit", "text": text, "resolved": text is not None}})
                    if pending_tab == "electives":
                        text = self.sql.get_electives(snies_num)
                        return self._return_no_llm(chat_session_id, question,
                                                   {"route": "ELECTIVES", "data": {"snies": snies_num, "source": "explicit", "text": text, "resolved": text is not None}})
                    if pending_tab == "degree_options":
                        text = self.sql.get_degree_options(snies_num)
                        return self._return_no_llm(chat_session_id, question,
                                                   {"route": "DEGREE_OPTIONS", "data": {"snies": snies_num, "source": "explicit", "text": text, "resolved": text is not None}})

                if self._get_pending_overview(chat_session_id):
                    self._clear_pending_overview(chat_session_id)
                    return self._return_no_llm(chat_session_id, question,
                                               {"route": "PROGRAM_SELECTED", "data": {"snies": snies_num, "message": "Listo. Ahora pregúntame: ¿De qué trata el programa?", "resolved": True}})

                return self._return_no_llm(
                    chat_session_id,
                    question,
                    {"route": "PROGRAM_SELECTED", "data": {
                        "snies": snies_num,
                        "message": "Listo. ¿Qué información necesitas (duración, costo, créditos, malla, etc.)?",
                        "resolved": True}},
                )

        if analysis.should_run_classifier:
            if not self.llm.classify_domain(question):
                logger.info(
                    "[CLASSIFIER] NOT_ACADEMIC. session=%s q=%r",
                    chat_session_id, question[:60],
                )
                return self._return_no_llm(
                    chat_session_id,
                    question,
                    {"route": "OUT_OF_DOMAIN", "data": {
                        "answer": _OUT_OF_DOMAIN_MSG,
                        "reason": "classifier",
                        "resolved": False,
                    }},
                )

        if self._looks_like_program_title(q_norm):
            maybe = self._try_set_active_program_from_title(
                question, chat_session_id)
            if maybe:
                return self._return_no_llm(chat_session_id, question, maybe)

        narr_field = analysis.narrative_field
        if narr_field:
            snies = self._get_active_snies(chat_session_id)
            source = "memory"
            if not snies:
                snies, source = self._ensure_program(
                    question, chat_session_id,
                    allow_embedding=analysis.has_program_reference,
                    allow_memory_fallback=analysis.can_use_memory_for_program_resolution,
                    min_sim_embedding=PROGRAM_RESOLVE_MIN_SIM["narrative"],
                )
            if not snies:
                self._set_pending_narrative(chat_session_id, narr_field)
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "NEED_PROGRAM", "data": {"message": "¿De qué programa necesitas esa información? Dime el nombre o el SNIES.", "resolved": False}})
            self._set_active_snies(chat_session_id, snies)
            text = self.sql.get_program_narrative_field(snies, narr_field)
            if not text:
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "NARRATIVE_NOT_FOUND", "data": {"snies": snies, "source": source, "field": narr_field, "message": "No tengo ese campo narrativo cargado para este programa.", "resolved": False}})
            self._clear_pending_narrative(chat_session_id)
            return self._return_no_llm(chat_session_id, question,
                                       {"route": "NARRATIVE_SQL", "data": {"snies": snies, "source": source, "field": narr_field, "text": text, "resolved": True}})

        if analysis.is_overview:
            snies, source = self._ensure_program(
                question, chat_session_id,
                allow_embedding=analysis.has_program_reference,
                allow_memory_fallback=analysis.can_use_memory_for_program_resolution,
                min_sim_embedding=PROGRAM_RESOLVE_MIN_SIM["default"],
            )
            if not snies:
                self._set_pending_overview(chat_session_id)
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "NEED_PROGRAM", "data": {"message": "¿De qué programa necesitas esa información? Dime el nombre o el SNIES.", "resolved": False}})
            self._set_active_snies(chat_session_id, snies)

            ctx = self.sql.get_program_brief_context(snies)
            if not ctx:
                ctx = self.sql.get_program_overview_context(snies)

            if not ctx:
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "PROGRAM_OVERVIEW_NOT_FOUND", "data": {"snies": snies, "source": source, "message": "No tengo información narrativa cargada para ese programa.", "resolved": False}})
            try:
                answer = self.llm.generate_general_controlled(
                    context=ctx, question=question, chat_session_id=chat_session_id,
                )
            except Exception as e:
                logger.error(
                    f"[PROGRAM_OVERVIEW] Error LLM: {e}", exc_info=True)
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "LLM_ERROR", "data": {"message": "Tuve un problema procesando tu consulta. Intenta de nuevo.", "resolved": False}})

            validation = validate_output(
                llm_answer=answer,
                best_similarity=1.0,
                program_resolved=True,
                program_source=source,
                domain_allowed=analysis.guard_allowed,
            )
            logger.debug("[OUTPUT_VALIDATOR] PROGRAM_OVERVIEW confidence=%.3f flags=%s",
                         validation.confidence, validation.flags)

            return {"route": "PROGRAM_OVERVIEW", "data": {
                "snies": snies, "source": source, "answer": answer,
                "confidence": validation.confidence,
                "resolved": True,
            }}

        asks_curriculum = analysis.asks_curriculum
        asks_electives = analysis.asks_electives
        asks_degree_options = analysis.asks_degree_options
        if asks_curriculum or asks_electives or asks_degree_options:
            snies = self._get_active_snies(chat_session_id)
            source = "memory"
            if not snies:
                snies, source = self._ensure_program(
                    question, chat_session_id,
                    allow_embedding=analysis.has_program_reference,
                    allow_memory_fallback=analysis.can_use_memory_for_program_resolution,
                    min_sim_embedding=PROGRAM_RESOLVE_MIN_SIM["tabular"],
                )
            if not snies:
                if asks_curriculum:
                    self._set_pending_tabular(chat_session_id, "curriculum")
                elif asks_electives:
                    self._set_pending_tabular(chat_session_id, "electives")
                elif asks_degree_options:
                    self._set_pending_tabular(
                        chat_session_id, "degree_options")
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "NEED_PROGRAM", "data": {"message": "¿De qué programa necesitas esa información? Dime el nombre o el SNIES.", "resolved": False}})
            self._set_active_snies(chat_session_id, snies)
            if asks_curriculum and semester is not None:
                text = self.sql.get_curriculum_semester(snies, semester)
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "CURRICULUM_SEMESTER", "data": {"snies": snies, "source": source, "semester": semester, "text": text, "resolved": text is not None}})
            if asks_curriculum:
                text = self.sql.get_curriculum(snies)
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "CURRICULUM", "data": {"snies": snies, "source": source, "text": text, "resolved": text is not None}})
            if asks_electives:
                text = self.sql.get_electives(snies)
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "ELECTIVES", "data": {"snies": snies, "source": source, "text": text, "resolved": text is not None}})
            if asks_degree_options:
                text = self.sql.get_degree_options(snies)
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "DEGREE_OPTIONS", "data": {"snies": snies, "source": source, "text": text, "resolved": text is not None}})

        if analysis.asks_inscription:
            cand = extract_program_candidate(question)
            cand_looks_like_program = bool(cand) and len(
                cand.split()) > 1 and len(cand) > 8
            allow_emb = cand_looks_like_program or analysis.has_program_reference

            snies, source = self._ensure_program(
                question, chat_session_id,
                allow_embedding=allow_emb,
                allow_memory_fallback=analysis.can_use_memory_for_program_resolution,
                min_sim_embedding=PROGRAM_RESOLVE_MIN_SIM["inscription"],
            )
            if not snies:
                return self._return_no_llm(chat_session_id, question, {
                    "route": "INSCRIPTION_NEED_PROGRAM",
                    "data": {
                        "hint": "Para enviarte el enlace oficial necesito que me indiques el programa (o su SNIES).",
                        "example": "Ej: ¿Cómo me inscribo a la Maestría en Administración?",
                        "resolved": False,
                    },
                })
            info = self.sql.get_inscription_info(snies)
            if info:
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "INSCRIPTION_LINK", "data": {**info, "resolved": True}})
            return self._return_no_llm(chat_session_id, question, {
                "route": "INSCRIPTION_NO_LINK",
                "data": {"snies": snies, "source": source, "message": "Encontré el programa, pero no tengo el enlace de inscripción cargado.", "resolved": False},
            })

        field_cmp = analysis.field
        if field_cmp in ("duration", "credits", "cost") and analysis.is_global_comparison:
            mode = analysis.comparison_mode
            info = self.sql.get_program_minmax(field_cmp, mode=mode)
            if info:
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "PROGRAM_MINMAX", "data": {**info, "resolved": True}})
            return self._return_no_llm(chat_session_id, question,
                                       {"route": "NOT_FOUND", "data": {"message": "No tengo datos suficientes para comparar los programas en ese campo.", "resolved": False}})

        if analysis.is_listing and not analysis.is_false_listing and analysis.field is None:
            topic = analysis.topic_for_listing
            if not topic:
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "LIST_PROGRAMS", "data": {"catalogUrl": self.catalog_url, "resolved": True}})
            programs = self.sql.list_programs_filtered(
                division_like=None, modality_like=None, location_like=None,
                type_like=None, name_like=topic, limit=80,
            )
            if programs:
                return self._return_no_llm(chat_session_id, question, {
                    "route": "LIST_PROGRAMS_FILTERED",
                    "data": {"intent": "LIST_PROGRAMS", "filters": {"topic": topic}, "items": programs, "resolved": True},
                })
            return self._return_no_llm(chat_session_id, question, {
                "route": "LIST_PROGRAMS_FILTERED_EMPTY",
                "data": {
                    "intent": "LIST_PROGRAMS", "filters": {"topic": topic},
                    "message": f'No encontré programas que coincidan con "{topic}". Revisa el catálogo oficial:',
                    "catalogUrl": self.catalog_url,
                    "resolved": False,
                },
            })

        field = analysis.field
        if field is not None:
            snies, source = self._ensure_program(
                question, chat_session_id,
                allow_embedding=analysis.has_program_reference,
                allow_memory_fallback=analysis.can_use_memory_for_program_resolution,
                min_sim_embedding=PROGRAM_RESOLVE_MIN_SIM["default"],
            )
            if not snies:
                self._set_pending_field(chat_session_id, field)
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "NEED_PROGRAM", "data": {"message": "¿De qué programa necesitas esa información? Dime el nombre o el SNIES.", "resolved": False}})
            self._set_active_snies(chat_session_id, snies)
            ans = self.sql.get_program_field(snies, field)
            if ans:
                self._clear_pending_field(chat_session_id)
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "PROGRAM_FIELD", "data": {"snies": snies, "source": source, "field": field, "text": ans, "resolved": True}})
            return self._return_no_llm(chat_session_id, question,
                                       {"route": "NOT_FOUND", "data": {"message": "No encontré ese dato específico en mis documentos.", "resolved": False}})

        pending_field = self._get_pending_field(chat_session_id)
        pending_narr = self._get_pending_narrative(chat_session_id)
        if pending_field or pending_narr:
            snies, source = self._ensure_program(
                question, chat_session_id,
                allow_embedding=True,
                allow_memory_fallback=True,
                min_sim_embedding=PROGRAM_RESOLVE_MIN_SIM["block_4b"],
            )
            if not snies:
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "NEED_PROGRAM", "data": {"message": "No pude identificar el programa. Escríbeme el nombre completo o el SNIES.", "resolved": False}})
            self._set_active_snies(chat_session_id, snies)
            if pending_field:
                ans = self.sql.get_program_field(snies, pending_field)
                self._clear_pending_field(chat_session_id)
                if ans:
                    return self._return_no_llm(chat_session_id, question,
                                               {"route": "PROGRAM_FIELD", "data": {"snies": snies, "source": source, "field": pending_field, "text": ans, "resolved": True}})
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "NOT_FOUND", "data": {"message": "Identifiqué el programa, pero no tengo ese dato cargado.", "resolved": False}})
            if pending_narr:
                text = self.sql.get_program_narrative_field(
                    snies, pending_narr)
                self._clear_pending_narrative(chat_session_id)
                if text:
                    return self._return_no_llm(chat_session_id, question,
                                               {"route": "NARRATIVE_SQL", "data": {"snies": snies, "source": source, "field": pending_narr, "text": text, "resolved": True}})
                return self._return_no_llm(chat_session_id, question,
                                           {"route": "NARRATIVE_NOT_FOUND", "data": {"snies": snies, "source": source, "field": pending_narr, "message": "No tengo ese campo narrativo cargado.", "resolved": False}})

        if analysis.is_general:
            active = self._get_active_snies(chat_session_id)
            if active:
                self._set_active_snies(chat_session_id, active)
                ctx = self.sql.get_program_overview_context(active)
                if not ctx:
                    ctx = self.sql.get_program_brief_context(active)
                if ctx:
                    try:
                        answer = self.llm.generate_general_controlled(
                            context=ctx, question=question, chat_session_id=chat_session_id,
                        )
                    except Exception as e:
                        logger.error(
                            f"[GENERAL_CONTROLLED] Error LLM: {e}", exc_info=True)
                        return self._return_no_llm(chat_session_id, question,
                                                   {"route": "LLM_ERROR", "data": {"message": "Tuve un problema procesando tu consulta. Intenta de nuevo.", "resolved": False}})

                    validation = validate_output(
                        llm_answer=answer,
                        best_similarity=1.0,
                        program_resolved=True,
                        program_source="memory",
                        domain_allowed=analysis.guard_allowed,
                    )
                    logger.debug("[OUTPUT_VALIDATOR] GENERAL_CONTROLLED confidence=%.3f flags=%s",
                                 validation.confidence, validation.flags)

                    return {"route": "GENERAL_CONTROLLED", "data": {
                        "answer": answer,
                        "snies": active,
                        "confidence": validation.confidence,
                        "resolved": True,
                    }}
            safe = (
                "Con la información disponible no puedo confirmarlo con certeza. "
                "Si me indicas el programa exacto (o su SNIES), puedo darte una respuesta más precisa."
            )
            return self._return_no_llm(chat_session_id, question,
                                       {"route": "GENERAL_CONTROLLED", "data": {"answer": safe, "resolved": False}})

        active_snies = self._get_active_snies(chat_session_id)
        if active_snies:
            self._set_active_snies(chat_session_id, active_snies)

        vec_ctx = self.vector.semantic_search(
            question,
            sections=_DEFAULT_RAG_SECTIONS,
            limit=8,
            min_similarity=0.55,
            include_program_meta=True,
        )

        active_program_id: Optional[int] = None
        if active_snies:
            active_program_id = self.sql._get_program_id_by_snies(active_snies)

        if active_program_id:
            anchored = self.vector.semantic_search(
                question,
                sections=_DEFAULT_RAG_SECTIONS,
                limit=6,
                min_similarity=0.50,
                program_id=active_program_id,
                include_program_meta=True,
            )
            seen = {
                (r.get("program_id"), r.get("section"),
                 (r.get("content") or "")[:120])
                for r in vec_ctx
            }
            for r in (anchored or []):
                key = (r.get("program_id"), r.get("section"),
                       (r.get("content") or "")[:120])
                if key not in seen:
                    vec_ctx.append(r)
                    seen.add(key)

        if not vec_ctx:
            return self._return_no_llm(chat_session_id, question, {
                "route": "NOT_FOUND",
                "data": {"message": "No encontré información específica sobre eso en los documentos oficiales.", "resolved": False},
            })

        best_sim = max((r.get("similarity") or 0) for r in vec_ctx)

        if analysis.is_reasoning and best_sim < 0.80:
            return self._return_no_llm(chat_session_id, question, {
                "route": "LOW_CONFIDENCE",
                "data": {
                    "message": (
                        "No encontré información suficientemente confiable para responder esa comparación. "
                        "Te recomiendo consultar directamente con la facultad o revisar el catálogo oficial."
                    ),
                    "catalogUrl": self.catalog_url,
                    "confidence": round(best_sim, 3),
                    "resolved": False,
                },
            })

        if best_sim < 0.60:
            return self._return_no_llm(chat_session_id, question, {
                "route": "LOW_CONFIDENCE",
                "data": {
                    "message": (
                        "Tengo información relacionada pero no con certeza suficiente. "
                        "Si me indicas el programa exacto o el SNIES, puedo ayudarte mejor."
                    ),
                    "catalogUrl": self.catalog_url,
                    "confidence": round(best_sim, 3),
                    "resolved": False,
                },
            })

        vec_ctx = rerank_top(
            vec_ctx, active_program_id=active_program_id, top_k=6)

        ctx_lines = ["INFORMACION RECUPERADA:"]
        for item in vec_ctx:
            prog_label = (
                item.get("program_name")
                or (f"Programa {item['program_id']}" if item.get("program_id") else "Programa")
            )
            ctx_lines.append(
                f"[{prog_label} — {item.get('section', '')}]\n{item.get('content', '')}"
            )
        full_context = "\n\n".join(ctx_lines)

        if best_sim < 0.75:
            full_context = (
                "NOTA: El contexto tiene similitud moderada. "
                "Responde solo si la información está explícita; si no, usa la frase de no encontrado.\n\n"
                + full_context
            )

        try:
            answer = self.llm.generate(
                context=full_context, question=question, chat_session_id=chat_session_id,
            )
        except Exception as e:
            logger.error(f"[DEFAULT_RAG] Error LLM: {e}", exc_info=True)
            return self._return_no_llm(chat_session_id, question,
                                       {"route": "LLM_ERROR", "data": {"message": "Tuve un problema procesando tu consulta. Intenta de nuevo.", "resolved": False}})

        if not active_snies and vec_ctx:
            top_pid = vec_ctx[0].get("program_id")
            if top_pid:
                snies_from_ctx = self.sql.get_snies_for_program(top_pid)
                if snies_from_ctx:
                    self._set_active_snies(chat_session_id, snies_from_ctx)

        _prog_source = "memory" if active_snies else "none"
        validation = validate_output(
            llm_answer=answer,
            best_similarity=best_sim,
            program_resolved=bool(active_snies or (
                vec_ctx and vec_ctx[0].get("program_id"))),
            program_source=_prog_source,
            domain_allowed=analysis.guard_allowed,
            context_chunks=vec_ctx,
        )
        logger.debug("[OUTPUT_VALIDATOR] DEFAULT_RAG confidence=%.3f flags=%s hallucination=%s",
                     validation.confidence, validation.flags, validation.hallucination_suspected)

        if validation.hallucination_suspected:
            logger.warning("[OUTPUT_VALIDATOR] Hallucination suspected. session=%s flags=%s",
                           chat_session_id, validation.flags)

        if validation.add_disclaimer and not validation.should_block:
            answer = (
                answer + "\n\n_Nota: Esta respuesta está basada en la información disponible. "
                "Si necesitas confirmación oficial, te recomiendo contactar directamente a la División de Posgrados._"
            )

        return {"route": "DEFAULT_RAG", "data": {
            "answer": answer,
            "confidence": validation.confidence,
            "resolved": best_sim >= 0.75 and not validation.should_block,
        }}
