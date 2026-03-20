from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class RequestGateDecision:
    is_grounded: bool
    can_use_memory: bool
    can_resolve_program: bool
    can_run_vector_search: bool
    should_ask_program: bool
    should_block: bool
    reason: str


_ACADEMIC_KEYWORDS = (
    "snies", "programa", "programas", "posgrado", "posgrados",
    "maestr", "especializ", "doctorad",
    "credit", "duracion", "duración", "semestre", "semestres",
    "malla", "pensum", "asignaturas", "materias", "electiv",
    "opcion de grado", "opciones de grado",
    "inscrip", "admision", "admisión", "matricul",
    "modalidad", "sede", "campus", "horario", "horarios",
    "perfil", "egres", "ocupacional", "requisitos",
    "division", "división", "facultad",
)

_SHORT_FOLLOWUP_PATTERNS = [
    r"^(y\s+)?cuanto\s+cuesta\??$",
    r"^(y\s+)?cuanto\s+dura\??$",
    r"^(y\s+)?cuantos?\s+creditos\s+tiene\??$",
    r"^(y\s+)?que\s+horario\s+tiene\??$",
    r"^(y\s+)?donde\s+se\s+dicta\??$",
    r"^(y\s+)?en\s+que\s+sede\s+se\s+dicta\??$",
    r"^(y\s+)?que\s+titulo\s+otorga\??$",
    r"^(y\s+)?que\s+registro\s+tiene\??$",
    r"^(y\s+)?de\s+que\s+trata\??$",
    r"^(y\s+)?dame\s+un\s+resumen(\s+general)?\??$",
    r"^(y\s+)?malla(\s+curricular)?\??$",
    r"^(y\s+)?pensum\??$",
    r"^(y\s+)?electivas\??$",
    r"^(y\s+)?opciones\s+de\s+grado\??$",
    r"^(y\s+)?como\s+me\s+puedo\s+graduar\??$",
    r"^(y\s+)?como\s+puedo\s+graduarme\??$",
    r"^(y\s+)?requisitos\??$",
    r"^(y\s+)?como\s+me\s+inscribo\??$",
]

_OBVIOUS_OFFDOMAIN_HINTS = (
    "pizza", "pollo", "hamburguesa", "empanada", "maraton", "maratón",
    "restaurante", "receta", "netflix", "spotify", "minecraft",
    "bitcoin", "forex", "nft",
)


def _has_academic_keyword(q_norm: str) -> bool:
    q = q_norm or ""
    return any(k in q for k in _ACADEMIC_KEYWORDS)


def _looks_like_short_followup(q_norm: str) -> bool:
    q = (q_norm or "").strip()
    return any(re.fullmatch(p, q) for p in _SHORT_FOLLOWUP_PATTERNS)


def _looks_obviously_offdomain(q_norm: str) -> bool:
    q = q_norm or ""
    return any(k in q for k in _OBVIOUS_OFFDOMAIN_HINTS)


def _looks_like_program_candidate(program_candidate: Optional[str]) -> bool:
    cand = (program_candidate or "").strip().lower()
    if not cand:
        return False

    if len(cand) < 8:
        return False

    words = cand.split()
    if len(words) < 2:
        return False

    if any(x in cand for x in _OBVIOUS_OFFDOMAIN_HINTS):
        return False

    return True


def validate_request_gate(
    *,
    q_norm: str,
    has_active_program: bool,
    has_program_reference: bool,
    program_candidate: Optional[str],
    field: Optional[str],
    narrative_field: Optional[str],
    is_listing: bool,
    is_overview: bool,
    is_general: bool,
    asks_curriculum: bool,
    asks_electives: bool,
    asks_degree_options: bool,
    asks_inscription: bool,
) -> RequestGateDecision:
    """
    Decide si la consulta está suficientemente anclada al dominio académico
    para permitir:
    - usar memoria de programa activo
    - resolver programa por nombre/embedding
    - ejecutar búsqueda vectorial
    - pedir aclaración de programa
    - o bloquear por fuera de alcance

    Esta capa NO consulta BD, NO usa embeddings y NO tiene estado.
    """

    q = (q_norm or "").strip()
    token_count = len(q.split())

    structured_intent = any([
        field is not None,
        narrative_field is not None,
        is_listing,
        is_overview,
        is_general,
        asks_curriculum,
        asks_electives,
        asks_degree_options,
        asks_inscription,
    ])

    has_academic_keyword = _has_academic_keyword(q)
    short_followup = _looks_like_short_followup(q)
    obvious_offdomain = _looks_obviously_offdomain(q)
    plausible_program_candidate = _looks_like_program_candidate(
        program_candidate)

    # 1) Referencia explícita a programa o SNIES -> grounded fuerte
    if has_program_reference:
        return RequestGateDecision(
            is_grounded=True,
            can_use_memory=False,
            can_resolve_program=True,
            can_run_vector_search=True,
            should_ask_program=False,
            should_block=False,
            reason="explicit_program_reference",
        )

    # 2) Follow-up corto con programa activo -> puede usar memoria
    if has_active_program and short_followup:
        return RequestGateDecision(
            is_grounded=True,
            can_use_memory=True,
            can_resolve_program=True,
            can_run_vector_search=True,
            should_ask_program=False,
            should_block=False,
            reason="short_followup_with_active_program",
        )

    # 3) Intención estructurada + vocabulario académico -> grounded
    if structured_intent and has_academic_keyword:
        # si requiere programa pero no lo trae, preguntar
        if field is not None or narrative_field is not None or asks_curriculum or asks_electives or asks_degree_options or asks_inscription or is_overview:
            return RequestGateDecision(
                is_grounded=True,
                can_use_memory=False,
                can_resolve_program=plausible_program_candidate,
                can_run_vector_search=plausible_program_candidate,
                should_ask_program=not plausible_program_candidate,
                should_block=False,
                reason="structured_academic_intent",
            )

        return RequestGateDecision(
            is_grounded=True,
            can_use_memory=False,
            can_resolve_program=plausible_program_candidate,
            can_run_vector_search=True,
            should_ask_program=False,
            should_block=False,
            reason="structured_academic_intent_general",
        )

    # 4) Candidato plausible a programa aunque no tenga trigger fuerte
    if plausible_program_candidate:
        return RequestGateDecision(
            is_grounded=True,
            can_use_memory=False,
            can_resolve_program=True,
            can_run_vector_search=True,
            should_ask_program=False,
            should_block=False,
            reason="plausible_program_candidate",
        )

    # 5) Fuera de dominio obvio
    if obvious_offdomain:
        return RequestGateDecision(
            is_grounded=False,
            can_use_memory=False,
            can_resolve_program=False,
            can_run_vector_search=False,
            should_ask_program=False,
            should_block=True,
            reason="obvious_offdomain",
        )

    # 6) Campo exacto/narrativo/tabular sin programa ni anclaje -> bloquear
    if structured_intent and not has_active_program:
        return RequestGateDecision(
            is_grounded=False,
            can_use_memory=False,
            can_resolve_program=False,
            can_run_vector_search=False,
            should_ask_program=False,
            should_block=True,
            reason="structured_but_ungrounded",
        )

    # 7) Caso neutro/ambiguo -> bloquear antes de recuperar por similitud
    return RequestGateDecision(
        is_grounded=False,
        can_use_memory=False,
        can_resolve_program=False,
        can_run_vector_search=False,
        should_ask_program=False,
        should_block=True,
        reason="ungrounded_query",
    )
