"""
Capa 4 — Context Reranker
=========================
Función pura. No tiene estado, no accede a BD ni a modelos.

Responsabilidad: reordenar los chunks de contexto devueltos por
semantic_search() ANTES de enviárselos al LLM, para que el programa
activo en la sesión tenga mayor peso que programas tangenciales.

Estrategia de puntuación compuesta:
  score = similarity_weight * similarity
        + active_program_bonus   (si el chunk pertenece al programa activo)
        + section_weight[section] (secciones de alta densidad informativa valen más)

El reranker NO descarta chunks; solo reordena. El pipeline decide cuántos
pasar al LLM.
"""

from __future__ import annotations

from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Pesos de sección
# ─────────────────────────────────────────────────────────────────────────────
# Las secciones con más densidad de datos concretos tienen peso más alto.
_SECTION_WEIGHT: dict[str, float] = {
    "info_general":        0.15,
    "program_name":        0.10,
    "division":            0.08,
    "descripcion":         0.12,
    "perfil_ingreso":      0.10,
    "perfil_egresado":     0.10,
    "perfil_ocupacional":  0.10,
    "diferencial":         0.08,
    "requisitos":          0.12,
    "course_row":          0.05,
    "elective_row":        0.03,
    "degree_option_row":   0.05,
}
_DEFAULT_SECTION_WEIGHT: float = 0.05

# Bonus por pertenecer al programa activo en sesión (0–1, se suma al score)
_ACTIVE_PROGRAM_BONUS: float = 0.20

# Peso base de la similitud coseno en el score final
_SIMILARITY_WEIGHT: float = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def rerank(
    chunks: list[dict],
    active_program_id: Optional[int] = None,
) -> list[dict]:
    """
    Reordena *chunks* según un score compuesto.

    Args:
        chunks: Lista de dicts tal como la devuelve RetrievalService.semantic_search()
                (campos: program_id, section, content, similarity, …).
        active_program_id: ID interno del programa activo en la sesión, si existe.
                           Si es None, solo se usa similitud + sección.

    Returns:
        Nueva lista ordenada de mayor a menor score.
        Cada dict recibe el campo extra ``rerank_score`` (float, 0–1.2 aprox).
    """
    if not chunks:
        return []

    scored = []
    for chunk in chunks:
        score = _compute_score(chunk, active_program_id)
        scored.append({**chunk, "rerank_score": round(score, 4)})

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored


def rerank_top(
    chunks: list[dict],
    active_program_id: Optional[int] = None,
    top_k: int = 6,
) -> list[dict]:
    """
    Equivalente a rerank() pero devuelve solo los primeros *top_k* chunks.
    Conveniente para limitar el contexto enviado al LLM.
    """
    return rerank(chunks, active_program_id)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _compute_score(chunk: dict, active_program_id: Optional[int]) -> float:
    similarity = float(chunk.get("similarity") or 0.0)
    section = chunk.get("section") or ""
    pid = chunk.get("program_id")

    score = _SIMILARITY_WEIGHT * similarity
    score += _SECTION_WEIGHT.get(section, _DEFAULT_SECTION_WEIGHT)

    if active_program_id is not None and pid == active_program_id:
        score += _ACTIVE_PROGRAM_BONUS

    return score
