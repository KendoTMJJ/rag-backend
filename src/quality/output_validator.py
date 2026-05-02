"""
Capa 5 — Output Validator
=========================
Función pura. No tiene estado, no accede a BD ni a modelos.

Responsabilidad: evaluar la respuesta generada por el LLM y producir
un confidence score compuesto que el pipeline puede usar para:
  - Decidir si mostrar la respuesta al usuario.
  - Agregar un disclaimer ("con la información disponible…").
  - Loggear para métricas de calidad.

Score compuesto (0.0 – 1.0):
  retrieval_score  × 0.40   → calidad del mejor chunk recuperado
  program_score    × 0.30   → ¿se identificó un programa con confianza?
  output_score     × 0.20   → ¿la respuesta parece anclada al contexto?
  domain_score     × 0.10   → ¿el guardrail confirmó in-domain?

Detección de alucinación ligera:
  - Respuesta contiene frases de incertidumbre propias del LLM
    ("no tengo información", "no puedo confirmar", etc.) → penaliza output_score.
  - Respuesta muy corta o genérica → penaliza output_score.
  - Respuesta menciona entidades no presentes en el contexto recuperado
    (solo nombres de programas muy distintos) → flag hallucination_suspected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds de decisión
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_SHOW: float = 0.55   # por debajo → agregar disclaimer
CONFIDENCE_BLOCK: float = 0.35   # por debajo → no mostrar, devolver fallback

# ─────────────────────────────────────────────────────────────────────────────
# Patrones de LLM "uncertainty phrases"
# ─────────────────────────────────────────────────────────────────────────────
_UNCERTAINTY_PHRASES: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"no\s+tengo\s+informaci[oó]n",
        r"no\s+cuento\s+con\s+informaci[oó]n",
        r"no\s+puedo\s+confirmar",
        r"no\s+encontr[eé]\s+informaci[oó]n",
        r"no\s+est[aá]\s+en\s+(mis\s+)?(documentos|la\s+informaci[oó]n)",
        r"lamentablemente\s+no",
        r"desafortunadamente\s+no",
        r"no\s+dispongo\s+de",
        r"no\s+tengo\s+acceso",
        r"as\s+an\s+ai",              # LLM hablando de sí mismo en inglés
        r"i\s+(don'?t|do\s+not)\s+have",
        r"i\s+cannot\s+provide",
    ]
]

# Frases que indican que el LLM no está anclado al contexto
_GENERIC_PHRASES: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"en\s+general",
        r"t[íi]picamente",
        r"por\s+lo\s+general",
        r"usualmente",
        r"en\s+la\s+mayor[íi]a\s+de\s+(los\s+)?casos",
        r"suele\s+ser",
        r"podr[íi]a\s+ser",
        r"depende\s+de\s+cada",
    ]
]


# ─────────────────────────────────────────────────────────────────────────────
# Resultado
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ValidationResult:
    confidence: float             # 0.0 – 1.0, score compuesto final
    retrieval_score: float        # componente retrieval (0–1)
    program_score: float          # componente program resolution (0–1)
    output_score: float           # componente calidad del output (0–1)
    domain_score: float           # componente dominio (0–1)
    hallucination_suspected: bool
    add_disclaimer: bool          # True si confidence < CONFIDENCE_SHOW
    should_block: bool            # True si confidence < CONFIDENCE_BLOCK
    flags: list[str] = field(default_factory=list)  # razones legibles


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def validate_output(
    llm_answer: str,
    best_similarity: float,
    program_resolved: bool,
    # "explicit"|"sql_full"|"sql_name"|"embedding"|"memory"|"none"
    program_source: Optional[str] = None,
    domain_allowed: bool = True,
    context_chunks: Optional[list[dict]] = None,
) -> ValidationResult:
    """
    Evalúa la respuesta del LLM y devuelve un ValidationResult.

    Args:
        llm_answer:       Texto generado por el LLM.
        best_similarity:  Mejor similitud coseno de los chunks recuperados (0–1).
        program_resolved: ¿Se identificó un programa concreto para la respuesta?
        program_source:   Cómo se resolvió el programa ("embedding" es menos confiable).
        domain_allowed:   Resultado del domain_guardrail (True = in-domain).
        context_chunks:   Lista de chunks usados como contexto (opcional, para
                          detección de alucinaciones).

    Returns:
        ValidationResult con el score compuesto y flags de decisión.
    """
    flags: list[str] = []

    # ── Componente 1: Retrieval (40%) ─────────────────────────────────────
    retrieval_score = float(min(1.0, max(0.0, best_similarity)))

    # ── Componente 2: Program resolution (30%) ───────────────────────────
    program_score = _compute_program_score(
        program_resolved, program_source, flags)

    # ── Componente 3: Output quality (20%) ───────────────────────────────
    output_score = _compute_output_score(llm_answer, flags)

    # ── Componente 4: Domain (10%) ────────────────────────────────────────
    domain_score = 1.0 if domain_allowed else 0.0
    if not domain_allowed:
        flags.append("domain:out_of_domain")

    # ── Score compuesto ───────────────────────────────────────────────────
    confidence = (
        retrieval_score * 0.40
        + program_score * 0.30
        + output_score * 0.20
        + domain_score * 0.10
    )
    confidence = round(min(1.0, max(0.0, confidence)), 4)

    # ── Detección de alucinación ligera ───────────────────────────────────
    hallucination_suspected = _detect_hallucination(
        llm_answer, context_chunks, flags)

    return ValidationResult(
        confidence=confidence,
        retrieval_score=round(retrieval_score, 4),
        program_score=round(program_score, 4),
        output_score=round(output_score, 4),
        domain_score=round(domain_score, 4),
        hallucination_suspected=hallucination_suspected,
        add_disclaimer=confidence < CONFIDENCE_SHOW,
        should_block=confidence < CONFIDENCE_BLOCK,
        flags=flags,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _compute_program_score(
    resolved: bool,
    source: Optional[str],
    flags: list[str],
) -> float:
    if not resolved:
        flags.append("program:not_resolved")
        return 0.0
    # El source más confiable es "explicit" o "sql_*"; embedding es más débil
    source_weights = {
        "explicit":  1.0,
        "sql_full":  0.95,
        "sql_name":  0.90,
        "memory":    0.80,
        "embedding": 0.60,   # más propenso a falsos positivos
    }
    score = source_weights.get(source or "", 0.70)
    if score < 0.70:
        flags.append(f"program:weak_source({source})")
    return score


def _compute_output_score(answer: str, flags: list[str]) -> float:
    if not answer or not answer.strip():
        flags.append("output:empty")
        return 0.0

    score = 1.0

    # Penalizar frases de incertidumbre del LLM
    uncertainty_hits = sum(
        1 for p in _UNCERTAINTY_PHRASES if p.search(answer))
    if uncertainty_hits:
        penalty = min(0.5, uncertainty_hits * 0.15)
        score -= penalty
        flags.append(f"output:uncertainty_phrases({uncertainty_hits})")

    # Penalizar respuestas genéricas no ancladas
    generic_hits = sum(1 for p in _GENERIC_PHRASES if p.search(answer))
    if generic_hits >= 2:
        score -= 0.15
        flags.append(f"output:generic_phrases({generic_hits})")

    # Penalizar respuestas demasiado cortas (< 20 chars = poco informativa)
    if len(answer.strip()) < 20:
        score -= 0.30
        flags.append("output:too_short")

    return round(max(0.0, score), 4)


def _detect_hallucination(
    answer: str,
    chunks: Optional[list[dict]],
    flags: list[str],
) -> bool:
    """
    Heurística simple: si el LLM menciona un programa por nombre
    que NO aparece en ninguno de los chunks recuperados → posible alucinación.
    """
    if not answer or not chunks:
        return False

    # Extraer nombres de programa del contexto
    context_names: set[str] = set()
    for chunk in chunks:
        pname = (chunk.get("program_name") or "").lower().strip()
        if pname:
            context_names.add(pname)

    if not context_names:
        return False

    # Buscar nombres en la respuesta que NO estén en el contexto
    # (solo buscamos "maestría|especialización|doctorado en X")
    answer_lower = answer.lower()
    found_in_answer = re.findall(
        r"(maestr[íi]a|especializaci[oó]n|doctorado)\s+en\s+([\w\s]{4,40})",
        answer_lower,
    )
    for _, prog_mention in found_in_answer:
        prog_mention = prog_mention.strip()
        # Si ningún nombre del contexto contiene esta mención → sospechoso
        if not any(prog_mention in ctx_name or ctx_name in prog_mention
                   for ctx_name in context_names):
            flags.append(
                f"hallucination:unknown_program_mention({prog_mention[:30]})")
            return True

    return False
