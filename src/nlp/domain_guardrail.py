"""
Capa de guardrail de dominio
============================
Detecta inyecciones de prompt, términos peligrosos y valida que la consulta
esté dentro del dominio académico.

CAMBIO: _ACADEMIC_SIGNALS ya no se define aquí. Se importa de domain_taxonomy
para mantener una única fuente de verdad. check_domain() ahora usa score_domain()
internamente y expone el confidence score en GuardrailResult.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet

from src.nlp.domain_taxonomy import score_domain, DomainScore


# ─────────────────────────────────────────────────────────────────────────────
# Patrones de inyección de prompt (sin cambios)
# ─────────────────────────────────────────────────────────────────────────────

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignora\s+(todas\s+)?(las\s+)?(tus\s+)?instrucciones",
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"olvida\s+(todo|tus\s+instrucciones|lo\s+que\s+sabes)",
        r"act[uú]a\s+como\s+(si\s+fueras|un\s+)",
        r"pretend\s+(you\s+are|to\s+be)",
        r"jailbreak",
        r"prompt\s*injection",
        r"bypass\s+(the\s+)?filter",
        r"you\s+are\s+now\s+(dan|evil|free)",
        r"</?(system|user|assistant|human)\s*>",
        r"\[INST\]|\[/?SYS\]|<<SYS>>",
    ]
]

_HARD_BLACKLIST: FrozenSet[str] = frozenset({
    "malware", "ransomware", "exploit", "rootkit",
})


# ─────────────────────────────────────────────────────────────────────────────
# Resultado — ahora incluye confidence score
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GuardrailResult:
    allowed: bool
    reason: str         # "ok" | "injection" | "blacklist" | "in_domain" | "unknown"
    detail: str = ""
    confidence: float = 0.0     # NUEVO: score del domain scorer (0–1)
    category: str = "unknown"   # NUEVO: categoría de intención detectada


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

# Umbral mínimo de confianza para considerar una consulta dentro del dominio.
# Ajústalo si ves demasiados falsos negativos (baja → más permisivo)
# o demasiados falsos positivos (sube → más estricto).
DOMAIN_CONFIDENCE_THRESHOLD: float = 0.25


def check_domain(text: str) -> GuardrailResult:
    """
    Evalúa si el texto está dentro del dominio y es seguro de procesar.

    Orden de evaluación (fail-fast):
      1. Inyección de prompt   → blocked
      2. Blacklist dura        → blocked
      3. Score de dominio      → allowed si confidence >= umbral, unknown si no
    """
    if not text or not text.strip():
        return GuardrailResult(allowed=True, reason="unknown", confidence=0.0)

    # ── 1. Inyección de prompt ────────────────────────────────────────────
    for pattern in _INJECTION_PATTERNS:
        m = pattern.search(text)
        if m:
            return GuardrailResult(
                allowed=False,
                reason="injection",
                detail=m.group(0)[:60],
                confidence=0.0,
                category="unknown",
            )

    # ── 2. Blacklist dura ────────────────────────────────────────────────
    tokens = _tokenize(text)
    for token in tokens:
        if token in _HARD_BLACKLIST:
            return GuardrailResult(
                allowed=False,
                reason="blacklist",
                detail=token,
                confidence=0.0,
                category="unknown",
            )

    # ── 3. Score de dominio ──────────────────────────────────────────────
    domain: DomainScore = score_domain(text)

    if domain.confidence >= DOMAIN_CONFIDENCE_THRESHOLD:
        return GuardrailResult(
            allowed=True,
            reason="in_domain",
            confidence=domain.confidence,
            category=domain.category,
        )

    # Confianza baja pero no bloqueado: se deja pasar con reason="unknown"
    # para que el pipeline decida qué hacer con él.
    return GuardrailResult(
        allowed=True,
        reason="unknown",
        confidence=domain.confidence,
        category=domain.category,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-záéíóúüñ0-9\s]", " ", text.lower()).split()
