"""
Capa 2 — Input Sanitizer
========================
Función pura. No tiene estado, no accede a BD ni a modelos.

Responsabilidad: limpiar y normalizar el texto del usuario ANTES de que
llegue a normalize_and_fix() y al resto del pipeline.

Operaciones:
  1. Truncar a MAX_INPUT_CHARS (evita flood / DoS al LLM).
  2. Strip de caracteres de control y Unicode sospechoso.
  3. Colapsar espacios y saltos de línea múltiples.
  4. Expandir abreviaciones frecuentes del dominio.
  5. Detectar patrones de inyección SQL / prompt residuales
     (complementa al domain_guardrail).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

MAX_INPUT_CHARS: int = 800  # suficiente para cualquier pregunta razonable

# ─────────────────────────────────────────────────────────────────────────────
# Abreviaciones del dominio → forma expandida
# ─────────────────────────────────────────────────────────────────────────────
# Se aplican con \b para no romper palabras más largas.
# Orden importa: más específicos primero.
_ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bmaestr[ií]a\b", re.I),       "maestría"),
    (re.compile(r"\bespecializ\w+\b", re.I),      "especialización"),
    (re.compile(r"\bdoctorad\w+\b", re.I),        "doctorado"),
    (re.compile(r"\bposgrado\b", re.I),           "posgrado"),
    (re.compile(r"\bpregrado\b", re.I),           "pregrado"),
    (re.compile(r"\bcred[eé]?itos?\b", re.I),     "créditos"),
    (re.compile(r"\bsem\.\b", re.I),              "semestre"),
    (re.compile(r"\binsc\.?\b", re.I),            "inscripción"),
    (re.compile(r"\badm\.?\b", re.I),             "admisión"),
    (re.compile(r"\bfac\.?\b", re.I),             "facultad"),
    (re.compile(r"\bprog\.?\b", re.I),            "programa"),
    (re.compile(r"\binfo\.?\b", re.I),            "información"),
    (re.compile(r"\btel\.?\b", re.I),             "teléfono"),
    (re.compile(r"\bpres\.?\b", re.I),            "presencial"),
    (re.compile(r"\bvirt\.?\b", re.I),            "virtual"),
    (re.compile(r"\bmod\.?\b", re.I),             "modalidad"),
    (re.compile(r"\breq\.?\b", re.I),             "requisitos"),
    (re.compile(r"\bdoc\.?\b", re.I),             "documento"),
    (re.compile(r"\bval\.?\b", re.I),             "valor"),
    (re.compile(r"\bdur\.?\b", re.I),             "duración"),
    # Siglas comunes
    (re.compile(r"\bUSTA\b"),                     "Universidad Santo Tomás"),
    (re.compile(r"\bUSTAT\b"),                    "Universidad Santo Tomás Tunja"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Patrones de inyección SQL residuales
# ─────────────────────────────────────────────────────────────────────────────
_SQL_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r";\s*(drop|delete|truncate|alter|insert|update)\s+",
        r"union\s+select",
        r"--\s*$",          # comentario SQL al final
        r"/\*.*?\*/",       # comentario SQL de bloque
        r"xp_cmdshell",
        r"exec\s*\(",
        r"benchmark\s*\(",
        r"sleep\s*\(\s*\d+",
        r"0x[0-9a-f]{4,}",  # hex encoding
    ]
]

# ─────────────────────────────────────────────────────────────────────────────
# Resultado
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SanitizeResult:
    text: str             # texto limpio y listo para normalize_and_fix()
    was_truncated: bool   # True si se truncó por longitud
    has_sql_injection: bool  # True si se detectó patrón SQL


# ─────────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────────

def sanitize(raw: str) -> SanitizeResult:
    """
    Limpia *raw* (texto crudo del usuario) y devuelve un SanitizeResult.

    El campo .text es el que debe pasarse al resto del pipeline.
    Si .has_sql_injection es True, el caller puede rechazar la petición
    o simplemente usar el texto ya purgado.
    """
    if not raw:
        return SanitizeResult(text="", was_truncated=False, has_sql_injection=False)

    # ── 1. Truncar ────────────────────────────────────────────────────────
    was_truncated = len(raw) > MAX_INPUT_CHARS
    text = raw[:MAX_INPUT_CHARS] if was_truncated else raw

    # ── 2. Eliminar caracteres de control (excepto \n, \t) ────────────────
    text = "".join(
        ch for ch in text
        if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
    )

    # ── 3. Normalizar saltos de línea y espacios múltiples ────────────────
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()

    # ── 4. Detección de SQL injection (antes de expandir, en texto original)
    has_sql = any(p.search(text) for p in _SQL_INJECTION_PATTERNS)
    if has_sql:
        # Purgar los patrones en lugar de bloquear duro
        # (el guardrail puede bloquear si quiere; aquí solo limpiamos)
        for p in _SQL_INJECTION_PATTERNS:
            text = p.sub(" ", text)
        text = re.sub(r" {2,}", " ", text).strip()

    # ── 5. Expandir abreviaciones ─────────────────────────────────────────
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)

    # ── 6. Strip final ────────────────────────────────────────────────────
    text = text.strip()

    return SanitizeResult(
        text=text,
        was_truncated=was_truncated,
        has_sql_injection=has_sql,
    )
