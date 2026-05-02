"""
Normalización de texto y chunking semántico
============================================
Exporta:
  - normalize_text()     limpia y normaliza para comparaciones
  - apply_typo_map()     corrige errores ortográficos frecuentes
  - normalize_and_fix()  convenience: normalize + typos
  - semantic_chunk()     NUEVO: divide texto respetando párrafos + overlap
"""

import re
import unicodedata
from typing import List

_COMMON_TYPOS = {
    "progrmas": "programas",
    "prgramas": "programas",
    "progrm": "programa",
    "progrma": "programa",
    "progamas": "programas",
    "programs": "programas",
    "posgrads": "posgrados",
    "postgrados": "posgrados",
    "ciencas": "ciencias",
    "profecional": "profesional",
    "profesional": "profesional",
    "ocuacional":  "ocupacional",
    "ocupasional": "ocupacional",
}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def normalize_text(s: str) -> str:
    """Lower + sin tildes + signos → espacios + colapsa espacios."""
    s = (s or "").strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"[¿?¡!.,;:()\[\]{}\"'`~^/\\|@#$%&+=<>]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def apply_typo_map(q_norm: str) -> str:
    """Reemplaza tokens exactos por su forma correcta."""
    if not q_norm:
        return q_norm
    tokens = q_norm.split()
    tokens = [_COMMON_TYPOS.get(t, t) for t in tokens]
    return " ".join(tokens)


def normalize_and_fix(text: str) -> str:
    """Convenience: normaliza + aplica corrección de typos."""
    return apply_typo_map(normalize_text(text))


# ─────────────────────────────────────────────────────────────────────────────
# Chunking semántico
# ─────────────────────────────────────────────────────────────────────────────

def semantic_chunk(
    text: str,
    max_chars: int = 800,
    overlap: int = 150,
) -> List[str]:
    """
    Divide texto en chunks respetando párrafos y con ventana de overlap.

    Algoritmo:
      1. Divide por párrafos (doble salto de línea).
      2. Acumula párrafos hasta alcanzar max_chars.
      3. Al cerrar un chunk, lleva las últimas `overlap` chars al siguiente
         para no perder contexto en los bordes.
      4. Si un párrafo solo ya supera max_chars, lo subdivide con overlap.

    Args:
        text:      Texto a dividir.
        max_chars: Tamaño máximo de cada chunk (caracteres).
        overlap:   Caracteres del chunk anterior que se repiten al inicio
                   del siguiente chunk.

    Returns:
        Lista de chunks no vacíos.

    Ejemplo:
        >>> chunks = semantic_chunk("Párrafo uno.\\n\\nPárrafo dos muy largo...", max_chars=50)
    """
    text = (text or "").strip()
    if not text:
        return []

    # Divide por párrafos — acepta \n\n, \r\n\r\n, o múltiples saltos
    paragraphs = [p.strip() for p in re.split(
        r"(?:\r?\n){2,}", text) if p.strip()]

    if not paragraphs:
        return []

    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para

        if len(candidate) <= max_chars:
            # Cabe: acumula
            current = candidate
        else:
            # No cabe: cierra el chunk actual (si existe)
            if current:
                chunks.append(current)
                # Overlap: arranca el nuevo chunk con el final del anterior
                overlap_seed = current[-overlap:].strip()
                current = (overlap_seed + "\n\n" + para).strip()
            else:
                # El párrafo solo ya es más grande que max_chars → subdivide
                current = ""
                for i in range(0, len(para), max_chars - overlap):
                    sub = para[i: i + max_chars].strip()
                    if sub:
                        chunks.append(sub)
                # No hay current pendiente tras esta subdivisión

    # Vuelca lo que quedó acumulado
    if current:
        chunks.append(current)

    return [c for c in chunks if c]
