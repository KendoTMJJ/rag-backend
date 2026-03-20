"""
Taxonomía de dominio centralizada
==================================
Fuente única de verdad para señales de dominio académico.

Reemplaza las listas dispersas en:
  - domain_guardrail.py  (_ACADEMIC_SIGNALS)
  - intent_utils.py      (_OVERVIEW_TRIGGERS, _FALSE_LISTING_PHRASES, etc.)
  - rag_pipeline.py      (detect_narrative_field, _detect_tabular_intent)

Uso:
    from src.nlp.domain_taxonomy import score_domain, DomainScore, INTENT_CATEGORY

    result = score_domain("cuánto cuesta la maestría en derecho")
    if result.confidence < 0.25:
        # fuera de dominio
        ...
    match result.category:
        case "info_general":  ...
        case "curricular":    ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet


# ─────────────────────────────────────────────────────────────────────────────
# Tipos de resultado
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class DomainScore:
    confidence: float           # 0.0 – 1.0  (≥ 0.25 = dentro de dominio)
    category: str               # ver INTENT_CATEGORY
    matched_signals: tuple[str, ...]  # señales que dispararon el match


# Categorías canónicas — úsalas como valores en cualquier rama del pipeline
INTENT_CATEGORY = {
    "info_general",      # costo, duración, modalidad, sede, horario, créditos
    "curricular",        # malla, pensum, asignaturas, electivas, opciones de grado
    "perfiles",          # perfil_ingreso, perfil_egresado, perfil_ocupacional
    # identificar/listar un programa (maestría, especialización…)
    "programa_id",
    "diferencial",       # por qué estudiar aquí, ventajas, valor agregado
    "admision",          # inscripción, requisitos, documentos, proceso
    "listing",           # enumerar programas (¿qué posgrados hay?)
    "comparison",        # comparar dos o más programas
    "unknown",           # no clasificado → manejo defensivo
}


# ─────────────────────────────────────────────────────────────────────────────
# Señales por categoría
# ─────────────────────────────────────────────────────────────────────────────
# Cada entrada es una cadena que se busca como subcadena en los tokens
# (no regex, búsqueda rápida O(n·m)). Ordena de más a menos específico.
# IMPORTANTE: no agregues stopwords genéricas aquí (el, la, que…).

DOMAIN_SIGNALS: dict[str, FrozenSet[str]] = {

    "info_general": frozenset({
        "snies", "credito", "creditos", "duracion", "dura", "semestre",
        "semestres", "modalidad", "presencial", "virtual", "semipresencial",
        "sede", "campus", "ciudad", "horario", "jornada", "sabado",
        "inversion", "costo", "cuesta", "precio", "valor", "pagar",
        "matricula", "titulo", "titulacion", "registro", "calificado",
        "enlace", "link", "inicio", "lanzamiento",
    }),

    "curricular": frozenset({
        "malla", "pensum", "asignatura", "asignaturas", "materia", "materias",
        "plan de estudios", "electiva", "electivas", "optativa", "optativas",
        "opcion de grado", "opciones de grado", "graduarme", "graduarse",
        "semestre", "credito", "curso", "cursos",
    }),

    "perfiles": frozenset({
        "perfil", "egresado", "egresados", "egreso", "ocupacional",
        "ingreso", "admision", "admitido", "competencia", "competencias",
        "habilidad", "habilidades", "laboral", "profesional", "ocupacion",
        "trabajo", "empleo", "campo laboral", "salida laboral",
    }),

    "programa_id": frozenset({
        "maestria", "especializacion", "doctorado", "posgrado", "posgrados",
        "programa", "programas", "postgrad", "postgrado",
        "facultad", "division",
    }),

    "diferencial": frozenset({
        "diferencial", "por que estudiar", "ventaja", "ventajas",
        "valor agregado", "especial", "unico", "unica", "hace diferente",
        "por que elegir", "por que este",
    }),

    "admision": frozenset({
        "inscripcion", "inscribir", "inscribirme", "inscribirse",
        "requisito", "requisitos", "documento", "documentos",
        "papel", "papeles", "proceso", "pasos", "matricular",
        "como entrar", "como ingresar", "como aplicar",
    }),

    "listing": frozenset({
        "que programas", "cuales programas", "que posgrados", "cuales posgrados",
        "que maestrias", "que especializaciones", "que doctorados",
        "hay programas", "existen programas", "ofrecen programas",
        "listado", "catalogo", "todos los programas",
    }),

    "comparison": frozenset({
        "diferencia entre", "diferencias entre", "comparar", "comparacion",
        "cual es mejor", "cual conviene", "me conviene", "mejor para",
        "ventajas y desventajas", "vs", "versus",
        "mas investigativ", "mas profesional",
    }),
}


# ─────────────────────────────────────────────────────────────────────────────
# Contexto descriptivo por sección narrativa
# (usado al construir anchors en el indexado)
# ─────────────────────────────────────────────────────────────────────────────

SECTION_CONTEXT: dict[str, str] = {
    "perfil_ingreso":      "Describe el tipo de estudiante que puede ingresar al programa.",
    "perfil_egresado":     "Describe las competencias y habilidades que desarrolla el graduado.",
    "perfil_ocupacional":  "Describe los campos laborales y empleadores del egresado.",
    "diferencial":         "Explica por qué estudiar este programa y sus ventajas diferenciales.",
    "requisitos":          "Lista los requisitos de admisión y documentos necesarios.",
    "descripcion":         "Descripción general del programa académico.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Normaliza sin tildes y devuelve lista de tokens."""
    import unicodedata
    nfkd = unicodedata.normalize("NFD", text.lower())
    clean = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9\s]", " ", clean).split()


def score_domain(text: str) -> DomainScore:
    """
    Evalúa qué tan dentro del dominio académico está el texto
    y qué categoría tiene más señales.

    Returns:
        DomainScore con:
          - confidence ∈ [0, 1]  (≥ 0.25 → probablemente dentro de dominio)
          - category: la categoría con más señales encontradas
          - matched_signals: señales que generaron el hit
    """
    if not text or not text.strip():
        return DomainScore(confidence=0.0, category="unknown", matched_signals=())

    tokens = _tokenize(text)
    # También evalúa bigrams y trigrams para señales de más de una palabra
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    trigrams = [
        f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens) - 2)]
    all_grams = set(tokens) | set(bigrams) | set(trigrams)

    category_hits: dict[str, list[str]] = {}

    for cat, signals in DOMAIN_SIGNALS.items():
        hits = [
            sig for sig in signals
            if any(sig in gram for gram in all_grams)
        ]
        if hits:
            category_hits[cat] = hits

    if not category_hits:
        return DomainScore(confidence=0.0, category="unknown", matched_signals=())

    # Categoría ganadora = la que más señales distintas disparó
    top_cat = max(category_hits, key=lambda c: len(category_hits[c]))
    all_matched = [s for hits in category_hits.values() for s in hits]

    # Confianza: escala suave — 1 señal=0.30, 2=0.55, 3+=0.80, 4+=1.0
    n = len(set(all_matched))
    confidence = min(1.0, 0.25 + (n - 1) * 0.25) if n >= 1 else 0.0

    return DomainScore(
        confidence=round(confidence, 3),
        category=top_cat,
        matched_signals=tuple(sorted(set(all_matched))),
    )
