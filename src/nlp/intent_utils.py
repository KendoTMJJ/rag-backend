import re
from typing import Optional

from src.nlp.text_normalizer import normalize_and_fix


# ─────────────────────────────────────────────────────────────────────────────
# Constantes compartidas
# ─────────────────────────────────────────────────────────────────────────────

_OVERVIEW_TRIGGERS = {
    "que me dices", "que me puede decir", "que me puedes decir",
    # FIX: "puedes decirme" y "puede decirme" se mantienen pero
    # looks_like_program_overview() los filtra si hay indicadores de listado
    "puedes decirme", "puede decirme",
    "de que trata", "de que se trata",
    "describeme", "descripcion",
    "hablame de", "hablame del", "hablame de la",
    "hableme de", "hableme del", "hableme de la",
    "dime del", "dime de la", "dime sobre",
    "cuentame del", "cuentame de la",
    "info del programa", "informacion del programa",
    "sobre el programa", "acerca del programa", "acerca de",
    "puedes describir", "resumen rapido", "dame un resumen",
    "cual es el enfoque", "enfoque del posgrado", "enfoque del programa",
    "que se aprende en", "que ensenan", "que se aprende",
    "cuentame del programa", "presentame el programa",
}

# FIX: Patrones que indican que la pregunta es un LISTADO aunque contenga
# un trigger de overview. Ej: "puedes decirme QUÉ PROGRAMAS hay de salud"
_LISTING_QUALIFIERS = [
    r"\bque\s+programas\b",
    r"\bcuales?\s+programas\b",
    r"\bque\s+posgrados\b",
    r"\bcuales?\s+posgrados\b",
    r"\bprogramas\s+(hay|son|existen|ofrece)\b",
    r"\bprogramas\s+(del?|de\s+la)\s+area\b",
]

# FIX: Palabras que indican pregunta TABULAR aunque contenga un trigger de overview.
# Ej: "puedes decirme el pensum" → CURRICULUM, no OVERVIEW
# Ej: "puedes decirme la malla curricular" → CURRICULUM, no OVERVIEW
_TABULAR_QUALIFIERS = [
    "malla", "pensum", "plan de estudios",
    "asignaturas", "materias", "electiv", "optativas",
    "opcion de grado", "opciones de grado",
    "como me puedo graduar", "como puedo graduarme", "como graduarse",
]

_REASONING_TRIGGERS = [
    "mas investigativ", "mas profesional",
    "profesional o investigativ", "investigativ o profesional",
    "diferencia entre", "diferencias entre",
    "mejor para", "cual es mejor", "cual conviene",
    "comparar", "comparacion", "ventajas y desventajas",
    "me conviene", "conviene mas", "que programa conviene",
    "que me conviene", "cual me conviene",
    "me conviene mas", "si trabajo en", "conviene si",
    "para alguien que trabaja", "para quien trabaja",
]

_FALSE_LISTING_PHRASES = [
    "de especial", "tiene de especial", "lo hace diferente",
    "competencias", "habilidades al salir", "habilidades que",
    "se aprende", "se estudia", "se ensena",
    "enfoque", "trata el programa", "trata la maestria",
    "trata la especializacion", "trata el doctorado",
    "investigativo o profesional", "profesional o investigativo",
    "tiene convenio", "tiene acreditacion", "esta acreditado",
    "tiene laboratorio", "tiene sede", "cuantos estudiantes",
]

# Stopwords que no son topics válidos para extract_topic_for_listing
_TOPIC_STOPWORDS = frozenset({
    "los", "las", "el", "la", "un", "una", "unos", "unas",
    "que", "de", "del", "en", "al", "a", "y", "o", "con",
    "para", "por", "se", "su", "sus",
})


# ─────────────────────────────────────────────────────────────────────────────
# Palabras de programa
# ─────────────────────────────────────────────────────────────────────────────

def looks_like_program_word(q_norm: str) -> bool:
    if not q_norm:
        return False
    if re.search(r"\b(program\w*|posgrad\w*|postgrad\w*|maestr\w*|doctorad\w*|especializ\w*)\b", q_norm):
        return True
    if re.search(r"\b(pr\w*gram\w*|prog\w*ram\w*)\b", q_norm):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Intención de listado
# ─────────────────────────────────────────────────────────────────────────────

def looks_like_programs_listing(q_norm: str) -> bool:
    q = (q_norm or "").strip()
    if not q:
        return False

    # Guardia 1: trigger de overview Y sin indicadores de listado → NO es listado
    # FIX: solo bloquear si el trigger de overview no va acompañado de
    # "que programas", "cuales programas", etc.
    if any(t in q for t in _OVERVIEW_TRIGGERS):
        if not any(re.search(lq, q) for lq in _LISTING_QUALIFIERS):
            return False

    # Guardia 2: razonamiento/comparación → NO es listado
    if is_reasoning_question(q):
        return False

    # Guardia 3: atributo singular de un programa → NO es listado
    if is_false_listing(q):
        return False

    # Caso A: palabra de programa + verbo de existencia/listado
    if looks_like_program_word(q):
        if re.search(
            r"\b(que|hay|existe|existen|ofrecen|ofrece|tienen|tiene|algun|alguna|alguno)\b", q
        ):
            if re.search(r"\bel\s+program\w+\s+tiene\b", q):
                return False
            return True

    # Caso B: búsqueda por área temática
    if looks_like_program_word(q):
        if re.search(
            r"\b(relacionad[oa]s?\s+con|del?\s+area\s+de"
            r"|en\s+el\s+area\s+de"
            r"|que\s+tienen\s+que\s+ver\s+con"
            r"|programas?\s+de\s+\w+|posgrados?\s+de\s+\w+)\b",
            q,
        ):
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Intención de overview
# ─────────────────────────────────────────────────────────────────────────────

def looks_like_program_overview(q_norm: str) -> bool:
    q = (q_norm or "").strip()
    if not q:
        return False

    # Si hay indicadores de listado → NO es overview
    if any(re.search(lq, q) for lq in _LISTING_QUALIFIERS):
        return False

    # Si hay intención tabular → NO es overview
    if any(w in q for w in _TABULAR_QUALIFIERS):
        return False

    # Si parece listado → NO es overview
    if looks_like_programs_listing(q):
        return False

    # FIX CLAVE: si hay campo exacto detectado, NO es overview
    # Ej: "puedes decirme cuánto cuesta..." debe ir a PROGRAM_FIELD
    if detect_field(q) is not None:
        return False

    return any(t in q for t in _OVERVIEW_TRIGGERS)


# ─────────────────────────────────────────────────────────────────────────────
# Pregunta general
# ─────────────────────────────────────────────────────────────────────────────

def looks_like_general_question(q_norm: str) -> bool:
    q = q_norm or ""
    phrase_triggers = [
        "en general", "es de la", "pertenece a la",
        "que postgrados", "que posgrados", "que programas hay",
        "a que facultad", "a que division",
        "cuantos programas", "cuantas maestrias", "cuantas especializaciones",
        "cuantos doctorados", "facultad de",
        "division de ingenieria", "division de ciencias",
        "division de salud", "division de derecho", "division de economia",
    ]
    return any(t in q for t in phrase_triggers)


# ─────────────────────────────────────────────────────────────────────────────
# False listing / reasoning
# ─────────────────────────────────────────────────────────────────────────────

def is_false_listing(q_norm: str) -> bool:
    return any(k in q_norm for k in _FALSE_LISTING_PHRASES)


def is_reasoning_question(q_norm: str) -> bool:
    return any(t in (q_norm or "") for t in _REASONING_TRIGGERS)


# ─────────────────────────────────────────────────────────────────────────────
# Extractores
# ─────────────────────────────────────────────────────────────────────────────

def extract_snies(text: str) -> Optional[str]:
    m = re.search(r"\b(\d{5,10})\b", text or "")
    return m.group(1) if m else None


def extract_semester(q_norm: str) -> Optional[int]:
    q = (q_norm or "").strip()
    m = re.search(r"(?:semestre|sem)\s*(\d{1,2})", q)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\b(\d{1,2})\s*(?:semestre|sem)\b", q)
    if m2:
        return int(m2.group(1))
    ord_map = {
        "primer": 1, "primero": 1, "segundo": 2,
        "tercer": 3, "tercero": 3, "cuarto": 4,
        "quinto": 5, "sexto": 6, "septimo": 7,
        "octavo": 8, "noveno": 9, "decimo": 10,
        "undecimo": 11, "duodecimo": 12,
    }
    m3 = re.search(r"\b(" + "|".join(ord_map.keys()) + r")\s+semestre\b", q)
    if m3:
        return ord_map[m3.group(1)]
    return None


def extract_program_candidate(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    m = re.search(r"\b(?:de|del|en|para|sobre)\s+(.+)$", t, re.IGNORECASE)
    if not m:
        return None
    cand = m.group(1).strip()
    cand = re.split(
        r"[?.!]|,|\b(semestre|materias|asignaturas|pensum|malla|requisitos|inscrip|inscripcion|pasos|proceso)\b",
        cand, maxsplit=1, flags=re.IGNORECASE,
    )[0].strip()
    cand_norm = re.sub(r"\s+", " ", cand).strip().lower()
    bad = {
        "total", "en total", "general", "en general",
        "eso", "en eso", "esto", "en esto", "este", "esta", "ese", "esa",
        "aqui", "aca", "alli", "alla", "tunja", "usta",
        "programa", "posgrados", "posgrado",
        "inscribirme", "inscribirse", "inscribir", "graduarme", "graduarse",
    }
    if cand_norm in bad:
        return None
    if len(cand_norm) < 6 and " " not in cand_norm:
        return None
    return cand


def extract_topic_for_listing(question: str) -> Optional[str]:
    qn = normalize_and_fix(question)
    if not qn:
        return None
    if any(t in qn for t in _OVERVIEW_TRIGGERS):
        # FIX: si hay trigger de overview pero también indicadores de listado,
        # sí extraer el topic (ej: "puedes decirme qué programas hay de salud")
        if not any(re.search(lq, qn) for lq in _LISTING_QUALIFIERS):
            return None
    if is_reasoning_question(qn):
        return None

    # FIX: patrones ordenados de mayor a menor especificidad.
    # Los patrones más específicos van primero para evitar que
    # r"\bde\s+(.+)$" capture demasiado ("de los programas que...")
    patterns = [
        # "que tienen que ver con salud"
        r"\bque\s+tienen\s+que\s+ver\s+con\s+(.+)$",
        # "relacionados con educacion"
        r"\brelacionad[oa]s?\s+con\s+(.+)$",
        # "vinculados con derecho"
        r"\bvinculad[oa]s?\s+con\s+(.+)$",
        # "del area de salud"
        r"\bdel?\s+area\s+de\s+(.+)$",
        # "en el area de sistemas"
        r"\ben\s+el\s+area\s+de\s+(.+)$",
        # "sobre psicologia"
        r"\bsobre\s+(.+)$",
        # FIX: "de" solo si NO sigue artículo+palabraPrograma
        # Antes: r"\bde\s+(.+)$" capturaba "de los programas que tienen que ver con salud"
        # → después de splits quedaba "los"
        r"\bde\s+(?!(?:los?|las?|un[oa]?)\s+(?:programas?|posgrados?|maestrias?|especializ\w*|doctorad\w*))\s*(.+)$",
        r"\ben\s+(.+)$",
    ]

    for p in patterns:
        m = re.search(p, qn, flags=re.IGNORECASE)
        if not m:
            continue
        topic = m.group(1).strip()
        # Cortar en verbos de existencia
        topic = re.split(
            r"\b(hay|existe|existen|ofrece|ofrecen|tiene|tienen|son|estan)\b",
            topic, maxsplit=1, flags=re.IGNORECASE,
        )[0].strip()
        # Cortar en palabras de programa
        topic = re.split(
            r"\b(posgrados?|programas?|maestrias?|especializ\w*|doctorad\w*)\b",
            topic, maxsplit=1, flags=re.IGNORECASE,
        )[0].strip()
        # Limpiar puntuación final y espacios
        topic = re.sub(r"[?.!,;]+$", "", topic).strip()
        topic = re.sub(r"\s+", " ", topic).strip()

        # Rechazar si el topic es solo stopwords (ej: "los", "el", "de")
        words = topic.lower().split()
        if not words or all(w in _TOPIC_STOPWORDS for w in words):
            continue
        if len(topic) >= 3:
            return topic

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Campo exacto
# ─────────────────────────────────────────────────────────────────────────────

# Patrones de "donde" que NO son preguntas de ubicación del programa
_DONDE_EXCLUSION_RE = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bdonde\s+(\w+\s+)?(puedo|puede|podria|podemos|pueden)\s+\w*\s*(trabajar|inscribir\w*|estudiar|conseguir|obtener|ver|encontrar|aplicar|registrar\w*)\b",
        r"\bdonde\s+(\w+\s+)?(inscribo|aplico|presento|registro|matriculo)\b",
        r"\bdonde\s+(se\s+)?consigue\b",
        r"\bdonde\s+\w*\s*inscribir\w*\b",
    ]
]


def detect_field(q_norm: str) -> Optional[str]:
    """
    Detecta el campo exacto de un programa que se está consultando.
    Retorna None si la pregunta es sobre contenido curricular (malla, pensum,
    electivas, opciones de grado) o no corresponde a ningún campo.
    """
    q = (q_norm or "").strip()
    if not q:
        return None

    # Guardia: contenido curricular → nunca un campo exacto
    if any(k in q for k in [
        "materias", "malla", "pensum", "asignaturas", "plan de estudios",
        "electiv", "optativas",
    ]):
        return None

    # ── DURATION ─────────────────────────────────────────────────────────────
    if re.search(r"\bduracion\b", q):
        return "duration"
    if re.search(r"\bcuanto\s+(tiempo\s+)?(dura|tarda|demora|toma)\b", q):
        return "duration"
    if re.search(r"\bdura\s", q) or q.endswith("dura"):
        return "duration"
    if re.search(r"\bcuantos?\s+(semestres?|anos?|periodos?)\b", q):
        return "duration"
    if re.search(r"\bsemestres?\s+(tiene|son|dura|tarda)\b", q):
        return "duration"
    if re.search(r"\bde\s+\d+\s+semestres?\b", q):
        return "duration"
    if re.search(r"\bcuanto\s+tiempo\b", q):
        return "duration"

    # ── CREDITS ──────────────────────────────────────────────────────────────
    if re.search(r"\bcreditos?\b", q):
        return "credits"

    # ── COST ─────────────────────────────────────────────────────────────────
    if re.search(r"\bvale\s+la\s+pena\b", q):
        return None

    # NUEVO: excluir contextos financieros fuera de dominio
    if re.search(r"\b(dolar|euro|divisa|bolsa|accion|cotizacion|tasa de cambio|peso|libra)\b", q):
        return None

    if re.search(r"\b(costo|cuesta|precio|valor|vale|inversion|matricula|pagar|pago)\b", q):
        return "cost"
    if re.search(r"\b(caro|cara|barato|barata|economico|economica|costoso|costosa)\b", q):
        return "cost"
    if re.search(r"\bfinanciacion\b", q):
        return "cost"

    # ── MODALITY ─────────────────────────────────────────────────────────────
    if re.search(r"\bmodalidad\b", q):
        return "modality"
    if re.search(r"\b(presencial|virtual|semipresencial|a\s+distancia|hibrido|hibrida)\b", q):
        return "modality"

    # ── LOCATION ─────────────────────────────────────────────────────────────
    # Excluir "donde puedo trabajar/inscribir/etc." que no son preguntas de sede
    if any(exc.search(q) for exc in _DONDE_EXCLUSION_RE):
        pass
    elif re.search(r"\b(ubicacion|ubicado|ubicada|sede|campus|ciudad)\b", q):
        return "location"
    elif re.search(r"\bdonde\b", q):
        return "location"

    # ── SCHEDULE ─────────────────────────────────────────────────────────────
    if re.search(r"\bhorarios?\b", q):
        return "schedule"
    if re.search(r"\b(jornada|sabados?|domingos?|nocturno|diurno)\b", q):
        return "schedule"
    if re.search(r"\b(dias?\s+de\s+(clase|clases)|clases?\s+(los|entre))\b", q):
        return "schedule"
    if re.search(r"\ben\s+que\s+dias?\b", q):
        return "schedule"

    # ── DEGREE ───────────────────────────────────────────────────────────────
    if re.search(r"\b(titulo|titulacion|titulaciones?)\b", q):
        return "degree"

    # ── REGISTRY ─────────────────────────────────────────────────────────────
    if re.search(r"\bregistro\b", q):
        return "registry"

    # ── YEAR UPDATE ──────────────────────────────────────────────────────────
    if re.search(r"\bano\s+(de\s+)?(inicio|actualizacion|resolucion|lanzamiento)\b", q):
        return "year_update"
    if re.search(r"\b(actualizacion|actualizado|actualizada)\b", q):
        return "year_update"
    if re.search(r"\bfecha\s+de\s+(inicio|lanzamiento|apertura|resolucion)\b", q):
        return "year_update"
    if re.search(r"\bcuando\s+(fue|inicio|empezo|comenzo|se\s+creo|se\s+abrio)\b", q):
        return "year_update"

    # ── DIVISION ─────────────────────────────────────────────────────────────
    if re.search(r"\bdivision\b", q) and not re.search(
        r"\b(en\s+general|a\s+que|que\s+division|pertenece|facultad)\b", q
    ):
        return "division"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Comparaciones globales
# ─────────────────────────────────────────────────────────────────────────────

def is_global_comparison(q_norm: str) -> bool:
    superlatives = [
        "mas ", "menos ", "mayor ", "menor ", "minimo", "maximo",
        "cual tiene", "cual dura", "cual cuesta",
        "cual es el mas", "cual es el menos",
        "top ", "ranking", "compar", "mejor ", "peor ", "barato", "caro",
    ]
    return any(t in q_norm for t in superlatives)


def resolve_minmax_mode(q_norm: str) -> str:
    if any(w in q_norm for w in ["economico", "economica", "barato", "barata", "menos ", "menor ", "minimo"]):
        return "min"
    if any(w in q_norm for w in ["mas ", "mayor ", "maximo", "caro", "cara"]):
        return "max"
    return "min"
