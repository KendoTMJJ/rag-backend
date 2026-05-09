import re
from typing import Optional

from src.nlp.text_normalizer import normalize_and_fix


# ─────────────────────────────────────────────────────────────────────────────
# Constantes compartidas
# ─────────────────────────────────────────────────────────────────────────────

# Detecta preguntas sobre la institución en general (no sobre un programa específico)
_INSTITUTIONAL_SIGNAL = re.compile(
    r"\b(en\s+la\s+universidad|de\s+la\s+universidad|la\s+universidad|"
    r"en\s+la\s+usta|de\s+la\s+usta)\b"
)
# Detecta contexto de programa específico que anula la señal institucional
_HAS_PROGRAM_CONTEXT = re.compile(
    r"\b(maestr|especializ|doctorad|posgrado|programa)\b"
)

_OVERVIEW_TRIGGERS = {
    "que me dices", "que me puede decir", "que me puedes decir",
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
    "darme un resumen", "me des un resumen", "puedes darme un resumen",
    "me puedes dar un resumen", "quiero un resumen", "necesito un resumen",
    "cual es el enfoque", "enfoque del posgrado", "enfoque del programa",
    "que se aprende en", "que ensenan", "que se aprende",
    "cuentame del programa", "presentame el programa",
    # solicitudes de información completa / más info
    "dame toda la informacion", "toda la informacion",
    "informacion completa", "informacion relevante",
    "dame mas informacion", "mas informacion sobre",
    "dime todo", "cuentame todo", "explicame todo",
    "todo sobre", "todo lo que sabes", "todo lo relacionado",
    "dame todo", "quiero saber todo", "informacion general",
}

_LISTING_QUALIFIERS = [
    r"\bque\s+programas\b",
    r"\bcuales?\s+programas\b",
    r"\bque\s+posgrados\b",
    r"\bcuales?\s+posgrados\b",
    r"\bprogramas\s+(hay|son|existen|ofrece)\b",
    r"\bprogramas\s+(del?|de\s+la)\s+area\b",
    # "¿cuáles son los programas disponibles?" — "son" entre cuáles y programas
    r"\bcuales?\s+son\s+(los\s+|las\s+)?(programas?|posgrados?)\b",
    # "programas disponibles", "programas que tienen"
    r"\bprogramas?\s+disponibles?\b",
    # "¿qué modalidades/horarios/sedes hay en la universidad?" → listado institucional
    r"\bque\s+(modalidades?|horarios?\s+de\s+clases?|sedes?|costos?|precios?)\s+(hay|existen|tienen|manejan|ofrecen)\b",
    r"\bcuales?\s+(modalidades?|sedes?)\s+(hay|existen|tienen|son)\b",
]

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
    if re.search(r"\b(program\w*|posgrad\w*|postgrad\w*|maestr\w*|doctorad\w*|especializ\w*|oferta\w*\s+academica\w*|oferta\w*\s+educativa\w*)\b", q_norm):
        return True
    if re.search(r"\b(pr\w*gram\w*|prog\w*ram\w*)\b", q_norm):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Intención de listado
# ─────────────────────────────────────────────────────────────────────────────

_BARE_PROGRAM_TYPES = frozenset({
    "maestria", "maestrias",
    "doctorado", "doctorados",
    "especializacion", "especializaciones",
})


def looks_like_programs_listing(q_norm: str) -> bool:
    q = (q_norm or "").strip()
    if not q:
        return False

    # "Maestria" / "Doctorado" / "Especializacion" solos → listado por tipo
    if q in _BARE_PROGRAM_TYPES:
        return True

    if any(t in q for t in _OVERVIEW_TRIGGERS):
        if not any(re.search(lq, q) for lq in _LISTING_QUALIFIERS):
            return False

    if is_reasoning_question(q):
        return False

    if is_false_listing(q):
        return False

    # Cualquier qualifier de listado es suficiente por sí solo
    if any(re.search(lq, q) for lq in _LISTING_QUALIFIERS):
        return True

    if looks_like_program_word(q):
        if re.search(
            r"\b(que|hay|existe|existen|ofrecen|ofrece|tienen|tiene|algun|alguna|alguno)\b", q
        ):
            if re.search(r"\bel\s+program\w+\s+tiene\b", q):
                return False
            return True

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

    if any(re.search(lq, q) for lq in _LISTING_QUALIFIERS):
        return False

    if any(w in q for w in _TABULAR_QUALIFIERS):
        return False

    if looks_like_programs_listing(q):
        return False

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


_TABULAR_PREFIX_RE = re.compile(
    r"^\s*(?:plan\s+de\s+estudios|malla(?:\s+curricular)?|pensum|materias?|asignaturas?)\s+",
    re.IGNORECASE,
)


def extract_program_candidate(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    # Strip leading tabular keywords so "plan de estudios de administracion"
    # becomes "de administracion" and we extract "administracion", not "estudios de administracion".
    t = _TABULAR_PREFIX_RE.sub("", t).strip()
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

    # "Maestria" / "Doctorado" / "Especializacion" solos → el tipo ES el topic
    if qn in _BARE_PROGRAM_TYPES:
        return qn

    if any(t in qn for t in _OVERVIEW_TRIGGERS):
        if not any(re.search(lq, qn) for lq in _LISTING_QUALIFIERS):
            return None
    if is_reasoning_question(qn):
        return None

    patterns = [
        r"\bque\s+tienen\s+que\s+ver\s+con\s+(.+)$",
        r"\brelacionad[oa]s?\s+con\s+(.+)$",
        r"\bvinculad[oa]s?\s+con\s+(.+)$",
        r"\bdel?\s+area\s+de\s+(.+)$",
        r"\ben\s+el\s+area\s+de\s+(.+)$",
        r"\bsobre\s+(.+)$",
        r"\bpara\s+(.+)$",
        r"\bde\s+(?!(?:los?|las?|un[oa]?)\s+(?:programas?|posgrados?|maestrias?|especializ\w*|doctorad\w*))\s*(.+)$",
        r"\ben\s+(.+)$",
    ]

    for p in patterns:
        m = re.search(p, qn, flags=re.IGNORECASE)
        if not m:
            continue
        topic = m.group(1).strip()
        # FIX G1: agregar "recomiendan/sugieren" al split de verbos
        topic = re.split(
            r"\b(hay|existe|existen|ofrece|ofrecen|tiene|tienen|son|estan"
            r"|recomiendan|recomienda|sugieren|sugiere|conviene|convendria)\b",
            topic, maxsplit=1, flags=re.IGNORECASE,
        )[0].strip()
        topic = re.split(
            r"\b(posgrados?|programas?|maestrias?|especializ\w*|doctorad\w*)\b",
            topic, maxsplit=1, flags=re.IGNORECASE,
        )[0].strip()
        topic = re.sub(r"[?.!,;]+$", "", topic).strip()
        topic = re.sub(r"\s+", " ", topic).strip()

        words = topic.lower().split()
        if not words or all(w in _TOPIC_STOPWORDS for w in words):
            continue
        if len(topic) >= 2:
            return topic

    # "Qué maestrías hay?" / "Que doctorados tienen?" → sin topic adicional,
    # pero el tipo de programa está en la query → usarlo como topic
    _singular_map = {
        "maestrias": "maestria",
        "doctorados": "doctorado",
        "especializaciones": "especializacion",
    }
    for bare in _BARE_PROGRAM_TYPES:
        if re.search(rf"\b{bare}\b", qn, re.IGNORECASE):
            return _singular_map.get(bare, bare)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Campo exacto
# ─────────────────────────────────────────────────────────────────────────────

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

    # Exclusión: "¿qué hora es/son?" nunca es una consulta de horarios académicos
    if re.search(r"\bque\s+hora\s+(es|son|era|eran|fue|fueron)\b", q):
        return None

    # Guard: pregunta sobre la institución en general, sin contexto de programa
    # → no puede resolverse como campo de un programa específico
    if _INSTITUTIONAL_SIGNAL.search(q) and not _HAS_PROGRAM_CONTEXT.search(q):
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
    # FIX D7: "qué tan largo es" → duration
    if re.search(r"\btan\s+largo\b", q):
        return "duration"

    # ── CREDITS ──────────────────────────────────────────────────────────────
    if re.search(r"\bcreditos?\b", q):
        return "credits"

    # ── COST ─────────────────────────────────────────────────────────────────
    if re.search(r"\bvale\s+la\s+pena\b", q):
        return None

    if re.search(r"\b(dolar|euro|divisa|bolsa|accion|cotizacion|tasa de cambio|peso|libra)\b", q):
        return None

    if re.search(r"\b(costo|cuesta|precio|valor|vale|inversion|matricula|pagar|pago)\b", q):
        return "cost"
    # FIX D8: "es caro" → cost
    if re.search(r"\b(caro|cara|barato|barata|economico|economica|costoso|costosa)\b", q):
        return "cost"
    if re.search(r"\bfinanciacion\b", q):
        return "cost"

    # ── MODALITY ─────────────────────────────────────────────────────────────
    if re.search(r"\bmodalidad(?:es)?\b", q):
        return "modality"
    if re.search(r"\b(presencial|virtual|semipresencial|a\s+distancia|hibrido|hibrida)\b", q):
        return "modality"

    # ── LOCATION ─────────────────────────────────────────────────────────────
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
    # FIX A8/C10: "a qué división pertenece" debe retornar "division"
    # Se eliminan "a\s+que" y "pertenece" de la lista de exclusión
    if re.search(r"\bdivision\b", q) and not re.search(
        r"\b(en\s+general|que\s+division|facultad)\b", q
    ):
        return "division"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Recomendaciones / orientación vocacional
# ─────────────────────────────────────────────────────────────────────────────

_RECOMMENDATION_RE = [
    re.compile(
        r"\b(que|qu[eé])\s+(puedo|podria|debo|deberia|me\s+conviene)\s+estudiar\b", re.IGNORECASE),
    re.compile(r"\b(que|qu[eé])\s+(programa|posgrado|maestr\w+|especializ\w+|doctorad\w+)\s+(me\s+)?(recomiendas?|sugieren?|aconsejan?|conviene|seria\s+(?:bueno|ideal|adecuado))\b", re.IGNORECASE),
    re.compile(
        r"\b(que|qu[eé])\s+me\s+(recomiend(?:a|as|an)|sugieren?|aconsejan?)\b", re.IGNORECASE),
    re.compile(
        r"\bpara\s+alguien\s+(como\s+yo|que\s+(?:es|trabaja|se\s+dedica|tiene\s+perfil))\b", re.IGNORECASE),
    re.compile(
        r"\btengo\s+(?:perfil|experiencia|formacion)\s+(?:de|en)\b", re.IGNORECASE),
    re.compile(
        r"\bsoy\s+\w+(\s+\w+)?\s+y\s+(?:quiero|busco|deseo|me\s+gustar[ií]a)\s+(?:estudiar|especializarme|hacer\s+(?:un|una)\s+posgrado)\b", re.IGNORECASE),
]

_PROFILE_PATTERNS = [
    r"\bsi\s+soy\s+(?:un[ao]?\s+)?(.+?)(?:[?,!]|$)",
    r"\bsiendo\s+(?:un[ao]?\s+)?(.+?)(?:[?,!]|$)",
    r"\bsi\s+me\s+gustan?\s+(.+?)(?:[?,!]|$)",
    r"\bsi\s+me\s+interesa(?:n)?\s+(.+?)(?:[?,!]|$)",
    r"\bme\s+gustan?\s+(.+?)\s+(?:que\s+puedo|que\s+debo|y\s+(?:quiero|busco|deseo))",
    r"\bque\s+le\s+gustan?\s+(.+?)(?:[?,!]|$)",
    r"\btengo\s+(?:perfil|experiencia|formacion)\s+(?:de|en)\s+(.+?)(?:[?,!]|$)",
    r"\bsoy\s+(?:un[ao]?\s+)?(.+?)\s+(?:y\s+(?:quiero|busco|deseo|me\s+gustar[ií]a)|[,]\s*que|que\s+(?:me\s+)?(?:recomiendas?|sugieren?|aconsejan?))",
    r"\bsoy\s+(?:un[ao]?\s+)?(.+?)(?:[?,!]|$)",
    r"\bme\s+dedico\s+(?:a|al?)\s+(.+?)(?:[?,!]|$)",
    r"\btrabajo\s+(?:en|como|de)\s+(.+?)(?:[?,!]|$)",
]

_PROFILE_LEADING_ARTICLES = re.compile(
    r"^(?:el|la|los|las|un|una|unos|unas)\s+", re.IGNORECASE)


def is_recommendation_query(q_norm: str) -> bool:
    """Detecta si la pregunta es una solicitud de orientación o recomendación
    de programas basada en el perfil o intereses del usuario."""
    return any(p.search(q_norm or "") for p in _RECOMMENDATION_RE)


def extract_profile_for_recommendation(question: str) -> Optional[str]:
    """
    Extrae el perfil o interés del usuario de una pregunta de recomendación.

    Usa regex estructural para capturar lo que venga después de patrones
    como "si soy", "siendo", "si me gusta", etc. sin listas quemadas.
    Si ningún patrón específico captura el perfil, retorna la pregunta
    normalizada completa para que el LLM decida.

    Returns:
        Texto del perfil/interés, o None si la pregunta no es de recomendación.
    """
    qn = normalize_and_fix(question)
    if not qn or not is_recommendation_query(qn):
        return None

    for pattern in _PROFILE_PATTERNS:
        m = re.search(pattern, qn, re.IGNORECASE)
        if not m:
            continue
        profile = m.group(1).strip()
        profile = re.sub(r"[?.!,;]+$", "", profile).strip()
        profile = _PROFILE_LEADING_ARTICLES.sub("", profile).strip()
        profile = re.sub(r"\s+", " ", profile)
        words = profile.lower().split()
        if words and not all(w in _TOPIC_STOPWORDS for w in words) and len(profile) >= 3:
            return profile

    return qn


# ─────────────────────────────────────────────────────────────────────────────
# Detección de escalación (contacto humano / canales de la institución)
# ─────────────────────────────────────────────────────────────────────────────

_ESCALATION_SIGNALS = frozenset({
    "contacto", "contactarlos", "contactarme", "contactar",
    "whatsapp", "wasap", "wsp",
    "telefono", "celular", "llamar",
    "correo", "email",
    "hablar con",
    "canales de contacto", "canales de comunicacion",
    "comunicarme", "comunicarlos", "comunicarse",
    "alguien de la", "una persona", "un asesor",
    "numero de contacto",
})


def looks_like_escalation_candidate(q_norm: str) -> bool:
    """Pre-filtro rápido: True si la pregunta contiene señales de escalación.
    Solo sirve para decidir si vale la pena llamar al LLM clasificador."""
    q = (q_norm or "").strip()
    return any(s in q for s in _ESCALATION_SIGNALS)


# ─────────────────────────────────────────────────────────────────────────────
# Comparaciones globales
# ─────────────────────────────────────────────────────────────────────────────

def is_global_comparison(q_norm: str) -> bool:
    superlatives = [
        "mas ", "menos ", "mayor ", "menor ", "minimo", "maximo",
        "cual tiene", "cual dura", "cual cuesta",
        "cual es el mas", "cual es el menos",
        "top ", "ranking", "compar", "mejor ", "peor ", "barato", "mas caro",
    ]
    return any(t in q_norm for t in superlatives)


def resolve_minmax_mode(q_norm: str) -> str:
    if any(w in q_norm for w in ["economico", "economica", "barato", "barata", "menos", "menor ", "minimo"]):
        return "min"
    if any(w in q_norm for w in ["mayor ", "maximo", "mas caro", "mas cara"]):
        return "max"
    return "min"
