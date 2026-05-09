# Pipeline RAG — Árbol de Decisión

Documentación del pipeline principal en [src/services/rag_pipeline.py](../src/services/rag_pipeline.py).

---

## Visión general

El pipeline recibe una pregunta y devuelve una ruta (`route`) con datos estructurados. No devuelve texto libre directamente — el NestJS (chat gateway) interpreta la ruta para mostrar la respuesta apropiada al usuario.

```
Usuario → ask(question, chatSessionId)
            │
            ├─ Sanitización (SQL injection, caracteres peligrosos)
            ├─ Normalización de texto
            ├─ Detección de ruido / gibberish
            ├─ Atajos rápidos (saludo, gracias, capacidades)
            ├─ Análisis centralizado (_analyze_request)
            │     ├─ Guardrail de dominio / inyección
            │     ├─ Extracción de intención (listing, overview, field...)
            │     └─ Validación de resolución de programa por memoria
            └─ Árbol de decisión (ver abajo)
```

---

## Árbol de decisión completo

El método `ask()` sigue este orden de prioridad. La primera rama que aplica devuelve el resultado:

```
1. Sanitización + normalización
2. Input vacío → EMPTY
3. Ruido / gibberish → NOT_FOUND
4. Saludo → GREETING
5. Agradecimiento → THANKS
6. "¿Qué puedes hacer?" → CAPABILITIES (mensaje con lista de capacidades)
7. Análisis centralizado (_analyze_request)
8. Guardrail fallido → INJECTION_BLOCKED o OUT_OF_DOMAIN
9. Señal de escalación → ESCALATION_INTENT (si LLM confirma)
10. Dominio mínimo con sesión activa (confidence < 10%) → OUT_OF_DOMAIN
11. "snies XXXXX" exacto → PROGRAM_SELECTED (+ resuelve pendientes)
12. Pendiente NEED_PROGRAM + respuesta sin intent estructurado → resolver programa
13. Título de programa detectado (sin keywords de consulta) → PROGRAM_SELECTED o NOT_FOUND
14. Narrative field detectado → NARRATIVE_SQL o NARRATIVE_NOT_FOUND o NEED_PROGRAM
15. Overview detectado → PROGRAM_OVERVIEW o NEED_PROGRAM
16. Malla / electivas / opciones de grado → CURRICULUM / CURRICULUM_SEMESTER / ELECTIVES / DEGREE_OPTIONS
17. Inscripción → INSCRIPTION_LINK / INSCRIPTION_NO_LINK / INSCRIPTION_NEED_PROGRAM
18. Comparación global (más caro, más largo...) → PROGRAM_MINMAX
19. Perfil de recomendación → LIST_PROGRAMS_FILTERED (o _EMPTY)
20. Listado de programas → LIST_PROGRAMS o LIST_PROGRAMS_FILTERED
21. Campo exacto detectado (duración, costo, créditos...) → PROGRAM_FIELD o NEED_PROGRAM
22. Pendientes de campo/narrativo con contexto → PROGRAM_FIELD / NARRATIVE_SQL
23. Pregunta general → GENERAL_CONTROLLED (usa LLM con contexto del programa activo)
24. Búsqueda semántica vectorial (fallback)
     ├─ Sin resultados → NOT_FOUND
     ├─ Pregunta de razonamiento con similitud < 0.80 → LOW_CONFIDENCE
     ├─ Similitud < 0.60 → LOW_CONFIDENCE
     └─ LLM genera respuesta → DEFAULT_RAG
```

---

## Rutas del pipeline

| Ruta | Trigger | Usa LLM |
|---|---|---|
| `GREETING` | Saludo exacto | No |
| `THANKS` | Agradecimiento exacto | No |
| `CAPABILITIES` | "¿Qué puedes hacer?" | No |
| `OUT_OF_DOMAIN` | Guardrail rechaza dominio | No |
| `INJECTION_BLOCKED` | Guardrail detecta inyección | No |
| `ESCALATION_INTENT` | Usuario pide contacto humano | Sí (clasificador) |
| `PROGRAM_SELECTED` | Programa identificado (sin pregunta específica) | No |
| `NEED_PROGRAM` | Requiere programa pero no fue especificado | No |
| `PROGRAM_FIELD` | Campo exacto consultado (costo, duración...) | No |
| `NARRATIVE_SQL` | Campo narrativo (perfil egreso, diferencial...) | No |
| `NARRATIVE_NOT_FOUND` | Campo narrativo vacío en DB | No |
| `PROGRAM_OVERVIEW` | "¿De qué trata el programa?" | Sí |
| `CURRICULUM` | Malla curricular completa | No |
| `CURRICULUM_SEMESTER` | Materias de un semestre específico | No |
| `ELECTIVES` | Electivas del programa | No |
| `DEGREE_OPTIONS` | Opciones de grado | No |
| `INSCRIPTION_LINK` | Cómo inscribirse (con enlace) | No |
| `INSCRIPTION_NO_LINK` | Cómo inscribirse (sin enlace en DB) | No |
| `LIST_PROGRAMS` | Listar todos los programas | No |
| `LIST_PROGRAMS_FILTERED` | Listar programas filtrados por tema/tipo | Sí (filtro) |
| `PROGRAM_MINMAX` | "¿Cuál es el más caro / largo?" | No |
| `DEFAULT_RAG` | Búsqueda semántica vectorial | Sí |
| `LOW_CONFIDENCE` | Similitud baja / pregunta de razonamiento | No |
| `NOT_FOUND` | Sin resultados relevantes | No |
| `GENERAL_CONTROLLED` | Pregunta general con programa activo en sesión | Sí |

---

## Estado de sesión

Cada sesión (`chatSessionId`) tiene un estado persistido en `TTLStore` (diccionario en memoria con TTL):

| Clave | Tipo | Descripción |
|---|---|---|
| `snies` | `str` | SNIES del programa activo — permite follow-ups sin repetir el nombre |
| `pending_field` | `str \| None` | Campo pendiente (ej: `"cost"`) cuando el usuario preguntó sin especificar programa |
| `pending_narrative` | `str \| None` | Campo narrativo pendiente (ej: `"graduated_profile"`) |
| `pending_overview` | `bool` | Overview pendiente — esperando que el usuario especifique el programa |
| `pending_tabular` | `str \| None` | Dato tabular pendiente: `"curriculum"`, `"electives"`, `"degree_options"` |

**Flujo de `pending`:** cuando el usuario pregunta "¿cuánto cuesta?" sin un programa activo, el pipeline guarda `pending_field = "cost"` y devuelve `NEED_PROGRAM`. En el siguiente turno, cuando el usuario dice el nombre del programa, el pipeline lo resuelve y usa `pending_field` para responder directamente con `PROGRAM_FIELD`.

**TTL de sesión:** configurable con `SESSION_TTL_SECONDS` (default: 2 horas). Sesiones antiguas se evictan automáticamente.

---

## Módulos NLP

### `intent_utils.py` — Extracción de intención

| Función | Detecta |
|---|---|
| `looks_like_programs_listing(q)` | Pregunta que pide listar programas |
| `looks_like_program_overview(q)` | "¿De qué trata...?", "Descríbeme..." |
| `looks_like_general_question(q)` | Preguntas sobre la institución en general |
| `is_false_listing(q)` | Frases que parecen listing pero son sobre contenido |
| `is_reasoning_question(q)` | Comparaciones, "¿cuál conviene?", "diferencia entre..." |
| `is_global_comparison(q)` | "el más caro", "el más largo" |
| `detect_field(q)` | Campo exacto: `duration`, `cost`, `credits`, `modality`, `location`, `schedule`, `degree`, `registry`, `division`, `year_update` |
| `extract_snies(q)` | Número SNIES en la pregunta |
| `extract_semester(q)` | "semestre 3", "tercer semestre" |
| `extract_program_candidate(q)` | "Maestría en X" → extrae "X" |
| `extract_topic_for_listing(q)` | Tema del listado: "programas de salud" → `"salud"` |
| `extract_profile_for_recommendation(q)` | Perfil del usuario para recomendación |
| `looks_like_escalation_candidate(q)` | Pre-filtro rápido de señales de escalación |

### `domain_taxonomy.py` + `domain_guardrail.py` — Control de dominio

- `check_domain(q)` — devuelve `DomainResult(allowed, reason, detail)`. Bloquea preguntas con `reason = "injection"` o `reason = "off_domain"`.
- `score_domain(q)` — puntuación de confianza en el dominio (0–1). Usado como verificación mínima cuando hay sesión activa.

**Importante:** Las señales de dominio académico están declaradas **solo** en `domain_taxonomy.py`. No agregar señales de dominio en otros archivos.

---

## Resolución de programa (`_ensure_program`)

Cuando el pipeline necesita identificar un programa, sigue este orden:

1. **SNIES explícito** — regex `\b\d{5,10}\b` en la pregunta → fuente `"explicit"`
2. **SQL por texto completo** — `resolve_program_by_name(full_question)` → fuente `"sql_full"`
3. **SQL por candidato extraído** — `extract_program_candidate()` + SQL → fuente `"sql_name"`
4. **Embedding** — búsqueda vectorial de similitud (si `allow_embedding=True`) → fuente `"embedding"`
   - Validación extra: el candidato debe compartir tokens con el nombre resuelto (`_candidate_matches_resolved_name`)
   - Validación extra: al menos un token del candidato debe estar en el vocabulario de programas reales (`_candidate_in_program_vocab`)
5. **Memoria de sesión** — SNIES activo de turno anterior (si `allow_memory_fallback=True`) → fuente `"memory"`

Si ninguna fuente resuelve el programa, devuelve `(None, "none")`.

---

## Embeddings

El modelo `intfloat/multilingual-e5-base` requiere prefijos específicos:

```python
# Para indexar documentos (al subir el Excel):
embed_document(text)   # antepone "passage: " al texto

# Para consultas del usuario:
embed_query(text)      # antepone "query: " al texto
```

**No invertir estos prefijos** — la similitud semántica se degrada significativamente.

Los vectores generados están normalizados → el producto punto equivale a la similitud coseno.

---

## Agregar una nueva ruta al pipeline

1. **Definir el trigger** en `intent_utils.py` (nueva función de detección o ampliar una existente).
2. **Agregar la señal de dominio** en `domain_taxonomy.py` si el nuevo tema puede confundirse con contenido fuera de dominio.
3. **Insertar la rama** en el método `ask()` de `rag_pipeline.py` en el orden de prioridad correcto.
4. **Devolver el payload** con `{"route": "NUEVA_RUTA", "data": {...}}` usando `_return_no_llm()` si no necesita LLM.
5. **Manejar la ruta** en el NestJS chat gateway (`chat.gateway.ts`) para que el socket envíe la respuesta correcta al cliente.
6. **Manejar el `responseType`** en el n8n workflow si la ruta también pasa por mesa de ayuda.

---

## Thresholds de similitud

Definidos en `src/services/retrieval_service.py` como `PROGRAM_RESOLVE_MIN_SIM`:

| Contexto | Threshold | Uso |
|---|---|---|
| `"default"` | 0.65 | Resolución estándar de programa |
| `"narrative"` | 0.70 | Campos narrativos (perfil, diferencial) |
| `"tabular"` | 0.68 | Malla, electivas, opciones de grado |
| `"inscription"` | 0.70 | Preguntas de inscripción |
| `"block_4b"` | 0.72 | Resolución cuando hay pendientes acumulados |

Para búsqueda semántica general (`semantic_search`):
- `min_similarity = 0.55` para resultados generales
- `min_similarity = 0.50` para búsqueda anclada al programa activo
- Respuesta con `LOW_CONFIDENCE` si `best_sim < 0.60`
- Aviso de contexto moderado si `best_sim < 0.75`

---

## Output validator

`src/quality/output_validator.py` — valida la respuesta del LLM antes de devolverla:

- `confidence` — puntuación combinada de similitud + señales de confianza
- `hallucination_suspected` — True si la respuesta contiene frases que indican invención
- `should_block` — True si la respuesta debe bloquearse (confianza muy baja)
- `add_disclaimer` — True si se debe agregar nota de verificación oficial

Umbrales definidos como `CONFIDENCE_SHOW` y `CONFIDENCE_BLOCK` en el módulo.
