# RAG Backend Santo Tomás Tunja

Backend de Recuperación Aumentada por Generación (RAG) para el Asistente de Posgrados de la Universidad Santo Tomás Seccional Tunja. Expone un pipeline conversacional que responde preguntas sobre programas de posgrado usando búsqueda semántica vectorial + LLM local, y un módulo de helpdesk con intenciones configurables dinámicamente desde base de datos.

## Stack

- **FastAPI** + Uvicorn
- **PostgreSQL** + **pgvector** — Base de datos con soporte vectorial para embeddings
- **SQLAlchemy** — ORM
- **LangChain** + **Ollama** — LLM local (`qwen2.5:3b-instruct-q8_0`)
- **intfloat/multilingual-e5-base** — Modelo de embeddings semánticos (768 dims)
- **Pandas** + **openpyxl** — Ingesta de conocimiento desde Excel

---

## Arquitectura del pipeline RAG

```mermaid
flowchart TD
    POST["POST /chat\n{question, chatSessionId}"]

    POST --> SAN["InputSanitizer\ndetección SQL injection"]
    SAN --> NORM["text_normalizer\nnormalize_and_fix()"]
    NORM --> NOISE{"_is_noise_input?\n(gibberish, >10 dígitos)"}
    NOISE -->|sí| NF["NOT_FOUND\n'No entendí tu mensaje'"]
    NOISE -->|no| TRIVIAL{"Atajos triviales\n(saludo, gracias, capacidades)"}
    TRIVIAL -->|match| TR["GREETING / THANKS /\nCAPABILITIES"]
    TRIVIAL -->|no match| GUARD["domain_guardrail\ncheck_domain()"]
    GUARD -->|bloqueado| OOD["OUT_OF_DOMAIN /\nINJECTION_BLOCKED"]
    GUARD -->|permitido| ANALYZE["_analyze_request()\nRequestAnalysis"]

    ANALYZE --> ESC{"looks_like_escalation?"}
    ESC -->|sí + LLM confirma| ESCI["ESCALATION_INTENT"]
    ESC -->|no| ROUTE{"Intent Router"}

    ROUTE -->|narrative_field| NARR["NARRATIVE_SQL\nperfil egreso / ingreso / ocupacional"]
    ROUTE -->|is_overview| OVW["PROGRAM_OVERVIEW\nLLM + contexto narrativo"]
    ROUTE -->|curriculum/electives/degree| TAB["CURRICULUM / ELECTIVES /\nDEGREE_OPTIONS"]
    ROUTE -->|inscription| INS["INSCRIPTION_LINK"]
    ROUTE -->|listing| LIST["LIST_PROGRAMS /\nLIST_PROGRAMS_FILTERED"]
    ROUTE -->|field| FLD["PROGRAM_FIELD\nduración / costo / créditos..."]
    ROUTE -->|default| VEC["RetrievalService\nsemantic_search() pgvector"]

    VEC --> SIMCHECK{"best_sim < 0.60?"}
    SIMCHECK -->|sí| LC["LOW_CONFIDENCE"]
    SIMCHECK -->|no| RERANK["context_reranker\nrerank_top()"]
    RERANK --> LLM["LLMService / Ollama\ngenerate()"]
    LLM --> VAL["output_validator\nconfianza + hallucination check"]
    VAL --> RAGOK["DEFAULT_RAG"]
```

---

## Arquitectura de servicios

```mermaid
graph TB
    subgraph API["API Layer (FastAPI)"]
        CR["/chat\nRAG pipeline"]
        HR["/helpdesk/classify\n/helpdesk/category/{intent}"]
        AHR["/helpdesk/admin/categories\nCRUD — x-internal-key"]
        AKR["/admin/knowledge/upload\nx-internal-key"]
        HLT["/health"]
    end

    subgraph Services["Services Layer"]
        RP["RAGPipeline\norquestador principal"]
        RS["RetrievalService\nbúsqueda vectorial"]
        LS["LLMService\nOllama wrapper + historial"]
        HS["HelpdeskService\nclasificación de intenciones"]
        KS["KnowledgeIngestionService\ningesta Excel → DB"]
    end

    subgraph NLP["NLP Layer"]
        TN["text_normalizer"]
        DG["domain_guardrail"]
        IU["intent_utils\nextract_snies, detect_field..."]
        IS["input_sanitizer"]
        PR["prompts.py\ntodos los prompts centralizados"]
    end

    subgraph Quality["Quality Layer"]
        RG["request_gate"]
        CR2["context_reranker"]
        OV["output_validator"]
    end

    subgraph DB["PostgreSQL + pgvector"]
        P["Program\nsnies · nombre · modalidad · duración"]
        C["Course\nsemestre · nombre · créditos"]
        E["Elective\nnombre · ciclo"]
        DO["DegreeOption\nnombre · requisitos"]
        PE["ProgramEmbedding\nsección · contenido · vector 768d"]
        PN["ProgramNarrative\nperfil_ingreso · egresado · ocupacional"]
        HC["HelpdeskCategory\nintent · label · pdf_url · descripción"]
    end

    CR --> RP
    HR --> HS
    AHR --> HC
    AKR --> KS

    RP --> RS & LS & NLP & Quality
    HS --> LS & HC
    KS --> P & C & E & DO & PE & PN
    RS --> PE

    LS -->|"ChatOllama\nOllama :11434"| OL[("Ollama\nqwen2.5:3b")]
    RS -->|"embed_query()"| EM[("multilingual-e5-base\nembedding model")]
    KS -->|"embed_document()"| EM
```

---

## Rutas RAG (respuestas posibles)

| Ruta | Cuándo se devuelve |
|---|---|
| `GREETING` | Saludo del usuario |
| `THANKS` | Agradecimiento |
| `CAPABILITIES` | Pregunta sobre qué puede hacer el bot |
| `PROGRAM_SELECTED` | Se identifica un programa sin pregunta específica |
| `PROGRAM_FIELD` | Consulta sobre campo exacto (duración, costo, créditos…) |
| `PROGRAM_OVERVIEW` | Descripción general del programa (LLM) |
| `NARRATIVE_SQL` | Perfil de egreso, ingreso u ocupacional (texto largo de DB) |
| `CURRICULUM` | Malla curricular completa |
| `CURRICULUM_SEMESTER` | Materias de un semestre específico |
| `ELECTIVES` | Electivas disponibles |
| `DEGREE_OPTIONS` | Opciones de grado |
| `LIST_PROGRAMS` | Listado de todos los programas |
| `LIST_PROGRAMS_FILTERED` | Listado filtrado por área o tipo |
| `PROGRAM_MINMAX` | Programa más caro/barato, más largo/corto |
| `INSCRIPTION_LINK` | Link e info de inscripción |
| `DEFAULT_RAG` | Respuesta generada por LLM con contexto vectorial |
| `LOW_CONFIDENCE` | Similitud insuficiente para responder con certeza |
| `NOT_FOUND` | Sin resultados en DB ni vectores |
| `NEED_PROGRAM` | Falta identificar el programa para continuar |
| `OUT_OF_DOMAIN` | Consulta fuera del dominio académico |
| `INJECTION_BLOCKED` | Intento de prompt injection detectado |
| `ESCALATION_INTENT` | Usuario quiere hablar con un agente humano |

---

## Endpoints

| Método | Ruta | Auth | Descripción |
|---|---|---|---|
| `POST` | `/chat` | — | Consulta al pipeline RAG |
| `POST` | `/helpdesk/classify` | — | Clasificar intención de pregunta helpdesk |
| `GET` | `/helpdesk/category/{intent}` | — | Obtener detalle de una categoría |
| `GET` | `/health` | — | Estado de embeddings, DB y Ollama |
| `GET` | `/helpdesk/admin/categories` | `x-internal-key` | Listar categorías helpdesk |
| `POST` | `/helpdesk/admin/categories` | `x-internal-key` | Crear categoría |
| `GET/PATCH/DELETE` | `/helpdesk/admin/categories/{id}` | `x-internal-key` | CRUD categoría |
| `GET` | `/helpdesk/admin/intents` | `x-internal-key` | Listar intenciones disponibles |
| `POST` | `/admin/knowledge/upload` | `x-internal-key` | Subir archivo `.xlsx` de conocimiento |

---

## Modelos de base de datos

```
Program          — id · snies · program_name · division · modality · duration_semesters
                   total_credits · investment_per_semester · location · schedule
                   degree_awarded · qualified_registry · link

Course           — id · program_id(FK) · semester · name · credits · type
Elective         — id · program_id(FK) · name · credits · cycle
DegreeOption     — id · program_id(FK) · option_name · requirements_description

ProgramNarrative — id · program_id(FK) · perfil_ingreso · perfil_egresado
                   perfil_ocupacional · diferencial · requisitos · descripcion

ProgramEmbedding — id · program_id(FK) · section · content · chunk_index
                   embedding Vector(768)   ← pgvector

HelpdeskCategory — id · intent(unique) · label · pdf_url · description · updated_at
```

---

## Estructura del proyecto

```
src/
├── api/
│   └── routes/
│       ├── chat.py               # Endpoint principal RAG
│       ├── helpdesk.py           # Clasificación helpdesk (público)
│       ├── health.py
│       └── admin/
│           ├── helpdesk.py       # CRUD categorías (x-internal-key)
│           └── knowledge.py      # Ingesta Excel (x-internal-key)
├── core/
│   └── config.py                 # Variables de entorno
├── database/
│   └── config.py                 # SessionLocal + Base declarativa
├── models/                       # Entidades SQLAlchemy
├── nlp/
│   ├── domain_guardrail.py       # Guardrail de dominio e inyección
│   ├── domain_taxonomy.py        # Señales de vocabulario académico
│   ├── intent_utils.py           # Extractores: SNIES, campo, semestre...
│   ├── input_sanitizer.py        # Sanitización de entrada
│   └── text_normalizer.py        # Normalización + corrección de typos
├── prompts.py                    # Todos los prompts LLM centralizados
├── quality/
│   ├── context_reranker.py       # Re-ranking de chunks recuperados
│   ├── output_validator.py       # Validación de confianza en salida LLM
│   └── request_gate.py           # Gate de análisis de solicitud
├── services/
│   ├── rag_pipeline.py           # Orquestador principal
│   ├── retrieval_service.py      # Búsqueda semántica vectorial
│   ├── sql_retrieval_service.py  # Consultas SQL directas
│   ├── llm_service.py            # Cliente Ollama + historial de sesión
│   ├── helpdesk_service.py       # Clasificación helpdesk con LLM
│   └── knowledge_ingestion_service.py  # Parser Excel → DB + embeddings
└── main.py
```

---

## Instalación

### 1. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Variables de entorno

```env
POSTGRESQL_URL=postgresql://usuario:password@localhost:5432/rag_db
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b-instruct-q8_0
EMBEDDINGS_MODEL=intfloat/multilingual-e5-base
RAG_INTERNAL_API_KEY=<clave-interna-compartida-con-chat-backend>

# Opcionales
SESSION_TTL_SECONDS=7200
SESSION_MAX_SIZE=500
```

### 3. Inicializar la base de datos

```bash
python scripts/bootstrap_db.py
```

### 4. Levantar el servidor

```bash
uvicorn src.main:app --reload --port 8000
```

Documentación interactiva: `http://localhost:8000/docs`

---

## Convenciones de embeddings

El modelo `multilingual-e5-base` requiere prefijos explícitos:

```python
embed_document("passage: <texto del documento>")  # indexado
embed_query("query: <pregunta del usuario>")       # consulta
```

Sin estos prefijos la similitud coseno se degrada notablemente.