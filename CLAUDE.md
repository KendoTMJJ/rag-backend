# RAG Backend — Asistente de Posgrados USTA Tunja

## Descripción
Backend RAG (Retrieval-Augmented Generation) para responder preguntas sobre 
programas de posgrado de la Universidad Santo Tomás Seccional Tunja.

## Stack
- FastAPI + Uvicorn
- PostgreSQL + pgvector (embeddings)
- SQLAlchemy (ORM)
- LangChain + Ollama (LLM local)
- Modelo de embeddings: intfloat/multilingual-e5-base
- Modelo LLM: llama3.1:8b-instruct-q4_K_M

## Estructura clave
- src/services/rag_pipeline.py     — Pipeline principal (rutas de decisión)
- src/services/retrieval_service.py — Búsqueda semántica + domain gate
- src/services/llm_service.py      — Cliente Ollama + generación
- src/nlp/domain_taxonomy.py       — Señales de dominio académico
- src/nlp/domain_guardrail.py      — Guardrail de inyección y dominio
- src/nlp/intent_utils.py          — Detección de intención
- src/extractors/program_excel_parser.py — Ingesta de datos desde Excel

## Rutas del pipeline
GREETING, THANKS, PROGRAM_SELECTED, PROGRAM_FIELD, CURRICULUM,
CURRICULUM_SEMESTER, ELECTIVES, DEGREE_OPTIONS, NARRATIVE_SQL,
NARRATIVE_NOT_FOUND, PROGRAM_OVERVIEW, INSCRIPTION_LINK,
INSCRIPTION_NO_LINK, LIST_PROGRAMS, LIST_PROGRAMS_FILTERED,
PROGRAM_MINMAX, DEFAULT_RAG, LOW_CONFIDENCE, OUT_OF_DOMAIN,
INJECTION_BLOCKED, NEED_PROGRAM, NOT_FOUND

## Convenciones
- Embeddings: embed_document() para indexado (prefijo "passage:")
- Embeddings: embed_query() para consultas (prefijo "query:")
- Memoria de sesión: TTLStore con TTL configurable por env
- Tests: test/eval/test_retrieval_quality.py (pytest -m integration)

## Variables de entorno requeridas
POSTGRESQL_URL, OLLAMA_BASE_URL, OLLAMA_MODEL, EMBEDDINGS_MODEL

## Lo que NO hacer
- No cambiar el esquema SQL sin migración
- No usar simple_chunk() — usar semantic_chunk() de text_normalizer.py
- No hardcodear señales de dominio fuera de domain_taxonomy.py