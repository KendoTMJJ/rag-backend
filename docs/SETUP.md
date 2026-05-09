# Guía de Instalación — RAG Backend

Guía de instalación local del servicio RAG. Si vas a instalar el sistema completo (los 3 servicios), usa `chat/docs/SETUP.md`.

---

## Requisitos previos

| Herramienta | Versión | Uso |
|---|---|---|
| Python | 3.11+ | Runtime del servidor |
| PostgreSQL | 15+ | Base de datos con extensión pgvector |
| Ollama | última | Servidor LLM local |

---

## 1. Clonar e instalar dependencias

```bash
git clone <repo-url> rag-backend
cd rag-backend
pip install -r requirements.txt
```

El primer `pip install` descarga PyTorch y transformers — puede tardar varios minutos.

---

## 2. Base de datos PostgreSQL con pgvector

```sql
-- Como superusuario de PostgreSQL:
CREATE DATABASE rag_db;
CREATE USER rag_user WITH PASSWORD 'tu_password';
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;

-- Conectarse a rag_db y habilitar pgvector:
\c rag_db
CREATE EXTENSION IF NOT EXISTS vector;
```

Si `pgvector` no está instalado en el servidor:

```bash
# Ubuntu / Debian
sudo apt install postgresql-15-pgvector

# macOS con Homebrew
brew install pgvector
```

---

## 3. Ollama (modelo LLM)

```bash
# Instalar Ollama (Linux / WSL)
curl -fsSL https://ollama.com/install.sh | sh

# Descargar el modelo (Q4_K_M: mejor balance calidad / memoria en CPU)
ollama pull llama3.1:8b-instruct-q4_K_M

# Iniciar el servidor
ollama serve
# Disponible en http://localhost:11434
```

Para configuración avanzada en CPU (hilos, paralelismo), ver [OLLAMA_CPU.md](OLLAMA_CPU.md).

---

## 4. Variables de entorno

Crear el archivo `.env` en la raíz del proyecto:

```env
# Base de datos
POSTGRESQL_URL=postgresql://rag_user:tu_password@localhost:5432/rag_db

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M

# Modelo de embeddings (se descarga automáticamente de HuggingFace)
EMBEDDINGS_MODEL=intfloat/multilingual-e5-base

# Seguridad — clave que el servidor NestJS debe enviar en x-internal-key
RAG_INTERNAL_API_KEY=clave-secreta-interna

# Sesiones (opcionales)
SESSION_TTL_SECONDS=7200      # Tiempo de vida de sesión en segundos (default: 2h)
SESSION_MAX_SIZE=500          # Número máximo de sesiones en memoria (default: 500)

# URL del catálogo oficial de posgrados (para botones de "ver catálogo")
CATALOG_POSGRADOS_URL=https://santotovirtual.edu.co/
```

---

## 5. Inicializar la base de datos

```bash
python -m scripts.bootstrap_db
```

Esto crea las tablas necesarias en `rag_db`. Debe ejecutarse una vez antes del primer inicio.

---

## 6. Iniciar el servidor

```bash
uvicorn src.main:app --reload --port 8000
```

**Primer inicio:** el modelo de embeddings `intfloat/multilingual-e5-base` (~550 MB) se descarga automáticamente desde HuggingFace. Puede tardar varios minutos. Verificar en los logs:

```
Downloading multilingual-e5-base...
Model loaded successfully.
Application startup complete.
```

Servidor disponible en:
- API: `http://localhost:8000`
- Documentación interactiva: `http://localhost:8000/docs`

---

## 7. Verificar la instalación

```bash
# Verificar que el servidor responde
curl http://localhost:8000/health
# → {"status":"ok"}

# Verificar que Ollama está accesible desde el servicio
curl http://localhost:11434
# → Ollama is running

# Probar el pipeline RAG (requiere que haya datos cargados)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "hola", "chatSessionId": "test-001"}'
```

---

## 8. Cargar datos de posgrados

Sin datos el bot solo puede responder saludos. Para cargar el catálogo:

```bash
# Via API con curl
curl -X POST http://localhost:8000/admin/knowledge/upload \
  -H "x-internal-key: clave-secreta-interna" \
  -F "file=@/ruta/al/archivo.xlsx"

# Respuesta esperada:
# {"inserted": 12, "updated": 0, "errors": []}
```

O desde el panel de administración del chat-widget (tab Posgrados → Subir Excel).

---

## 9. Estructura del proyecto

```
src/
├── main.py                          Punto de entrada FastAPI
├── api/
│   └── routes/
│       ├── chat.py                  POST /chat (pipeline posgrados)
│       ├── helpdesk.py              GET /helpdesk/*, POST /helpdesk/classify
│       └── admin/
│           ├── helpdesk.py          CRUD /admin/helpdesk/*
│           └── knowledge.py         POST /admin/knowledge/upload
│
├── services/
│   ├── rag_pipeline.py              Pipeline principal (árbol de decisión)
│   ├── retrieval_service.py         Búsqueda semántica + resolución de programas
│   ├── sql_retrieval_service.py     Consultas SQL estructuradas
│   └── llm_service.py               Cliente Ollama, historial de chat
│
├── nlp/
│   ├── intent_utils.py              Detección de intención y extractores de entidades
│   ├── domain_taxonomy.py           Señales del dominio académico
│   ├── domain_guardrail.py          Guardrail de dominio e inyección de prompt
│   ├── text_normalizer.py           Normalización y limpieza de texto
│   └── input_sanitizer.py           Sanitización de entrada (SQL injection, etc.)
│
├── extractors/
│   └── program_excel_parser.py      Parser del Excel de programas de posgrado
│
├── quality/
│   ├── request_gate.py              Gate de validación de solicitud
│   ├── context_reranker.py          Reranking del contexto recuperado
│   └── output_validator.py          Validación de la respuesta del LLM
│
└── models/
    └── helpdesk.py                  Modelos SQLAlchemy (HelpdeskCategory)
```

---

## 10. Tests

```bash
# Instalar dependencias de test
pip install pytest pytest-asyncio

# Correr tests de calidad de retrieval (requiere DB con datos)
pytest test/eval/test_retrieval_quality.py -m integration -v

# Correr todos los tests
pytest
```
