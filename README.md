# RAG Backend Santo Tomás Tunja

Backend para un sistema de Recuperación Aumentada por Generación (RAG) orientado a programas académicos institucionales. Permite ingestar información desde archivos Excel, generar embeddings semánticos y exponerlos vía API REST.

---

## Tecnologías

- **FastAPI** — API REST
- **PostgreSQL + pgvector** — Base de datos con soporte vectorial
- **SQLAlchemy** — ORM
- **sentence-transformers** — Embeddings locales (`multilingual-e5-base`)
- **LangChain** — Integración con modelos de lenguaje
- **Pandas** — Procesamiento de archivos Excel

---

## Estructura del proyecto

```
rag-backend/
├── src/
│   ├── core/           # Configuración y variables de entorno
│   ├── database/       # Conexión a PostgreSQL e inicialización
│   ├── models/         # Modelos SQLAlchemy
│   ├── services/       # Servicio de embeddings
│   ├── extractors/     # Parser de Excel institucional
│   └── main.py         # Aplicación FastAPI
├── scripts/
│   └── bootstrap_db.py # Inicialización de la base de datos
├── temp/               # Archivos temporales (no versionado)
├── .env.example        # Variables de entorno de ejemplo
└── requirements.txt
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/rag-backend.git
cd rag-backend
```

### 2. Crear entorno virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tus credenciales
```

### 5. Inicializar la base de datos

> Requiere PostgreSQL corriendo con la extensión `pgvector` disponible.

```bash
python scripts/bootstrap_db.py
```

---

## Uso

### Levantar el servidor

```bash
uvicorn src.main:app --reload
```

La API estará disponible en `http://localhost:8000`.  
Documentación interactiva: `http://localhost:8000/docs`

### Subir un archivo Excel de programas

```
POST /admin/knowledge/upload
Content-Type: multipart/form-data
```

El archivo debe contener las hojas: `INFO_GENERAL`, `MALLA_CURRICULAR`, `ELECTIVAS`, `OPCIONES_GRADO`, `NARRATIVA`.

---

## Variables de entorno

Crea un archivo `.env` basado en `.env.example`:

```env
DATABASE_URL=postgresql://usuario:contraseña@localhost:5432/nombre_db
EMBEDDINGS_MODEL=intfloat/multilingual-e5-base
HF_TOKEN=hf_...   # Opcional, mejora velocidad de descarga desde HuggingFace
```

---

## Notas sobre embeddings

El modelo `multilingual-e5-base` requiere prefijos explícitos:

- **Documentos indexados:** `passage: <texto>`
- **Consultas del usuario:** `query: <texto>`

Sin estos prefijos la similitud coseno se degrada notablemente.
