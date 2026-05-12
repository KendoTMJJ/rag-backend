import os
from dotenv import load_dotenv

load_dotenv()


class Config:

    # Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

    # PostgreSQL
    POSTGRESQL_URL = os.getenv("POSTGRESQL_URL")

    # Internal API Key (para rutas admin llamadas desde Chat Backend)
    RAG_INTERNAL_API_KEY = os.getenv("RAG_INTERNAL_API_KEY")

    # URL pública base del backend RAG (usada para construir enlaces de descarga)
    PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")


Config = Config()
