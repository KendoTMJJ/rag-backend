import os
from dotenv import load_dotenv

load_dotenv()


class Config:

    # Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

    # PostgreSql

    POSTGRESQL_URL = os.getenv("POSTGRESQL_URL")


Config = Config()
