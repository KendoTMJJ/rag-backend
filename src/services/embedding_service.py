import threading
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Optional
import logging
import torch
from src.core.config import Config

logger = logging.getLogger(__name__)


class LocalEmbeddings:
    """
    Servicio de embeddings local usando E5.

    IMPORTANTE:
    - Consultas → "query: ..."
    - Documentos → "passage: ..."
    """

    def __init__(self, device: str = None):
        self.model_name = Config.EMBEDDINGS_MODEL
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Cargando modelo de embeddings: {self.model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # prueba de carga
        test = self.embed_query("test")
        logger.info(f"Embeddings listos. Dimensión: {len(test)}")

    # ---------------------------------------------------------
    # Indexado (documentos)
    # ---------------------------------------------------------

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        return self.embeddings.embed_documents(prefixed)

    def embed_document(self, text: str) -> List[float]:
        return self.embeddings.embed_query(f"passage: {text}")

    # ---------------------------------------------------------
    # Consulta (usuario)
    # ---------------------------------------------------------

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(f"query: {text}")


_lock: threading.Lock = threading.Lock()
_instance: Optional[LocalEmbeddings] = None


def get_embedding_model() -> LocalEmbeddings:
    """Singleton del modelo de embeddings. Thread-safe (double-checked locking).

    Carga el modelo la primera vez y reutiliza la misma instancia en llamadas
    posteriores. Llamar desde startup garantiza que el primer request no pague
    el costo de carga.
    """
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = LocalEmbeddings()
    return _instance
