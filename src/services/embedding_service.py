from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
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
