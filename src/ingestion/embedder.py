# ingestion/embedder.py
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


class Embedder:

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return np.array(embeddings).astype('float32')

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            query,
            normalize_embeddings=True
        )

    def get_embedding_dimension(self) -> int:
        return len(self.embed_query("test"))


def get_embedder(model_name: str = DEFAULT_MODEL, **kwargs) -> Embedder:
    return Embedder(model_name=model_name, **kwargs)