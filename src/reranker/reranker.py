# reranker/reranker.py
import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder reranker using a small sentence-transformers model.
    Reranks retrieved chunks by actual relevance to the query.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model      = CrossEncoder(model_name)
        self.model_name = model_name
        logger.info(f"Loaded reranker model: {model_name}")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
     
        if not chunks:
            return []

        # Build (query, chunk_text) pairs for cross-encoder
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach rerank score to each chunk
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        # Sort by rerank score descending
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        final = reranked[:top_k]
        logger.info(f"Reranked {len(chunks)} → kept top {len(final)}")
        for i, chunk in enumerate(final):
            logger.info(
                f"[Reranker] #{i+1} | score={chunk.get('rerank_score', 0):.4f} | "
                f"collection={chunk.get('metadata', {}).get('collection', 'unknown')} | "
                f"text_preview={chunk.get('text', '')[:100]}"
            )
        return final