# retriever/retriever.py

import sys
from pathlib import Path
import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# from retrievers.router import route_query
from ingestion.embedder import get_embedder

logger = logging.getLogger(__name__)

VECTOR_STORE_PATH = project_root / "data" / "vector_store"


class Retriever:

    def __init__(self, embedder=None):
        self.embedder = embedder or get_embedder()
        self.indexes = {}      # {collection_name: faiss_index}
        self.documents = {}    # {collection_name: documents}

    # ==========================================================
    # Index Loading
    # ==========================================================

    def load_index(self, collection_name: str) -> bool:

        index_path = VECTOR_STORE_PATH / f"{collection_name}_index.bin"
        metadata_path = VECTOR_STORE_PATH / f"{collection_name}_documents.pkl"

        if not index_path.exists():
            logger.warning(
                f"No index found for '{collection_name}' at {index_path}"
            )
            return False

        self.indexes[collection_name] = faiss.read_index(str(index_path))

        with open(metadata_path, "rb") as f:
            self.documents[collection_name] = pickle.load(f)

        logger.info(
            f"Loaded '{collection_name}' "
            f"({self.indexes[collection_name].ntotal} vectors)"
        )

        return True

    def load_all_indexes(self, collection_names: List[str]):

        for collection_name in collection_names:
            self.load_index(collection_name)

        logger.info(f"Loaded {len(self.indexes)} indexes")

    # ==========================================================
    # Query Cleaning
    # ==========================================================

    def clean_query_for_retrieval(self, query: str) -> str:

        forbidden_patterns = [
            "User:",
            "Assistant:",
            "Conversation History:",
            "Context:",
            "Source:"
        ]

        cleaned = query

        for pattern in forbidden_patterns:
            if pattern in cleaned:
                logger.warning(
                    f"Query contaminated with '{pattern}'"
                )
                cleaned = cleaned.split(pattern)[-1]

        return cleaned.strip()

    # ==========================================================
    # Core Search
    # ==========================================================

    def search_collection(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:

        if collection_name not in self.indexes:

            loaded = self.load_index(collection_name)

            if not loaded:
                return []

        query_embedding = self.embedder.embed_query(query)

        query_vector = np.array(
            [query_embedding],
            dtype=np.float32
        )

        faiss.normalize_L2(query_vector)

        index = self.indexes[collection_name]

        actual_k = min(top_k, index.ntotal)

        scores, indices = index.search(
            query_vector,
            actual_k
        )

        docs = self.documents[collection_name]

        results = []

        for idx, score in zip(indices[0], scores[0]):

            if idx < 0:
                continue

            if idx >= len(docs):
                continue

            result = docs[idx].copy()
            result["similarity_score"] = float(score)

            results.append(result)

        logger.info(
            f"[{collection_name}] "
            f"hits={len(results)} "
            f"query='{query}'"
        )

        return results

    # ==========================================================
    # Single Collection Retrieval
    # ==========================================================

    def retrieve_single_collection(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Use when you have ONE knowledge base collection.

        No routing.
        Direct retrieval.
        """

        # clean_query = self.clean_query_for_retrieval(query)
        clean_query = query.strip()

        logger.info(f"Single Collection Retrieval")
        logger.info(f"Query: {clean_query}")
        logger.info(f"Collection: {collection_name}")

        results = self.search_collection(
            query=clean_query,
            collection_name=collection_name,
            top_k=top_k
        )

        results.sort(
            key=lambda x: x["similarity_score"],
            reverse=True
        )

        final = results[:top_k]

        logger.info(
            f"Retrieved {len(final)} chunks "
            f"from '{collection_name}'"
        )

        return final

    # ==========================================================
    # Multi Collection Retrieval
    # ==========================================================

    def retrieve_multi_collection(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:

        # clean_query = self.clean_query_for_retrieval(query)
        clean_query = query.strip()

        logger.info("Multi Collection Retrieval")
        logger.info(f"Query: {clean_query}")

        if len(clean_query.split()) < 2:
            logger.warning("Query too vague")
            return []

        # load all indexes if not loaded
        if not self.indexes:
            self.load_all_indexes()

        all_results = []

        for collection_name in self.indexes.keys():

            hits = self.search_collection(
                query=clean_query,
                collection_name=collection_name,
                top_k=top_k
            )

            all_results.extend(hits)

        all_results.sort(
            key=lambda x: x["similarity_score"],
            reverse=True
        )

        final = all_results[:top_k]

        logger.info(
            f"Retrieved {len(final)} chunks across "
            f"{len(self.indexes)} collections"
        )

        return final

    # ==========================================================
    # Backward Compatible Alias
    # ==========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        collection_name: Optional[str] = None
    ):
        """
        Convenience wrapper.

        If collection_name is provided:
            -> single collection retrieval

        Otherwise:
            -> multi collection retrieval
        """

        if collection_name:
            return self.retrieve_single_collection(
                query=query,
                collection_name=collection_name,
                top_k=top_k
            )

        return self.retrieve_multi_collection(
            query=query,
            top_k=top_k
        )