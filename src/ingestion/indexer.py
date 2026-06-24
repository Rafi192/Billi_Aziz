# ingestion/indexer.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any
import numpy as np
import faiss
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MongoDBVectorIndexer:
    def __init__(
        self,
        embedder,
        vector_store_path: str = "data/vector_store"
    ):
        self.embedder = embedder
        project_root = Path(__file__).parent.parent
        self.vector_store_path = (project_root / vector_store_path).resolve()
        self.indexes = {}       # { collection_name: faiss_index }
        self.documents = {}     # { collection_name: list of docs }

        os.makedirs(self.vector_store_path, exist_ok=True)
        logger.info(f"Vector store path: {self.vector_store_path}")

    def create_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        texts = [doc['text'] for doc in documents]
        logger.info(f"Creating embeddings for {len(texts)} documents...")

        if hasattr(self.embedder, 'embed_documents'):
            embeddings = self.embedder.embed_documents(texts)
        else:
            embeddings = []
            for i, text in enumerate(texts):
                if i % 50 == 0:
                    logger.info(f"  Embedded {i}/{len(texts)} documents...")
                embedding = self.embedder.embed_query(text)
                embeddings.append(embedding)

        embeddings_array = np.array(embeddings).astype('float32')
        logger.info(f"Created embeddings with shape: {embeddings_array.shape}")
        return embeddings_array

    def build_index(self, documents: List[Dict[str, Any]], collection_name: str):
      
        logger.info(f"Building index for '{collection_name}' ({len(documents)} documents)...")

        embeddings = self.create_embeddings(documents)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # inner product for cosine similarity
        index.add(embeddings)                 # embedder already normalized, no double norm

        self.indexes[collection_name] = index
        self.documents[collection_name] = documents

        logger.info(f"Built FAISS index for '{collection_name}' — {index.ntotal} vectors (dim: {dimension})")

    def build_all_indexes(self, collection_data: Dict[str, List[Dict[str, Any]]]):
    
        for collection_name, documents in collection_data.items():
            if not documents:
                logger.warning(f"Skipping '{collection_name}' — no documents")
                continue
            self.build_index(documents, collection_name)

        logger.info(f"Built {len(self.indexes)} indexes total")



    def save_index(self, collection_name: str):
      
        if collection_name not in self.indexes:
            raise ValueError(f"No index found for '{collection_name}'. Build it first.")

        index_path = self.vector_store_path / f"{collection_name}_index.bin"
        metadata_path = self.vector_store_path / f"{collection_name}_documents.pkl"

        faiss.write_index(self.indexes[collection_name], str(index_path))

        with open(metadata_path, 'wb') as f:
            pickle.dump(self.documents[collection_name], f)

        logger.info(f"Saved '{collection_name}' index → {index_path}")
        logger.info(f"Saved '{collection_name}' metadata → {metadata_path}")


    def save_all_indexes(self):
        """Save all built indexes to disk."""
        for collection_name in self.indexes:
            self.save_index(collection_name)
        logger.info(f"Saved {len(self.indexes)} indexes to {self.vector_store_path}")


    def load_index(self, collection_name: str):
       
        index_path = self.vector_store_path / f"{collection_name}_index.bin"
        metadata_path = self.vector_store_path / f"{collection_name}_documents.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found for '{collection_name}' at {index_path}\n"
                f"Please run ingestion first."
            )

        self.indexes[collection_name] = faiss.read_index(str(index_path))

        with open(metadata_path, 'rb') as f:
            self.documents[collection_name] = pickle.load(f)

        logger.info(f"Loaded '{collection_name}' index — {self.indexes[collection_name].ntotal} vectors")
        logger.info(f"Loaded '{collection_name}' metadata — {len(self.documents[collection_name])} documents")



    def load_all_indexes(self, collection_names: List[str]):
     
        for collection_name in collection_names:
            try:
                self.load_index(collection_name)
            except FileNotFoundError as e:
                logger.warning(str(e))

        logger.info(f"Loaded {len(self.indexes)} indexes")


    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
      
        if self.index is None:
            raise ValueError(
                "Index not built or loaded. "
                "Call build_index() or load_index() first."
            )
        
        # Create query embedding
        query_embedding = self.embedder.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        
        # Search the index
        scores, indices = self.index.search(query_vector, top_k)
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        logger.info(f"Found {len(results)} similar documents for query: '{query[:50]}...'")
        return results