# ingestion/chunker.py
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50      # overlap between chunks to preserve context

class Chunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # skip chunks that are too short to be meaningful
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a single formatted document into chunks.
        Each chunk inherits the parent document's metadata.
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})

        if not text:
            logger.warning(f"Empty text in document: {document.get('id')}")
            return []

        text_chunks = self.split_text(text)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,                  # carry all original metadata forward
                    "chunk_index": i,            # position of chunk in original doc
                    "total_chunks": len(text_chunks),
                    "parent_doc_id": document.get("id"),
                }
            })

        return chunks

    def chunk_collection(self, documents: List[Dict[str, Any]], collection_name: str) -> List[Dict[str, Any]]:
        """
        Chunk all documents from a single collection.
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(f"'{collection_name}': {len(documents)} docs → {len(all_chunks)} chunks")
        return all_chunks

    def chunk_all_collections(self, collection_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Chunk all collections. Returns dict keyed by collection name.
        Input comes directly from load_data.py's load_and_format_all_collections().
        """
        all_chunked = {}

        for collection_name, documents in collection_data.items():
            chunks = self.chunk_collection(documents, collection_name)
            if chunks:
                all_chunked[collection_name] = chunks

        total = sum(len(c) for c in all_chunked.values())
        logger.info(f"Total chunks across all collections: {total}")
        return all_chunked