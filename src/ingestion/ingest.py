# ingestion/ingest.py
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingestion.load_data import MultiCollectionMongoDBLoader
from ingestion.chunker import Chunker
from ingestion.embedder import get_embedder
from ingestion.indexer import MongoDBVectorIndexer
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


def main():

    print("=====================")
    print("RAG INGESTION PIPELINE")
    print("=======================")

    # my configs 
    MONGODB_URI      = os.getenv("MONGODB_URI")
    DATABASE_NAME    = os.getenv("MONGODB_DATABASE")
    VECTOR_STORE     = "data/vector_store"
    EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"
    CHUNK_SIZE       = 100
    CHUNK_OVERLAP    = 10

    if not MONGODB_URI:
        print("ERROR: MONGODB_URI not found in .env")
        return

    if not DATABASE_NAME:
        print("ERROR: MONGODB_DATABASE not found in .env")
        return

    print(f"\nDatabase       : {DATABASE_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Vector store   : {VECTOR_STORE}")
    print(f"Chunk size     : {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")

    # ── step 1: loading + format ────────────────────────────────────
    print("\n[1] Connecting to MongoDB and loading documents...")

    loader = MultiCollectionMongoDBLoader(
        connection_string=MONGODB_URI,
        database_name=DATABASE_NAME
    )

    available = loader.get_available_collections()
    if not available:
        print("ERROR: No RAG-compatible collections found.")
        print("Check COLLECTION_SCHEMAS in load_data.py")
        loader.close()
        return

    print(f"    Found collections: {', '.join(available)}")

    formatted_data = loader.load_and_format_all_collections()
    loader.close()

    if not formatted_data:
        print("ERROR: No documents loaded.")
        return

    total_docs = sum(len(docs) for docs in formatted_data.values())
    print(f"    Loaded {total_docs} documents across {len(formatted_data)} collections")

    for name, docs in formatted_data.items():
        print(f"      - {name}: {len(docs)} docs")

    # ── step 2: chunk ────────────────────────────────────────────
    print("\n[2] Chunking documents...")

    chunker = Chunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunked_data = chunker.chunk_all_collections(formatted_data)

    total_chunks = sum(len(chunks) for chunks in chunked_data.values())
    print(f"    {total_docs} docs → {total_chunks} chunks")

    for name, chunks in chunked_data.items():
        print(f"      - {name}: {len(chunks)} chunks")

    # ── step 3: embed ────────────────────────────────────────────
    print("\n[3] Loading embedding model...")
    embedder = get_embedder(model_name=EMBEDDING_MODEL)
    print(f"    Embedding dimension: {embedder.get_embedding_dimension()}")

    # ── step 4: build indexes ────────────────────────────────────
    print("\n[4] Building FAISS indexes (one per collection)...")

    indexer = MongoDBVectorIndexer(
        embedder=embedder,
        vector_store_path=VECTOR_STORE
    )

    try:
        indexer.build_all_indexes(chunked_data)
    except Exception as e:
        print(f"ERROR building indexes: {e}")
        logger.exception("Index building failed")
        return

    # ── step 5: save ─────────────────────────────────────────────
    print("\n[5] Saving indexes to memory...")

    try:
        indexer.save_all_indexes()
    except Exception as e:
        print(f"ERROR saving indexes: {e}")
        logger.exception("Index saving failed")
        return

    # ── complete ─────────────────────────────────────────────────────
    print("===================")
    print("INGESTION COMPLETE")
    print("====================")
    print(f"  Documents loaded : {total_docs}")
    print(f"  Chunks created   : {total_chunks}")
    print(f"  Collections      : {len(chunked_data)}")
    print(f"  Indexes saved to : {VECTOR_STORE}/")
    print(f"  Model            : {EMBEDDING_MODEL}")
    print(f"  Dim              : {embedder.get_embedding_dimension()}")
    print("\nReady for retrieval.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        logger.exception("Pipeline failed")
        sys.exit(1)