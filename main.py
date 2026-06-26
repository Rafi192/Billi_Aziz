# main.py
import os
import sys
import uuid
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import logging

from src.memory.chat_history import ChatHistory
from src.retrievers.retriever import Retriever
from src.reranker.reranker import Reranker
from src.llm.generator import generate_response

from src.ingestion.schema import is_casual_query
from src.ingestion.load_data import MongoDBLoader
from src.ingestion.embedder import get_embedder
from src.ingestion.chunker import Chunker
from src.ingestion.indexer import MongoDBVectorIndexer
# from llm.query_rewriter import rewrite_query
from contextlib import asynccontextmanager

ADMIN_API_KEY    = os.getenv("ADMIN_API_KEY")
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE       = 400
CHUNK_OVERLAP    = 40
VECTOR_STORE     = "data/vector_store"

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ADMIN_API_KEY   = os.getenv("ADMIN_API_KEY")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE      = 400
CHUNK_OVERLAP   = 40
VECTOR_STORE    = "src/data/vector_store"

# initialise once — shared across all requests
chat_history = ChatHistory()
retriever    = Retriever()
reranker     = Reranker()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App starting — loading FAISS indexes...")
    retriever.load_all_indexes()
    if not retriever.indexes:
        logger.warning("No FAISS indexes on disk. POST /api/admin/reindex to build them.")
    else:
        logger.info(f"Loaded indexes: {list(retriever.indexes.keys())}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialise once at startup — shared across all requests
chat_history = ChatHistory()
retriever    = Retriever()
reranker     = Reranker()


@app.post("/api/chat/")
def chat(user_id: str = Form(), query: str = Form()):
    try:

        # 1. get history FIRST
        history = chat_history.get_history(user_id)

        logger.info(f"Original query: {query}")

        # 3. casual handling
        if is_casual_query(query):
            answer = generate_response(
                query=query,
                chunks=[],
                chat_history=[]
            )

        else:
            # 4. retrieve USING CLEAN QUERY ONLY
            chunks = retriever.retrieve(query, top_k=10)

            # 5. rerank
            reranked = reranker.rerank(query, chunks, top_k=5)

            # 6. generate
            answer = generate_response(
                query=query,
                chunks=reranked,
                chat_history=history
            )

        # 7. save chat
        chat_history.add_message(user_id, "user", query)
        chat_history.add_message(user_id, "assistant", answer)

        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "statuscode": 200,
                "text": {
                    "user_id": user_id,
                    "query": query,
                    # "rewritten_query": clean_query,
                    "answer": answer
                }
            }
        )

    except Exception as ex:
        logger.exception("Chat endpoint failed")
        return JSONResponse(
            status_code=500,
            content={"status": False, "error": str(ex)}
        )


@app.delete("/api/chat/clear/")
def clear_chat(user_id: str = Form()):
    try:
        chat_history.clear_session(user_id)
        return JSONResponse(
            status_code=200,
            content={
                "status":     True,
                "statuscode": 200,
                "text":       f"History cleared for user {user_id}"
            }
        )

    except Exception as ex:
        logger.exception("Clear chat failed")
        return JSONResponse(
            status_code=500,
            content={
                "status":     False,
                "statuscode": 500,
                "text":       str(ex)
            }
        )

@app.get("/api/chat/history/")
def get_history(user_id: str):   
    try:
        history = chat_history.get_history(user_id)
        if not history:
            return JSONResponse(
                status_code=404,
                content={
                    "status":     False,
                    "statuscode": 404,
                    "text":       "No history found for this user"
                }
            )
        return JSONResponse(
            status_code=200,
            content={
                "status":     True,
                "statuscode": 200,
                "text": {
                    "user_id": user_id,
                    "history": history
                }
            }
        )
    except Exception as ex:
        logger.exception("Get history failed")
        return JSONResponse(
            status_code=500,
            content={
                "status":     False,
                "statuscode": 500,
                "text":       str(ex)
            }
        )

@app.post("/api/admin/reindex")
def reindex(api_key: str = Form()):
    try:
        if api_key != os.getenv("ADMIN_API_KEY"):
            return JSONResponse(
                status_code=401,
                content={
                    "status": False,
                    "statuscode": 401,
                    "text": "Unauthorized: Invalid API key"
                }
            )

        logger.info("Admin reindexing triggered via API")

        loader = MongoDBLoader(
            connection_string=os.getenv("MONGODB_URI"),
            database_name=os.getenv("DB_NAME", "billaziz")
        )

        collection_names = loader.get_collection_names()
        if not collection_names:
            raise ValueError("No collections found in the database")
        logger.info(f"Discovered collections: {collection_names}")

        formatted_data = loader.load_multiple_collections(collection_names=collection_names)
        loader.close()

        if not formatted_data:
            logger.error("No data loaded from MongoDB during reindexing")
            return JSONResponse(
                status_code=500,
                content={
                    "status": False,
                    "statuscode": 500,
                    "text": "Reindexing failed: No data loaded from MongoDB"
                }
            )

        chunker = Chunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunked_data = chunker.chunk_all_collections(formatted_data)

        embedder = get_embedder(model_name=EMBEDDING_MODEL)
        indexer = MongoDBVectorIndexer(
            embedder=embedder,
            vector_store_path=VECTOR_STORE
        )
        indexer.build_all_indexes(chunked_data)
        indexer.save_all_indexes()

        # hot-reload the live retriever
        retriever.indexes = {}
        retriever.documents = {}
        retriever.load_all_indexes()
        logger.info(f"Retriever hot-reloaded: {list(retriever.indexes.keys())}")

        total_docs = sum(len(docs) for docs in formatted_data.values())
        total_chunks = sum(len(chunk) for chunk in chunked_data.values())
        logger.info(f"Reindex completed: {total_docs} docs, {total_chunks} chunks")

        return JSONResponse(
            status_code=200,
            content={
                "status": True,
                "statuscode": 200,
                "text": {
                    "total_docs": total_docs,
                    "total_chunks": total_chunks,
                    "collections": collection_names
                }
            }
        )

    except Exception as e:
        logger.exception("Reindexing failed")
        return JSONResponse(
            status_code=500,
            content={
                "status": False,
                "statuscode": 500,
                "text": f"Reindexing failed: {str(e)}"
            }
        )


@app.get("/api/health/")
def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status":     True,
            "statuscode": 200,
            "text":       "RAG chatbot is running"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)