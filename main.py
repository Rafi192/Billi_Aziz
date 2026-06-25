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

from memory.chat_history import ChatHistory
from retrievers.retriever import Retriever
from reranker.reranker import Reranker
from llm.generator import generate_response

from ingestion.schema import is_casual_query
from ingestion.load_data import MongoDBLoader
from ingestion.embedder import get_embedder
from ingestion.chunker import Chunker
from ingestion.indexer import MongoDBVectorIndexer
# from llm.query_rewriter import rewrite_query

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

app = FastAPI()

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

# @app.post("/api/chat/")
# def chat(
#     user_id: str = Form(),
#     query:   str = Form()
# ):
#     try:
#         if is_casual_query(query):
#             answer = generate_response(
#                 query=query,
#                 chunks=[],          # empty — no context needed
#                 chat_history=[]
#             )
#             chat_history.add_message(user_id, "user",      query)
#             chat_history.add_message(user_id, "assistant", answer)

#             return JSONResponse(
#                 status_code=200,
#                 content={
#                     "status":     True,
#                     "statuscode": 200,
#                     "text": {
#                         "user_id": user_id,
#                         "query":   query,
#                         "answer":  answer
#                     }
#                 }
#             )
        
#         #normal RAG flow for real queries
#         # 1  get the user's previous history
#         history = chat_history.get_history(user_id)

#         # 2 — retrieve + rerank
#         chunks   = retriever.retrieve(query, top_k=10)
#         reranked = reranker.rerank(query, chunks, top_k=5)

#         # 3 — generate
#         answer = generate_response(
#             query=query,
#             chunks=reranked,
#             chat_history=history
#         )

#         # 4 — save both turns
#         chat_history.add_message(user_id, "user",      query)
#         chat_history.add_message(user_id, "assistant", answer)

#         return JSONResponse(
#             status_code=200,
#             content={
#                 "status":     True,
#                 "statuscode": 200,
#                 "text": {
#                     "user_id": user_id,
#                     "query":   query,
#                     "answer":  answer
#                 }
#             }
#         )

#     except Exception as ex:
#         logger.exception("Chat endpoint failed")
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "status":     False,
#                 "statuscode": 500,
#                 "text":       str(ex)
#             }
#         )


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
def reindex(api_key:str = Form()):
    try:

        if api_key != os.getenv("ADMIN_API_KEY"):
            return JSONResponse(
                status_code=401,
                content={
                    "status":     False,
                    "statuscode": 401,
                    "text":       "Unauthorized: Invalid API key"
                }
            )
        logger.info("Admin reindexing triggered via API")

        #1.calling mongoDB multi collection loader
        loader = MongoDBLoader(
            connection_string=os.getenv("MONGODB_URI"),
            database_name=os.getenv("DB_NAME", "billaziz")
        )
        formatted_data = loader.load_multiple_collections(collection_names=os.getenv("COLLECTION_NAMES").split(","))
        loader.close()
        #handling empty data case
        if not formatted_data:
            logger.error("No data loaded from MongoDB during reindexing")
            return JSONResponse(
                status_code=500,
                content={
                    "status":     False,
                    "statuscode": 500,
                    "text":       "Reindexing failed: No data loaded from MongoDB"
                }
            )
        
        #.2. chunking 
        chunker = Chunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunked_data = chunker.chunk_all_collections(formatted_data)

        # 3. embeddding + index builkdig
        embedder = get_embedder(model_name=EMBEDDING_MODEL)

        indexer = MongoDBVectorIndexer(
            embedder=embedder,
            vector_store_path=VECTOR_STORE
        )
        indexer.build_all_indexes(chunked_data)
        indexer.save_all_indexes()

        total_docs = sum(len(docs) for docs in formatted_data.values())
        total_chunks = sum(len(chunk) for chunk in chunked_data.values())

        logger.info(f'reindex completed with total docs of {total_docs} and total chunks of {total_chunks} chunks')

        return JSONResponse(
            status_code=200,
            content={
                "status":True,
                "statuscode":200,
                "text":{
                    "total_docs": total_docs,
                    "total_chunks": total_chunks
                }
            }
        )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status":     False,
                "statuscode": 500,
                "text":       f"Reindexing failed: {str(e)}"
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