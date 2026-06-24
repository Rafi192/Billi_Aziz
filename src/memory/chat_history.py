# chat/chat_history.py
import os
import logging
from pyexpat.errors import messages
from typing import List, Dict
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10


class ChatHistory:

    def __init__(self):
        client      = MongoClient(os.getenv("MONGODB_URI"))
        db          = client[os.getenv("CHATBOT_DB", "chatbot_db")]
        self.col    = db[os.getenv("CHATBOT_HISTORY_COLLECTION", "history")]

        self.col.create_index([
            ("session_id", ASCENDING),
            ("timestamp",  ASCENDING)
        ])
        logger.info(f"ChatHistory connected → chatbot_db.history")

    def add_message(self, session_id: str, role: str, content: str):
        self.col.insert_one({
            "session_id": session_id,
            "role":       role,
            "content":    content,
            "timestamp":  datetime.now()
        })
        logger.info(f"[{session_id}] Saved {role} message")

    def get_history(self, session_id: str) -> List[Dict]:

        messages = list(
        self.col.find({"session_id": session_id}, {"_id": 0, "role": 1, "content": 1})
        .sort("timestamp", -1)   
        .limit(MAX_HISTORY_TURNS)
       )
        return list(reversed(messages))
    
    def clear_session(self, session_id: str):
        result = self.col.delete_many({"session_id": session_id})
        logger.info(f"Cleared {result.deleted_count} messages for [{session_id}]")

    def get_session_count(self, session_id: str) -> int:
        return self.col.count_documents({"session_id": session_id})