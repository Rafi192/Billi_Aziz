# # memory/chat_history.py
# import os
# import logging
# from pyexpat.errors import messages
# from typing import List, Dict
# from datetime import datetime
# from pymongo import MongoClient, ASCENDING
# from dotenv import load_dotenv

# load_dotenv()
# logger = logging.getLogger(__name__)

# MAX_HISTORY_TURNS = 20


# class ChatHistory:

#     def __init__(self):
#         client      = MongoClient(os.getenv("MONGODB_URI"))
#         db          = client[os.getenv("CHATBOT_DB", "chatbot_db")]
#         self.col    = db[os.getenv("CHATBOT_HISTORY_COLLECTION", "history")]

#         self.col.create_index([
#             ("session_id", ASCENDING),
#             ("timestamp",  ASCENDING)
#         ])
#         logger.info(f"ChatHistory connected → chatbot_db.history")

#     def add_message(self, session_id: str, role: str, content: str):
#         self.col.insert_one({
#             "session_id": session_id,
#             "role":       role,
#             "content":    content,
#             "timestamp":  datetime.now()
#         })
#         logger.info(f"[{session_id}] Saved {role} message")

#     def get_history(self, session_id: str) -> List[Dict]:

#         messages = list(
#         self.col.find({"session_id": session_id}, {"_id": 0, "role": 1, "content": 1})
#         .sort("timestamp", -1)   
#         .limit(MAX_HISTORY_TURNS)
#        )
#         return list(reversed(messages))
    
#     def clear_session(self, session_id: str):
#         result = self.col.delete_many({"session_id": session_id})
#         logger.info(f"Cleared {result.deleted_count} messages for [{session_id}]")

#     def get_session_count(self, session_id: str) -> int:
#         return self.col.count_documents({"session_id": session_id})

# memory/chat_history.py
import os
import logging
from pyexpat.errors import messages
from typing import List, Dict
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING  # UPDATED: added DESCENDING
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 20


class ChatHistory:

    def __init__(self):
        client      = MongoClient(os.getenv("MONGODB_URI"))
        db          = client[os.getenv("CHATBOT_DB", "chatbot_db")]
        self.col    = db[os.getenv("CHATBOT_HISTORY_COLLECTION", "history")]

        self.col.create_index([
            ("session_id", ASCENDING),
            ("timestamp",  ASCENDING)
        ])
        # NEW: Index on timestamp alone — powers the get_all_sessions sort efficiently
        self.col.create_index([("timestamp", DESCENDING)], name="timestamp_desc")
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

    # NEW: Return all unique session_ids sorted by most recent activity
    def get_all_sessions(self) -> List[str]:
        """
        Aggregates the chat history collection to return unique session_ids,
        ordered by the timestamp of their most recent message (descending).

        Pipeline:
          1. $sort     — sort all docs newest-first so $last picks the latest timestamp
          2. $group    — group by session_id, keep the max (latest) timestamp
          3. $sort     — sort groups by that latest timestamp, newest session first
          4. $project  — return only the session_id field
        """
        pipeline = [
            # Step 1: pre-sort so $group can rely on document order if needed
            {"$sort": {"timestamp": DESCENDING}},
            # Step 2: one document per session_id, capturing the most recent timestamp
            {
                "$group": {
                    "_id": "$session_id",
                    "last_active": {"$max": "$timestamp"}
                }
            },
            # Step 3: newest-active session appears first
            {"$sort": {"last_active": DESCENDING}},
            # Step 4: only session_id needed in the result
            {"$project": {"_id": 1}}
        ]

        results = list(self.col.aggregate(pipeline))
        session_ids = [doc["_id"] for doc in results]
        logger.info(f"get_all_sessions → {len(session_ids)} unique sessions found")
        return session_ids