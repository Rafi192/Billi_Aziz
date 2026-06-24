import os
from dotenv import load_dotenv
from pymongo import MongoClient
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DB_NAME") 
print(f"database name : {DATABASE_NAME}")

client = MongoClient(MONGODB_URI)
DB = client[DATABASE_NAME]

collections = DB.list_collection_names()
print(f"Collections in database '{DATABASE_NAME}': {collections}")

for coll in collections:
    count = DB[coll].count_documents({})
    print(f"Collection '{coll}' has {count} documents.")

client.close()