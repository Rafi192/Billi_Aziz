
#ingestion/load_data.py
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
import logging
import re

logger = logging.getLogger(__name__)


class MongoDBLoader:



    def __init__(self, connection_string: str, database_name: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.database_name = database_name

        logger.info(f"Connected to MongoDB: {database_name}")



    def clean_text(self, text: Any) -> str:
        if text is None:
            return ""

        text = str(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    

    # important method if i want to format multiple collections and return a single flat list of documents

    # def format_document_for_rag(
    #     self,
    #     document: Dict[str, Any],
    #     collection_name: str
    # ) -> Optional[Dict[str, Any]]:

    #     text_parts = []

    #     for key, value in document.items():

    #         if key == "_id":
    #             continue

    #         if value is None:
    #             continue

    #         if isinstance(value, str):
    #             cleaned = self.clean_text(value)
    #             if cleaned:
    #                 text_parts.append(f"{key}: {cleaned}")

    #         elif isinstance(value, (int, float, bool)):
    #             text_parts.append(f"{key}: {value}")

    #         elif isinstance(value, list):
    #             values = [str(v) for v in value if v]
    #             if values:
    #                 text_parts.append(f"{key}: {', '.join(values)}")

    #         elif isinstance(value, dict):
    #             values = [f"{k}: {v}" for k, v in value.items()]
    #             if values:
    #                 text_parts.append(f"{key}: {', '.join(values)}")

    #     combined_text = "\n".join(text_parts).strip()

    #     if not combined_text:
    #         return None

    #     return {
    #         "id": str(document.get("_id", "")),
    #         "text": combined_text,
    #         "metadata": {
    #             "source": "mongodb",
    #             "database": self.database_name,
    #             "collection": collection_name,
    #             "document_id": str(document.get("_id", "")),
    #         }
    #     }

    def flatten_document(
    self,
    data: Any,
    prefix: str = ""
) -> List[str]:
        
        output = []

        if isinstance(data, dict):

            for key, value in data.items():

                if key == "_id":
                    continue

                new_prefix = f"{prefix}.{key}" if prefix else key

                output.extend(
                    self.flatten_document(
                        value,
                        new_prefix
                    )
                )

        elif isinstance(data, list):

            values = []

            for item in data:

                if isinstance(item, (dict, list)):
                    output.extend(
                        self.flatten_document(
                            item,
                            prefix
                        )
                    )
                else:
                    values.append(str(item))

            if values:
                output.append(
                    f"{prefix}: {', '.join(values)}"
                )

        else:

            value = self.clean_text(data)

            if value:
                output.append(
                    f"{prefix}: {value}"
                )

        return output

    def format_document_for_rag(
    self,
    document: Dict[str, Any],
    collection_name: str
):
         
    
    
        flattened = self.flatten_document(document)

        if not flattened:
            return None

        text = "\n".join(flattened)

        return {
            "id": str(document.get("_id", "")),
            "text": text,
            "metadata": {
                "source": "mongodb",
                "database": self.database_name,
                "collection": collection_name,
                "document_id": str(document.get("_id", ""))
            }
        }


    # Single Collection Loader


    def load_single_collection(
        self,
        collection_name: str,
        filter_query: Optional[Dict] = None,
        projection: Optional[Dict] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        
        """
        Load and format documents from a single collection.
        """

        collection = self.db[collection_name]

        query = collection.find(
            filter_query or {},
            projection
        )

        if limit:
            query = query.limit(limit)

        raw_documents = list(query)

        logger.info(
            f"Loaded {len(raw_documents)} documents from '{collection_name}'"
        )

        formatted_documents = []

        for doc in raw_documents:
            formatted = self.format_document_for_rag(
                doc,
                collection_name
            )

            if formatted:
                formatted_documents.append(formatted)

        logger.info(
            f"Formatted {len(formatted_documents)} documents from '{collection_name}'"
        )

        return formatted_documents


    # Multi Collection Loader

    def load_multiple_collections(
        self,
        collection_names: List[str],
        filter_query: Optional[Dict] = None,
        limit_per_collection: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        
        """
        Load and format documents from multiple collections.

        Returns:
        {
            "products": [...],
            "blogs": [...],
            "faq": [...]
        }
        """

        results = {}

        for collection_name in collection_names:

            docs = self.load_single_collection(
                collection_name=collection_name,
                filter_query=filter_query,
                limit=limit_per_collection
            )

            results[collection_name] = docs

        total_docs = sum(len(v) for v in results.values())

        logger.info(
            f"Loaded {total_docs} documents from "
            f"{len(collection_names)} collections"
        )

        return results


    # Flatten Multi Collection Result

    def load_multiple_collections_flat(
        self,
        collection_names: List[str],
        filter_query: Optional[Dict] = None,
        limit_per_collection: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        
        """
        Load multiple collections and return a single flat list.
        """

        collection_data = self.load_multiple_collections(
            collection_names=collection_names,
            filter_query=filter_query,
            limit_per_collection=limit_per_collection
        )

        flattened = []

        for docs in collection_data.values():
            flattened.extend(docs)

        return flattened


    def get_collection_names(self) -> List[str]:
        return self.db.list_collection_names()

    def close(self):
        self.client.close()
        logger.info("MongoDB connection closed")