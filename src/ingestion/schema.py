CASUAL_PATTERNS = [
    "hello", "hi", "hey", "how are you", "how can you help",
    "what can you do", "who are you", "good morning", "good evening",
    "thanks", "thank you", "bye", "goodbye",
    "how can you assist me", "what can you do for me",
]

def is_casual_query(query: str) -> bool:
    query_lower = query.lower().strip()
    return any(pattern in query_lower for pattern in CASUAL_PATTERNS)