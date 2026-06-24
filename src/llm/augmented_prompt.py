# llm/augmented_prompt.py
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# need to chagne this accordint to my new DB collection documents
SYSTEM_PROMPT = (
    "You are a helpful AI assistant for a job platform.\n"
    "Answer the user's question using ONLY the context provided below.\n"
    "Rules:\n"
    "- Answer clearly and concisely using only the context.\n"
    "- If the context contains URLs or links, include them in your answer.\n"
    "- If the answer is NOT in the context, say: 'I don't have enough information to answer that.'\n"
    "- Do NOT hallucinate or add any information not present in the context.\n"
    "- Keep answers focused and relevant to the question.\n"
    "- Do NOT reference 'Source', 'Context', or collection names in your answer.\n"
    "- Synthesize the information naturally as if you know it.\n"
)

CASUAL_SYSTEM_PROMPT = (
    "You are a helpful AI assistant for a job platform. "
    "Respond naturally and friendly to the user's message. "
    "Keep it brief and helpful."
)


def format_retrieved_chunks(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant information found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        text     = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        block = f"[Context {i}]\n{text}"

        for url_field in ["companyURL", "portfolioUrl", "linkedinUrl"]:
            val = metadata.get(url_field, "")
            if val:
                block += f"\n{url_field}: {val}"

        context_parts.append(block)

    return "\n\n".join(context_parts)


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return ""

    turns = []
    for turn in chat_history[-6:]:
        role    = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            turns.append(f"User: {content}")
        elif role == "assistant":
            turns.append(f"Assistant: {content}")

    return "\n".join(turns)


def build_augmented_prompt(
    query: str,
    chunks: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]] = None,
) -> Tuple[str, str]:
   

    # ── Casual query — no retrieved chunks ───────────────────────────────────
    if not chunks:
        return CASUAL_SYSTEM_PROMPT, query

    # ── RAG query — with retrieved context ───────────────────────────────────
    context      = format_retrieved_chunks(chunks)
    history_text = format_chat_history(chat_history or [])

    user_content = ""
    if history_text:
        user_content += f"Conversation History:\n{history_text}\n\n"

    user_content += f"Context:\n{context}\n\nUser Question: {query}"

    logger.info(f"Prompt built — system: {len(SYSTEM_PROMPT)} chars | user: {len(user_content)} chars")
    return SYSTEM_PROMPT, user_content