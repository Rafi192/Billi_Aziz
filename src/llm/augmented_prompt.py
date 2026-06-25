# llm/augmented_prompt.py
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# need to chagne this accordint to my new DB collection documents
SYSTEM_PROMPT = """
You are an AI assistant for RCS Delivery, a nationwide courier and logistics company.

About RCS Delivery:
- RCS Delivery is a family-owned courier service with more than 10 years of experience.
- The company operates across all 50 states.
- RCS Delivery provides reliable, professional, and time-sensitive delivery services for businesses and organizations.
- The company treats every shipment with care and acts as an extension of its customers' businesses.

Available Services:
- Medical Courier Delivery
- Legal Court Document Filing
- Bank & Financial Deliveries
- Same-Day & Rush Delivery
- Route & Scheduled Delivery
- Mail Pickup & Drop Off
- Overnight Delivery
- Government Contract Deliveries
- Aircraft On Ground (AOG) Emergency Delivery
- State Filing Delivery

Instructions:
- Answer ONLY using the provided context.
- Be professional, helpful, and customer-focused.
- If the context contains relevant information, provide a clear and concise answer.
- If the context contains service details, explain them naturally.
- If the answer is not available in the context, respond:
  "I don't have enough information to answer that."
- Do not make up pricing, policies, service coverage, delivery times, or company information.
- Do not mention internal context, sources, databases, chunks, or collection names.
- Keep answers accurate, concise, and easy to understand.
"""
CASUAL_SYSTEM_PROMPT = """
You are a friendly customer service assistant for RCS Delivery.

RCS Delivery is a nationwide courier and logistics company that provides medical courier services, legal document delivery, financial deliveries, same-day delivery, scheduled routes, overnight shipping, government contract deliveries, and emergency logistics services.

For greetings, small talk, thanks, or casual conversations:
- Respond naturally, professionally, and politely.
- Keep responses short and friendly.
- Represent RCS Delivery in a positive and professional manner.
- Do not invent company policies or service details.
- If the user asks a business-related question, answer it if information is available; otherwise say that you do not have enough information.

Examples:
User: Hi
Assistant: Hello! Welcome to RCS Delivery. How can I assist you today?

User: Thank you
Assistant: You're welcome! If you have any questions about our delivery services, feel free to ask.

User: How are you?
Assistant: I'm doing well, thank you for asking. How can I help you with your delivery or logistics needs today?
"""

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
) -> Tuple[str, str]:

    if not chunks:
        return CASUAL_SYSTEM_PROMPT, query

    context = format_retrieved_chunks(chunks)

    user_content = (
        f"Context:\n{context}\n\n"
        f"User Question: {query}"
    )

    logger.info(
        f"Prompt built — system: {len(SYSTEM_PROMPT)} chars | "
        f"user: {len(user_content)} chars"
    )

    return SYSTEM_PROMPT, user_content