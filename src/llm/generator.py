
# llm/generator.py
import time
import logging
from typing import List, Dict, Any
from llm.llm_client import get_client, get_model_name, active_model, LLMProvider
from llm.augmented_prompt import build_augmented_prompt

logger = logging.getLogger(__name__)

MAX_RETRIES  = 2
RETRY_DELAY  = 1.5   # seconds between retries


def generate_response(
    query: str,
    chunks: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]] = None,
    max_new_tokens: int = 512,
) -> str:

    try:
        # Step 1 — build prompt (returns two strings, not one merged blob)
        system_prompt, user_content = build_augmented_prompt(
            query=query,
            chunks=chunks,
            chat_history=chat_history or [],
        )

        logger.info(
            f"Prompt built — system: {len(system_prompt)} chars | "
            f"user: {len(user_content)} chars | provider: {active_model}"
        )

        client     = get_client()
        model_name = get_model_name()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        # Step 2 — call LLM with retry/backoff
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                result = _call_llm(client, model_name, messages, max_new_tokens)
                logger.info(f"Generated {len(result)} chars (attempt {attempt})")
                return result

            except Exception as e:
                if attempt <= MAX_RETRIES:
                    logger.warning(f"Attempt {attempt} failed: {e} — retrying in {RETRY_DELAY}s")
                    time.sleep(RETRY_DELAY)
                else:
                    raise

    except Exception as e:
        logger.error(f"Generation failed after all retries: {e}")
        return "I'm sorry, I was unable to generate a response. Please try again."


def _call_llm(client, model_name: str, messages: list, max_tokens: int) -> str:

    # OpenAI and Groq both implement the OpenAI chat completions spec
    if active_model in (LLMProvider.OPENAI, LLMProvider.GROQ):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    # HuggingFace InferenceClient
    response = client.chat_completion(
        messages=messages,
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()