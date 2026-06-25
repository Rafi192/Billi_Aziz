# llm/generator.py

import time
import logging
from typing import List, Dict, Any

from llm.llm_client import (
    get_client,
    get_model_name,
    active_model,
    LLMProvider
)

from llm.augmented_prompt import build_augmented_prompt

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY = 1.5


def generate_response(
    query: str,
    chunks: List[Dict[str, Any]],
    chat_history: List[Dict[str, str]] = None,
    max_new_tokens: int = 256,
) -> str:

    try:

        system_prompt, user_content = build_augmented_prompt(
            query=query,
            chunks=chunks
        )

        client = get_client()
        model_name = get_model_name()

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        # Add previous conversation turns as real chat messages
        if chat_history:
            for turn in chat_history[-6:]:

                role = turn.get("role")
                content = turn.get("content")

                if not role or not content:
                    continue

                if role not in ("user", "assistant"):
                    continue

                messages.append({
                    "role": role,
                    "content": content
                })

        # Current user message
        messages.append({
            "role": "user",
            "content": user_content
        })

        logger.info(
            f"Messages: {len(messages)} | "
            f"Provider: {active_model} | "
            f"System chars: {len(system_prompt)} | "
            f"User chars: {len(user_content)}"
        )

        for attempt in range(1, MAX_RETRIES + 2):

            try:

                result = _call_llm(
                    client=client,
                    model_name=model_name,
                    messages=messages,
                    max_tokens=max_new_tokens
                )

                logger.info(
                    f"Generated response ({len(result)} chars) "
                    f"attempt={attempt}"
                )

                return result

            except Exception as e:

                if attempt <= MAX_RETRIES:

                    logger.warning(
                        f"Generation attempt {attempt} failed: {e}. "
                        f"Retrying in {RETRY_DELAY}s..."
                    )

                    time.sleep(RETRY_DELAY)

                else:
                    raise

    except Exception as e:

        logger.exception(f"Generation failed: {e}")

        return (
            "I'm sorry, I was unable to generate a response. "
            "Please try again."
        )


def _call_llm(
    client,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int
) -> str:

    if active_model in (
        LLMProvider.OPENAI,
        LLMProvider.GROQ
    ):

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("LLM returned empty response")

        return content.strip()

    response = client.chat_completion(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("LLM returned empty response")

    return content.strip()