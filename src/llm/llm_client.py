import os
import logging
from enum import Enum
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    OPENAI = "openai"


OPENAI_MODEL = "gpt-4o-mini"

active_model = LLMProvider.OPENAI

_openai_client = None


def get_openai_client():
    global _openai_client

    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")

        _openai_client = OpenAI(api_key=api_key)

    return _openai_client


def get_client():
    return get_openai_client()


def get_model_name():
    return OPENAI_MODEL