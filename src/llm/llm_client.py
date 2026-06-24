import os
import logging
from enum import Enum
import openai
from groq import Groq

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    HF = "huggingface"
    GROQ = "groq"
    OPENAI = "openai"


# HF_MODEL   = "meta-llama/Llama-3.2-3B-Instruct"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
GROQ_MODEL = "llama-3.1-8b-instant"

OPENAI_MODEL = "gpt-4o-mini"

active_model = LLMProvider.OPENAI
_hf_client = None
_groq_client = None
_openai_client = None


def get_hf_client():
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceClient
        api_key = os.getenv("HF_API_KEY")
        if not api_key:
            raise ValueError("HF_API_KEY not found in .env")
        # return InferenceClient(token=api_key)
        _hf_client = InferenceClient(token=api_key)
    return _hf_client


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")
        _groq_client = Groq(api_key=api_key)
    
    return _groq_client
        

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        _openai_client = openai.OpenAI(api_key=api_key)
    return _openai_client

def get_client():
    if active_model == LLMProvider.GROQ:
        return get_groq_client()
    elif active_model == LLMProvider.OPENAI:
        return get_openai_client()
    return get_hf_client()


def get_model_name():
    if active_model == LLMProvider.GROQ:
        return GROQ_MODEL
    elif active_model == LLMProvider.OPENAI:
        return OPENAI_MODEL
    return HF_MODEL
