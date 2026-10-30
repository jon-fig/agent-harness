from openai import OpenAI
from src.config import (
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

def get_llm_client():
    """Returns the correct OpenAI-compatible client and active model name."""
    if LLM_PROVIDER == "ollama":
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        print(f"[Client] Connected to local Ollama (Model: {OLLAMA_MODEL})")
        return client, OLLAMA_MODEL
    else:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing in your .env file!")
        
        # Point the OpenAI client to Groq's API endpoint
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=OPENAI_API_KEY
        )
        print(f"[Client] Connected to Groq API (Model: {OPENAI_MODEL})")
        return client, OPENAI_MODEL