from openai import OpenAI

from src import config, models

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_llm_client(provider: str | None = None, model: str | None = None):
    """Returns the correct OpenAI-compatible client and active model name.

    ``provider`` lets the CLI pick Ollama for iteration-heavy work without editing
    .env; the model comes from ``/model`` state (Groq) or OLLAMA_MODEL. Reading
    ``config`` at call time rather than import time means runtime changes stick.
    max_retries=0 keeps retries in the agent loop (src/agent.py), which knows how to
    shrink a 413 and how to wait out a 429 using the provider's own headers.
    """
    provider = (provider or config.LLM_PROVIDER).lower()

    if provider == "ollama":
        active = model or config.OLLAMA_MODEL
        client = OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=config.LLM_TIMEOUT,
            max_retries=0,
        )
        print(f"[Client] Connected to local Ollama (Model: {active})")
        return client, active

    # "openai" is kept as a silent legacy alias because earlier .env files used it;
    # both names point at Groq's endpoint.
    if provider not in ("groq", "openai"):
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use 'groq' (remote) or 'ollama' (local)."
        )
    if not config.AI_API_KEY:
        raise ValueError(
            "AI_API_KEY is missing: set it in .env or export it (the legacy OPENAI_API_KEY name is still read)."
        )

    active = model or models.active()
    client = OpenAI(
        base_url=GROQ_BASE_URL,
        api_key=config.AI_API_KEY,
        timeout=config.LLM_TIMEOUT,
        max_retries=0,
    )
    print(f"[Client] Connected to Groq API (Model: {active})")
    return client, active
