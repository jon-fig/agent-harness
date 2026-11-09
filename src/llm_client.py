"""OpenAI-compatible client factory.

Every provider is just a base URL + API key + model id (see ``src/providers.py``),
so there is no vendor-specific branch here. ``max_retries=0`` keeps retries in
the agent loop (``src/agent.py``), which knows how to shrink a 413 and how to
wait out a 429 using the provider's own headers.
"""

from openai import OpenAI

from src import config, models, providers


def build_client(
    provider: str | None = None,
    timeout: float | None = None,
) -> tuple[OpenAI, providers.Provider]:
    """Builds the client and its resolved endpoint without choosing a model."""
    endpoint = providers.resolve(provider)
    client = OpenAI(
        base_url=endpoint.base_url,
        # The SDK rejects an empty key; local endpoints do not need a real one.
        api_key=endpoint.api_key or "local",
        timeout=config.LLM_TIMEOUT if timeout is None else timeout,
        max_retries=0,
    )
    return client, endpoint


def get_llm_client(
    provider: str | None = None,
    model: str | None = None,
) -> tuple[OpenAI, str]:
    """Returns the correct client and active model name.

    ``provider`` lets the CLI or /provider choose an endpoint without editing
    .env; the model comes from the provider's own /model state. Reading config at
    call time rather than import time means runtime changes stick.
    """
    client, endpoint = build_client(provider)
    active = model or models.active(endpoint.name)
    print(f"[Client] Connected to {endpoint.label} ({endpoint.base_url}) (Model: {active})")
    return client, active


def list_models(provider: str | None = None) -> list[str]:
    """Asks the provider for its catalogue over GET /models.

    This is the same OpenAI-compatible endpoint every preset exposes, so model
    discovery works for any provider without a bundled table. Providers differ
    in what they advertise (some list models the key cannot actually call), so
    the list is informational; /model <id> remains the source of truth.
    """
    client, endpoint = build_client(provider, timeout=config.MODEL_DISCOVERY_TIMEOUT)
    try:
        page = client.models.list()
    except Exception as exc:
        raise RuntimeError(f"could not list models from {endpoint.base_url}: {exc}") from exc

    data = getattr(page, "data", None) or []
    ids = sorted({getattr(item, "id", "") for item in data if getattr(item, "id", "")})
    return ids
