"""
Model catalog, per-provider selection and request budget.

A provider decides which model ids exist; the harness only needs limits when a
provider publishes them. Groq's free tier is the one bundled catalog: its token
window is *per model, per organisation*, so the active model decides both the
request budget and which pacing window the agent should consult.

Other providers (NVIDIA NIM, OpenAI, OpenRouter, a self-hosted endpoint, ...)
work without a catalog entry. An unknown model simply falls back to
``MAX_INPUT_TOKENS`` and has no daily guard, which keeps the harness usable while
``/model list`` queries the provider's real catalogue.

Nothing here talks to the network: discovery lives in ``src/llm_client.py``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from src import config, providers


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    tpm: int  # tokens per minute (input + output, cached tokens excluded)
    rpm: int  # requests per minute
    rpd: int  # requests per day
    tpd: int  # tokens per day
    context: int | None = None


@dataclass(frozen=True)
class RateLimits:
    """Provider-enforced limits, separate from the request-size budget.

    A context window says "how much can one request carry"; a rate limit says
    "how often / how much may this account send". They fail differently (413 vs
    429) and reset on different clocks, so they are kept apart.
    """

    tpm: int | None = None  # tokens per minute
    rpm: int | None = None  # requests per minute
    rpd: int | None = None  # requests per day
    tpd: int | None = None  # tokens per day

    @property
    def any(self) -> bool:
        return any(value is not None for value in (self.tpm, self.rpm, self.rpd, self.tpd))

    def describe(self) -> str:
        if not self.any:
            return "no rate limits configured"
        parts = []
        for label, value in (
            ("TPM", self.tpm),
            ("RPM", self.rpm),
            ("RPD", self.rpd),
            ("TPD", self.tpd),
        ):
            if value is not None:
                parts.append(f"{value:,} {label}")
        return " | ".join(parts)


@dataclass(frozen=True)
class RequestBudget:
    """Everything that bounds a single request, with no enforcement inside.

    ``context_window`` is the model's total window (input + output);
    ``max_input_tokens`` is what this harness allows for the request body;
    ``max_output_tokens`` is the generation cap. ``rate_limits`` belongs to the
    provider account and is enforced by ``src/usage.py``.
    """

    model_id: str
    provider: str
    context_window: int | None
    max_input_tokens: int
    max_output_tokens: int
    rate_limits: RateLimits
    source: str = "default"

    def describe(self) -> str:
        context = f"{self.context_window:,} ctx" if self.context_window else "context unknown"
        return (
            f"{context} | input ≤ {self.max_input_tokens:,} | "
            f"output ≤ {self.max_output_tokens:,} | {self.rate_limits.describe()}"
        )


# Published free-tier limits (console.groq.com/docs/rate-limits). All three
# coding models share the same 8K TPM / 200K TPD bucket - a smaller model does
# not buy more room, but each model gets its own bucket.
GROQ_MODELS: dict[str, ModelSpec] = {
    "openai/gpt-oss-120b": ModelSpec(
        "openai/gpt-oss-120b", "gpt-oss-120b (strongest)", 8_000, 30, 1_000, 200_000, 131_072
    ),
    "openai/gpt-oss-20b": ModelSpec(
        "openai/gpt-oss-20b", "gpt-oss-20b (fast)", 8_000, 30, 1_000, 200_000, 131_072
    ),
    "qwen/qwen3.8-27b": ModelSpec(
        "qwen/qwen3.8-27b", "qwen3.8-27b (reasoning)", 8_000, 30, 1_000, 200_000, None
    ),
    "openai/gpt-oss-safeguard-20b": ModelSpec(
        "openai/gpt-oss-safeguard-20b", "gpt-oss-safeguard-20b (2K TPM)", 2_000, 3, 1_000, 200_000, 131_072
    ),
}

# Provider name -> models whose limits are known. Missing providers are valid;
# they just have no bundled limits.
PROVIDER_MODELS: dict[str, dict[str, ModelSpec]] = {
    "groq": GROQ_MODELS,
}

ALIASES_BY_PROVIDER: dict[str, dict[str, str]] = {
    "groq": {
        "120b": "openai/gpt-oss-120b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "oss-120b": "openai/gpt-oss-120b",
        "20b": "openai/gpt-oss-20b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "oss-20b": "openai/gpt-oss-20b",
        "qwen": "qwen/qwen3.8-27b",
        "qwen3.8-27b": "qwen/qwen3.8-27b",
        "safeguard": "openai/gpt-oss-safeguard-20b",
    },
}

# Previous versions stored the single Groq choice here. It is still read for
# Groq when models.json has no entry, so an upgrade does not forget the model.
_LEGACY_STATE_FILE = config.ROOT_DIR / ".agent" / "model.txt"

# Runtime overrides for this process only; the persisted copy is models.json.
_override: dict[str, str] = {}


def _provider(provider: str | None) -> str:
    return providers.normalize(provider or config.LLM_PROVIDER)


def _read_state() -> dict[str, str]:
    try:
        data = json.loads(config.MODEL_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if value}


def _write_state(state: dict[str, str]) -> None:
    config.MODEL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.MODEL_STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, config.MODEL_STATE_FILE)


def resolve(name: str, provider: str | None = None) -> str:
    """Turns an alias or full id into a model id. Unknown ids pass through."""
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("no model name given")
    aliases = ALIASES_BY_PROVIDER.get(_provider(provider), {})
    return aliases.get(cleaned.lower(), cleaned)


def spec_for(model_id: str, provider: str | None = None) -> ModelSpec | None:
    return PROVIDER_MODELS.get(_provider(provider), {}).get(model_id)


def active(provider: str | None = None) -> str:
    """The model this provider would use right now."""
    key = _provider(provider)
    if _override.get(key):
        return _override[key]

    saved = _read_state().get(key)
    if saved:
        return saved

    if key == "groq":
        try:
            legacy = _LEGACY_STATE_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            legacy = ""
        if legacy:
            return legacy

    return providers.resolve(key).default_model


def set_active(name: str, provider: str | None = None) -> str:
    """Persists the choice per provider, so switching does not clobber the other."""
    key = _provider(provider)
    model_id = resolve(name, key)
    state = _read_state()
    state[key] = model_id
    _write_state(state)
    _override[key] = model_id
    return model_id


def _env_int(*names: str) -> int | None:
    """First non-blank, integer-valued environment variable in ``names``."""
    for name in names:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            continue
        try:
            return int(float(raw))
        except ValueError:
            continue
    return None


def _model_tag(model_id: str) -> str:
    """``minimaxai/minimax-m3.1-flash`` -> ``MINIMAXAI_MINIMAX_M3_1_FLASH``."""
    return re.sub(r"[^A-Z0-9]+", "_", model_id.upper()).strip("_")


def _rate_limit(env_name: str, fallback: int | None) -> int | None:
    value = _env_int(env_name)
    return fallback if value is None else value


# A request is input + reserved output. Both share the context window, and on a
# metered provider both share the per-minute token bucket, so the input ceiling is
# never allowed to eat the whole allowance: at least this much prompt room is kept,
# and a small safety margin covers the provider's own accounting.
MIN_REQUEST_INPUT_TOKENS = 400
RATE_LIMIT_HEADROOM = 200


def request_budget(model_id: str, provider: str | None = None) -> RequestBudget:
    """Builds the size and rate budget for one provider/model pair.

    Precedence for each value: a model-specific environment override, then the
    provider-prefixed override, then bundled metadata (Groq), then the global
    defaults. Context window, input ceiling and output ceiling are computed
    separately: a request has to fit the context window *and* the per-minute
    bucket, and the output cap is reserved before measuring the input.
    """
    key = _provider(provider)
    spec = spec_for(model_id, key)
    local = providers.is_local(key)
    prefix = providers.env_prefix(key)
    tag = _model_tag(model_id)

    max_output = _env_int(
        f"MODEL_{tag}_MAX_OUTPUT_TOKENS",
        f"{prefix}_MAX_OUTPUT_TOKENS",
    )
    if max_output is None:
        max_output = config.MAX_OUTPUT_TOKENS
    max_output = max(1, max_output)

    context = _env_int(
        f"MODEL_{tag}_CONTEXT_WINDOW",
        f"{prefix}_CONTEXT_WINDOW",
        "MODEL_CONTEXT_WINDOW",
    )
    if context is None and spec is not None:
        context = spec.context
    if context is not None:
        context = max(context, 1)

    bundled = RateLimits(
        tpm=spec.tpm if spec else None,
        rpm=spec.rpm if spec else None,
        rpd=spec.rpd if spec else None,
        tpd=spec.tpd if spec else None,
    )
    rate_limits = RateLimits(
        tpm=_rate_limit(f"{prefix}_TPM", bundled.tpm),
        rpm=_rate_limit(f"{prefix}_RPM", bundled.rpm),
        rpd=_rate_limit(f"{prefix}_RPD", bundled.rpd),
        tpd=_rate_limit(f"{prefix}_TPD", bundled.tpd),
    )

    configured_input = _env_int(
        f"MODEL_{tag}_MAX_INPUT_TOKENS",
        f"{prefix}_MAX_INPUT_TOKENS",
    )
    if configured_input is None:
        configured_input = (
            config.OLLAMA_MAX_INPUT_TOKENS if local else config.MAX_INPUT_TOKENS
        )

    # A generic MAX_OUTPUT_TOKENS is a request, not a guarantee: clamp it to what
    # this model/provider can actually emit. Without this, raising the default to
    # 4000 would reserve more than a 2000-TPM bucket (leaving a 400-token input
    # budget and an output request the provider must reject) and would silently
    # override a lower model-specific or provider-specific maximum.
    output_ceiling: int | None = None
    if context is not None:
        output_ceiling = context - MIN_REQUEST_INPUT_TOKENS
    if rate_limits.tpm is not None:
        tpm_output = rate_limits.tpm - MIN_REQUEST_INPUT_TOKENS - RATE_LIMIT_HEADROOM
        output_ceiling = tpm_output if output_ceiling is None else min(output_ceiling, tpm_output)
    if output_ceiling is not None:
        max_output = max(1, min(max_output, output_ceiling))

    # The request's input and its reserved output share the context window, and
    # on a metered provider they also share the per-minute token bucket.
    ceiling: int | None = None
    if context is not None:
        ceiling = context - max_output
    if rate_limits.tpm is not None:
        tpm_room = rate_limits.tpm - max_output - RATE_LIMIT_HEADROOM
        ceiling = tpm_room if ceiling is None else min(ceiling, tpm_room)

    max_input = configured_input if ceiling is None else min(configured_input, ceiling)
    max_input = max(MIN_REQUEST_INPUT_TOKENS, max_input)

    if spec is not None:
        source = "bundled"
    elif local:
        source = "local"
    elif any(
        _env_int(name) is not None
        for name in (
            f"MODEL_{tag}_MAX_OUTPUT_TOKENS",
            f"MODEL_{tag}_MAX_INPUT_TOKENS",
            f"MODEL_{tag}_CONTEXT_WINDOW",
            f"{prefix}_MAX_OUTPUT_TOKENS",
            f"{prefix}_MAX_INPUT_TOKENS",
            f"{prefix}_CONTEXT_WINDOW",
            f"{prefix}_TPM",
            f"{prefix}_RPM",
            f"{prefix}_RPD",
            f"{prefix}_TPD",
        )
    ):
        source = "environment"
    else:
        source = "default"

    return RequestBudget(
        model_id=model_id,
        provider=key,
        context_window=context,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        rate_limits=rate_limits,
        source=source,
    )


def input_budget(model_id: str, provider: str | None = None) -> int:
    """Input tokens this model can take in one request.

    Kept as a thin compatibility helper; callers that need the output cap,
    context window or rate limits should ask for ``request_budget`` instead.
    """
    return request_budget(model_id, provider).max_input_tokens


def describe_active(provider: str | None = None) -> str:
    key = _provider(provider)
    model_id = active(key)
    try:
        endpoint = providers.resolve(key)
        where = f"{endpoint.label} ({endpoint.base_url})"
    except ValueError:
        where = key

    budget = request_budget(model_id, key)
    label = spec_for(model_id, key)
    title = f"{label.label} ({label.id})" if label is not None else model_id
    source = "" if budget.source == "bundled" else f" [{budget.source}]"
    return (
        f"[Model] Active -> {title} on {where}\n"
        f"        {budget.describe()}{source}"
    )


def table(provider: str | None = None) -> str:
    """One line per bundled model, marking the active one."""
    key = _provider(provider)
    current = active(key)
    bundled = PROVIDER_MODELS.get(key, {})

    if not bundled:
        try:
            endpoint = providers.resolve(key)
            where = f" from {endpoint.base_url}/models"
        except ValueError:
            where = ""
        return (
            f"[Model] No bundled catalog for {key}.\n"
            f"        Query the provider with: /model list{where}\n"
            f"        Then switch with: /model <id>"
        )

    rows = [f"[Model] Available ({key} catalog):"]
    for spec in bundled.values():
        marker = "*" if spec.id == current else " "
        rows.append(f" {marker} {spec.label:<28} {spec.id:<30} {spec.tpm:,} TPM")
    rows.append(
        "Switch with: /model <id|alias>   (e.g. /model 20b, /model qwen)\n"
        "Ask the provider for its full list with: /model list"
    )
    return "\n".join(rows)
