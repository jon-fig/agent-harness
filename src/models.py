"""
Model catalog and runtime selection.

Groq's free tier applies its token window *per model, per organisation*, so the
active model decides both the request budget and which pacing window the agent
should consult. Keeping that metadata in one place makes switching models
mid-session a single assignment instead of a config reload.

Nothing here talks to the network: the catalog is the published free-tier table,
and an unknown id is accepted (with unknown limits) so new models can be tried
without waiting for a release.
"""

from dataclasses import dataclass

from src import config

DEFAULT_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    tpm: int  # tokens per minute (input + output, cached tokens excluded)
    rpm: int  # requests per minute
    rpd: int  # requests per day
    tpd: int  # tokens per day
    context: int | None = None


# Published free-tier limits (console.groq.com/docs/rate-limits). All three coding
# models share the same 8K TPM / 200K TPD bucket - a smaller model does not buy
# more room, but each model gets its own bucket.
KNOWN_MODELS: dict[str, ModelSpec] = {
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

ALIASES = {
    "120b": "openai/gpt-oss-120b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "oss-120b": "openai/gpt-oss-120b",
    "20b": "openai/gpt-oss-20b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "oss-20b": "openai/gpt-oss-20b",
    "qwen": "qwen/qwen3.8-27b",
    "qwen3.8-27b": "qwen/qwen3.8-27b",
    "safeguard": "openai/gpt-oss-safeguard-20b",
}

_STATE_FILE = config.ROOT_DIR / ".agent" / "model.txt"
_override: str | None = None


def resolve(name: str) -> str:
    """Turns an alias or full id into a model id. Unknown ids pass through."""
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("no model name given")
    return ALIASES.get(cleaned.lower(), cleaned)


def spec_for(model_id: str) -> ModelSpec | None:
    return KNOWN_MODELS.get(model_id)


def active() -> str:
    """The model a Groq request would use right now."""
    if _override:
        return _override
    try:
        saved = _STATE_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        saved = ""
    return saved or config.AI_MODEL or DEFAULT_MODEL


def set_active(name: str) -> str:
    """Persists the choice in .agent/ so it survives a restart."""
    global _override
    model_id = resolve(name)
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(model_id + "\n", encoding="utf-8")
    _override = model_id
    return model_id


def input_budget(model_id: str) -> int:
    """Input tokens this model can take in one request.

    The whole request (input + max output) has to fit the per-minute bucket, and a
    request that alone exceeds it is rejected outright rather than queued.
    """
    spec = spec_for(model_id)
    if spec is None:
        return config.MAX_INPUT_TOKENS
    room = spec.tpm - config.MAX_OUTPUT_TOKENS - 200
    return max(400, min(config.MAX_INPUT_TOKENS, room))


def describe_active() -> str:
    model_id = active()
    spec = spec_for(model_id)
    if spec is None:
        return f"[Model] Active -> {model_id} (not in the bundled catalog; budget {config.MAX_INPUT_TOKENS} tokens/request)"

    budget = input_budget(model_id)
    context = f"{spec.context:,} ctx" if spec.context else "context n/a"
    return (
        f"[Model] Active -> {spec.label} ({spec.id})\n"
        f"        {spec.tpm:,} TPM | {spec.rpm} RPM | {spec.rpd:,} RPD | {spec.tpd:,} TPD | {context}\n"
        f"        input budget: {budget} tokens/request"
    )


def table() -> str:
    """One line per bundled model, marking the active one."""
    current = active()
    rows = ["[Model] Available (Groq free tier):"]
    for spec in KNOWN_MODELS.values():
        marker = "*" if spec.id == current else " "
        rows.append(f" {marker} {spec.label:<28} {spec.id:<30} {spec.tpm:,} TPM")
    rows.append("Switch with: /model <id|alias>   (e.g. /model 20b, /model qwen)")
    return "\n".join(rows)
