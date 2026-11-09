"""
Provider registry for OpenAI-compatible APIs.

The harness only needs three things from a provider: a base URL, an API key and a
model id. Everything else (Groq's free-tier limits, Ollama's local context) is
optional metadata. This module turns a provider name into that triple so the rest
of the code never hardcodes a vendor.

Configuration precedence for a provider called ``NAME``:

1. ``NAME_BASE_URL`` / ``NAME_API_KEY`` / ``NAME_MODEL`` (provider-specific)
2. ``AI_BASE_URL`` / ``AI_API_KEY`` / ``AI_MODEL``     (the active provider's values)
3. the preset in ``PRESETS``                             (known providers only)

That means a single ``AI_API_KEY``/``AI_BASE_URL``/``AI_MODEL`` block works for
any provider, while a .env that holds several keys can use the prefixed names and
switch with ``--provider`` or ``/provider``. Names are never limited to the
presets: an unknown name simply has no defaults, so it must be given a URL and a
model (this is how a self-hosted OpenAI-compatible server is added).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from src import config


@dataclass(frozen=True)
class ProviderPreset:
    """Built-in defaults for a well-known provider."""

    name: str
    label: str
    base_url: str | None
    default_model: str | None
    api_key_envs: tuple[str, ...] = ()
    local: bool = False


# Key names are intentionally the ones each vendor documents. A generic
# AI_API_KEY still works with any of these when only one provider is configured.
PRESETS: dict[str, ProviderPreset] = {
    "groq": ProviderPreset(
        "groq",
        "Groq",
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        ("GROQ_API_KEY",),
    ),
    "nvidia": ProviderPreset(
        "nvidia",
        "NVIDIA NIM",
        "https://integrate.api.nvidia.com/v1",
        "deepseek-ai/deepseek-v4.1-flash",
        ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"),
    ),
    "openai": ProviderPreset(
        "openai",
        "OpenAI",
        "https://api.openai.com/v1",
        "gpt-4o-mini",
        ("OPENAI_API_KEY",),
    ),
    "openrouter": ProviderPreset(
        "openrouter",
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        "openrouter/auto",
        ("OPENROUTER_API_KEY",),
    ),
    "together": ProviderPreset(
        "together",
        "Together AI",
        "https://api.together.xyz/v1",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        ("TOGETHER_API_KEY",),
    ),
    "deepseek": ProviderPreset(
        "deepseek",
        "DeepSeek",
        "https://api.deepseek.com/v1",
        "deepseek-chat",
        ("DEEPSEEK_API_KEY",),
    ),
    "mistral": ProviderPreset(
        "mistral",
        "Mistral AI",
        "https://api.mistral.ai/v1",
        "mistral-large-latest",
        ("MISTRAL_API_KEY",),
    ),
    "cerebras": ProviderPreset(
        "cerebras",
        "Cerebras",
        "https://api.cerebras.ai/v1",
        "llama3.3-70b",
        ("CEREBRAS_API_KEY",),
    ),
    "fireworks": ProviderPreset(
        "fireworks",
        "Fireworks AI",
        "https://api.fireworks.ai/inference/v1",
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        ("FIREWORKS_API_KEY",),
    ),
    "perplexity": ProviderPreset(
        "perplexity",
        "Perplexity",
        "https://api.perplexity.ai",
        "sonar",
        ("PERPLEXITY_API_KEY",),
    ),
    "ollama": ProviderPreset(
        "ollama",
        "Ollama (local)",
        None,  # OLLAMA_BASE_URL is the legacy spelling, read in resolve()
        None,  # OLLAMA_MODEL
        (),
        local=True,
    ),
    "custom": ProviderPreset(
        "custom",
        "OpenAI-compatible",
        None,
        None,
    ),
}

# Common shorthands. Anything not listed here is still accepted as a custom
# provider name, so this map is convenience only.
ALIASES = {
    "nim": "nvidia",
    "nvidia-nim": "nvidia",
    "nvidia_nim": "nvidia",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "compatible": "custom",
    "local": "ollama",
    "open-router": "openrouter",
    "together-ai": "together",
    "deep-seek": "deepseek",
}

GENERIC_BASE_URL_ENVS = ("AI_BASE_URL", "LLM_BASE_URL")
GENERIC_API_KEY_ENVS = ("AI_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
GENERIC_MODEL_ENVS = ("AI_MODEL", "LLM_MODEL")


@dataclass(frozen=True)
class Provider:
    """A fully resolved endpoint, ready for the OpenAI client."""

    name: str
    label: str
    base_url: str
    api_key: str | None
    api_key_source: str | None
    default_model: str
    local: bool

    def describe(self) -> str:
        key = self.api_key_source or ("not required" if self.local else "missing")
        return (
            f"{self.label} ({self.name}) | {self.base_url} | "
            f"key: {key} | default model: {self.default_model}"
        )


def normalize(name: str | None = None) -> str:
    """Lowercases a provider name and applies the shorthand aliases."""
    cleaned = (name or "").strip().lower().replace(" ", "-")
    return ALIASES.get(cleaned, cleaned)


def _prefix(name: str) -> str:
    """``open-router`` -> ``OPEN_ROUTER`` for its environment variables."""
    return "".join(character if character.isalnum() else "_" for character in name).upper()


def env_prefix(name: str | None = None) -> str:
    """Public spelling of the environment-variable prefix for a provider.

    ``env_prefix("open-router")`` is ``OPEN_ROUTER``; omitting the name uses the
    active provider. Budget overrides (``NVIDIA_MAX_OUTPUT_TOKENS``, ...) are read
    with this prefix so they stay symmetric with the connection variables.
    """
    return _prefix(normalize(name or config.LLM_PROVIDER))


def _first_env(names: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Returns (value, variable) for the first non-blank variable in ``names``."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip(), name
    return None, None


def _looks_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".local")


def resolve(name: str | None = None) -> Provider:
    """Resolves a provider name into base URL, key and default model.

    Raises ValueError with an actionable message when a required value is
    missing; the CLI prints that instead of letting the SDK fail with a 401.
    """
    requested = (name or config.LLM_PROVIDER or "ollama").strip()
    key = normalize(requested)
    if not key:
        raise ValueError("no provider name given")

    preset = PRESETS.get(key)
    prefix = _prefix(key)

    # The generic AI_* block belongs to the provider named by LLM_PROVIDER (or to
    # a custom name with no preset). It is also the fallback when LLM_PROVIDER is
    # local, which is the usual "Ollama by default, remote with --provider" setup.
    # Without this rule, switching from NVIDIA to Groq would send the NVIDIA key
    # and model to Groq instead of saying that Groq is not configured.
    configured_key = normalize(config.LLM_PROVIDER)
    configured_preset = PRESETS.get(configured_key)
    configured_is_local = bool(configured_preset and configured_preset.local)
    use_generic = (
        key == configured_key
        or configured_is_local
        or preset is None
        or key == "custom"
    )
    generic_base = GENERIC_BASE_URL_ENVS if use_generic else ()
    generic_key = GENERIC_API_KEY_ENVS if use_generic else ()
    generic_model = GENERIC_MODEL_ENVS if use_generic else ()

    # Base URL. Ollama keeps its historical name; every other provider uses the
    # prefixed name, then the generic one, then the preset.
    base_names = (
        ("OLLAMA_BASE_URL",)
        if key == "ollama"
        else (f"{prefix}_BASE_URL",) + generic_base
    )
    base_url, _ = _first_env(base_names)
    if base_url is None and preset is not None:
        base_url = preset.base_url
    if base_url is None and key == "ollama":
        base_url = config.OLLAMA_BASE_URL
    if not base_url:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(
            f"provider '{requested}' has no base URL. Set {prefix}_BASE_URL or "
            f"AI_BASE_URL (e.g. https://host/v1). Known presets: {known}."
        )
    base_url = base_url.rstrip("/")

    local = bool(preset and preset.local) or _looks_local(base_url)

    # API key. A local endpoint is allowed to run without one; the SDK still
    # needs a non-empty placeholder. Ollama in particular must not inherit a
    # remote AI_API_KEY just because one is present in .env.
    if key == "ollama":
        api_key, key_source = "local", None
    else:
        key_names = (f"{prefix}_API_KEY",) + generic_key
        # Older Groq setups stored the key under OPENAI_API_KEY (and the old
        # provider name was an "openai" alias). Keep that spelling working for
        # the groq preset even when the generic block belongs to another vendor.
        if key == "groq":
            key_names += ("OPENAI_API_KEY",)
        api_key, key_source = _first_env(key_names)
        if api_key is None and preset is not None:
            api_key, key_source = _first_env(preset.api_key_envs)
        if (
            key == "openai"
            and api_key
            and api_key.startswith("gsk_")
        ):
            raise ValueError(
                "OPENAI_API_KEY looks like a Groq key. Set LLM_PROVIDER=groq "
                "(or GROQ_API_KEY=...), or point OPENAI_BASE_URL/AI_BASE_URL at "
                "the endpoint this key belongs to."
            )
        if api_key is None:
            if local:
                api_key, key_source = "local", None
            else:
                raise ValueError(
                    f"provider '{requested}' has no API key. Set {prefix}_API_KEY or "
                    "AI_API_KEY (the legacy OPENAI_API_KEY name also works)."
                )

    # Model id. Generic AI_MODEL is the escape hatch for a custom provider.
    model_names = (
        ("OLLAMA_MODEL",) if key == "ollama" else (f"{prefix}_MODEL",) + generic_model
    )
    default_model, _ = _first_env(model_names)
    if default_model is None and key == "ollama":
        default_model = config.OLLAMA_MODEL
    if default_model is None and preset is not None:
        default_model = preset.default_model
    if not default_model:
        raise ValueError(
            f"provider '{requested}' has no model. Set {prefix}_MODEL or AI_MODEL."
        )

    label = preset.label if preset is not None else requested
    return Provider(
        name=key,
        label=label,
        base_url=base_url,
        api_key=api_key,
        api_key_source=key_source,
        default_model=default_model,
        local=local,
    )


def is_local(name: str | None = None) -> bool:
    """True when the provider runs locally (no metered token window)."""
    try:
        return resolve(name).local
    except ValueError:
        return False


def configured(active: str | None = None) -> list[str]:
    """Provider names that have a key (or are local) and can be selected now.

    Generic AI_API_KEY is only counted for the active provider: it cannot say
    which of the other presets the user meant. Provider-specific keys are what
    make a second provider visible here.
    """
    names: list[str] = []
    active = normalize(active or config.LLM_PROVIDER)
    for name, preset in PRESETS.items():
        if preset.local:
            names.append(name)
            continue
        # The old "openai" provider alias actually pointed at Groq, and stored
        # its key under OPENAI_API_KEY. Show it as Groq, not as OpenAI.
        legacy_groq_key = os.getenv("OPENAI_API_KEY") or ""
        if name == "groq" and legacy_groq_key.startswith("gsk_"):
            names.append(name)
            continue
        if (
            name == "openai"
            and legacy_groq_key.startswith("gsk_")
            and not os.getenv("OPENAI_BASE_URL")
        ):
            continue
        if name == active and config.AI_API_KEY:
            names.append(name)
            continue
        if any(os.getenv(variable) for variable in preset.api_key_envs):
            names.append(name)
    if active and active not in names:
        # A custom provider selected in .env is still a valid target.
        try:
            resolve(active)
        except ValueError:
            pass
        else:
            names.append(active)
    return names


def describe_all(active: str | None = None) -> str:
    """Multi-line summary for /provider and --help."""
    active = normalize(active or config.LLM_PROVIDER)
    rows = []
    for name in configured(active):
        marker = "*" if name == active else " "
        try:
            provider = resolve(name)
        except ValueError as exc:
            rows.append(f" {marker} {name}: {exc}")
            continue
        key = provider.api_key_source or ("not required" if provider.local else "missing")
        rows.append(f" {marker} {name:<12} {provider.base_url}  (key: {key})")
    if not rows:
        rows.append(" (none configured; set AI_API_KEY/AI_BASE_URL or a provider-specific key)")
    rows.append(
        "Switch with: /provider <name>, --provider <name>, or LLM_PROVIDER in .env. "
        "Any OpenAI-compatible name works with NAME_BASE_URL/NAME_API_KEY/NAME_MODEL."
    )
    return "\n".join(rows)
