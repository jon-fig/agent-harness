"""
Model-aware token estimation.

There are three distinct things the harness must not confuse:

* an *estimate* made locally before a request is sent;
* the *reported* usage returned by the provider after a request;
* the *budget* the request is checked against.

This module owns only the first. It resolves the best tokenizer it can for a
model, and every estimate carries an ``exact`` flag so callers never present a
fallback as if it were provider-reported usage.

Exact tokenizers are practical only for model families with a matching local
tokenizer. ``tiktoken`` covers OpenAI's encodings; other families fall back to a
pessimistic character estimate unless the user supplies a compatible tokenizer
through one of these environment variables:

* ``TOKENIZER_<SANITIZED_MODEL_ID>`` - e.g. ``TOKENIZER_QWEN_QWEN3_8_27B``
* ``TOKENIZER_MAP`` - a JSON object of ``fnmatch`` pattern -> tokenizer spec
* ``TOKENIZER_DEFAULT`` - fallback tokenizer spec

A spec can be ``tiktoken:<encoding>``, ``approx[:chars_per_token]``,
``hf:<repo>`` or ``file:<tokenizer.json>``. The HuggingFace forms are optional:
if ``tokenizers`` is not installed, resolution falls back to the approximate
tokenizer instead of failing.
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from src import config

try:  # optional dependency
    import tiktoken
except ImportError:  # pragma: no cover - depends on the environment
    tiktoken = None

DEFAULT_CHARS_PER_TOKEN = 4.0

# The fallback ratio is learned from what the provider actually reports (see
# record_calibration): a family tokenizer can differ from chars/4 by a wide
# margin. Observations are smoothed and clamped so one odd response cannot skew
# every later estimate.
CALIBRATION_FILE = config.SESSION_FILE.parent / "tokenizer.json"
MIN_CHARS_PER_TOKEN = 1.5
MAX_CHARS_PER_TOKEN = 8.0
CALIBRATION_SMOOTHING = 0.5

# The regexes are deliberately about model families, not vendors: the same model
# id can be served by Groq, NVIDIA, OpenRouter or a local gateway.
_OPENAI_ENCODINGS: tuple[tuple[str, str], ...] = (
    (r"(^|/)gpt-oss", "o200k_harmony"),
    (r"(^|/)(gpt-4o|gpt-4\.1|gpt-5|chatgpt|o[1-9])", "o200k_base"),
    (r"(^|/)(gpt-4|gpt-3\.5|text-embedding-3|text-embedding-ada|davinci|curie|babbage)", "cl100k_base"),
    (r"(^|/)(codex|code-davinci)", "p50k_base"),
)


@dataclass(frozen=True)
class TokenEstimate:
    """A token count plus enough provenance to report it honestly."""

    tokens: int
    exact: bool
    tokenizer: str
    note: str = ""

    def describe(self) -> str:
        state = "exact" if self.exact else "approximate"
        suffix = f"; {self.note}" if self.note else ""
        return f"~{self.tokens} tokens ({state} via {self.tokenizer}{suffix})"


class Tokenizer:
    """Small interface so tiktoken, HF and approximate counts look the same."""

    name: str
    exact: bool

    def count(self, text: str) -> int:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class TiktokenTokenizer(Tokenizer):
    name: str
    encoding_name: str
    encoding: Any
    exact: bool = True

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))


@dataclass(frozen=True)
class ApproximateTokenizer(Tokenizer):
    name: str = "approx"
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
    exact: bool = False

    def count(self, text: str) -> int:
        # Round up: a guard rail should overestimate rather than undercount.
        return int(math.ceil(len(text) / self.chars_per_token))


@dataclass(frozen=True)
class CallableTokenizer(Tokenizer):
    name: str
    counter: Callable[[str], int]
    exact: bool = True

    def count(self, text: str) -> int:
        return int(self.counter(text))


_CACHE: dict[tuple[str, str, float], Tokenizer] = {}


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _model_key(model_id: str | None, provider: str | None) -> str:
    return f"{provider or ''}/{model_id or ''}".lower()


_CALIBRATION_CACHE: dict[str, dict] | None = None


def _read_calibration() -> dict:
    try:
        data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = data.get("models") if isinstance(data, dict) else None
    return entries if isinstance(entries, dict) else {}


def _calibration_entries() -> dict:
    global _CALIBRATION_CACHE
    if _CALIBRATION_CACHE is None:
        _CALIBRATION_CACHE = _read_calibration()
    return _CALIBRATION_CACHE


def _write_calibration(entries: dict) -> None:
    try:
        CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = CALIBRATION_FILE.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "models": entries}, indent=2), encoding="utf-8"
        )
        os.replace(temporary, CALIBRATION_FILE)
    except OSError:
        # Calibration is an optimisation: a read-only .agent/ must not break a turn.
        pass


def calibrated_chars_per_token(
    model_id: str | None, provider: str | None = None
) -> float | None:
    """Learned chars-per-token for a model, or None when it has no samples."""
    if not model_id:
        return None
    entry = _calibration_entries().get(_model_key(model_id, provider))
    if not isinstance(entry, dict):
        return None
    try:
        value = float(entry.get("chars_per_token"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return min(max(value, MIN_CHARS_PER_TOKEN), MAX_CHARS_PER_TOKEN)


def record_calibration(
    model_id: str | None,
    provider: str | None,
    characters: int,
    reported_input_tokens: int | None,
) -> float | None:
    """Learns a model's real chars/token ratio from the provider's own count.

    Only the approximate tokenizer needs this; an exact one already agrees with
    the provider. The observation is clamped and smoothed, then persisted so a
    later process starts from the learned value instead of chars/4.
    """
    if not config.TOKENIZER_CALIBRATION:
        return None
    if not model_id or characters <= 0:
        return None
    if not reported_input_tokens or reported_input_tokens <= 0:
        return None

    observed = characters / float(reported_input_tokens)
    if not math.isfinite(observed) or observed <= 0:
        return None
    observed = min(max(observed, MIN_CHARS_PER_TOKEN), MAX_CHARS_PER_TOKEN)

    key = _model_key(model_id, provider)
    entries = _calibration_entries()
    entry = entries.get(key)
    previous = calibrated_chars_per_token(model_id, provider)
    samples = 0
    if isinstance(entry, dict):
        try:
            samples = int(entry.get("samples", 0))
        except (TypeError, ValueError):
            samples = 0

    value = (
        observed
        if previous is None
        else previous * (1 - CALIBRATION_SMOOTHING) + observed * CALIBRATION_SMOOTHING
    )
    value = min(max(value, MIN_CHARS_PER_TOKEN), MAX_CHARS_PER_TOKEN)
    entries[key] = {"chars_per_token": round(value, 4), "samples": samples + 1}
    _write_calibration(entries)

    for cache_key in [cached for cached in _CACHE if cached[0] == key]:
        _CACHE.pop(cache_key, None)
    return value


def _explicit_spec(model_id: str | None, provider: str | None) -> str | None:
    if model_id:
        variable = f"TOKENIZER_{_sanitize(model_id)}"
        value = os.getenv(variable)
        if value and value.strip():
            return value.strip()

        mapping = os.getenv("TOKENIZER_MAP")
        if mapping:
            try:
                patterns = json.loads(mapping)
            except ValueError:
                patterns = None
            if isinstance(patterns, dict):
                for pattern, spec in patterns.items():
                    if not isinstance(pattern, str) or not isinstance(spec, str):
                        continue
                    if fnmatch.fnmatchcase(model_id, pattern):
                        return spec.strip()

    value = os.getenv("TOKENIZER_DEFAULT")
    return value.strip() if value and value.strip() else None


def _get_encoding(name: str):
    """Loads a tiktoken encoding; separated so tests can patch it."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return None


def _load_optional_hf(spec: str) -> Tokenizer | None:
    """Loads a HuggingFace tokenizer when the optional package is available."""
    try:
        from tokenizers import Tokenizer as HFTokenizer  # type: ignore
    except ImportError:
        return None

    try:
        if spec.startswith("file:"):
            backend = HFTokenizer.from_file(spec[len("file:"):])
        else:
            backend = HFTokenizer.from_pretrained(spec[len("hf:"):])
        return CallableTokenizer(
            name=spec,
            counter=lambda text: len(backend.encode(text, add_special_tokens=False).ids),
            exact=True,
        )
    except Exception:
        return None


def _build_tokenizer(spec: str) -> Tokenizer:
    cleaned = spec.strip()
    lowered = cleaned.lower()

    if lowered.startswith("approx"):
        _, _, size = lowered.partition(":")
        try:
            chars = max(1, int(size)) if size else DEFAULT_CHARS_PER_TOKEN
        except ValueError:
            chars = DEFAULT_CHARS_PER_TOKEN
        return ApproximateTokenizer(chars_per_token=chars)

    if lowered.startswith("hf:") or lowered.startswith("file:"):
        optional = _load_optional_hf(cleaned)
        if optional is not None:
            return optional
        return ApproximateTokenizer(name="approx (optional tokenizer unavailable)")

    encoding_name = cleaned[len("tiktoken:"):] if lowered.startswith("tiktoken:") else cleaned
    encoding = _get_encoding(encoding_name)
    if encoding is not None:
        return TiktokenTokenizer(name=f"tiktoken:{encoding_name}", encoding_name=encoding_name, encoding=encoding)
    return ApproximateTokenizer(name="approx (unknown tokenizer)")


def _default_tiktoken_name(model_id: str | None) -> str | None:
    if not model_id:
        return None
    lowered = model_id.lower()
    for pattern, encoding in _OPENAI_ENCODINGS:
        if re.search(pattern, lowered):
            return encoding
    return None


def resolve_tokenizer(model_id: str | None = None, provider: str | None = None) -> Tokenizer:
    """Resolves the best available tokenizer for ``model_id``.

    Resolution order: explicit per-model env, ``TOKENIZER_MAP``,
    ``TOKENIZER_DEFAULT``, a known OpenAI family, then the approximate fallback.
    """
    explicit = _explicit_spec(model_id, provider)
    calibrated = calibrated_chars_per_token(model_id, provider)
    cache_key = (
        _model_key(model_id, provider),
        explicit or "",
        round(calibrated or 0.0, 4),
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    if explicit:
        tokenizer = _build_tokenizer(explicit)
    else:
        encoding_name = _default_tiktoken_name(model_id)
        encoding = _get_encoding(encoding_name) if encoding_name else None
        if encoding is not None and encoding_name is not None:
            tokenizer = TiktokenTokenizer(
                name=f"tiktoken:{encoding_name}", encoding_name=encoding_name, encoding=encoding
            )
        else:
            tokenizer = ApproximateTokenizer()

    # A fallback is re-based on the ratio learned from this model's reported
    # usage; an exact tokenizer already matches the provider and is left alone.
    if calibrated is not None and isinstance(tokenizer, ApproximateTokenizer):
        tokenizer = ApproximateTokenizer(
            name=f"approx (calibrated {calibrated:.2f} chars/token)",
            chars_per_token=calibrated,
        )

    _CACHE[cache_key] = tokenizer
    return tokenizer


def clear_cache() -> None:
    """Drops resolved tokenizers; useful after changing tokenizer env vars."""
    global _CALIBRATION_CACHE
    _CACHE.clear()
    _CALIBRATION_CACHE = None


def _serialize(value: Any) -> tuple[str, bool]:
    if isinstance(value, str):
        return value, True
    return json.dumps(value, default=str, ensure_ascii=False), False


def request_characters(payload: Any, tool_schemas: Any = None) -> int:
    """Characters in the serialised request, the basis of every local estimate."""
    characters = len(_serialize(payload)[0])
    if tool_schemas:
        characters += len(_serialize(tool_schemas)[0])
    return characters


def estimate_value(value: Any, model_id: str | None = None, provider: str | None = None) -> TokenEstimate:
    """Estimates a string or a JSON-serialisable payload.

    A string counted with an exact tokenizer is exact. A structured payload is
    always marked approximate because chat framing, message separators and tool
    wrappers are not represented by the raw tokenizer.
    """
    text, is_text = _serialize(value)
    tokenizer = resolve_tokenizer(model_id, provider)
    tokens = tokenizer.count(text)
    exact = bool(is_text and tokenizer.exact)
    note = "" if exact else ("chat framing estimated" if tokenizer.exact else "character-based fallback")
    return TokenEstimate(tokens=tokens, exact=exact, tokenizer=tokenizer.name, note=note)


def estimate_tokens(value: Any, model_id: str | None = None, provider: str | None = None) -> int:
    """Compatibility helper returning only the number of tokens."""
    return estimate_value(value, model_id, provider).tokens
