"""
Normalised provider-reported token usage.

Providers describe the same quantities with different names. OpenAI-compatible
APIs use ``prompt_tokens`` / ``completion_tokens``; some gateways use
``input_tokens`` / ``output_tokens``; the details live in a nested object whose
shape also varies. This module turns any of those into one immutable
``TokenUsage`` so the rest of the harness never reads provider-specific fields.

The distinction that matters most:

* ``None`` means "the provider did not report this".
* ``0`` means "the provider reported zero".

Nothing here invents a value. If a provider returns incomplete metadata, the
missing fields stay ``None`` and the caller can fall back to a local estimate,
which is always labelled as an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _get(source: Any, key: str, default=None):
    """Reads ``key`` from a mapping or an SDK object, uniformly."""
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _as_int(value: Any) -> int | None:
    """Returns an int for a plausible token count, otherwise ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return None
    return max(number, 0)


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported input/output/total/cached/reasoning tokens."""

    input: int | None = None
    output: int | None = None
    total: int | None = None
    cached: int | None = None
    reasoning: int | None = None

    @property
    def reported(self) -> bool:
        """True when the provider supplied at least one usage field."""
        return any(
            value is not None
            for value in (self.input, self.output, self.total, self.cached, self.reasoning)
        )

    @property
    def effective_total(self) -> int | None:
        """The best total the provider gave us, deriving it only when safe.

        Deriving ``input + output`` is safe when both are present. A lone input
        count is not a total: treating the missing output as zero would undercount
        the request.
        """
        if self.total is not None:
            return self.total
        if self.input is not None and self.output is not None:
            return self.input + self.output
        return None

    @property
    def billable_total(self) -> int | None:
        """Total minus cached tokens, which providers usually exclude from limits."""
        total = self.effective_total
        if total is None:
            return None
        return max(total - (self.cached or 0), 0)

    def describe(self) -> str:
        if not self.reported:
            return "not reported"
        parts = []
        for label, value in (
            ("input", self.input),
            ("output", self.output),
            ("total", self.effective_total),
            ("cached", self.cached),
            ("reasoning", self.reasoning),
        ):
            if value is not None:
                parts.append(f"{label}={value}")
        return " ".join(parts)

    def as_dict(self) -> dict[str, int]:
        """Only the fields the provider actually reported."""
        data: dict[str, int] = {}
        for key, value in (
            ("input", self.input),
            ("output", self.output),
            ("total", self.effective_total),
            ("cached", self.cached),
            ("reasoning", self.reasoning),
        ):
            if value is not None:
                data[key] = value
        return data

    @classmethod
    def from_mapping(cls, data: Any) -> "TokenUsage":
        """Normalises a usage mapping or object from any compatible provider."""
        if data is None:
            return cls()

        prompt_details = _get(data, "prompt_tokens_details") or _get(
            data, "input_tokens_details"
        )
        completion_details = _get(data, "completion_tokens_details") or _get(
            data, "output_tokens_details"
        )

        input_tokens = _as_int(_get(data, "prompt_tokens"))
        if input_tokens is None:
            input_tokens = _as_int(_get(data, "input_tokens"))

        output_tokens = _as_int(_get(data, "completion_tokens"))
        if output_tokens is None:
            output_tokens = _as_int(_get(data, "output_tokens"))

        total_tokens = _as_int(_get(data, "total_tokens"))

        cached_tokens = _as_int(_get(prompt_details, "cached_tokens"))
        if cached_tokens is None:
            cached_tokens = _as_int(_get(data, "cached_tokens"))

        reasoning_tokens = _as_int(_get(completion_details, "reasoning_tokens"))
        if reasoning_tokens is None:
            reasoning_tokens = _as_int(_get(data, "reasoning_tokens"))

        return cls(
            input=input_tokens,
            output=output_tokens,
            total=total_tokens,
            cached=cached_tokens,
            reasoning=reasoning_tokens,
        )

    @classmethod
    def from_response(cls, response: Any) -> "TokenUsage":
        """Reads ``response.usage`` without assuming a particular SDK object."""
        return cls.from_mapping(_get(response, "usage"))
