"""
Per-provider/model daily usage accounting and the projected request guard.

Providers such as Groq enforce requests per day (RPD) and tokens per day (TPD) on
top of the per-minute token bucket. Unlike TPM, those cannot be paced away: the
reset is hours, so the harness counts its own usage and refuses a request it knows
will be rejected.

Two kinds of token data are stored separately:

* **reported** - what the provider returned in ``response.usage``;
* **estimated** - the local estimate, used only when the provider returned no
  usable total.

Reported usage is what history is built from. Estimates are never presented as
exact and never double-count a reported response. Counters live in
``.agent/usage.json`` keyed by UTC date and ``provider:model``. They are a local
guard rail, not a billing system.

The guard is *projective*: it adds the request about to be sent (input estimate
plus reserved output) to today's totals before comparing them with the limit.
Checking only past totals would let the last request that fits slip over.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from src import config, models, providers
from src.config import USAGE_FILE
from src.usage_report import TokenUsage


@dataclass(frozen=True)
class UsageTotals:
    """Today's counters, separating exact reported values from estimates."""

    reported_requests: int = 0
    estimated_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_tokens: int = 0

    @property
    def requests(self) -> int:
        return self.reported_requests + self.estimated_requests

    @property
    def tokens(self) -> int:
        return self.total_tokens + self.estimated_tokens

    @property
    def billable_tokens(self) -> int:
        """Tokens counted against a rate limit (cached tokens are excluded)."""
        return max(self.tokens - self.cached_tokens, 0)


def _today() -> str:
    """Groq resets at midnight UTC; matching that avoids off-by-a-day counters."""
    return time.strftime("%Y-%m-%d", time.gmtime())


def _key(model_id: str, provider: str | None = None) -> str:
    name = providers.normalize(provider or config.LLM_PROVIDER)
    return f"{name}:{model_id}"


def _read() -> dict:
    """Counters for today only; anything else is treated as empty."""
    try:
        data = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("date") != _today():
        return {}
    entries = data.get("models")
    return entries if isinstance(entries, dict) else {}


def _write(entries: dict) -> None:
    try:
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = USAGE_FILE.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"date": _today(), "models": entries}, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, USAGE_FILE)
    except OSError:
        # Accounting is best-effort: a read-only .agent/ must not break the turn.
        pass


def _int(entry: dict, key: str, default: int = 0) -> int:
    try:
        return int(entry.get(key, default))
    except (TypeError, ValueError):
        return default


def _totals_from_entry(entry: dict) -> UsageTotals:
    """Reads both the current schema and the older ``requests``/``tokens`` one."""
    input_tokens = _int(entry, "input_tokens")
    output_tokens = _int(entry, "output_tokens")
    total_tokens = _int(entry, "total_tokens", -1)
    if total_tokens < 0:
        # Versions before reported/estimated separation stored one total.
        total_tokens = _int(entry, "tokens", input_tokens + output_tokens)

    reported_requests = _int(entry, "reported_requests", _int(entry, "requests"))
    return UsageTotals(
        reported_requests=reported_requests,
        estimated_requests=_int(entry, "estimated_requests"),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=_int(entry, "cached_tokens"),
        reasoning_tokens=_int(entry, "reasoning_tokens"),
        estimated_tokens=_int(entry, "estimated_tokens"),
    )


def totals(model_id: str, provider: str | None = None) -> UsageTotals:
    """Today's usage for one provider/model pair."""
    entry = _read().get(_key(model_id, provider)) or {}
    return _totals_from_entry(entry if isinstance(entry, dict) else {})


def used(model_id: str, provider: str | None = None) -> tuple[int, int]:
    """(requests, tokens) today. Kept for callers that only need the summary."""
    summary = totals(model_id, provider)
    return summary.requests, summary.tokens


def guard(
    model_id: str,
    provider: str | None = None,
    projected_tokens: int = 0,
    projected_requests: int = 1,
) -> str | None:
    """The reason the next request would exceed a local limit, or ``None``.

    ``projected_tokens`` should include the estimated input *and* the maximum
    output the request may generate, so the check covers the request being sent
    rather than only the requests already recorded.
    """
    budget = models.request_budget(model_id, provider)
    limits = budget.rate_limits
    if not limits.any:
        return None  # no published limits to enforce

    summary = totals(model_id, provider)
    future_requests = summary.requests + max(projected_requests, 0)
    # Cached prompt tokens are excluded from the provider's daily token bucket,
    # so the guard compares billable (uncached) usage - the same number /model
    # reports. Counting cached tokens here would refuse requests the provider
    # would have accepted.
    future_tokens = summary.billable_tokens + max(projected_tokens, 0)

    if limits.rpd is not None and future_requests > limits.rpd:
        return (
            f"the daily request limit for {model_id} would be exceeded "
            f"({summary.requests:,} used + {max(projected_requests, 0)} projected "
            f"> {limits.rpd:,}). It resets at midnight UTC."
        )
    if limits.tpd is not None and future_tokens > limits.tpd:
        return (
            f"the daily token limit for {model_id} would be exceeded "
            f"({summary.billable_tokens:,} billable used + "
            f"{max(projected_tokens, 0)} projected > {limits.tpd:,}). "
            "It resets at midnight UTC."
        )
    return None


def record(
    model_id: str,
    token_usage: TokenUsage | None = None,
    provider: str | None = None,
    estimated_tokens: int = 0,
) -> None:
    """Adds one request to today's counters.

    Reported values are accumulated as reported. A request without a usable
    reported total contributes its local estimate instead, in a separate field,
    so the two are never confused or double-counted.
    """
    token_usage = token_usage or TokenUsage()
    entries = _read()
    key = _key(model_id, provider)
    entry = entries.get(key) or {}
    if not isinstance(entry, dict):
        entry = {}
    summary = _totals_from_entry(entry)

    input_tokens = summary.input_tokens + (token_usage.input or 0)
    output_tokens = summary.output_tokens + (token_usage.output or 0)
    cached_tokens = summary.cached_tokens + (token_usage.cached or 0)
    reasoning_tokens = summary.reasoning_tokens + (token_usage.reasoning or 0)

    reported_total = token_usage.effective_total
    if reported_total is not None:
        reported_requests = summary.reported_requests + 1
        estimated_requests = summary.estimated_requests
        total_tokens = summary.total_tokens + reported_total
        estimated = summary.estimated_tokens
    else:
        reported_requests = summary.reported_requests
        estimated_requests = summary.estimated_requests + 1
        total_tokens = summary.total_tokens
        estimated = summary.estimated_tokens + max(int(estimated_tokens or 0), 0)

    entry.update(
        {
            "reported_requests": reported_requests,
            "estimated_requests": estimated_requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "estimated_tokens": estimated,
        }
    )
    # Mirror the summary for older readers. New readers prefer the fields above.
    entry["requests"] = reported_requests + estimated_requests
    entry["tokens"] = total_tokens + estimated
    entries[key] = entry
    _write(entries)


def describe(model_id: str, provider: str | None = None) -> str:
    """One line for /model: how much of today's allowance is gone."""
    limits = models.request_budget(model_id, provider).rate_limits
    summary = totals(model_id, provider)

    if not limits.any:
        return (
            f"today: {summary.requests:,} requests "
            f"({summary.reported_requests:,} reported, "
            f"{summary.estimated_requests:,} estimated), "
            f"{summary.tokens:,} tokens "
            f"({summary.total_tokens:,} reported + "
            f"{summary.estimated_tokens:,} estimated; no published limits)"
        )

    parts = []
    if limits.rpd is not None:
        parts.append(f"{summary.requests:,}/{limits.rpd:,} requests")
    if limits.tpd is not None:
        parts.append(f"{summary.billable_tokens:,}/{limits.tpd:,} tokens")
    if limits.tpm is not None:
        parts.append(f"{limits.tpm:,} TPM")
    detail = " | ".join(parts) if parts else "no daily limits"
    return f"today: {detail}"
