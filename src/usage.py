"""
Per-model daily usage accounting.

Groq's free tier enforces requests per day (RPD) and tokens per day (TPD) on top
of the per-minute token bucket. Unlike TPM, those cannot be paced away: the reset
is hours, so the harness counts its own usage and refuses a request it knows will
be rejected.

Counters live in ``.agent/usage.json`` keyed by UTC date, so a new day starts from
zero without any cleanup. They are local estimates (what this process sent, plus
the provider's reported totals when a response succeeds); a second machine or an
older version sharing the same key is invisible to them. They are a guard rail,
not a billing system.

This module intentionally does not import ``src.agent``: the loop asks it two
questions (may I send this? / record what it cost) and nothing else.
"""

from __future__ import annotations

import json
import os
import time

from src import models
from src.config import USAGE_FILE


def _today() -> str:
    """Groq resets at midnight UTC; matching that avoids off-by-a-day counters."""
    return time.strftime("%Y-%m-%d", time.gmtime())


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


def used(model_id: str) -> tuple[int, int]:
    """(requests, tokens) this machine has spent on ``model_id`` today."""
    entry = _read().get(model_id) or {}
    try:
        return int(entry.get("requests", 0)), int(entry.get("tokens", 0))
    except (TypeError, ValueError):
        return 0, 0


def guard(model_id: str) -> str | None:
    """The reason this model cannot be called again today, or None.

    Checked before building a request, so the failure arrives as a clear stop
    instead of a 429 that the retry loop would wait minutes for.
    """
    spec = models.spec_for(model_id)
    if spec is None:  # unknown model: no published limits to enforce
        return None

    requests, tokens = used(model_id)
    if requests >= spec.rpd:
        return (
            f"the daily request limit for {model_id} is spent "
            f"({requests:,}/{spec.rpd:,} requests). It resets at midnight UTC."
        )
    if tokens >= spec.tpd:
        return (
            f"the daily token limit for {model_id} is spent "
            f"({tokens:,}/{spec.tpd:,} tokens). It resets at midnight UTC."
        )
    return None


def record(model_id: str, prompt_tokens, completion_tokens) -> None:
    """Adds one request and its reported tokens to today's counters."""
    try:
        total = int(prompt_tokens or 0) + int(completion_tokens or 0)
    except (TypeError, ValueError):
        total = 0

    entries = _read()
    entry = entries.get(model_id) or {}
    try:
        requests = int(entry.get("requests", 0))
        tokens = int(entry.get("tokens", 0))
    except (TypeError, ValueError):
        requests = tokens = 0

    entry["requests"] = requests + 1
    entry["tokens"] = tokens + max(total, 0)
    entries[model_id] = entry
    _write(entries)


def describe(model_id: str) -> str:
    """One line for /model: how much of today's allowance is gone."""
    spec = models.spec_for(model_id)
    requests, tokens = used(model_id)
    if spec is None:
        return f"today: {requests:,} requests, {tokens:,} tokens (no published limits)"
    return f"today: {requests:,}/{spec.rpd:,} requests, {tokens:,}/{spec.tpd:,} tokens"
