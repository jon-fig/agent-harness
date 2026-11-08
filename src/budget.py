"""
Input-token budget for the agent loop.

Groq's free tier rejects a request whose input does not fit the per-minute token
bucket (HTTP 413) and throttles the requests that follow (429). Because the whole
conversation is resent on every iteration, keeping each individual message small
is not enough: old tool exchanges have to collapse, or the turn grows
quadratically and trips the limit three or four iterations in.

This module owns the estimate, the compaction and the pre-flight fit. The agent
loop only asks it for a payload.
"""

from __future__ import annotations

import json
from typing import Any

from src.config import HISTORY_KEEP_EXCHANGES, MAX_INPUT_TOKENS

# tiktoken gives exact counts for OpenAI-compatible tokenizers (gpt-oss uses
# o200k_harmony). It is optional: without it, or offline on first use (the BPE
# files are downloaded lazily), we fall back to a pessimistic char estimate.
try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

CHARS_PER_TOKEN = 4

# A collapsed old call keeps this much of each long string argument (file
# contents, code, patterns) so the model still knows what it did.
ARG_HEAD_CHARS = 160

# Upper bound for an old message's text (assistant reasoning or tool error).
SUMMARY_CHARS = 240


def _load_encoding():
    if tiktoken is None:
        return None
    for name in ("o200k_harmony", "o200k_base", "cl100k_base"):
        try:
            return tiktoken.get_encoding(name)
        except Exception:
            continue
    return None


_ENCODING = _load_encoding()


def estimate_tokens(value: Any) -> int:
    """Token count for a string or a JSON-serialisable object.

    Exact when tiktoken is available, otherwise chars/4. A slightly pessimistic
    estimate is what a rate-limit guard wants, so the fallback rounds up.
    """
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text, disallowed_special=()))
        except Exception:
            pass
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _as_dict(message: Any) -> dict:
    """Normalises a message to a dict, including SDK objects that slipped through."""
    if isinstance(message, dict):
        return message

    dump = getattr(message, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except TypeError:
            return dump()

    return {"role": getattr(message, "role", "assistant"), "content": str(message)}


def _tool_calls(message: dict) -> list[dict]:
    """Reads a message's tool calls in a uniform dict shape."""
    calls = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            call = _as_dict(call)
        function = call.get("function")
        if not isinstance(function, dict):
            function = _as_dict(function)
        calls.append(
            {
                "id": call.get("id", ""),
                "type": "function",
                "function": {
                    "name": function.get("name", "tool"),
                    "arguments": function.get("arguments", "{}"),
                },
            }
        )
    return calls


def _collapse_arguments(raw: Any) -> str:
    """Keeps the head of long string arguments (file bodies, code) only."""
    if not isinstance(raw, str):
        raw = json.dumps(raw or {}, default=str)

    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return json.dumps({"note": "arguments collapsed"}, ensure_ascii=False)

    if not isinstance(parsed, dict):
        return json.dumps({"note": "arguments collapsed"}, ensure_ascii=False)

    collapsed = {}
    for key, value in parsed.items():
        if isinstance(value, str) and len(value) > ARG_HEAD_CHARS:
            omitted = len(value) - ARG_HEAD_CHARS
            collapsed[key] = f"{value[:ARG_HEAD_CHARS]}…[{omitted} chars collapsed]"
        else:
            collapsed[key] = value
    return json.dumps(collapsed, ensure_ascii=False, default=str)


def compact_history(messages: list, keep_exchanges: int = HISTORY_KEEP_EXCHANGES) -> list[dict]:
    """Collapses every tool exchange except the newest ``keep_exchanges`` ones.

    An exchange is the assistant message carrying tool_calls plus the tool
    messages answering it. The newest exchanges stay verbatim because the model is
    still working from them; older ones become the head of the call plus an
    outcome one-liner. The full text stays on disk, which is where the agent
    should be reading it from anyway.
    """
    copied = [dict(_as_dict(message)) for message in messages]
    keep = max(0, keep_exchanges)
    starts = [
        index
        for index, message in enumerate(copied)
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    stale = starts[: max(0, len(starts) - keep)]
    if not stale:
        return copied

    names: dict[str, str] = {}
    for index in stale:
        message = copied[index]
        calls = _tool_calls(message)
        for call in calls:
            names[call["id"]] = call["function"]["name"]

        message["tool_calls"] = [
            {
                **call,
                "function": {
                    "name": call["function"]["name"],
                    "arguments": _collapse_arguments(call["function"]["arguments"]),
                },
            }
            for call in calls
        ]

        content = str(message.get("content") or "").strip().replace("\n", " ")
        message["content"] = content[:SUMMARY_CHARS] + ("…" if len(content) > SUMMARY_CHARS else "")

    for message in copied:
        if message.get("role") != "tool":
            continue
        name = names.get(message.get("tool_call_id", ""))
        if name is None:
            continue

        text = str(message.get("content") or "")
        if text.startswith("Error"):
            # Keep the failure text: it is what the model needs to self-correct.
            message["content"] = text[:SUMMARY_CHARS] + ("…" if len(text) > SUMMARY_CHARS else "")
        else:
            message["content"] = f"{name}(...) -> ok ({len(text) / 1024:.1f} KB omitted)"

    return copied


def build_payload(
    system_prompt: str,
    messages: list,
    tool_schemas: list,
    budget: int = MAX_INPUT_TOKENS,
    keep_options: tuple[int, ...] = (HISTORY_KEEP_EXCHANGES, 1, 0),
    include_tools: bool = True,
) -> tuple[list[dict], int, bool]:
    """Builds the request messages, compacting until the input fits ``budget``.

    Returns ``(payload, estimated_input_tokens, fits)``. When nothing fits, the
    smallest attempt comes back with ``fits=False`` so the caller can refuse
    instead of sending a request the provider will reject.

    ``include_tools=False`` prices the request without the tool block, which is
    the cheapest way to still get an answer when the conversation alone is close
    to the limit.
    """
    schema_tokens = estimate_tokens(tool_schemas) if include_tools else 0
    payload: list[dict] = []
    estimated = 0

    for keep in keep_options:
        payload = [{"role": "system", "content": system_prompt}] + compact_history(messages, keep)
        estimated = estimate_tokens(payload) + schema_tokens
        if estimated <= budget:
            return payload, estimated, True

    return payload, estimated, False
