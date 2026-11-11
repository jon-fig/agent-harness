"""
Request measurement, history compaction and automatic context compaction.

Three responsibilities live in this module, deliberately without enforcing a
daily quota:

* **estimation** - delegated to ``src.tokenization`` so it is model-aware;
* **measurement** - prices the *complete* request (system prompt + tool schemas +
  messages) before it is sent;
* **compaction** - collapses older tool exchanges, either to fit the hard budget
  or proactively when the configured threshold is reached.

The agent loop decides whether a measured request may be sent; ``src.usage``
decides whether the account's daily allowance can absorb it. Keeping those apart
means an estimate is never mistaken for provider-reported usage, and a local
budget cannot silently override a provider limit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src import config, models, tokenization
from src.config import HISTORY_KEEP_EXCHANGES, MAX_INPUT_TOKENS

# A collapsed old call keeps this much of each long string argument (file
# contents, code, patterns) so the model still knows what it did.
ARG_HEAD_CHARS = 160

# Upper bound for an old message's text (assistant reasoning or tool error).
SUMMARY_CHARS = 240

# An old tool result summarised at the ``summary`` level keeps its first line
# (a DataFrame header, a row count, a saved path) up to this many characters.
RESULT_FIRST_LINE_CHARS = 120

# How much of an old tool result survives compaction. ``summary`` keeps the first
# line of the output, ``terse`` keeps only the outcome and the original size. The
# aggressive level is used only when the cheap one cannot reach the target.
DETAIL_SUMMARY = "summary"
DETAIL_TERSE = "terse"
DETAIL_LEVELS = (DETAIL_SUMMARY, DETAIL_TERSE)

# A compaction that saves fewer tokens than this is noise: retrying it on every
# iteration would not make the next request meaningfully cheaper.
MIN_REDUCTION_TOKENS = 4


@dataclass(frozen=True)
class RequestMeasurement:
    """The payload to send and the estimate attached to it."""

    payload: list[dict]
    tokens: int
    estimate: tokenization.TokenEstimate
    fits: bool
    includes_tools: bool
    limit: int


@dataclass(frozen=True)
class CompactionResult:
    """Outcome of one automatic-compaction decision."""

    messages: list[dict]
    changed: bool
    eligible: bool
    turns: int
    before_tokens: int
    after_tokens: int
    threshold_tokens: float
    target_tokens: float
    before_estimate: tokenization.TokenEstimate
    after_estimate: tokenization.TokenEstimate
    reason: str = ""
    # The smallest request this policy can produce. Everything above it is
    # protected (system prompt, tool schemas, user messages, assistant prose, the
    # newest exchange), which is why a target may be unreachable.
    floor_tokens: int = 0
    reached_target: bool = False

    @property
    def reduction(self) -> int:
        return max(self.before_tokens - self.after_tokens, 0)

    @property
    def percent_reduction(self) -> float:
        if self.before_tokens <= 0:
            return 0.0
        return 100.0 * self.reduction / self.before_tokens


def estimate_tokens(
    value: Any,
    model_id: str | None = None,
    provider: str | None = None,
) -> int:
    """Token count for a string or a JSON-serialisable object."""
    return tokenization.estimate_tokens(value, model_id, provider)


def estimate_value(
    value: Any,
    model_id: str | None = None,
    provider: str | None = None,
) -> tokenization.TokenEstimate:
    """Token count plus its provenance (exact tokenizer vs. fallback)."""
    return tokenization.estimate_value(value, model_id, provider)


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


def _is_collapsed(text: str, name: str) -> bool:
    """True when this tool message already *is* a compaction summary.

    Without this check every later compaction would rewrite "ok (3.1 KB)" as
    "ok (0.0 KB omitted)", losing information while saving nothing.
    """
    return text.startswith(f"{name}(...) ->")


def _summarize_outcome(name: str, text: str, detail: str) -> str:
    """One line that keeps the fact of the call and a little of its result.

    Errors are kept (truncated rather than summarised) because the model may still
    have to correct them. For successful output the ``summary`` level keeps the
    first non-empty line — often a header row, a row count or a saved path — while
    ``terse`` drops it. Both keep the original size so the model can tell that it
    is looking at an excerpt and re-read the file instead of guessing.
    """
    stripped = text.strip()
    if stripped.startswith("Error") or "Traceback (most recent call last)" in stripped[:400]:
        return stripped[:SUMMARY_CHARS] + ("…" if len(stripped) > SUMMARY_CHARS else "")
    if not stripped:
        return f"{name}(...) -> ok (no output)"

    size = len(text) / 1024
    if detail == DETAIL_SUMMARY:
        first = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
        if first:
            return f"{name}(...) -> ok ({size:.1f} KB); first line: {first[:RESULT_FIRST_LINE_CHARS]}"
    return f"{name}(...) -> ok ({size:.1f} KB omitted)"


def compact_history(
    messages: list,
    keep_exchanges: int = HISTORY_KEEP_EXCHANGES,
    detail: str = DETAIL_TERSE,
) -> list[dict]:
    """Collapses every tool exchange except the newest ``keep_exchanges`` ones.

    An exchange is the assistant message carrying tool_calls plus the tool
    messages answering it. The newest exchanges stay verbatim because the model is
    still working from them; older ones become the head of the call plus an
    outcome summary (see ``_summarize_outcome``). The full text stays on disk,
    which is where the agent should be reading it from anyway.

    User messages and assistant prose are never rewritten, so the requirements
    that frame the conversation survive every compaction. Already-collapsed
    messages are left untouched, which makes repeated compaction idempotent.
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
        if _is_collapsed(text, name):
            continue
        message["content"] = _summarize_outcome(name, text, detail)

    return copied


def _budget_limit(budget: int | models.RequestBudget) -> int:
    if isinstance(budget, models.RequestBudget):
        return budget.max_input_tokens
    try:
        return int(budget)
    except (TypeError, ValueError):
        return MAX_INPUT_TOKENS


def _measure(
    system_prompt: str,
    body: list[dict],
    tool_schemas: list,
    limit: int,
    model_id: str | None,
    provider: str | None,
    include_tools: bool,
) -> RequestMeasurement:
    payload = [{"role": "system", "content": system_prompt}] + body
    payload_estimate = tokenization.estimate_value(payload, model_id, provider)

    schema_tokens = 0
    if include_tools:
        schema_tokens = tokenization.estimate_value(
            tool_schemas, model_id, provider
        ).tokens

    tokens = payload_estimate.tokens + schema_tokens
    note = "chat framing estimated"
    if include_tools:
        note = "chat framing and tool schemas estimated"
    combined = tokenization.TokenEstimate(
        tokens=tokens,
        exact=False,
        tokenizer=payload_estimate.tokenizer,
        note=note,
    )
    return RequestMeasurement(
        payload=payload,
        tokens=tokens,
        estimate=combined,
        fits=tokens <= limit,
        includes_tools=include_tools,
        limit=limit,
    )


def measure_request(
    system_prompt: str,
    messages: list,
    tool_schemas: list,
    budget: int | models.RequestBudget = MAX_INPUT_TOKENS,
    keep_exchanges: int = HISTORY_KEEP_EXCHANGES,
    include_tools: bool = True,
    model_id: str | None = None,
    provider: str | None = None,
    compact: bool = True,
) -> RequestMeasurement:
    """Prices the complete request, optionally compacting history first.

    ``compact=False`` measures the raw conversation and is what the automatic
    compactor uses to decide whether its threshold was reached; ``compact=True``
    measures what would actually be sent.
    """
    limit = _budget_limit(budget)
    if compact:
        body = compact_history(messages, keep_exchanges)
    else:
        body = [dict(_as_dict(message)) for message in messages]
    return _measure(system_prompt, body, tool_schemas, limit, model_id, provider, include_tools)


def build_payload(
    system_prompt: str,
    messages: list,
    tool_schemas: list,
    budget: int | models.RequestBudget = MAX_INPUT_TOKENS,
    keep_options: tuple[int, ...] = (HISTORY_KEEP_EXCHANGES, 1, 0),
    include_tools: bool = True,
    model_id: str | None = None,
    provider: str | None = None,
) -> tuple[list[dict], int, bool]:
    """Builds request messages, compacting until the input fits ``budget``.

    Returns ``(payload, estimated_input_tokens, fits)``. When nothing fits, the
    smallest attempt comes back with ``fits=False`` so the caller can refuse
    instead of sending a request the provider will reject.

    ``include_tools=False`` prices the request without the tool block, which is
    the cheapest way to still get an answer when the conversation alone is close
    to the limit.
    """
    limit = _budget_limit(budget)
    last: RequestMeasurement | None = None

    for keep in keep_options:
        measurement = measure_request(
            system_prompt,
            messages,
            tool_schemas,
            limit,
            keep_exchanges=keep,
            include_tools=include_tools,
            model_id=model_id,
            provider=provider,
            compact=True,
        )
        last = measurement
        if measurement.fits:
            return measurement.payload, measurement.tokens, True

    if last is None:  # keep_options was empty
        last = measure_request(
            system_prompt,
            messages,
            tool_schemas,
            limit,
            include_tools=include_tools,
            model_id=model_id,
            provider=provider,
            compact=False,
        )
    return last.payload, last.tokens, False


def count_turns(messages: list) -> int:
    """User turns in the session, including the prompt for the current turn."""
    return sum(1 for message in messages if _as_dict(message).get("role") == "user")


def resolve_threshold(value: float | int | None, limit: int, default: float) -> float:
    """Resolves a fraction (``<= 1``) or an absolute token count."""
    try:
        number = float(value) if value is not None else float(default)
    except (TypeError, ValueError):
        number = float(default)
    if number <= 1:
        return max(number, 0.0) * limit
    return max(number, 0.0)


def _auto_keep_options() -> tuple[int, ...]:
    """Never collapse the newest exchange during proactive compaction.

    Hard fitting (``build_payload``) may still go to zero as a last resort, but
    automatic compaction keeps at least one exchange verbatim so the model
    retains the most recent context.
    """
    first = max(1, HISTORY_KEEP_EXCHANGES)
    return (first,) if first == 1 else (first, 1)


def _unchanged(
    messages: list,
    eligible: bool,
    turns: int,
    measurement: RequestMeasurement,
    threshold: float,
    target: float,
    reason: str,
    floor: int = 0,
) -> CompactionResult:
    body = [dict(_as_dict(message)) for message in messages]
    return CompactionResult(
        messages=body,
        changed=False,
        eligible=eligible,
        turns=turns,
        before_tokens=measurement.tokens,
        after_tokens=measurement.tokens,
        threshold_tokens=threshold,
        target_tokens=target,
        before_estimate=measurement.estimate,
        after_estimate=measurement.estimate,
        reason=reason,
        floor_tokens=floor or measurement.tokens,
        reached_target=measurement.tokens <= target,
    )


def maybe_auto_compact(
    system_prompt: str,
    messages: list,
    tool_schemas: list,
    budget: int | models.RequestBudget = MAX_INPUT_TOKENS,
    model_id: str | None = None,
    provider: str | None = None,
    allow_tools: bool = True,
) -> CompactionResult:
    """Compacts the conversation only when it is eligible and over threshold.

    The complete raw request is measured first (system prompt + tool schemas +
    every message), then the newest exchange is kept verbatim and older tool
    exchanges are collapsed progressively until the target is reached or no
    meaningful reduction remains. The caller replaces its history only when
    ``changed`` is true.
    """
    limit = _budget_limit(budget)
    turns = count_turns(messages)
    threshold = resolve_threshold(
        config.AUTO_COMPACT_THRESHOLD, limit, 0.6
    )
    target = resolve_threshold(config.AUTO_COMPACT_TARGET, limit, 0.5)

    # Measure the complete request *before* compaction decides anything.
    before = measure_request(
        system_prompt,
        messages,
        tool_schemas,
        limit,
        include_tools=allow_tools,
        model_id=model_id,
        provider=provider,
        compact=False,
    )

    if not config.AUTO_COMPACT:
        return _unchanged(messages, False, turns, before, threshold, target, "disabled")

    minimum_turns = max(1, config.AUTO_COMPACT_MIN_TURNS)
    if turns < minimum_turns:
        return _unchanged(
            messages,
            False,
            turns,
            before,
            threshold,
            target,
            f"turn {turns} < minimum {minimum_turns}",
        )

    eligible = True
    if before.tokens <= threshold:
        return _unchanged(
            messages,
            eligible,
            turns,
            before,
            threshold,
            target,
            f"~{before.tokens} tokens <= threshold {threshold:.0f}",
        )

    # The cheapest request this policy can produce. Anything above it is protected
    # text, which is what makes an unreachable target explainable instead of
    # mysterious.
    floor = measure_request(
        system_prompt,
        compact_history(messages, 1, DETAIL_TERSE),
        tool_schemas,
        limit,
        include_tools=allow_tools,
        model_id=model_id,
        provider=provider,
        compact=False,
    ).tokens

    best_messages: list[dict] | None = None
    best: RequestMeasurement = before
    # Cheap first, then aggressive: keeping the first line of an old result is
    # worth a few tokens, so it is only dropped when it cannot reach the target.
    for detail in DETAIL_LEVELS:
        for keep in _auto_keep_options():
            candidate = compact_history(messages, keep, detail)
            measured = measure_request(
                system_prompt,
                candidate,
                tool_schemas,
                limit,
                include_tools=allow_tools,
                model_id=model_id,
                provider=provider,
                compact=False,
            )
            if measured.tokens < best.tokens:
                best = measured
                best_messages = candidate
            if measured.tokens <= target:
                break
        if best.tokens <= target:
            break

    if best_messages is None or best.tokens >= before.tokens:
        return _unchanged(
            messages,
            eligible,
            turns,
            before,
            threshold,
            target,
            "no stale tool exchanges to collapse",
            floor=floor,
        )

    reduction = before.tokens - best.tokens
    if reduction < MIN_REDUCTION_TOKENS:
        return _unchanged(
            messages,
            eligible,
            turns,
            before,
            threshold,
            target,
            f"compaction saved only ~{reduction} tokens (minimum {MIN_REDUCTION_TOKENS})",
            floor=floor,
        )

    reached = best.tokens <= target
    if reached:
        reason = (
            f"~{before.tokens} > threshold {threshold:.0f}; "
            f"target {target:.0f} reached"
        )
    else:
        reason = (
            f"~{before.tokens} > threshold {threshold:.0f}; ~{best.tokens} left, "
            f"~{floor} of them protected (system prompt, tool schemas, "
            "user/assistant text, newest exchange)"
        )

    return CompactionResult(
        messages=best_messages,
        changed=True,
        eligible=True,
        turns=turns,
        before_tokens=before.tokens,
        after_tokens=best.tokens,
        threshold_tokens=threshold,
        target_tokens=target,
        before_estimate=before.estimate,
        after_estimate=best.estimate,
        reason=reason,
        floor_tokens=floor,
        reached_target=reached,
    )
