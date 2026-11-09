"""
Core agentic loop.

Every iteration resends the whole conversation, so on a token-limited provider
the cost of a turn grows with the square of the iteration count. This module keeps
that growth in check at the four points that matter:

* the tool surface - a filtered, compressed presentation of the server's schemas
  (Stages 1-2);
* the system prompt - short, with persistent memory loaded once per turn and
  placed after the static block (Stages 3 and 6);
* tool output - capped at the MCP bridge (Stage 4, see src/mcp_client.py);
* history - old tool exchanges collapsed (Stage 5, see src/budget.py);
* the request itself - bounded output, pacing against the provider's own
  rate-limit headers, retries that respect 413/429, and a pre-flight refusal
  instead of an oversized request (Stage 7).
"""

import json
import random
import re
import time
import types
import uuid

from src import config, models, providers, tokenization, usage
from src.budget import build_payload, compact_history, maybe_auto_compact
from src.config import LLM_MAX_RETRIES, MEMORY_MAX_CHARS
from src.llm_client import get_llm_client
from src.memory import backup_current, load_memory, save_memory
from src.usage_report import TokenUsage

# Harness state (memory) is not a file/exec operation, so it stays in-process even
# though every filesystem and Python tool is served over MCP.
HARNESS_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Save a short note, preference or insight to persistent memory.",
            "parameters": {
                "type": "object",
                "properties": {"memory": {"type": "string"}},
                "required": ["memory"],
            },
        },
    },
]
HARNESS_TOOL_MAP = {"save_memory": save_memory}

# Stage 1: only the tools a coding turn actually needs. install_package is the
# only tool that can mutate the server's venv, and the environment/package/session
# listing helpers are diagnostics the loop never needs. clear_session stays: it is
# the only way to reset the persistent default REPL session, and the harness calls
# it at the start of every turn.
KEEP_TOOLS = {
    "read_file",
    "write_file",
    "list_directory",
    "run_python_code",
    "run_python_file",
    "clear_session",
}

# Stage 2: presentation-only short descriptions. The server remains the source of
# truth for names, types and required arguments; this map only replaces prose that
# would otherwise be resent on every request (run_python_code's is 825 chars).
SHORT_DESCRIPTIONS = {
    "read_file": "Read a text file (never a CSV) from the workspace.",
    "write_file": "Write a file in the workspace; overwrite to replace.",
    "list_directory": "List files in a workspace directory.",
    "run_python_code": "Run Python inline; print only what you need to see.",
    "run_python_file": "Run a workspace .py file and return its output.",
    "clear_session": "Reset the Python session and its variables.",
}
MAX_DESCRIPTION_CHARS = 90

# Stage 3: role, sandbox scope, and the two rules that prevent token blowups.
SYSTEM_PROMPT = """You are an autonomous coding and data-science agent working in a sandboxed workspace served by MCP tools.
Use read_file / list_directory for text files; use run_python_code or run_python_file for anything involving datasets.
Never read_file a CSV or other dataset: inspect it with pandas and print only what you need.
After write_file, do not read the file back to verify; run it instead.
Issue independent tool calls together in one response.
Reply with a short summary of what you did and what is left."""

# Appended (after the cacheable prefix) only to the request that has no tools, so
# the model writes the turn's summary instead of another plan it cannot execute.
FINAL_ITERATION_NOTE = """

--- Final iteration ---
No tools are available for this response. Do not describe a call you cannot make.
Summarise what you completed, what is left, and which files hold the result."""

# One identical repeat within a turn is always a loop, never a coincidence: the
# conversation has not changed, so the tool cannot return anything new. The head
# of the earlier result is enough for the model to continue.
DUPLICATE_HEAD_CHARS = 400


class BudgetExceeded(RuntimeError):
    """The conversation cannot be sent without the provider rejecting it."""


class AgentError(RuntimeError):
    """The turn did not finish; the CLI reports this instead of printing an answer."""


# --- Rate-limit awareness ----------------------------------------------------
# Groq reports the remaining tokens of the rolling window on every response. Using
# them turns "the model stops around 8K" from a guess into a number we can act on.
_DURATION = re.compile(r"(?:(\d+)m)?([\d.]+)s")
# Windows are per provider *and* per model, so pacing state is keyed by both:
# switching model or /provider must not inherit another endpoint's budget.
_RATE: dict[str, dict] = {}


def _rate_key(provider: str, model_name: str) -> str:
    return f"{provider}:{model_name}"


def _rate_state(model_key: str) -> dict:
    return _RATE.setdefault(model_key, {"remaining": None, "reset_at": 0.0})


def _seconds_from_header(value) -> float:
    if not value:
        return 0.0
    match = _DURATION.fullmatch(str(value).strip())
    if match:
        minutes, seconds = match.groups()
        return float(minutes or 0) * 60 + float(seconds)
    try:
        return float(str(value).strip().rstrip("s"))
    except ValueError:
        return 0.0


def _note_rate_limit(model_key: str, headers) -> None:
    """Remembers how much of this model's per-minute token bucket is left."""
    if headers is None:
        return
    try:
        raw_remaining = headers.get("x-ratelimit-remaining-tokens")
    except AttributeError:
        return
    if raw_remaining is None:
        return
    state = _rate_state(model_key)
    try:
        state["remaining"] = int(float(raw_remaining))
    except (TypeError, ValueError):
        return
    reset = _seconds_from_header(headers.get("x-ratelimit-reset-tokens"))
    if reset:
        state["reset_at"] = time.monotonic() + reset


def _pace_before_request(model_key: str, approx_input: int, max_output: int) -> None:
    """Waits out the window instead of provoking a 429 the retry loop must absorb."""
    state = _rate_state(model_key)
    remaining = state["remaining"]
    if remaining is None:
        return
    needed = approx_input + max_output
    if remaining >= needed:
        return

    wait = min(max(state["reset_at"] - time.monotonic(), 1.0), 60.0)
    print(
        f"[Agent] Token window has {remaining} left, next request needs ~{needed}; "
        f"pausing {wait:.0f}s."
    )
    time.sleep(wait)
    state["remaining"] = None


def _present_schema(schema: dict) -> dict:
    """Compresses one server schema for the wire (presentation only)."""
    function = schema.get("function") or {}
    name = function.get("name", "")

    description = SHORT_DESCRIPTIONS.get(name) or (function.get("description") or "")
    description = " ".join(description.split())[:MAX_DESCRIPTION_CHARS]

    # Deep copy so the server's own definitions are never mutated.
    parameters = json.loads(json.dumps(function.get("parameters") or {}, default=str))
    parameters.pop("description", None)
    parameters.pop("title", None)
    for prop in (parameters.get("properties") or {}).values():
        if isinstance(prop, dict):
            prop.pop("description", None)
            prop.pop("title", None)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def get_tooling():
    """Returns (openai_schemas, dispatch) for the MCP tool server.

    dispatch(name, arguments) -> str, and is responsible for its own errors.
    """
    # Imported lazily so importing this module never pays the server startup cost.
    from src.mcp_client import get_bridge

    bridge = get_bridge()
    schemas = [
        _present_schema(schema)
        for schema in bridge.openai_schemas()
        if schema["function"]["name"] in KEEP_TOOLS
    ] + HARNESS_TOOL_SCHEMAS
    parameters_by_name = {
        schema["function"]["name"]: schema["function"].get("parameters") or {}
        for schema in schemas
    }

    def dispatch(name, arguments):
        # Enforce the filter on the call path too: hiding a tool from the schema
        # does not stop a hallucinated call from reaching the server.
        if name not in KEEP_TOOLS and name not in HARNESS_TOOL_MAP:
            return f"Error: tool '{name}' is not enabled in this harness."

        arguments = arguments if isinstance(arguments, dict) else {}
        _strip_sandbox_prefix(arguments)
        problem = _validate_arguments(name, parameters_by_name.get(name), arguments)
        if problem:
            return problem

        # memory.md is inside the workspace, so the agent's own write_file can
        # replace everything it has saved. Keep the previous copy first.
        if name == "write_file" and _touches_memory(arguments):
            if backup_current():
                print("[Agent Info] Backed up memory.md to .agent/memory.bak before overwriting.")

        if name in HARNESS_TOOL_MAP:
            return HARNESS_TOOL_MAP[name](**arguments)
        # Resolved per call so a bridge that died mid-turn is replaced instead of
        # failing every later tool call in the turn.
        return get_bridge().call(name, arguments)

    return schemas, dispatch


_JSON_TYPES = {
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "array": list,
    "object": dict,
}


def _strip_sandbox_prefix(arguments: dict) -> None:
    """Drops a redundant leading 'workspace/' from path arguments.

    The MCP server is started with --dir pointing at the workspace, so a path is
    already relative to it. Models see the word "workspace" in the system prompt
    and call write_file("workspace/notes.md"), which the server resolves under its
    own root and lands in workspace/workspace/notes.md. The prefix is meaningless
    to the server, so removing it keeps every file in the workspace the user sees.
    """
    for key in ("file_path", "directory_path"):
        value = arguments.get(key)
        if not isinstance(value, str):
            continue
        text = value.replace("\\", "/")
        while text.startswith("./"):
            text = text[2:]
        if text == "workspace" or text.startswith("workspace/"):
            arguments[key] = text[len("workspace"):].lstrip("/")


def _touches_memory(arguments: dict) -> bool:
    """True when a filesystem tool targets the memory file by any usual key."""
    for key in ("file_path", "path", "filename", "name"):
        value = arguments.get(key)
        if isinstance(value, str):
            normalized = value.replace("\\", "/").rstrip("/")
            if normalized.split("/")[-1] == config.MEMORY_FILE.name:
                return True
    return False


def _validate_arguments(name: str, parameters: dict | None, arguments: dict) -> str | None:
    """Checks a call against the tool's advertised schema, cheaply.

    The server does validate, but its answer is a generic MCP error the model has
    to guess from. Naming the offending argument and the allowed set lets it
    self-correct on the next iteration instead of repeating the mistake.

    Only the plain JSON-Schema subset the servers actually advertise is checked;
    schemas using $ref/combinators or accepting extra properties are left to the
    server rather than rejected on a guess.
    """
    parameters = parameters if isinstance(parameters, dict) else {}
    properties = parameters.get("properties")
    required = parameters.get("required") or []
    strict = (
        isinstance(properties, dict)
        and parameters.get("additionalProperties") in (None, False)
        and not parameters.get("$ref")
        and not parameters.get("oneOf")
    )

    # Unknown keys first: "filepath" instead of "file_path" is a typo the model
    # can fix immediately, while "missing required" alone hides the cause.
    if strict:
        unknown = [key for key in arguments if key not in properties]
        if unknown:
            listed = ", ".join(f"'{key}'" for key in unknown)
            allowed = ", ".join(sorted(properties)) or "(none)"
            return (
                f"Error: tool '{name}' does not accept argument(s) {listed}. "
                f"Allowed: {allowed}."
            )

    if isinstance(required, list):
        missing = [key for key in required if key not in arguments]
        if missing:
            listed = ", ".join(f"'{key}'" for key in missing)
            return f"Error: tool '{name}' is missing required argument(s) {listed}."

    if not isinstance(properties, dict):
        return None  # nothing declared, so nothing to type-check

    for key, value in arguments.items():
        spec = properties.get(key)
        if value is None or not isinstance(spec, dict):
            continue
        declared = spec.get("type")
        expected = declared if isinstance(declared, list) else [declared]
        expected = [item for item in expected if item in _JSON_TYPES]
        if not expected:
            continue
        # bool is an int in Python, but not in JSON: {"n": true} is not a count.
        matches = any(
            isinstance(value, _JSON_TYPES[item]) and not (item in ("integer", "number") and isinstance(value, bool))
            for item in expected
        )
        if not matches:
            return (
                f"Error: argument '{key}' for tool '{name}' must be "
                f"{'/'.join(expected)}, got {type(value).__name__}."
            )

    return None


def build_system_prompt() -> str:
    """Stage 6: memory is read once per turn, capped, and kept after the static block.

    Placing it after the constant text keeps the cacheable prefix byte-identical
    across iterations, and providers exclude cached tokens from rate limits.
    """
    memory = load_memory().strip()
    if len(memory) > MEMORY_MAX_CHARS:
        memory = memory[:MEMORY_MAX_CHARS] + "\n…[memory truncated]"
    return f"{SYSTEM_PROMPT}\n\n--- Persistent Memory ---\n{memory or '(empty)'}"


def _message_dict(message) -> dict:
    """Normalises an SDK message into the dict shape stored in history.

    The last branch covers duck-typed message objects: dropping their tool_calls
    would leave a role:"tool" message answering a call the history does not
    contain, which the provider rejects.
    """
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)

    normalised = {
        "role": getattr(message, "role", "assistant"),
        "content": str(getattr(message, "content", "") or ""),
    }
    calls = getattr(message, "tool_calls", None)
    if calls:
        normalised["tool_calls"] = [
            {
                "id": getattr(call, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(getattr(call, "function", None), "name", ""),
                    "arguments": getattr(getattr(call, "function", None), "arguments", "{}"),
                },
            }
            for call in calls
        ]
    return normalised


def _parse_text_tool_call(content: str) -> tuple[str, str] | None:
    """Reads a tool call a model wrote as JSON text instead of the native field.

    Returns (tool name, arguments as a JSON string), or None when the text is a
    normal answer. Only a bare JSON object (optionally fenced) is accepted, so a
    prose reply that happens to mention a tool is left alone.
    """
    text = (content or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        ).strip()
    if not text.startswith("{"):
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    # gpt-oss writes {"name", "arguments"}; other models use tool/function and
    # parameters. Placeholders such as "<tool_name>" are not a call.
    name = parsed.get("name") or parsed.get("tool") or parsed.get("function")
    if not isinstance(name, str) or not name.strip() or name.strip().startswith("<"):
        return None

    arguments = parsed.get("arguments", parsed.get("parameters", {}))
    if isinstance(arguments, dict):
        arguments_json = json.dumps(arguments)
    elif isinstance(arguments, str):
        arguments_json = arguments
    else:
        return None
    return name.strip(), arguments_json


def _retry_after_seconds(headers, fallback: float) -> float:
    """Uses the provider's own retry hint instead of guessing."""
    for key in ("retry-after", "x-ratelimit-reset-tokens"):
        try:
            raw = headers.get(key)
        except AttributeError:
            return fallback
        if not raw:
            continue
        text = str(raw).strip()
        # Groq writes durations like "7.66s" or "1m2.4s""; plain seconds and
        # milliseconds are handled too.
        seconds = _seconds_from_header(text)
        if seconds <= 0 and text.endswith("ms"):
            try:
                seconds = float(text[:-2]) / 1000
            except ValueError:
                continue
        if seconds <= 0:
            continue
        return max(1.0, min(seconds, 60.0))
    return fallback


def _log_usage(
    approx_input: int,
    token_usage: TokenUsage,
    headers,
    model_name: str | None = None,
) -> None:
    if not config.DEBUG_TOKENS:
        return

    # A large gap between the estimate and the reported prompt means the
    # tokenizer needs attention; the headers show how much room is left.
    headroom = ""
    if headers is not None:
        try:
            remaining = headers.get("x-ratelimit-remaining-tokens")
            reset = headers.get("x-ratelimit-reset-tokens")
        except AttributeError:
            remaining = reset = None
        if remaining is not None or reset is not None:
            headroom = f" remaining={remaining} reset={reset}"

    prefix = f"{model_name} " if model_name else ""
    print(
        f"[Tokens] {prefix}estimated_input={approx_input} "
        f"reported={token_usage.describe()}{headroom}"
    )


def _create_completion(client, **kwargs):
    """Returns (response, headers); headers feed the pacing logic when available."""
    try:
        raw = client.chat.completions.with_raw_response.create(**kwargs)
    except AttributeError:
        # Older/alternate SDKs: no raw response handle, so no rate-limit headers.
        return client.chat.completions.create(**kwargs), None
    return raw.parse(), getattr(raw, "headers", None)


def _prepare_payload(
    system_prompt,
    messages,
    tool_schemas,
    budget,
    allow_tools,
    keep_options=(config.HISTORY_KEEP_EXCHANGES, 1, 0),
    model_id=None,
    provider=None,
):
    """Returns (payload, approx_tokens, fits, use_tools).

    Dropping the tool block is the cheapest way to keep a conversation that has
    grown close to the limit: fewer tokens, and a model that has to answer in
    prose instead of calling another tool.
    """
    if allow_tools:
        payload, approx, fits = build_payload(
            system_prompt,
            messages,
            tool_schemas,
            budget,
            keep_options,
            model_id=model_id,
            provider=provider,
        )
        if fits:
            return payload, approx, True, True

    payload, approx, fits = build_payload(
        system_prompt,
        messages,
        tool_schemas,
        budget,
        keep_options,
        include_tools=False,
        model_id=model_id,
        provider=provider,
    )
    if allow_tools and fits:
        print("[Agent] Input does not fit with the tool block; sending without tools to force an answer.")
    return payload, approx, fits, False


def _request_completion(
    client,
    provider,
    model_name,
    system_prompt,
    messages,
    tool_schemas,
    budget: models.RequestBudget,
    allow_tools=True,
):
    """Calls the provider once, surviving the free tier's limits.

    413 means this request's input is too large: compact everything and retry
    once. 429 means the rolling window is exhausted: wait for as long as the
    provider says. 5xx: bounded backoff. The client is built with max_retries=0 so
    these decisions are made here, visibly, instead of inside the SDK.
    """
    model_key = _rate_key(provider, model_name)

    # Automatic compaction is a request-path decision: it must see the system
    # prompt, the tool schemas and the whole history before it can know whether
    # the next request is expensive.
    compaction = maybe_auto_compact(
        system_prompt,
        messages,
        tool_schemas,
        budget,
        model_id=model_name,
        provider=provider,
        allow_tools=allow_tools,
    )
    if compaction.changed:
        messages[:] = compaction.messages
        print(
            f"[Session] Auto-compacted ({compaction.reason}; {compaction.turns} turns): "
            f"~{compaction.before_tokens} -> ~{compaction.after_tokens} tokens "
            f"(target {compaction.target_tokens:.0f})."
        )
    elif (
        config.DEBUG_TOKENS
        and compaction.eligible
        and compaction.before_tokens > compaction.threshold_tokens
    ):
        print(
            f"[Session] Auto-compaction did not reduce the request further "
            f"(~{compaction.after_tokens} tokens): {compaction.reason}."
        )

    payload, approx, fits, use_tools = _prepare_payload(
        system_prompt,
        messages,
        tool_schemas,
        budget,
        allow_tools,
        model_id=model_name,
        provider=provider,
    )
    if not fits:
        raise BudgetExceeded(
            f"the complete request needs ~{approx} input tokens but the budget is "
            f"{budget.max_input_tokens} (context "
            f"{budget.context_window or 'unknown'}, max output "
            f"{budget.max_output_tokens}); auto-compaction could not reduce it "
            "enough. Use /clear or choose a model with a larger context window."
        )

    # Project the request that is about to be sent, not only the totals already
    # recorded: input estimate plus the reserved output.
    projected_tokens = approx + budget.max_output_tokens
    blocked = usage.guard(
        model_name,
        provider,
        projected_tokens=projected_tokens,
        projected_requests=1,
    )
    if blocked:
        raise BudgetExceeded(blocked)

    attempt = 0
    delay = 2.0
    shrunk = False
    # OpenAI's newer reasoning models reject max_tokens and want
    # max_completion_tokens instead. Most compatible providers accept the old
    # name, so it is the default and the fallback is driven by the error text.
    output_field = "max_tokens"

    while attempt <= LLM_MAX_RETRIES:
        attempt += 1
        _pace_before_request(model_key, approx, budget.max_output_tokens)

        kwargs = {
            "model": model_name,
            "messages": payload,
            output_field: budget.max_output_tokens,
        }
        if use_tools:
            kwargs["tools"] = tool_schemas
            kwargs["tool_choice"] = "auto"

        try:
            response, headers = _create_completion(client, **kwargs)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            headers = getattr(getattr(exc, "response", None), "headers", None) or {}
            error_text = str(exc).lower()

            if (
                status == 400
                and output_field == "max_tokens"
                and "max_tokens" in error_text
                and (
                    "max_completion_tokens" in error_text
                    or "not supported" in error_text
                    or "unsupported" in error_text
                )
            ):
                output_field = "max_completion_tokens"
                print(
                    "[Agent] Provider rejected max_tokens; retrying with "
                    "max_completion_tokens."
                )
                continue

            if status == 413 and not shrunk:
                shrunk = True
                messages[:] = compact_history(messages, 0)
                payload, approx, fits, use_tools = _prepare_payload(
                    system_prompt,
                    messages,
                    tool_schemas,
                    budget,
                    allow_tools and use_tools,
                    keep_options=(0,),
                    model_id=model_name,
                    provider=provider,
                )
                if not fits:
                    raise BudgetExceeded(
                        f"the provider rejected the request as too large and "
                        f"compaction still leaves ~{approx} input tokens "
                        f"(budget {budget.max_input_tokens})."
                    ) from exc
                print(
                    f"[Agent] Request too large; compacted history to ~{approx} tokens and retrying."
                )
                continue

            if status == 429:
                wait = _retry_after_seconds(headers, delay)
                print(f"[Agent] Rate limited; waiting {wait:.0f}s (attempt {attempt}).")
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue

            if status is not None and 500 <= int(status) < 600:
                wait = min(delay, 30.0) + random.uniform(0, 0.5)
                print(f"[Agent] Provider error {status}; retrying in {wait:.1f}s.")
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue

            raise
        else:
            token_usage = TokenUsage.from_response(response)
            _note_rate_limit(model_key, headers)
            _log_usage(approx, token_usage, headers, model_key)
            # Learn this model's real chars/token ratio from the provider's own
            # count, so a fallback estimate stops undercounting later requests.
            if token_usage.input:
                tokenization.record_calibration(
                    model_name,
                    provider,
                    tokenization.request_characters(
                        payload, tool_schemas if use_tools else None
                    ),
                    token_usage.input,
                )
            usage.record(
                model_name,
                token_usage,
                provider=provider,
                estimated_tokens=approx,
            )
            return response

    raise RuntimeError(f"LLM provider kept failing after {LLM_MAX_RETRIES} retries.")


def _truncation_notice(max_output: int) -> str:
    """Marks an answer the provider cut off, instead of passing it off as complete."""
    return (
        f"\n\n[Answer truncated at the model's {max_output}-token output limit. "
        "Ask for the rest, or raise that limit for one turn.]"
    )


def run_agent(user_prompt: str, messages: list, provider: str | None = None) -> str:
    """
    Takes a user prompt and the conversation message history, then executes
    the agentic loop (calling tools iteratively) until the model provides a final text response.

    Raises AgentError when the turn could not finish (provider failure, spent
    daily allowance, truncated tool call). Returning that text as if it were an
    answer made a failed turn look successful, and saved it as history.
    """
    provider = providers.normalize(provider or config.LLM_PROVIDER)
    client, model_name = get_llm_client(provider)
    tool_schemas, dispatch_tool = get_tooling()

    # Split the limits into the request-size budget (context/input/output) and
    # the provider's rate limits. Each provider/model can override either without
    # changing the other.
    budget = models.request_budget(model_name, provider)

    # Append the new user message to the history
    messages.append({"role": "user", "content": user_prompt})

    # Stage 6: load memory once per turn instead of once per iteration.
    system_prompt = build_system_prompt()

    # The server's default REPL session persists across turns; reset it so code
    # state from an earlier turn cannot leak into this one.
    try:
        dispatch_tool("clear_session", {})
    except Exception:
        pass  # older servers may not expose it

    max_iterations = config.MAX_ITERATIONS
    # MAX_ITERATIONS bounds model calls, not tool executions: a single response can
    # ask for many calls, so executions have their own ceiling per turn.
    max_tool_calls = max(1, config.MAX_TOOL_CALLS_PER_TURN)
    executed_tool_calls = 0
    # Only used when DEDUPE_TOOL_CALLS is enabled: identical arguments do not
    # guarantee identical results, so repeats execute by default.
    seen: dict[tuple[str, str], str] = {}

    # Agentic loop. The last iteration gets no tools, so the model has to write
    # its summary from what it already has instead of opening new work.
    for current_iteration in range(1, max_iterations + 1):
        final_iteration = current_iteration == max_iterations
        iteration_prompt = system_prompt + (FINAL_ITERATION_NOTE if final_iteration else "")

        try:
            response = _request_completion(
                client,
                provider,
                model_name,
                iteration_prompt,
                messages,
                tool_schemas,
                budget,
                allow_tools=not final_iteration,
            )
        except BudgetExceeded as exc:
            message = (
                f"I stopped before sending the request: {exc}. "
                "Everything done so far is saved in the workspace; ask for a smaller step."
            )
            print(f"[Agent Warning] {message}")
            messages.append({"role": "assistant", "content": message})
            raise AgentError(message) from exc
        except Exception as exc:
            raise AgentError(f"Error communicating with LLM provider: {exc}") from exc

        choice = response.choices[0]
        response_message = choice.message
        content = response_message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)

        # Check native tool_calls first; on the final iteration tools were not
        # offered, so anything tool-shaped in the text is not executed.
        detected_tool_calls = [] if final_iteration else (response_message.tool_calls or [])

        # A call the provider cut off mid-arguments cannot be run: half a
        # write_file body would replace a good file with a broken one.
        if finish_reason == "length" and detected_tool_calls:
            message = (
                f"I reached the {budget.max_output_tokens}-token output limit while "
                "composing a tool call, so I did not run it. Ask me to retry that "
                "step in smaller pieces."
            )
            print(f"[Agent Warning] {message}")
            messages.append({"role": "assistant", "content": message})
            raise AgentError(message)

        # Fallback for models that write the tool call as JSON text instead of using
        # the native field. The result is replayed as a real assistant.tool_calls
        # entry: a role:"tool" message without one is rejected with a 400, which is
        # why the old fabricated id stopped the loop on the next iteration.
        fallback_assistant = None
        if not detected_tool_calls and not final_iteration:
            fallback = _parse_text_tool_call(content)
            if fallback is not None:
                func_name, arguments_json = fallback
                call_id = f"call_{uuid.uuid4().hex[:12]}"
                print(f"[Agent Info] Captured text-based tool call -> '{func_name}'")

                detected_tool_calls = [
                    types.SimpleNamespace(
                        id=call_id,
                        function=types.SimpleNamespace(
                            name=func_name, arguments=arguments_json
                        ),
                    )
                ]
                # No content: the JSON is already in tool_calls and would otherwise
                # be resent (and paid for) on every later iteration.
                fallback_assistant = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": func_name, "arguments": arguments_json},
                        }
                    ],
                }

        # Execute tools if detected (native or fallback)
        if detected_tool_calls:
            # Append the assistant's response to history so the model remembers its
            # action (as a dict: pydantic objects do not serialise for accounting).
            messages.append(fallback_assistant or _message_dict(response_message))

            for tool_call in detected_tool_calls:
                function_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments

                print(f"\n[Agent Action] Executing tool -> '{function_name}'")

                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else (raw_arguments or {})
                except json.JSONDecodeError as jde:
                    error_feedback = f"Error: Failed to parse tool arguments as valid JSON: {str(jde)}."
                    print(f"[Agent Warning] {error_feedback}")
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": error_feedback})
                    continue

                key = (function_name, json.dumps(arguments, sort_keys=True, default=str))
                if executed_tool_calls >= max_tool_calls:
                    tool_result = (
                        f"Error: this turn reached its limit of {max_tool_calls} tool "
                        f"executions, so '{function_name}' was not run. Summarise what "
                        "is done and continue in the next turn."
                    )
                    print(
                        f"[Agent Warning] Per-turn tool-call limit reached "
                        f"({max_tool_calls}); '{function_name}' was not executed."
                    )
                elif config.DEDUPE_TOOL_CALLS and key in seen:
                    earlier = seen[key]
                    head = (
                        earlier
                        if len(earlier) <= DUPLICATE_HEAD_CHARS
                        else earlier[:DUPLICATE_HEAD_CHARS] + "…"
                    )
                    tool_result = (
                        f"[Not executed again: '{function_name}' already ran with these exact "
                        f"arguments this turn. Earlier result:]\n{head}"
                    )
                    print(f"[Agent Info] Skipped duplicate '{function_name}' call.")
                else:
                    # Execute through the MCP bridge
                    try:
                        tool_result = dispatch_tool(function_name, arguments)
                    except KeyError:
                        tool_result = f"Error: Tool '{function_name}' does not exist in the environment."
                    except Exception as tool_exc:
                        tool_result = f"Error executing tool '{function_name}': {str(tool_exc)}"
                    executed_tool_calls += 1
                    seen[key] = str(tool_result)

                preview = str(tool_result)[:150].replace('\n', ' ')
                print(f"[Agent Result] Output -> {preview}...")

                # Feed the tool execution result back into the loop
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })
        else:
            # No tool calls required; model returned a final text answer to the user
            if not content.strip():
                content = (
                    "I hit the iteration limit for this turn without a final answer. "
                    "The work so far is in the workspace; ask me to continue."
                )
                messages.append({"role": "assistant", "content": content})
                return content

            messages.append({"role": "assistant", "content": content})
            if finish_reason == "length":
                print(
                    f"[Agent Warning] Answer hit the {budget.max_output_tokens}-token "
                    "output limit; it is truncated."
                )
                return content + _truncation_notice(budget.max_output_tokens)
            return content
