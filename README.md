# Custom Python AI Agent Harness

A local, single-process coding agent with no external orchestration. It runs an
OpenAI-compatible chat model in a tool loop, where every filesystem and Python
operation is served by an [MCP](https://modelcontextprotocol.io) server confined
to `workspace/`.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt      # Windows
.venv/bin/python        -m pip install -r requirements.txt       # POSIX

cp .env.example .env        # then set AI_API_KEY for Groq
.venv/Scripts/python.exe -m src.main
```

`python src/main.py` does not work (absolute `src.*` imports); use `-m src.main`.

The MCP tool server lives in its own virtualenv because it requires `mcp<2` while
the harness uses `mcp>=2`:

```bash
python -m venv .venv-mcp
.venv-mcp/Scripts/python.exe -m pip install "mcp<2" mcp-python-interpreter
```

The harness finds the console script automatically; override with
`MCP_SERVER_COMMAND`, `MCP_SERVER_ARGS` (JSON list), `MCP_STARTUP_TIMEOUT` and
`MCP_CALL_TIMEOUT` (a timed-out call is cancelled, not left running).

## Commands

| Command | Effect |
|---|---|
| `/model` | Show the active model, its free-tier limits and the bundled options |
| `/model <id-or-alias>` | Switch model (`20b`, `qwen`, `120b`, or a full Groq id). Persisted in `.agent/model.txt` |
| `/compact` | Collapse older tool exchanges to one line each, keeping the latest verbatim (also automatic, see below) |
| `/clear` | Forget the conversation and delete the saved session |
| `/help` | List commands |
| `exit` / `quit` | Quit (history is saved automatically) |

CLI flags: `--provider {groq,ollama}`, `--model <id>`, `--debug-tokens`,
`--no-session`.

## Providers and models

| Provider | Endpoint | Model selection |
|---|---|---|
| `groq` | `https://api.groq.com/openai/v1` | `/model`, `--model`, `AI_MODEL` |
| `ollama` | `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) | `OLLAMA_MODEL`, or `/model <name>` at runtime |

The remote provider uses one key, **`AI_API_KEY`**, and one model variable,
`AI_MODEL`.

Older variable spellings still work as legacy aliases: `LLM_PROVIDER=openai` maps
to `groq`, and `OPENAI_API_KEY` is read when `AI_API_KEY` is unset. They are only
kept so an existing `.env` or shell export does not break; new setup should use
the names above.

Groq's free tier applies its limits **per model, per organisation**: every coding
model gets its own 8K TPM / 200K TPD bucket, so switching models is the only way
to get more throughput — a smaller model does not give a bigger budget.
`src/models.py` holds the published numbers and derives the per-request input
budget from them.

## Token budget

Every iteration resends the whole conversation, so a turn grows with the square of
its iteration count. The harness bounds this in five places:

1. **Tool surface** — only the six coding tools are advertised, with short
   descriptions (schema block ~620 tokens instead of ~1,400).
2. **System prompt** — short and static; memory comes after it, so the cacheable
   prefix stays byte-identical (cached tokens do not count toward Groq limits).
3. **Tool output** — middle-truncated to `TOOL_OUTPUT_MAX_CHARS` (default 1,600) in
   `src/mcp_client.py`, the single point where tool output enters history.
4. **History** — older tool exchanges are collapsed to one line; the last
   `HISTORY_KEEP_EXCHANGES` are replayed verbatim.
5. **The request itself** — exact `tiktoken` counting, a pre-flight refusal instead
   of an oversized request, pacing from the provider's `x-ratelimit-remaining-tokens`
   header, 413 → compact and retry, 429 → wait for `retry-after`, bounded 5xx backoff,
   and a final iteration that runs without tools so the model has to summarise.

Two more rules keep a turn from wasting its budget:

- **Identical calls run once.** Repeating a tool call with the same arguments in
  one turn cannot return anything new, so the earlier result (its first 400 chars)
  is replayed instead of the tool running again.
- **The last iteration is told to wrap up.** It gets no tools plus an explicit
  "summarise what you completed and what is left" note, appended after the
  cacheable prefix.

### Auto-compaction

Old tool exchanges are collapsed automatically before a request when either
trigger fires: **3 user turns** in the session (`AUTOCOMPACT_AFTER_TURNS`), or the
history passing 60% of the input budget (`AUTOCOMPACT_TOKEN_FRACTION`). On the 8K
TPM free tier the token trigger usually fires first, because one turn of tool
output can cost more than several ordinary turns; either way older tool exchanges
are dropped early enough to keep the request inside the 8K window. `/compact`
still runs it on demand.

`MAX_ITERATIONS` (default 5) is a hard cap of model calls per user turn.
`MAX_OUTPUT_TOKENS` (default 1,200) is part of the per-minute bucket too.
`--debug-tokens` prints estimated vs. reported usage and remaining headroom.

### Output and daily limits

- `finish_reason=length` is now handled: a truncated **answer** comes back with a
  `[Answer truncated at MAX_OUTPUT_TOKENS=…]` marker instead of looking complete.
  A truncated **tool call** is not executed at all (half a `write_file` body would
  replace a good file with a broken one) and the turn reports that instead.
- Requests and tokens are counted per model per day in `.agent/usage.json`
  (Groq's RPD/TPD). A request that would exceed either limit is refused with the
  reset time (midnight UTC) instead of provoking a 429 the loop would wait out.
  The counters are local to this machine, so they are a guard rail, not billing.

Local Ollama has no per-minute ceiling, so `--provider ollama` uses
`OLLAMA_MAX_INPUT_TOKENS` (default 32,000) instead.

## Memory

`save_memory` is the one in-process tool (everything else is MCP). It appends a
one-line note to **`workspace/memory.md`** — inside the sandbox, so the agent can
read it with `read_file`, but the harness is the only intended writer.

- Loaded once per turn, after the static system prompt.
- Capped at `MEMORY_MAX_CHARS` (default 1,200 chars, roughly 300 tokens): the
  writer drops the oldest notes instead of cutting a note in half.
- Writes are atomic (temp file + `os.replace`), and the `# Agent Memory` header is
  preserved even when the file is empty.
- The previous version is copied to `.agent/memory.bak` before every save, and
  before any `write_file` that targets `memory.md` (the agent can overwrite its
  own memory; restore by copying the backup back).

## Layout

```
.
├── .agent/                 # session.json + model.txt (gitignored runtime state)
├── .env.example
├── requirements.txt
├── workspace/              # sandbox: the agent's files + memory.md
└── src/
    ├── agent.py            # tool loop, pacing, retries, fallback tool calls
    ├── budget.py           # tiktoken estimate, compaction, payload building
    ├── config.py           # paths and every tunable
    ├── llm_client.py       # OpenAI-compatible client factory
    ├── main.py             # CLI and slash commands
    ├── mcp_client.py       # stdio MCP bridge with output truncation
    ├── memory.py           # capped, atomic memory writer
    ├── models.py           # model catalog, /model state, budgets
    ├── session.py          # persisted conversation history
    └── usage.py            # per-model daily request/token accounting
```

## Tool chain

Tools come from the MCP server at runtime (`tools/list`), filtered to
`read_file`, `write_file`, `list_directory`, `run_python_code`, `run_python_file`
and `clear_session`, then flattened into OpenAI `tools=` payloads. Calls route
back through the same stdio session. Inspect what the server actually exposes:

```bash
python -m src.mcp_client --list
python -m src.mcp_client --call read_file '{"file_path": "analysis.py"}'
```

`run_python_code` executes in the server's virtualenv; `run_python_file` uses the
interpreter passed as `--python-path` (the harness venv). Install data-science
packages into the environment you actually use.

## Configuration

Everything is optional; see `.env.example`. Notable keys: `LLM_PROVIDER`,
`AI_API_KEY`, `AI_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `MAX_ITERATIONS`,
`MAX_OUTPUT_TOKENS`, `MAX_INPUT_TOKENS`, `OLLAMA_MAX_INPUT_TOKENS`, `TOOL_OUTPUT_MAX_CHARS`,
`MEMORY_MAX_CHARS`, `HISTORY_KEEP_EXCHANGES`, `AUTOCOMPACT_AFTER_TURNS`,
`AUTOCOMPACT_TOKEN_FRACTION`, `LLM_TIMEOUT`, `LLM_MAX_RETRIES`,
`MCP_STARTUP_TIMEOUT`, `MCP_CALL_TIMEOUT`, `DEBUG_TOKENS`.

`.env` precedence: a non-blank value in the file **overrides** the shell
environment; a blank assignment (`AI_API_KEY=`) means "not set here" and leaves a
shell value in place. This exists because a shell-level `LLM_PROVIDER` used to
mask the file silently.

## Known limitations

- Free-tier Groq is 8K TPM: a long conversation eventually cannot be sent, and the
  harness reports that instead of firing a request that would be rejected.
- Tool calls written as prose rather than JSON are not executed.
- Auto-compaction only collapses tool exchanges; long assistant prose and user
  prompts stay verbatim until `/clear`.
- Daily usage counters are local: a second machine using the same key is invisible
  to them, and clearing `.agent/usage.json` resets the guard (not the real quota).
- MCP `list_directory` and `run_python_code` output is capped by characters, not by
  rows, so a wide DataFrame is still expensive.
- There is no automated test suite; the harness is verified manually per change.
