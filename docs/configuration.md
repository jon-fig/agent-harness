# Configuration reference

Everything is configured through environment variables, typically via a `.env`
file copied from `.env.example`. Every value is read **once, at startup**:
editing `.env` while the agent runs has no effect on budgets, compaction,
timeouts or the tool server until the process is restarted. The REPL's
`/provider` and `/model` commands change the live connection, not these limits.

## `.env` precedence

A non-blank value in `.env` **overrides** the shell environment; a blank
assignment (`AI_API_KEY=`) means "not set here" and leaves a shell value in
place. This exists because a shell-level `LLM_PROVIDER` used to silently mask
the file.

## Provider selection and credentials

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Any OpenAI-compatible provider name: a preset (`groq`, `nvidia`, `openai`, `openrouter`, `together`, `deepseek`, `mistral`, `cerebras`, `fireworks`, `perplexity`, `ollama`) or an arbitrary custom name. Overridable per run with `--provider` or in the REPL with `/provider <name>`. |
| `NAME_BASE_URL` / `NAME_API_KEY` / `NAME_MODEL` | — | Provider-specific connection values, where `NAME` is the upper-cased provider name (e.g. `GROQ_API_KEY`, `NVIDIA_BASE_URL`, `MYHOST_MODEL`). Highest precedence. |
| `AI_BASE_URL` | *(empty)* | Generic base URL for the active provider; also reads legacy `LLM_BASE_URL`. |
| `AI_API_KEY` | *(empty)* | Generic API key; also reads `OPENAI_API_KEY`. A `gsk_…` value pointed at the `openai` preset is rejected with a hint. |
| `AI_MODEL` | *(empty)* | Generic model id for the active provider; also reads `LLM_MODEL`. Empty means the preset default. |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local Ollama endpoint. |
| `OLLAMA_MODEL` | `qwen2.5:coder` (code default; `.env.example` suggests `qwen3:1.7b`) | Local Ollama model. |

Precedence per value: `NAME_*` → `AI_*` → built-in preset. See
[providers-and-models.md](providers-and-models.md).

## Model selection and request limits

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_CONTEXT_WINDOW` | *(none)* | Global context-window override when no per-model/provider value applies. |
| `NAME_CONTEXT_WINDOW` | *(none)* | Context window for one provider. |
| `MODEL_<TAG>_CONTEXT_WINDOW` | *(none)* | Context window for one model (`<TAG>` = model id upper-cased, non-alphanumerics → `_`). |
| `NAME_MAX_INPUT_TOKENS` / `MODEL_<TAG>_MAX_INPUT_TOKENS` | *(none)* | Input ceiling override for a provider / model. |
| `NAME_MAX_OUTPUT_TOKENS` / `MODEL_<TAG>_MAX_OUTPUT_TOKENS` | *(none)* | Output cap override for a provider / model. |
| `NAME_TPM` / `NAME_RPM` / `NAME_RPD` / `NAME_TPD` | *(none)* | Rate-limit overrides for a provider; otherwise bundled metadata (Groq only) applies. |
| `MAX_INPUT_TOKENS` | `6500` | Global input ceiling for the estimated request body. |
| `MAX_OUTPUT_TOKENS` | `4000` | Generation cap requested from the model; clamped per model/provider. |
| `OLLAMA_MAX_INPUT_TOKENS` | `32000` | Input budget for local providers (no per-minute ceiling). |
| `MODEL_DISCOVERY_TIMEOUT` | `20` | Seconds for the `GET /models` discovery call. |
| `LLM_TIMEOUT` | `120` | Seconds for a chat completion. |

Budget precedence: `MODEL_<TAG>_*` → `NAME_*` → bundled metadata → globals.
Output and input are additionally clamped by the context window and the TPM
bucket (never below a 400-token minimum input, with 200 tokens of headroom).

## Token budgeting and compaction

| Variable | Default | Purpose |
|---|---|---|
| `TOOL_OUTPUT_MAX_CHARS` | `4000` | **Character** cap on one tool result as it enters history (middle-truncated). |
| `MEMORY_MAX_CHARS` | `1200` | **Character** cap on the memory block in the system prompt (~300 tokens). |
| `HISTORY_KEEP_EXCHANGES` | `2` | Recent tool exchanges replayed verbatim; older ones are collapsed. |
| `AUTO_COMPACT` | on (`1`) | Enables automatic compaction on the request path. |
| `AUTO_COMPACT_MIN_TURNS` | `2` | Eligibility starts at this many user turns. Legacy spelling `AUTOCOMPACT_AFTER_TURNS` is still accepted. |
| `AUTO_COMPACT_THRESHOLD` | `0.6` | Compaction fires above this fraction of the input budget (values above 1 = absolute tokens). Legacy spelling `AUTOCOMPACT_TOKEN_FRACTION` is still accepted. |
| `AUTO_COMPACT_TARGET` | `0.5` | Compaction stops once the request is at or below this fraction. |

## Agent iteration and tool execution

| Variable | Default | Purpose |
|---|---|---|
| `MAX_ITERATIONS` | `8` | Model calls per user turn. |
| `MAX_TOOL_CALLS_PER_TURN` | `20` | Tool *executions* per user turn (one model call can request many). |
| `DEDUPE_TOOL_CALLS` | `0` | By default identical calls re-execute (results can differ). `1`/`true`/`yes`/`on` replays the earlier result instead. |
| `LLM_MAX_RETRIES` | `2` | Retries per response class in the agent loop (413 shrink, 429 wait, 5xx backoff). The OpenAI SDK itself is built with `max_retries=0`. |

## Usage accounting and tokenizer calibration

| Variable | Default | Purpose |
|---|---|---|
| `TOKENIZER_CALIBRATION` | on (`1`) | Learns chars/token ratios for the approximate tokenizer from provider-reported prompt tokens; stored in `.agent/tokenizer.json`. `0` keeps the fixed chars/4 estimate. |
| `TOKENIZER_<SANITIZED_MODEL_ID>` | *(none)* | Explicit tokenizer spec for one model, e.g. `TOKENIZER_MINIMAXAI_MINIMAX_M3_1_FLASH=hf:some/repo`. |
| `TOKENIZER_MAP` | *(none)* | JSON object of `fnmatch` pattern → tokenizer spec, e.g. `{"qwen/*": "tiktoken:cl100k_base"}`. |
| `TOKENIZER_DEFAULT` | *(none)* | Fallback tokenizer spec for all other models. |

Specs: `tiktoken:<encoding>`, `approx[:chars_per_token]`, `hf:<repo>`,
`file:<path>`. The `hf:`/`file:` forms need the optional `tokenizers` package;
without it they degrade to the approximate tokenizer. There is no usage-related
configuration beyond this: counters live in `.agent/usage.json` and reset
automatically at midnight UTC.

## MCP server configuration and timeouts

| Variable | Default | Purpose |
|---|---|---|
| `MCP_SERVER_COMMAND` | resolved automatically | Path to the `mcp-python-interpreter` console script. Resolution looks only inside the environment running the harness (`sys.prefix`), never `PATH` or another `.venv`. Set it only to override that. |
| `MCP_SERVER_ARGS` | `["--dir", "<workspace>", "--python-path", "<harness interpreter>"]` | JSON list of server arguments. |
| `MCP_STARTUP_TIMEOUT` | `60` | Seconds for the server to become ready. |
| `MCP_CALL_TIMEOUT` | `300` | Seconds for one tool call. A timed-out call is cancelled best-effort. |

## Package installation approval

| Variable | Default | Purpose |
|---|---|---|
| `MCP_INSTALL_APPROVAL` | `ask` | `ask` prompts on the terminal that started the agent; `deny` refuses every install without asking. A session without an interactive terminal denies by default. |

## Optional sandbox configuration

| Variable | Default | Purpose |
|---|---|---|
| `MCP_SANDBOX_COMMAND` | *(empty)* | OS-level launcher (e.g. `bwrap`, `firejail`, `nsjail`) prepended to the MCP server command. |
| `MCP_SANDBOX_ARGS` | *(empty)* | JSON list of launcher arguments, prepended before the server command. |

Note that `workspace/` is **not** a security boundary by itself; see
[tools-and-mcp.md](tools-and-mcp.md#workspace-behaviour-and-limitations).

## Debugging and diagnostics

| Variable | Default | Purpose |
|---|---|---|
| `DEBUG_TOKENS` | `0` | Logs estimated vs. reported usage per request, rate-limit headroom, and why an eligible compaction reduced nothing. Also settable with `--debug-tokens`. |

## CLI flags (not environment variables)

| Flag | Effect |
|---|---|
| `--provider <name>` | Provider for this run. |
| `--model <id>` | Model for this run (overrides `.env`, persisted per provider). |
| `--list-models` | Print the provider's catalogue and exit. |
| `--list-providers` | Print configured providers and exit. |
| `--debug-tokens` | Enable token logging for this run. |
| `--no-session` | Start empty; do not load or save `.agent/session.json` for this process. |
