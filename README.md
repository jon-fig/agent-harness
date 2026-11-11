# Custom Python AI Agent Harness

A local, single-process coding agent with no external orchestration. It runs an
OpenAI-compatible chat model in a tool loop, where every filesystem and Python
operation is served by an [MCP](https://modelcontextprotocol.io) server confined
to `workspace/`.

The model endpoint is **not tied to a vendor**: Groq, NVIDIA NIM, OpenAI,
OpenRouter, Together, DeepSeek, Mistral, a self-hosted vLLM/LM Studio server, or
any other OpenAI-compatible API can be selected with a base URL, key and model.

## Quick start

**One virtual environment serves everything.** The MCP tool server is installed
next to the interpreter that runs the harness, so the harness interpreter, the
server executable, the interpreter used by `run_python_code` / `run_python_file`,
the pip target and the package query are all the same `.venv`.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt      # Windows
.venv/bin/python        -m pip install -r requirements.txt       # POSIX

cp .env.example .env        # then set LLM_PROVIDER and the key/URL/model
.venv/Scripts/python.exe -m src.main          # Windows
.venv/bin/python        -m src.main           # POSIX
```

`python src/main.py` does not work (absolute `src.*` imports); use `-m src.main`.
Start the agent with the venv's interpreter: that is how the environment resolves
itself. Launching it from a system Python is refused with a clear message instead
of mixing the client and tool environments.

`requirements.txt` pins the tested releases `mcp==1.30.0` and
`mcp-python-interpreter==1.2.3`, because `mcp-python-interpreter` imports
`mcp.server.fastmcp`, which `mcp` 2.x removed (the 2.x module raises
`ModuleNotFoundError` and points at the replacement). Client and server therefore
have to share one `mcp` release, which is why they are installed together into the
main environment instead of into a second venv.

The console script is resolved inside the running interpreter's environment
(`sys.prefix`), never from `PATH` and never from another checkout's `.venv`, so a
globally installed server cannot shadow the project's one. Starting the harness
from a system Python is refused with an install/start command rather than silently
mixing two environments. Override the resolution with `MCP_SERVER_COMMAND`,
`MCP_SERVER_ARGS` (JSON list), `MCP_STARTUP_TIMEOUT` and `MCP_CALL_TIMEOUT` (a
timed-out call is cancelled on a best-effort basis: a running tool may ignore the
cancellation, but the harness stops waiting for it).

An obsolete `.venv-mcp` directory from an earlier version is no longer used: it
can be deleted by hand (`rm -rf .venv-mcp`) and nothing in the project creates or
reads it.

## Commands

| Command | Effect |
|---|---|
| `/model` | Show the active provider, model, usage and bundled options |
| `/model list` | Query the provider's catalogue over `GET /models` |
| `/model <id\|alias\|#>` | Switch model (`/model 20b`, or a number from the last list) |
| `/provider` | Show the active endpoint and every provider configured in this environment |
| `/provider <name>` | Switch provider for this session (the conversation is kept) |
| `/compact` | Collapse older tool exchanges now, keeping the latest verbatim |
| `/clear` | Forget the conversation and delete the saved session |
| `/help` | List commands |
| `exit` / `quit` | Quit (history is saved automatically) |

CLI flags: `--provider <name>`, `--model <id>`, `--list-models`,
`--list-providers`, `--debug-tokens`, `--no-session`.

`--no-session` disables both loading and saving: the run starts empty, `/clear`
does not delete the saved file, and exiting does not overwrite it. Use it for
diagnostics or a scratch conversation; the existing `.agent/session.json` is left
byte-for-byte untouched.

`--list-models` prints one model id per line and exits (useful in scripts);
`/model list` prints a numbered chooser and remembers it, so `/model 7` works
without retyping a long vendor id.

## Providers and models

A provider is three values: **base URL**, **API key** and **model id**. The
resolved values come from the first source that has one:

1. provider-specific `NAME_BASE_URL`, `NAME_API_KEY`, `NAME_MODEL`
2. the active provider's `AI_BASE_URL`, `AI_API_KEY`, `AI_MODEL`
3. the built-in preset defaults

So one generic `AI_*` block is enough for a single provider, while a `.env` with
several keys can use the prefixed names and switch at runtime. A provider name
does not have to be a preset: `LLM_PROVIDER=myhost` with
`MYHOST_BASE_URL`/`MYHOST_API_KEY`/`MYHOST_MODEL` works too. Generic `AI_BASE_URL`
is only needed for presets you override or for custom providers.

| Preset | Default endpoint | Default model |
|---|---|---|
| `groq` | `https://api.groq.com/openai/v1` | `openai/gpt-oss-120b` |
| `nvidia` | `https://integrate.api.nvidia.com/v1` | `deepseek-ai/deepseek-v4.1-flash` |
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` |
| `openrouter` | `https://openrouter.ai/api/v1` | `openrouter/auto` |
| `together` | `https://api.together.xyz/v1` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `deepseek` | `https://api.deepseek.com/v1` | `deepseek-chat` |
| `mistral` | `https://api.mistral.ai/v1` | `mistral-large-latest` |
| `cerebras` | `https://api.cerebras.ai/v1` | `llama3.3-70b` |
| `fireworks` | `https://api.fireworks.ai/inference/v1` | `accounts/fireworks/models/llama-v3p3-70b-instruct` |
| `perplexity` | `https://api.perplexity.ai` | `sonar` |
| `ollama` | `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`) | `OLLAMA_MODEL` |
| `custom` | `AI_BASE_URL` / `CUSTOM_BASE_URL` | `AI_MODEL` / `CUSTOM_MODEL` |

`openai` now means OpenAI itself. Older `.env` files used `LLM_PROVIDER=openai`
as an alias for Groq; use `LLM_PROVIDER=groq` instead. A legacy
`OPENAI_API_KEY=gsk_...` value is still accepted as a Groq key, and a Groq key
mistakenly pointed at the `openai` preset produces a clear error instead of a
401 from the wrong host.

Example with NVIDIA NIM:

```dotenv
LLM_PROVIDER=nvidia
AI_BASE_URL=https://integrate.api.nvidia.com/v1
AI_API_KEY=nvapi-...
AI_MODEL=minimaxai/minimax-m3.1-flash
```

Example with two providers in one `.env`:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL=minimaxai/minimax-m3.1-flash
```

Then `/provider nvidia` changes the endpoint for later turns without restarting.

The request loop uses OpenAI's `max_tokens` by default and automatically retries
with `max_completion_tokens` when a newer reasoning model rejects the old name.

### Model discovery

`/model list` (or `--list-models`) calls the provider's OpenAI-compatible
`GET /models` endpoint. This works without a bundled table, so new providers and
new model ids do not need a code change. Two caveats:

- A catalogue can advertise models the key cannot actually call; `/model <id>`
  is the source of truth.
- The list is fetched on demand, not at startup, so a blocked discovery endpoint
  does not delay the REPL. `MODEL_DISCOVERY_TIMEOUT` bounds the call.

Model choices are stored per provider in `.agent/models.json`, so switching
providers does not overwrite the other provider's selection.

### Bundled limits

Only Groq's published free-tier numbers are bundled (`src/models.py`). Any other
provider/model has no bundled limits and falls back to the global
`MAX_INPUT_TOKENS`; the agent still reacts to the provider's own 429 and rate-limit
headers. Groq applies its limits **per model, per organization**: every coding
model gets its own 8K TPM / 200K TPD bucket, so switching models is the only way
to get more throughput — a smaller model does not give a bigger budget.

A request is bounded by four separate values, resolved per provider **and** model:

- **context window** — the model's total input + output capacity;
- **maximum input** — what this harness allows the request body to carry;
- **maximum output** — the generation cap reserved inside the window;
- **rate limits** — the account's TPM/RPM/RPD/TPD buckets.

A model-specific `MODEL_<TAG>_*` variable wins over a provider-prefixed `NAME_*`
variable, which wins over the bundled metadata, which wins over the global default.
Adding a provider never assumes another provider's limits or tokenization.

## Token budget

Every iteration resends the whole conversation, so a turn grows with the square of
its iteration count. The harness bounds this in five places:

1. **Tool surface** — the coding tools plus four diagnostics are advertised, each
   with a one-line description instead of the server's long prose (the schema
   block stays around 1,000 tokens instead of 2,500+).
2. **System prompt** — short and static; memory comes after it, so the cacheable
   prefix stays byte-identical (cached tokens are often excluded from limits).
3. **Tool output** — middle-truncated to `TOOL_OUTPUT_MAX_CHARS` (default 4,000
   **characters**) in `src/mcp_client.py`, the single point where tool output
   enters history.
4. **History** — older tool exchanges are collapsed to one line; the last
   `HISTORY_KEEP_EXCHANGES` are replayed verbatim.
5. **The request itself** — a model-aware token estimate (exact `tiktoken` for
   known OpenAI families, an explicit chars/4 fallback otherwise), a pre-flight
   refusal instead of an oversized request, pacing from the provider's
   `x-ratelimit-remaining-tokens` header, 413 → compact and retry, 429 → wait for
   `retry-after`, bounded 5xx backoff, and a final iteration that runs without
   tools so the model has to summarise.

Two more rules keep a turn from wasting its budget:

- **Duplicate calls re-execute by default.** Identical arguments do not guarantee
  identical results (`run_python_code` can read a clock, a file another call just
  wrote, or a random seed), so repeats are executed again; set
  `DEDUPE_TOOL_CALLS=1` to replay the earlier result instead. Either way,
  `MAX_TOOL_CALLS_PER_TURN` (default 20) stops one response from spending a whole
  turn on executions.
- **The last iteration is told to wrap up.** It gets no tools plus an explicit
  "summarise what you completed and what is left" note, appended after the
  cacheable prefix.

### Tokens and characters are different units

The settings that look interchangeable are not:

| Setting | Unit | What it bounds |
|---|---|---|
| `MAX_INPUT_TOKENS` | tokens (estimated) | the whole request the harness will send |
| `MAX_OUTPUT_TOKENS` | tokens (requested) | the generation cap, clamped per model |
| `TOOL_OUTPUT_MAX_CHARS` | characters | one tool result as it enters history |
| `MEMORY_MAX_CHARS` | characters | the memory block appended to the prompt |
| the provider's context window | tokens | input + reserved output together |
| the provider's TPM/RPM/RPD/TPD | tokens or requests | the account's rolling/day bucket |

A 4,000-character cap is **not** a 4,000-token cap: characters per token vary
with content, language and code (roughly 4 for English prose, noticeably fewer for
JSON or code, and more for non-Latin scripts). Tool output is therefore capped in
characters where it is produced, and measured in tokens only as part of the
request estimate.

The generic `MAX_OUTPUT_TOKENS=4000` is also only a request. `src/models.py`
clamps it against the model's context window and the provider's TPM bucket (which
holds input *and* output), so a lower model-specific or provider-specific maximum
always wins, and an 8K-per-minute bucket is never asked for a 6,500-token input
plus a 4,000-token output.

### Auto-compaction

Compaction is a request-path decision: the **complete** request (system prompt +
tool schemas + every user/assistant message, tool call, argument and result) is
priced before it is sent, and older tool exchanges are collapsed only when both
conditions hold:

- the session has at least `AUTO_COMPACT_MIN_TURNS` user turns (default **2**);
- the measured request exceeds `AUTO_COMPACT_THRESHOLD` (default `0.6`, i.e. 60%
  of the model's input budget; a value above 1 is an absolute token count).

Being eligible does not mean compacting: a turn below the threshold is sent
untouched. When it does fire, compaction is progressive and stops as soon as the
target is reached:

1. **summary** — each older tool result becomes one line that keeps the fact of
   the call, the original size and the result's first line (a header row, a row
   count, a saved path), and long arguments keep their head;
2. **terse** — if that is not enough, the first line is dropped and only the
   outcome and size remain.

The newest `HISTORY_KEEP_EXCHANGES` exchanges are replayed verbatim, user messages
and assistant prose are never rewritten, and error output is preserved (truncated)
because it is what the model needs to correct itself. The original request, file
paths, decisions and outstanding errors therefore survive compaction.

Estimated size is recalculated after every attempt. A compaction that saves less
than a meaningful number of tokens is neither applied nor retried, already
collapsed messages are never rewritten again (so repeated attempts are no-ops),
and the log line reports tokens before/after, tokens saved, percentage, whether
the target was reached and why it stopped. When most of the request is protected
text — the system prompt, tool schemas, the user's own messages and the newest
exchange — the report says so and names that floor instead of pretending the
target is reachable. If the request still does not fit, the turn is refused with a
clear error instead of sending something the provider will reject. `/compact` runs
the same collapse on demand; `AUTO_COMPACT=0` disables the automatic path.

Estimates are never presented as provider usage: each carries its provenance
(exact tokenizer or approximate fallback), and `.agent/usage.json` stores reported
and estimated tokens in separate fields. Historical accounting always prefers the
provider's reported usage.

Models without a compatible tokenizer start from a chars/4 estimate and
**self-calibrate**: after each response the observed characters per
reported-input-token ratio is smoothed, clamped and stored per provider/model in
`.agent/tokenizer.json`, so the estimate converges on the provider's own count.
`TOKENIZER_CALIBRATION=0` disables this and keeps the fixed ratio.

`MAX_ITERATIONS` (default 8) is a hard cap of model calls per user turn.
`MAX_TOOL_CALLS_PER_TURN` (default 20) separately caps individual tool executions,
because one model call can request many calls. `MAX_OUTPUT_TOKENS` (default 4,000,
clamped per model) is part of the per-minute bucket too.
`--debug-tokens` prints estimated vs. reported usage and remaining headroom, and
why an eligible compaction did not reduce anything further.

### Output and daily limits

- `finish_reason=length` is handled: a truncated **answer** comes back with a
  `[Answer truncated at MAX_OUTPUT_TOKENS=…]` marker instead of looking complete.
  A truncated **tool call** is not executed at all (half a `write_file` body would
  replace a good file with a broken one) and the turn reports that instead.
- Requests and tokens are counted per provider and model per day in
  `.agent/usage.json`. Where bundled limits exist (Groq), a request that would
  exceed RPD/TPD is refused with the reset time (midnight UTC) instead of
  provoking a 429 the loop would wait out. The check is projective: it adds the
  request about to be sent (estimated input plus reserved output) to today's
  totals, so the last request that fits cannot push usage over the limit. Cached
  prompt tokens are excluded from the daily token check, matching how `/model`
  reports usage. The counters are local to this machine, so they are a guard rail,
  not billing.
- Providers without bundled limits still benefit from header-based pacing, 429
  handling and retry-after waiting.

Local Ollama has no per-minute ceiling, so it uses
`OLLAMA_MAX_INPUT_TOKENS` (default 32,000) instead.

## Memory

`save_memory` is harness state (`python_environment_info` is the other harness
tool; it answers by running code through the MCP server, so it describes the real
execution environment). It appends a one-line note to
**`workspace/memory.md`** — inside the sandbox, so the agent can read it with
`read_file`, but the harness is the only intended writer.

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
├── .agent/                 # session.json, models.json, usage.json, tokenizer.json
├── .env.example
├── requirements.txt
├── tests/                  # unittest suite: budgets, compaction, env, approval, usage
├── workspace/              # sandbox: the agent's files + memory.md
└── src/
    ├── agent.py            # tool loop, pacing, retries, fallback tool calls
    ├── approval.py         # harness-enforced yes/no before a mutating tool call
    ├── budget.py           # request measurement, compaction, payload building
    ├── config.py           # paths and every tunable
    ├── environment.py      # environment diagnostic + confirmed package install
    ├── llm_client.py       # OpenAI-compatible client factory + model discovery
    ├── main.py             # CLI, slash commands, provider/model switching
    ├── mcp_client.py       # stdio MCP bridge with output truncation
    ├── memory.py           # capped, atomic memory writer
    ├── models.py           # per-provider model state, bundled limits, budgets
    ├── providers.py        # provider presets and env resolution
    ├── session.py          # persisted conversation history
    ├── tokenization.py     # model-aware tokenizer resolution and estimates
    ├── usage.py            # per-provider/model daily request/token accounting
    └── usage_report.py     # normalised provider-reported token usage
```

## Tool chain

Tools come from the MCP server at runtime (`tools/list`), then are flattened into
OpenAI `tools=` payloads. Calls route back through the same stdio session.

Exposed to the model (with short descriptions that name the interpreter or
directory each one uses):

| Tool | What it does |
|---|---|
| `read_file`, `write_file` | read/write a text file in the workspace |
| `list_directory` | list a workspace directory |
| `run_python_code` | inline Python in the agent venv, starting in the workspace |
| `run_python_file` | run a workspace `.py` file with the agent venv |
| `clear_session` | reset the persistent Python session |
| `list_installed_packages` | packages in the venv that runs agent code |
| `list_python_environments` | interpreters the server can execute in |
| `install_package` | pip install into the agent venv, **only after the user approves** |
| `python_environment_info` | interpreter, version, virtualenv, cwd, workspace, key package versions (harness tool, answered by running code through the server) |
| `save_memory` | append a note to `workspace/memory.md` (harness tool) |

Intentionally hidden: `list_sessions` and the server's code-templating helpers,
which no coding turn needs. Hiding is enforced twice — the schema filter and the
dispatch path — so a hallucinated call cannot reach a hidden tool either.

All of these execute in **one** environment. `run_python_code` and
`run_python_file` both report `sys.executable` = `.venv/bin/python` (Windows:
`.venv\Scripts\python.exe`), start in `workspace/`, and see the same
`site-packages`, so a package installed once is importable from both.

Inspect what the server actually exposes:

```bash
.venv/bin/python -m src.mcp_client --list
.venv/bin/python -m src.mcp_client --call read_file '{"file_path": "analysis.py"}'
```

### Installing packages

The agent cannot install anything on its own say-so. When a missing dependency
blocks it, `install_package` goes through the harness, which

1. validates the argument (`name` or `name==version`; no URLs, paths, extras or
   pip flags — it is never run through a shell),
2. refuses to install into a global interpreter (it must be running in a venv),
3. asks the person at the terminal, showing the package and the target
   interpreter, and treats anything but an explicit `y`/`yes` as no,
4. installs into the environment that runs agent code,
5. verifies the import with that same interpreter and reports the version.

`MCP_INSTALL_APPROVAL=deny` refuses every request without prompting, and a session
with no interactive terminal (piped output, cron) denies by default rather than
installing unattended. Upgrades are refused: ask for an explicit version instead,
so unrelated packages are never replaced.

If the child process exits or its MCP session drops, the next tool call starts a
fresh server; the interrupted call reports that it may or may not have run rather
than pretending it succeeded. `get_bridge()` never returns a dead bridge.

### Sandbox scope

`workspace/` is a working directory, not a security boundary: `run_python_code`
executes real Python with the host user's filesystem and network access. For an
OS-level sandbox, wrap the MCP server with a launcher you trust:

```dotenv
MCP_SANDBOX_COMMAND=bwrap
MCP_SANDBOX_ARGS=["--ro-bind","/","/","--bind","/abs/path/workspace","/abs/path/workspace","--unshare-net","--"]
```

The launcher and its arguments are prepended to the server command, so any tool
that takes `launcher [args] command [args]` works (bwrap, firejail, nsjail, a
container exec). Without one, the tools run unconfined.

## Configuration

Everything is optional; see `.env.example`. Notable keys: `LLM_PROVIDER`,
`AI_BASE_URL`, `AI_API_KEY`, `AI_MODEL`, the provider-specific
`NAME_BASE_URL`/`NAME_API_KEY`/`NAME_MODEL` variants, `OLLAMA_BASE_URL`,
`OLLAMA_MODEL`, `MAX_ITERATIONS`, `MAX_OUTPUT_TOKENS`, `MAX_INPUT_TOKENS`,
`OLLAMA_MAX_INPUT_TOKENS`, `MAX_TOOL_CALLS_PER_TURN`, `DEDUPE_TOOL_CALLS`,
`TOOL_OUTPUT_MAX_CHARS`, `MEMORY_MAX_CHARS`,
`HISTORY_KEEP_EXCHANGES`, `AUTO_COMPACT`, `AUTO_COMPACT_MIN_TURNS`,
`AUTO_COMPACT_THRESHOLD`, `AUTO_COMPACT_TARGET`, the provider/model budget
overrides (`NAME_*` and `MODEL_<TAG>_*`), the `TOKENIZER_*` variables and
`TOKENIZER_CALIBRATION`,
`LLM_TIMEOUT`, `LLM_MAX_RETRIES`, `MODEL_DISCOVERY_TIMEOUT`, `MCP_STARTUP_TIMEOUT`,
`MCP_CALL_TIMEOUT`, `MCP_SERVER_COMMAND`, `MCP_SERVER_ARGS`,
`MCP_INSTALL_APPROVAL`, `MCP_SANDBOX_COMMAND`, `MCP_SANDBOX_ARGS`,
`DEBUG_TOKENS`.

`.env` precedence: a non-blank value in the file **overrides** the shell
environment; a blank assignment (`AI_API_KEY=`) means "not set here" and leaves a
shell value in place. This exists because a shell-level `LLM_PROVIDER` used to
mask the file silently.

`.env` is read **once, at startup**. Editing it while the agent runs has no effect
on any setting — budgets, compaction, timeouts or the tool server — until the
process is restarted. `/provider` and `/model` change the live connection, not the
values above.

## Known limitations

- Model discovery lists what a provider advertises, not what the key can call;
  some catalogues include retired or unavailable models. `/model <id>` confirms.
- Providers without bundled limits or `NAME_*`/`MODEL_<TAG>_*` overrides fall
  back to `MAX_INPUT_TOKENS` and have no local daily guard. The provider's own 429
  and headers are still respected.
- A model that hangs without sending response bytes cannot be distinguished from
  a slow one; the harness waits up to `LLM_TIMEOUT` (default 120s). Pick a
  responsive model with `/model list`.
- Tool calls written as prose rather than JSON are not executed.
- Auto-compaction only collapses tool exchanges; long user prompts and assistant
  prose stay verbatim until `/clear`. That is deliberate (they carry the
  requirements), but it also means a session can hit the input budget with a
  single huge prompt: the compaction report then names that floor instead of
  shrinking it.
- Daily usage counters are local: a second machine using the same key is invisible
  to them, and clearing `.agent/usage.json` resets the guard (not the real quota).
- Non-OpenAI model families start from the approximate chars/4 estimate unless a
  compatible tokenizer is configured through `TOKENIZER_*`. The fallback
  self-calibrates from reported usage, but it stays labelled approximate and is
  never recorded as reported usage.
- `workspace/` confines the agent's file tools by convention, not by OS policy:
  `run_python_code` can read or write anything the harness user can. Use
  `MCP_SANDBOX_COMMAND` when that matters.
- MCP `list_directory` and `run_python_code` output is capped by characters, not by
  rows, so a wide DataFrame is still expensive.

## Testing

The suite uses the standard library's `unittest` (no extra dependency). Run it
with the project's interpreter:

```bash
.venv/bin/python -m unittest discover -s tests -v          # POSIX
.venv\Scripts\python.exe -m unittest discover -s tests -v  # Windows
```

Coverage: tokenizer resolution and fallback, provider-usage normalisation,
per-provider/model budgets and their clamping, the projected usage guard,
configuration defaults (checked against `.env.example`), auto-compaction
(eligibility from turn 2, threshold, target, preservation of instructions/errors/
newest exchanges, idempotency, before/after accounting, unreachable targets),
session safety (`--no-session` never loads, saves or clears the real file),
provider retries (413 compaction, 429 retry-after, bounded 5xx), native and
MiniMax/JSON text tool calls, tool-output truncation (including truncated JSON),
package-install approval and import-name verification, workspace path errors, and
— in `tests/test_unified_env.py` — resolution of the server executable from the
active environment plus a live-server check that both Python tools and the
package query use the harness interpreter, with `pandas` importable from both.

That last file starts the real MCP server, so it is skipped automatically when the
server is not installed; it is the end-to-end proof that one environment serves
the whole agent.
