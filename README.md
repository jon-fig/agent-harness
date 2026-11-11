# Custom Python AI Agent Harness

A local, single-process coding agent with no external orchestration. It runs an
OpenAI-compatible chat model in a tool loop, where every filesystem and Python
operation is served by an [MCP](https://modelcontextprotocol.io) server scoped
to `workspace/`.

## What it does

- **Vendor-agnostic model access** — Groq, NVIDIA NIM, OpenAI, OpenRouter,
  Together, DeepSeek, Mistral, a self-hosted vLLM/LM Studio server, or any other
  OpenAI-compatible API, selected with a base URL, key and model. Switching
  providers or models at runtime keeps the conversation.
- **Agentic coding loop** — read/write/list files, run Python code or files,
  install packages (with your approval), and keep persistent memory, all driven
  by tool calls with argument validation and self-correcting error messages.
- **Token discipline** — a model-aware token estimate, history compaction,
  tool-output truncation, pre-flight refusals, provider rate-limit pacing,
  413/429/5xx handling, and daily usage accounting where limits are published.
- **Session persistence** — conversation history, model choices, usage counters
  and learned tokenizer ratios survive a restart under `.agent/`.

## Architecture in one paragraph

`src/main.py` runs the REPL and calls `run_agent` in `src/agent.py` for each
turn. The agent builds a system prompt (static text + capped memory), measures
the complete request with a model-aware tokenizer (`src/budget.py`,
`src/tokenization.py`), compacts older tool exchanges when it exceeds the
configured threshold, checks the daily usage guard (`src/usage.py`), and sends
the request through an OpenAI-compatible client (`src/llm_client.py`, endpoint
resolved by `src/providers.py`, limits by `src/models.py`). Tool calls are
executed over a stdio MCP bridge (`src/mcp_client.py`) or handled in-process
(`save_memory`, `python_environment_info`), and results are appended to the
history until the model answers in prose. Details:
[docs/architecture.md](docs/architecture.md).

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

- `python src/main.py` does not work (absolute `src.*` imports); use
  `-m src.main` with the venv's interpreter. A system Python is refused with a
  clear message instead of mixing the client and tool environments.
- `requirements.txt` pins `mcp==1.30.0` and `mcp-python-interpreter==1.2.3`
  (the server imports `mcp.server.fastmcp`, removed in `mcp` 2.x), so client and
  server share one `mcp` release in one environment.

## Essential commands

| Command | Effect |
|---|---|
| `/model` | Show the active provider, model, usage and bundled options |
| `/model list` | Query the provider's catalogue over `GET /models` |
| `/model <id\|alias\|#>` | Switch model (`/model 20b`, or a number from the last list) |
| `/provider` | Show the active endpoint and every provider configured here |
| `/provider <name>` | Switch provider for this session (the conversation is kept) |
| `/compact` | Collapse older tool exchanges now |
| `/clear` | Forget the conversation and delete the saved session |
| `/help` | List commands |
| `exit` / `quit` | Quit (history is saved automatically) |

CLI flags: `--provider <name>`, `--model <id>`, `--list-models`,
`--list-providers`, `--debug-tokens`, `--no-session`.

`--no-session` disables both loading and saving: the run starts empty, `/clear`
does not delete the saved file, and exiting does not overwrite it. Use it for
diagnostics or a scratch conversation; the existing `.agent/session.json` is
left byte-for-byte untouched.

## Providers

A provider is three values: **base URL**, **API key**, **model id**, resolved
from `NAME_BASE_URL`/`NAME_API_KEY`/`NAME_MODEL`, then the generic
`AI_*` block, then the built-in preset. Any custom name works as a
self-hosted endpoint. Only Groq has bundled free-tier limits; every other
provider falls back to the global defaults and reacts to the provider's own 429s
and rate-limit headers. Examples, the preset table and per-model budget
overrides: [docs/providers-and-models.md](docs/providers-and-models.md).

## Features at a glance

- **Token management** — estimates carry provenance (exact tokenizer vs
  approximate, with self-calibration from reported usage), tool output is
  character-truncated where it enters history, older tool exchanges are
  collapsed, auto-compaction fires against the complete request, and 413/429/5xx
  are handled with provider-specific hints. See
  [docs/token-budget-and-compaction.md](docs/token-budget-and-compaction.md).
- **Memory** — `save_memory` appends capped, atomic notes to
  `workspace/memory.md`, injected into the system prompt each turn; the
  previous version is backed up to `.agent/memory.bak` before every save.
- **MCP tooling** — tools are discovered at runtime from the server, filtered
  and compressed for the wire, with package installs gated by an explicit
  terminal prompt. See [docs/tools-and-mcp.md](docs/tools-and-mcp.md).
- **Configuration** — every tunable is an environment variable, documented with
  defaults in [docs/configuration.md](docs/configuration.md) and
  [.env.example](.env.example).

## Project structure

```
.
├── .agent/                 # session.json, models.json, usage.json, tokenizer.json
├── .env.example
├── docs/                   # the documentation described below
├── requirements.txt
├── tests/                  # unittest suite
├── workspace/              # working directory: the agent's files + memory.md
└── src/
    ├── main.py             # CLI, slash commands, provider/model switching
    ├── agent.py            # tool loop, pacing, retries, text tool-call fallbacks
    ├── llm_client.py       # OpenAI-compatible client factory + model discovery
    ├── providers.py        # provider presets and env resolution
    ├── models.py           # per-provider model state, bundled limits, budgets
    ├── budget.py           # request measurement, compaction, payload building
    ├── tokenization.py     # model-aware tokenizer resolution and calibration
    ├── usage.py            # daily request/token accounting and projected guard
    ├── usage_report.py     # normalised provider-reported token usage
    ├── mcp_client.py       # stdio MCP bridge with output truncation
    ├── environment.py      # environment diagnostic + confirmed package install
    ├── approval.py         # harness-enforced yes/no before a mutating call
    ├── memory.py           # capped, atomic memory writer
    ├── session.py          # persisted conversation history
    └── config.py           # paths, .env loading, every tunable
```

## Known limitations

- **`workspace/` is not a security boundary.** It is a working directory, not an
  OS-level sandbox: `run_python_code` executes real Python with the host user's
  filesystem and network access. Wrap the MCP server with a launcher you trust
  (`MCP_SANDBOX_COMMAND` / `MCP_SANDBOX_ARGS`, e.g. bwrap/firejail/nsjail) when
  that matters — see [docs/tools-and-mcp.md](docs/tools-and-mcp.md).
- Model discovery lists what a provider advertises, not what the key can call.
- Providers without bundled limits or budget overrides have no local daily
  guard; the provider's own 429s and headers are still respected.
- Auto-compaction only collapses tool exchanges; long user prompts and assistant
  prose stay verbatim until `/clear` (deliberate: they carry the requirements).
- Daily usage counters are local to this machine; they are a guard rail, not
  billing. A second machine using the same key is invisible to them.
- Non-OpenAI model families start from an approximate chars/4 estimate unless a
  compatible tokenizer is configured via `TOKENIZER_*`.
- A model that hangs without sending response bytes is indistinguishable from a
  slow one; the harness waits up to `LLM_TIMEOUT` (default 120 s).
- The agent cannot install packages without explicit terminal approval, and a
  session without a terminal denies by default.

## Documentation

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Components, execution flow, state files, design boundaries |
| [docs/providers-and-models.md](docs/providers-and-models.md) | Presets, custom providers, precedence, model discovery, budgets |
| [docs/token-budget-and-compaction.md](docs/token-budget-and-compaction.md) | Estimation, calibration, truncation, compaction, retries, usage |
| [docs/tools-and-mcp.md](docs/tools-and-mcp.md) | MCP server lifecycle, tool surface, validation, installs, sandbox |
| [docs/configuration.md](docs/configuration.md) | Complete environment-variable reference with defaults |
| [docs/development-and-testing.md](docs/development-and-testing.md) | Dev setup, running and organizing the test suite |
