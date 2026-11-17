# Development and testing

## Setting up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
Python 3.12 or later. One uv-managed virtual environment serves the harness, MCP
server executable, Python tools and approved package installs. `uv.lock` pins
the complete resolved graph; `pyproject.toml` pins the direct dependencies.

```bash
uv sync --locked
# Optional pandas and NumPy for data-analysis tasks:
uv sync --locked --extra data

cp .env.example .env        # POSIX; then configure provider and credentials
```

On Windows PowerShell, use `Copy-Item .env.example .env` for the copy step.
`uv sync` creates and synchronizes `.venv`; use `--locked` to fail if the lock
file needs an update. Syncing an existing `.venv` may remove packages that are
not in the project lock. The `data` extra is optional because core modules do
not import pandas or NumPy.

The compatible MCP pair is intentionally pinned: `mcp==1.30.0` and
`mcp-python-interpreter==1.2.3`. The interpreter server imports
`mcp.server.fastmcp`, which the MCP SDK 2.x line removed. Keep both in one
environment and rerun the suite before changing either pin. `pip==26.3` is
also a direct runtime pin because the upstream server's package-listing tool
calls `python -m pip list`; uv-created environments otherwise need not include
pip.

## Starting the harness

Always start it with the venv's interpreter, as a module:

```bash
.venv/Scripts/python.exe -m src.main          # Windows PowerShell
.venv/bin/python -m src.main                  # POSIX
```

- `python src/main.py` does not work (absolute `src.*` imports); use
  `-m src.main`.
- A global interpreter is refused with an install/start command instead of
  silently mixing the client and tool environments. This is enforced by
  `config.require_supported_interpreter()`.
- The MCP server executable is resolved from the running interpreter's
  environment (`sys.prefix`), never from `PATH` or another checkout's `.venv`.

## Running the tests

The suite uses the standard library's `unittest` (no extra dependency):

```bash
uv run --locked python -m unittest discover -s tests -v
```

The equivalent direct interpreter commands are
`.venv/bin/python -m unittest discover -s tests -v` (POSIX) and
`.venv\Scripts\python.exe -m unittest discover -s tests -v` (Windows).
Run one test module with
`uv run --locked python -m unittest discover -s tests -p 'test_budgets.py' -v`.

## Test organization

| File | Covers |
|---|---|
| `tests/test_agent_limits.py` | Iteration limits, per-turn tool-call cap, duplicate-call handling |
| `tests/test_argument_validation.py` | Harness-side argument validation and workspace path errors |
| `tests/test_budgets.py` | Per-provider/model request budgets and their clamping |
| `tests/test_calibration.py` | Chars/token calibration learning and clamping |
| `tests/test_compaction.py` | Auto-compaction eligibility, thresholds, targets, preservation rules, idempotency, unreachable targets |
| `tests/test_config_defaults.py` | Configuration defaults checked against `.env.example` |
| `tests/test_environment_tools.py` | `python_environment_info`, requirement parsing, import-name resolution, install approval flow |
| `tests/test_mcp_lifecycle.py` | Bridge lifecycle: startup failure reporting, dead-bridge replacement, truncation |
| `tests/test_provider_retries.py` | 413 compaction retry, 429 retry-after waiting, bounded 5xx backoff |
| `tests/test_readline_editing.py` | Line-editing import behaviour in the REPL |
| `tests/test_session_safety.py` | `--no-session` never loads, saves or clears the real session file |
| `tests/test_tokenization.py` | Tokenizer resolution, explicit specs, fallbacks, caching |
| `tests/test_tool_call_parsing.py` | Native, bare-JSON and MiniMax text tool-call parsing |
| `tests/test_tool_output_truncation.py` | Middle truncation, including the truncated-JSON marker |
| `tests/test_unified_env.py` | Server resolution from the active environment **plus a live-server check** that both Python tools and the package query use the harness interpreter |
| `tests/test_usage_guard.py` | The projected daily usage guard (RPD/TPD) |
| `tests/test_usage_report.py` | Normalisation of provider-reported token usage |

## MCP integration tests

`tests/test_unified_env.py` starts the **real** MCP server. It is skipped
automatically when `mcp-python-interpreter` is not installed in the active
environment. It checks that `run_python_code`, `run_python_file` and package
queries use the harness interpreter. Its pandas import check runs only when the
optional package is installed (`uv sync --locked --extra data`).

Because it launches a child process, it is slower than the rest of the suite;
everything else runs in-process with fakes.

## Common environment issues

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: the harness is not running inside a virtual environment` | Started with a system Python. Run `uv sync --locked`, then use `.venv/bin/python -m src.main` (or the Windows equivalent). |
| `MCP server executable not found` | `uv sync --locked` was not run for the environment running the harness, or a custom `MCP_SERVER_COMMAND` points nowhere. |
| Startup error mentioning a `TaskGroup` | The server failed during the stdio handshake; the bridge prints the server's stderr and probes the executable directly to separate "cannot run" from "handshake failed". |
| `ModuleNotFoundError` for `mcp.server.fastmcp` | The MCP SDK and interpreter server versions are incompatible. Restore the locked environment with `uv sync --locked`. |
| An obsolete `.venv-mcp` directory | Not used by any current version; it can be deleted by hand. |
| Imports work in one tool and fail in another | Two different environments are involved. Verify with `python_environment_info`, which runs code through the actual server and warns about interpreter mismatches. |

## Dependency updates

Direct dependencies use exact versions in `pyproject.toml`; `uv.lock` records
the complete graph. For a deliberate direct-dependency update, change its
specifier, run `uv lock`, inspect the lockfile diff and upstream release notes,
then run `uv sync --locked` and the test suite. Use `uv audit --locked` to query
OSV for known advisories when network access is available. Do not treat a
successful audit as proof that packages are safe or that the tool server is
sandboxed.

Agent-approved runtime installs use `uv pip install` in the active environment.
They are not added to `pyproject.toml` or `uv.lock`; a later `uv sync` may remove
them. Add durable dependencies to `pyproject.toml` instead.

## Conventions for contributions

- Tests do not call an LLM API. Tokenizer tests may download tiktoken encoding
  data on first use if it is not already cached. The unified-environment test
  spawns the real local server and skips cleanly when it is unavailable.
- `tests/test_config_defaults.py` cross-checks defaults against `.env.example`,
  so changing a default means updating both.
- Best-effort writers (usage, calibration, session) must treat a read-only
  `.agent/` directory as non-fatal; the existing tests assert this.
