# Development and testing

## Setting up

One virtual environment serves the whole project: the harness interpreter, the
MCP server executable, the interpreter used by the Python tools, the pip target
and the package query must all be the same `.venv`.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt      # Windows
.venv/bin/python        -m pip install -r requirements.txt       # POSIX

cp .env.example .env        # then set LLM_PROVIDER and the key/URL/model
```

`requirements.txt` pins `mcp==1.30.0` and `mcp-python-interpreter==1.2.3`
because the server imports `mcp.server.fastmcp`, which `mcp` 2.x removed. Client
and server must share one `mcp` release, which is why they are installed
together into the main environment.

## Starting the harness

Always start it with the venv's interpreter, as a module:

```bash
.venv/Scripts/python.exe -m src.main          # Windows
.venv/bin/python        -m src.main           # POSIX
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
.venv/bin/python -m unittest discover -s tests -v          # POSIX
.venv\Scripts\python.exe -m unittest discover -s tests -v  # Windows
```

Run a single file with `python -m unittest tests.test_budgets -v`.

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
environment. It is the end-to-end proof that one environment serves the whole
agent: `run_python_code`, `run_python_file` and the package query all report the
harness interpreter, with `pandas` importable from both roles.

Because it launches a child process, it is slower than the rest of the suite;
everything else runs in-process with fakes.

## Common environment issues

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: the harness is not running inside a virtual environment` | Started with a system Python. Use `.venv/bin/python -m src.main` (or the Windows equivalent). |
| `MCP server executable not found` | `requirements.txt` not installed into the environment running the harness, or a custom `MCP_SERVER_COMMAND` points nowhere. Install with the harness interpreter, or set `MCP_SERVER_COMMAND`. |
| Startup error mentioning a `TaskGroup` | The server failed during the stdio handshake; the bridge prints the server's stderr and probes the executable directly to separate "cannot run" from "handshake failed". |
| `ModuleNotFoundError` for `mcp.server.fastmcp` | A `mcp` 2.x release got installed. Reinstall the pinned versions from `requirements.txt`. |
| An obsolete `.venv-mcp` directory | Not used by any current version; it can be deleted by hand. |
| Imports work in one tool and fail in another | Two different environments are involved. Verify with `python_environment_info`, which runs code through the actual server and warns about interpreter mismatches. |

## Conventions for contributions

- Tests must not require network access; the unified-environment test is the
  only one that spawns the real server, and it skips cleanly without it.
- `tests/test_config_defaults.py` cross-checks defaults against `.env.example`,
  so changing a default means updating both.
- Best-effort writers (usage, calibration, session) must treat a read-only
  `.agent/` directory as non-fatal; the existing tests assert this.
