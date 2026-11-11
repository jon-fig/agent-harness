# Tools and MCP

Every filesystem and Python operation is served by an
[MCP](https://modelcontextprotocol.io) server over stdio. Using MCP means the
tool implementation lives in a separate, replaceable process with a standard
protocol: the harness only needs `tools/list` and `tools/call`, so the tool
server can be updated or swapped without touching the agent loop.

## Server launch

`src/mcp_client.py` resolves the server command at startup:

- The console script `mcp-python-interpreter` is looked up **only inside the
  environment running the harness** (`sys.prefix`), never from `PATH` and never
  from another checkout's `.venv`. A globally installed server cannot shadow the
  project's one; if the active environment lacks the server, the bridge says how
  to install it instead of starting a mismatched one.
- `MCP_SERVER_COMMAND` overrides the resolution; `MCP_SERVER_ARGS` (a JSON list)
  overrides the default arguments `["--dir", <workspace>, "--python-path",
  <harness interpreter>]`.
- The child process is started with its working directory set to `workspace/`,
  so relative paths resolve the same way for every tool.
- An optional `MCP_SANDBOX_COMMAND` (+ `MCP_SANDBOX_ARGS`) is prepended to the
  command line, so the whole server can run inside `bwrap`, firejail, `nsjail`
  or a container exec.

The harness runs `require_supported_interpreter()` before starting the server:
a global interpreter is refused because it would put the MCP client and the
code-execution environment in different site-packages.

## Bridge lifecycle

`get_bridge()` starts the server lazily, once per process:

- The MCP session lives on its own asyncio event loop in a background thread;
  the synchronous agent loop marshals each call with
  `asyncio.run_coroutine_threadsafe()`.
- Startup is bounded by `MCP_STARTUP_TIMEOUT` (default 60 s). A failed startup
  is reported with the actual cause: the server's stderr is captured, and a
  probe run (`--help` outside the stdio client) separates "the executable cannot
  run" from "the stdio handshake failed".
- A tool call is bounded by `MCP_CALL_TIMEOUT` (default 300 s). A timed-out call
  is cancelled best-effort — a running tool may ignore cancellation, but the
  harness stops waiting and the abandoned coroutine cannot queue up later calls.
- If the child dies or its session drops, `get_bridge()` never returns a dead
  bridge: the interrupted call reports that it *may or may not have run*, and
  the next tool call starts a fresh server.
- Shutdown is registered with `atexit`.

You can inspect what the server exposes without starting the agent:

```bash
.venv/bin/python -m src.mcp_client --list
.venv/bin/python -m src.mcp_client --call read_file '{"file_path": "notes.txt"}'
```

## Tool discovery and exposure

Tools come from the server at runtime (`tools/list`) and are flattened into
OpenAI `tools=` function schemas. The harness filters and compresses this
presentation:

| Tool | What it does |
|---|---|
| `read_file` / `write_file` | Read / write a text file in the workspace |
| `list_directory` | List a workspace directory |
| `run_python_code` | Inline Python in the agent venv, starting in the workspace |
| `run_python_file` | Run a workspace `.py` file with the agent venv |
| `clear_session` | Reset the persistent Python session |
| `list_installed_packages` | Packages installed in the venv that runs agent code |
| `list_python_environments` | Interpreters the tool server can use |
| `install_package` | pip install into the agent venv, **only after user approval** |
| `python_environment_info` | Harness tool: interpreter, version, virtualenv, cwd, workspace, key package versions |
| `save_memory` | Harness tool: append a note to `workspace/memory.md` |

Intentionally hidden: `list_sessions` and the server's code-templating helpers.
Hiding is enforced twice — the schema filter **and** the dispatch path — so a
hallucinated call cannot reach a hidden tool either.

Descriptions are replaced with short, presentation-only one-liners that name the
interpreter or directory each tool uses (the server's own long prose is resent
on every request otherwise). The server remains the source of truth for names,
types and required arguments.

## One environment, three roles

```
harness interpreter (REPL, agent loop, MCP client)
        │  same .venv
        ├── mcp-python-interpreter console script (the server executable)
        └── interpreter used by run_python_code / run_python_file,
            the pip target and the package query
```

All three roles resolve to the same virtual environment. That is what keeps
`import pandas` from succeeding in one tool and failing in another: a package
installed once is importable from everywhere, and
`python_environment_info` verifies it by running code through the server rather
than by the harness guessing.

## Tool-call parsing and validation

Before any call reaches the server:

1. **Extraction** — native `tool_calls` are used first; otherwise two text
   protocols are parsed as a fallback (bare JSON object; MiniMax XML-ish
   serialization with its token-noise string). Prose is never rewritten, and a
   malformed or truncated text call fails the turn instead of executing half a
   call.
2. **Filter** — the enabled-tool check is enforced on the call path too.
3. **Argument validation** — the harness checks the call against the tool's
   advertised schema: unknown keys, missing required keys and plain JSON type
   mismatches are named explicitly so the model can self-correct. Only the
   simple JSON-Schema subset is checked; `$ref`/combinators are left to the
   server, which owns the real validation.
4. **Path normalisation** — a redundant leading `workspace/` is stripped from
   path arguments, and missing relative paths are answered with the workspace
   location and a listing of the nearest existing directory, instead of a
   generic server error.
5. **Result formatting** — the MCP result is flattened to text, and long output
   is middle-truncated (see
   [token budget](token-budget-and-compaction.md)).

## Package installation

The agent cannot install anything on its own say-so. `install_package` is
intercepted in the dispatch path and routed through `src/environment.py`, which:

1. validates the argument (`name` or `name==version`; URLs, local paths, extras
   and pip flags are rejected — the requirement is passed to pip as a single
   argv element, never through a shell);
2. refuses to target a global interpreter;
3. asks the person at the terminal (`src/approval.py`), showing the package and
   the target interpreter. Anything but an explicit `y`/`yes` is a no; a session
   without an interactive terminal denies by default;
4. installs through the MCP server into the environment that runs agent code;
5. verifies the import with that same interpreter.

`MCP_INSTALL_APPROVAL` controls step 3: `ask` (default) prompts on the terminal;
`deny` refuses every request without prompting. Upgrades are refused on purpose
(the agent must name an explicit version, so unrelated packages are never
silently replaced).

Import verification is careful about names: distribution names and import names
are different namespaces (`scikit-learn` → `sklearn`, `python-dotenv` →
`dotenv`). Installed metadata is asked for its real top-level modules first,
with a small explicit table and a legacy `-`→`_` transform as fallbacks. A wrong
guess is reported as *unverified*, never as an installation failure.

## Workspace behaviour and limitations

The MCP server is launched with `--dir` pointing at `workspace/`, so:

- tool paths are workspace-relative;
- `run_python_code` starts in the workspace directory;
- `memory.md` lives in the workspace so the agent can read it with its own
  tools (but the harness is the writer).

**`workspace/` is a working directory, not a security boundary.**
`run_python_code` executes real Python with the host user's filesystem and
network access. The `workspace/` restriction holds only for the file tools'
*default* path resolution — it is a convention, not OS policy. For an OS-level
sandbox, wrap the MCP server with a launcher you trust:

```dotenv
MCP_SANDBOX_COMMAND=bwrap
MCP_SANDBOX_ARGS=["--ro-bind","/","/","--bind","/abs/path/workspace","/abs/path/workspace","--unshare-net","--"]
```

The launcher and its arguments are prepended to the server command, so any
program of the form `launcher [args] command [args]` works. Without one, the
tools run unconfined.
