"""
MCP bridge.

Spawns a Model Context Protocol server as a child process (stdio transport) and
exposes the tools it advertises to the synchronous agent loop:

    bridge = get_bridge()                 # started once per process
    bridge.openai_schemas()               # tools/list -> OpenAI function schemas
    bridge.call("read_file", {...})       # tools/call -> text result

The agent loop is synchronous, so the MCP session lives on its own asyncio
event loop inside a background thread and each call is marshalled across with
asyncio.run_coroutine_threadsafe().

Run this module directly to inspect what the server actually exposes:

    python -m src.mcp_client --list
    python -m src.mcp_client --call read_file '{"file_path": "notes.txt"}'
"""

import argparse
import asyncio
import atexit
import concurrent.futures
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

from src.config import (
    MCP_CALL_TIMEOUT,
    MCP_SERVER_ARGS,
    MCP_SERVER_COMMAND,
    MCP_STARTUP_TIMEOUT,
    ROOT_DIR,
    TOOL_OUTPUT_MAX_CHARS,
)

logger = logging.getLogger(__name__)


def _describe_exception(exc: BaseException) -> str:
    """Flattens an exception group into its leaf messages.

    anyio reports a failed server startup as "unhandled errors in a TaskGroup (1
    sub-exception)", which says nothing about the actual cause (a missing
    executable, a server that exited during initialize(), a protocol mismatch).
    Only the leaves of the group carry that information, so they are named here.
    """
    leaves: list[str] = []
    queue: list[BaseException] = [exc]
    while queue:
        current = queue.pop(0)
        inner = getattr(current, "exceptions", None)
        if inner:
            queue[0:0] = list(inner)
            continue
        text = str(current).strip()
        leaves.append(f"{type(current).__name__}: {text}" if text else type(current).__name__)
    return "; ".join(leaves) or type(exc).__name__


def _truncate_middle(text: str, limit: int = TOOL_OUTPUT_MAX_CHARS) -> str:
    """Caps one tool result while keeping both ends.

    The head carries the column names or the top of the output, the tail carries
    the traceback or the last rows, so the middle is what gets dropped. This is
    the single point where tool output enters the chat history, which makes it the
    only place a 20 KB CSV read can be stopped from eating a whole turn.
    """
    if len(text) <= limit:
        return text

    head = (limit * 2) // 3
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n…[ {omitted} chars omitted ]…\n{text[-tail:]}"


def _format_result(result: Any) -> str:
    """Flattens an MCP CallToolResult into a single string for the chat history."""
    parts: list[str] = []

    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
            continue

        data = getattr(block, "data", None)
        if data is not None:
            parts.append(f"[binary content omitted: {len(data)} bytes]")
            continue

        parts.append(str(block))

    if not parts:
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            # No indent: pretty-printing structured results costs ~25% more tokens.
            parts.append(json.dumps(structured, default=str))

    return _truncate_middle("\n".join(parts).strip())


class MCPToolBridge:
    """Synchronous facade over one MCP server connection."""

    def __init__(
        self,
        command: str,
        args: list[str],
        cwd: str | None = None,
        call_timeout: float = 300.0,
        startup_timeout: float = 60.0,
    ) -> None:
        self.command = command
        self.args = list(args)
        self.cwd = cwd
        self.call_timeout = call_timeout
        self.startup_timeout = startup_timeout

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._session: ClientSession | None = None
        self._tools: list[Any] = []
        self._stop: asyncio.Event | None = None
        self._closed = False
        # A server that dies during initialize() says why on its stderr, and that
        # is the only place the reason exists: the stdio client only reports a
        # closed connection. Collected in a temp file (a real fd is required for
        # the child's stderr) and quoted when startup fails.
        self._errlog = tempfile.TemporaryFile(mode="w+b")

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> "MCPToolBridge":
        if self._thread is not None:
            return self

        # A mistyped MCP_SERVER_COMMAND, or a clone that never created the server
        # venv, otherwise surfaces as an opaque TaskGroup error from the client.
        if not os.path.exists(self.command) and shutil.which(self.command) is None:
            raise RuntimeError(
                f"MCP server executable not found: {self.command}. Create the server venv: "
                "python -m venv .venv-mcp && "
                ".venv-mcp/Scripts/python.exe -m pip install 'mcp<2' mcp-python-interpreter "
                "(or point MCP_SERVER_COMMAND at an existing install)."
            )

        self._thread = threading.Thread(target=self._run_loop, name="mcp-bridge", daemon=True)
        self._thread.start()

        if not self._ready.wait(self.startup_timeout):
            raise RuntimeError(
                f"MCP server did not become ready within {self.startup_timeout:.0f}s: "
                f"{self.command} {' '.join(self.args)}"
            )
        if self._startup_error is not None:
            detail = _describe_exception(self._startup_error)
            stderr = self._server_stderr()
            if stderr:
                detail = f"{detail}. Server stderr:\n{stderr}"
            else:
                # Nothing on stderr means the process died before Python could say
                # anything: a broken launcher, a missing DLL, a denied spawn.
                detail = f"{detail}. {self._probe_server()}"
            raise RuntimeError(f"MCP server failed to start ({self.command}): {detail}") from self._startup_error
        return self

    def _probe_server(self) -> str:
        """Runs the server once outside the stdio client to see if it can start.

        The probe uses the same executable, cwd and environment the client passes,
        so a non-zero exit here separates "the executable cannot run" from "the
        handshake failed", which a closed connection alone never does.
        """
        try:
            probe = subprocess.run(
                [self.command, *self.args, "--help"],
                cwd=self.cwd,
                env=get_default_environment(),
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"It could not be run directly either: {type(exc).__name__}: {exc}"

        if probe.returncode == 0:
            return (
                "It does start when run directly with --help, so the failure is in the "
                "stdio handshake rather than the executable."
            )
        output = (probe.stderr or probe.stdout or "").strip()
        return f"Run directly it exits {probe.returncode}: {output[-800:] or '(no output)'}"

    def _server_stderr(self, limit: int = 2000) -> str:
        """Returns the tail of what the server process wrote to stderr."""
        try:
            self._errlog.seek(0)
            text = self._errlog.read().decode("utf-8", "replace").strip()
        except (OSError, ValueError):
            return ""
        if len(text) > limit:
            text = "…" + text[-limit:]
        return text

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except BaseException as exc:  # noqa: BLE001 - surfaced to start()
            self._startup_error = exc
            self._ready.set()
        finally:
            try:
                self._loop.close()
            except Exception:  # pragma: no cover - best effort shutdown
                pass

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        params = StdioServerParameters(command=self.command, args=self.args, cwd=self.cwd)

        try:
            async with stdio_client(params, errlog=self._errlog) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self._tools = list(listed.tools)
                    self._session = session

                    names = ", ".join(t.name for t in self._tools) or "(none)"
                    logger.debug("MCP server ready with tools: %s", names)
                    self._ready.set()

                    await self._stop.wait()
        except BaseException as exc:  # noqa: BLE001
            if not self._ready.is_set():
                self._startup_error = exc
                self._ready.set()
            else:
                logger.error("MCP session ended: %s%s", exc, f"\n{self._server_stderr()}" if self._server_stderr() else "")
            return

        self._session = None

    def close(self) -> None:
        global _bridge

        if self._closed:
            return
        self._closed = True

        if self._loop is not None and self._stop is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:
                pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

        # Drop the references so a closed bridge cannot be handed out again: a
        # restart used to keep returning the dead object because the module
        # global was never cleared.
        self._session = None
        self._loop = None
        self._stop = None
        self._tools = []
        self._errlog.close()
        if _bridge is self:
            _bridge = None

    # --------------------------------------------------------------------- tools

    @property
    def tools(self) -> list[Any]:
        return list(self._tools)

    @property
    def tool_names(self) -> set[str]:
        return {t.name for t in self._tools}

    def openai_schemas(self) -> list[dict]:
        """Converts MCP tool definitions into OpenAI `tools=` entries."""
        schemas = []
        for tool in self._tools:
            description = (getattr(tool, "description", None) or "").strip()
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": description,
                        "parameters": getattr(tool, "input_schema", None)
                        or {"type": "object", "properties": {}, "required": []},
                    },
                }
            )
        return schemas

    def call(self, name: str, arguments: dict | None = None) -> str:
        """Calls one MCP tool and returns its result as text."""
        if self._session is None or self._loop is None:
            raise RuntimeError("MCP bridge is not running. Call start() first.")

        future = asyncio.run_coroutine_threadsafe(
            self._call_tool(name, arguments or {}), self._loop
        )
        try:
            return future.result(self.call_timeout)
        except concurrent.futures.TimeoutError as exc:
            # Cancelling is best-effort (a running tool may ignore it) but it
            # stops the abandoned coroutine from holding the session open and
            # making every later call queue behind it.
            future.cancel()
            raise RuntimeError(
                f"MCP tool '{name}' did not finish within {self.call_timeout:.0f}s."
            ) from exc

    async def _call_tool(self, name: str, arguments: dict) -> str:
        result = await self._session.call_tool(name, arguments)
        text = _format_result(result)
        if getattr(result, "is_error", False):
            return f"Error from MCP tool '{name}': {text}"
        return text or f"MCP tool '{name}' completed with no output."


_bridge: MCPToolBridge | None = None
_bridge_lock = threading.Lock()


def get_bridge() -> MCPToolBridge:
    """Returns the process-wide MCP bridge, starting the server on first use."""
    global _bridge

    with _bridge_lock:
        if _bridge is None:
            bridge = MCPToolBridge(
                command=MCP_SERVER_COMMAND,
                args=MCP_SERVER_ARGS,
                cwd=str(ROOT_DIR),
                call_timeout=MCP_CALL_TIMEOUT,
                startup_timeout=MCP_STARTUP_TIMEOUT,
            )
            bridge.start()
            atexit.register(bridge.close)
            _bridge = bridge
        return _bridge


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the MCP server used by this harness.")
    parser.add_argument("--list", action="store_true", help="list discovered tools (default)")
    parser.add_argument(
        "--call",
        nargs=2,
        metavar=("TOOL", "JSON_ARGS"),
        help="call one tool, e.g. --call read_file '{\"file_path\": \"notes.txt\"}'",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    bridge = get_bridge()
    print(f"Server: {bridge.command} {' '.join(bridge.args)}")

    if args.call:
        name, raw = args.call
        print(bridge.call(name, json.loads(raw)))
    else:
        schemas = bridge.openai_schemas()
        print(f"Discovered {len(schemas)} tools:\n")
        for schema in schemas:
            fn = schema["function"]
            params = ", ".join((fn["parameters"] or {}).get("properties", {}))
            first_line = (fn["description"] or "").splitlines()[0] if fn["description"] else ""
            print(f"- {fn['name']}({params})")
            if first_line:
                print(f"    {first_line}")

    bridge.close()


if __name__ == "__main__":
    _main()
