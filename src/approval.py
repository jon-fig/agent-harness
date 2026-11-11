"""Harness-enforced approval for tool calls that change the environment.

A model writing "may I install pandas?" is not approval: it can skip the question
and call the tool on the next iteration, and nothing else in the harness would
stop it. This module is the gate in the tool-dispatch path, so an install reaches
the MCP server only after the person at the terminal says yes.

The prompt runs where the agent runs (the REPL owns the terminal), and a session
without one denies by default instead of installing unattended.
"""

from __future__ import annotations

import sys
from typing import Callable, TextIO

from src import config

# input() only gets line editing when the readline module is loaded, and that
# happens automatically only in the interactive REPL. The approval prompt can be
# the first thing the user types into (the REPL owns the terminal), so import it
# here too; on Windows the console provides its own editing and the missing
# module is ignored.
try:
    import readline  # noqa: F401
except ImportError:  # pragma: no cover - Windows console editing stays native
    pass

# A programmatic approver (tests, or an embedding UI) replaces the terminal
# prompt. set_approver(None) restores it.
_approver: Callable[[str], bool] | None = None

_AFFIRMATIVE = {"y", "yes"}
_DENY_VALUES = {"deny", "never", "no", "off", "0", "false"}

# Messages are kept here so the wording stays consistent with what the model is
# told and what a user sees on the terminal.
NO_TERMINAL_NOTE = (
    "[Approval] No interactive terminal is attached, so the request was denied. "
    "Run the agent from a terminal to approve installs, or set "
    "MCP_INSTALL_APPROVAL=deny to stop asking."
)


def set_approver(approver: Callable[[str], bool] | None) -> None:
    """Installs (or removes) a programmatic approver."""
    global _approver
    _approver = approver


def denied_by_config() -> bool:
    """True when MCP_INSTALL_APPROVAL turns every request down."""
    return config.MCP_INSTALL_APPROVAL in _DENY_VALUES


def request(prompt: str, stream: TextIO | None = None) -> bool:
    """Asks for approval of one action; anything but an explicit yes is a no."""
    out = stream or sys.stdout
    if denied_by_config():
        print(
            "[Approval] Denied: MCP_INSTALL_APPROVAL is set to deny.",
            file=out,
            flush=True,
        )
        return False
    if _approver is not None:
        return bool(_approver(prompt))

    print(prompt, file=out, flush=True)
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            print(NO_TERMINAL_NOTE, file=out, flush=True)
            return False
        answer = input("Approve? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print(file=out, flush=True)
        return False
    return answer.strip().lower() in _AFFIRMATIVE
