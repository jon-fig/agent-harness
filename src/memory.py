"""
Persistent memory: a capped, human-readable note file.

The file lives in ``workspace/memory.md``, so the agent can open it with its own
tools, but the harness is the only writer (``save_memory`` is an in-process tool,
not an MCP call). Memory is injected into the system prompt on every request, so
its size is a recurring token cost: the writer caps it at ``MEMORY_MAX_CHARS``
(~300 tokens) by dropping the oldest notes, never by cutting a note in half.
"""

import os

from src.config import MEMORY_BACKUP, MEMORY_FILE, MEMORY_MAX_CHARS

HEADER = "# Agent Memory"
MAX_NOTE_CHARS = 400


def _read() -> str:
    try:
        return MEMORY_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""


def _notes(content: str) -> list[str]:
    return [line.strip() for line in content.splitlines() if line.strip().startswith("-")]


def _render(notes: list[str]) -> str:
    return HEADER + "\n\n" + "\n".join(notes) + "\n"


def backup_current() -> bool:
    """Copies the memory file aside before something else rewrites it.

    The agent can overwrite ``memory.md`` with its own ``write_file``, which would
    silently destroy everything it had learned. The backup is the way back: copy
    ``.agent/memory.bak`` over ``workspace/memory.md``.
    """
    content = _read()
    if not content.strip():
        return False
    try:
        MEMORY_BACKUP.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_BACKUP.write_text(content, encoding="utf-8")
        return True
    except OSError:
        return False


def load_memory() -> str:
    """The full memory block, header included, even for a missing/empty file."""
    content = _read()
    notes = _notes(content)
    if not notes:
        return HEADER + "\n"
    return _render(notes)


def save_memory(memory: str) -> str:
    """Appends one note, keeping the file capped and the header intact.

    Writes atomically (temp file + os.replace): a crash mid-write must not leave a
    truncated memory behind, since it would be replayed on every request.
    """
    note = " ".join(str(memory or "").split())
    if not note:
        return "Error: nothing to save (the note was empty)."
    if len(note) > MAX_NOTE_CHARS:
        note = note[:MAX_NOTE_CHARS].rstrip() + "…"

    previous = _read()
    notes = _notes(previous)
    notes.append(f"- {note}")

    dropped = 0
    rendered = _render(notes)
    while len(rendered) > MEMORY_MAX_CHARS and len(notes) > 1:
        notes.pop(0)  # oldest first: recent context is the useful part
        dropped += 1
        rendered = _render(notes)

    if len(rendered) > MEMORY_MAX_CHARS:  # a single note larger than the cap
        rendered = _render(notes)[:MEMORY_MAX_CHARS].rstrip() + "\n"

    if previous.strip():
        backup_current()

    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = MEMORY_FILE.with_suffix(".md.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.replace(temporary, MEMORY_FILE)
    except OSError as exc:
        return f"Error writing memory file: {exc}"

    suffix = f", {dropped} oldest note(s) dropped" if dropped else ""
    # Deliberately does not echo the note back: that is pure repeated input tokens.
    return f"Saved to memory ({len(notes)} notes, {len(rendered)}/{MEMORY_MAX_CHARS} chars{suffix})."
