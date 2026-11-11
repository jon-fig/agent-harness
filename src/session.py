"""
Persisted conversation state.

The whole history is resent on every iteration, so it is also the main token
cost. Keeping it on disk means a restart does not lose context, and that
``/clear`` and ``/compact`` operate on something durable.
"""

import json
import time

from src.config import SESSION_FILE

# Normal launches load and save the conversation. A `--no-session` launch turns
# both off at this boundary, so no caller (the REPL, /compact, /clear or the exit
# path) can overwrite the user's real session just because it did not know about
# the flag. The flag is per process: main() sets it once, before anything reads
# or writes a session file.
_enabled = True


def set_session_enabled(enabled: bool) -> None:
    """Enables or disables reading *and* writing the persisted conversation."""
    global _enabled
    _enabled = bool(enabled)


def session_enabled() -> bool:
    return _enabled


def load_session() -> list:
    """Returns the saved messages, or [] when there is none or it is unreadable."""
    if not _enabled:
        return []
    if not SESSION_FILE.exists():
        return []
    try:
        data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    messages = data.get("messages") if isinstance(data, dict) else data
    return messages if isinstance(messages, list) else []


def save_session(messages: list) -> None:
    """Writes the history atomically: a crash mid-write keeps the previous file."""
    if not _enabled:
        return
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "saved_at": time.time(),
        "messages": messages,
    }
    temporary = SESSION_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8"
    )
    temporary.replace(SESSION_FILE)


def clear_session_file() -> None:
    if not _enabled:
        return
    SESSION_FILE.unlink(missing_ok=True)
