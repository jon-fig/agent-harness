"""
Persisted conversation state.

The whole history is resent on every iteration, so it is also the main token
cost. Keeping it on disk means a restart does not lose context, and that
``/clear`` and ``/compact`` operate on something durable.
"""

import json
import time

from src.config import SESSION_FILE


def load_session() -> list:
    """Returns the saved messages, or [] when there is none or it is unreadable."""
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
    SESSION_FILE.unlink(missing_ok=True)
