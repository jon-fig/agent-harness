"""
Manages reading from and appending to the persistent memory file (memory.md).
"""

import os
from src.config import MEMORY_FILE

def load_memory() -> str:
    """Reads and returns the full content of the persistent memory file."""
    try:
        if MEMORY_FILE.exists():
            return MEMORY_FILE.read_text(encoding="utf-8")
        return "# Agent Memory\n"
    except Exception as e:
        return f"Error reading memory file: {str(e)}"

def save_memory(note: str) -> str:
    """Appends a new note or detail to the persistent memory file."""
    try:
        content = load_memory()
        updated_content = content.strip() + f"\n- {note}\n"
        MEMORY_FILE.write_text(updated_content, encoding="utf-8")
        return f"Success: Added note to memory: '{note}'"
    except Exception as e:
        return f"Error updating memory file: {str(e)}"