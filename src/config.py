import json
import os
import sys
from pathlib import Path
from dotenv import dotenv_values

# Project root directories
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"


def _load_env(path: Path) -> None:
    """Loads the project's .env with predictable precedence.

    python-dotenv's default search starts at the CWD and never overrides an
    existing variable, so a shell-level LLM_PROVIDER silently masked the file and
    every os.getenv fell back to defaults. Here a non-blank value in .env wins; a
    blank assignment (AI_API_KEY=) means "not configured here" and leaves the
    shell value in place.
    """
    if not path.exists():
        return
    for key, value in dotenv_values(path).items():
        if value:
            os.environ[key] = value


_load_env(ENV_FILE)
WORKSPACE_DIR = ROOT_DIR / "workspace"
# Memory lives inside the workspace so the agent's own tools can read it; the
# harness writes it directly (it is not an MCP call). Conversation history stays
# outside the workspace so the agent cannot rewrite it.
MEMORY_FILE = WORKSPACE_DIR / "memory.md"
SESSION_FILE = ROOT_DIR / ".agent" / "session.json"
# Last known-good memory, written before every save. The agent can clobber
# memory.md with write_file; this copy is the way back.
MEMORY_BACKUP = SESSION_FILE.parent / "memory.bak"
# Per-model daily request/token counters (Groq's RPD and TPD are per day).
USAGE_FILE = SESSION_FILE.parent / "usage.json"

# Ensure workspace and memory exist
WORKSPACE_DIR.mkdir(exist_ok=True)
if not MEMORY_FILE.exists():
    MEMORY_FILE.write_text("# Agent Memory\n")

# LLM Configurations
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:coder")
# One key for the remote provider, whatever OpenAI-compatible endpoint it talks
# to. The legacy OPENAI_API_KEY spelling is still read so an existing shell
# export keeps working; AI_API_KEY wins when both are set.
AI_API_KEY = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")
# Model id for the remote provider, using that provider's own naming.
AI_MODEL = os.getenv("AI_MODEL") or "openai/gpt-oss-120b"

# ----------------------------------------------------------------- Tool backend
# Every file/execution tool is served by an MCP server over stdio. The only
# in-process tool is save_memory (harness state), implemented in src/memory.py.

# The MCP server ships its own dependency set (it needs mcp<2), so it may live in a
# dedicated virtualenv. It must be launched through its console script: the package
# has no __main__, so `python -m mcp_python_interpreter` is not supported upstream.
_SERVER_VENV_BIN = (
    ROOT_DIR / ".venv-mcp" / "Scripts",  # Windows
    ROOT_DIR / ".venv-mcp" / "bin",      # POSIX
)
_SERVER_SCRIPT = "mcp-python-interpreter.exe" if os.name == "nt" else "mcp-python-interpreter"
_DEFAULT_SERVER_COMMAND = next(
    (str(d / _SERVER_SCRIPT) for d in _SERVER_VENV_BIN if (d / _SERVER_SCRIPT).exists()),
    _SERVER_SCRIPT,
)

MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND", _DEFAULT_SERVER_COMMAND)
MCP_SERVER_ARGS = json.loads(os.getenv("MCP_SERVER_ARGS", "null")) or [
    "--dir", str(WORKSPACE_DIR),
    "--python-path", sys.executable,
]
MCP_CALL_TIMEOUT = float(os.getenv("MCP_CALL_TIMEOUT", "300"))
MCP_STARTUP_TIMEOUT = float(os.getenv("MCP_STARTUP_TIMEOUT", "60"))

# ------------------------------------------------------- Token budget / LLM
# Groq's free tier for openai/gpt-oss-120b is ~8K tokens per rolling minute,
# counting input AND output. A request whose input does not fit is rejected with
# 413; a drained window throttles the rest with 429. These caps exist so a whole
# coding turn stays inside one bucket.
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1200"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "6500"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Enforced before a request is built: one tool result, the memory block, and how
# many recent tool exchanges are replayed verbatim.
TOOL_OUTPUT_MAX_CHARS = int(os.getenv("TOOL_OUTPUT_MAX_CHARS", "1600"))
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "1200"))
HISTORY_KEEP_EXCHANGES = int(os.getenv("HISTORY_KEEP_EXCHANGES", "2"))
# Auto-compaction: after this many user turns in the session, older tool
# exchanges are collapsed before the next request. It also fires early when the
# history alone would take more than AUTOCOMPACT_TOKEN_FRACTION of the budget.
AUTOCOMPACT_AFTER_TURNS = int(os.getenv("AUTOCOMPACT_AFTER_TURNS", "3"))
AUTOCOMPACT_TOKEN_FRACTION = float(os.getenv("AUTOCOMPACT_TOKEN_FRACTION", "0.6"))
DEBUG_TOKENS = os.getenv("DEBUG_TOKENS", "0") == "1"

# Ollama has no tokens-per-minute ceiling, only a context window, so local runs
# use a much larger budget (qwen2.5-coder is commonly served with 32K context).
OLLAMA_MAX_INPUT_TOKENS = int(os.getenv("OLLAMA_MAX_INPUT_TOKENS", "32000"))