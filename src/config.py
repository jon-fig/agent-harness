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
# Active model per provider, as {"nvidia": "vendor/model", ...}. Keeping it in
# one file means switching providers does not overwrite the other's choice.
MODEL_STATE_FILE = SESSION_FILE.parent / "models.json"

# Ensure workspace and memory exist
WORKSPACE_DIR.mkdir(exist_ok=True)
if not MEMORY_FILE.exists():
    MEMORY_FILE.write_text("# Agent Memory\n")

# LLM Configurations
# LLM_PROVIDER is any OpenAI-compatible provider name: a built-in preset
# (groq, nvidia, openai, openrouter, together, deepseek, mistral, cerebras,
# fireworks, perplexity, ollama) or an arbitrary name configured with
# NAME_BASE_URL / NAME_API_KEY / NAME_MODEL. See src/providers.py.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:coder")
# One key/URL/model block for whichever remote provider is active. A provider's
# own prefixed variables (NVIDIA_API_KEY, GROQ_BASE_URL, ...) take precedence.
AI_API_KEY = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")
AI_BASE_URL = os.getenv("AI_BASE_URL") or os.getenv("LLM_BASE_URL") or ""
# Model id for the active provider, using that provider's own naming. When this
# is empty the provider preset supplies a default.
AI_MODEL = os.getenv("AI_MODEL") or ""

# ----------------------------------------------------------------- Tool backend
# Every file/execution tool is served by an MCP server over stdio. The two
# in-process tools are save_memory (harness state) and python_environment_info.
#
# One virtual environment serves the whole agent. The server is a console script
# installed next to the interpreter that runs the harness, so the harness
# interpreter, the server executable, the interpreter used by run_python_code /
# run_python_file, the pip target and the package query all name the same
# environment (`.venv` by default). That is what keeps `import pandas` from
# succeeding in one tool and failing in another.
#
# It must be launched through its console script: the package has no __main__, so
# `python -m mcp_python_interpreter` is not supported upstream.
SERVER_SCRIPT_NAME = "mcp-python-interpreter"
# The interpreter the tools execute code with, and the one pip installs into.
MCP_PYTHON = sys.executable
# Virtualenv layout: Scripts\ on Windows, bin/ everywhere else.
BIN_DIR_NAME = "Scripts" if os.name == "nt" else "bin"


def environment_bin_dir(prefix: str | os.PathLike[str]) -> Path:
    """The script directory of a virtual environment on this platform."""
    return Path(prefix) / BIN_DIR_NAME


def project_python() -> Path:
    """The interpreter of the project's documented ``.venv`` on this platform."""
    name = "python.exe" if os.name == "nt" else "python"
    return environment_bin_dir(ROOT_DIR / ".venv") / name


def interpreter_problem() -> str | None:
    """Explains why this interpreter cannot serve the harness, or returns None.

    The harness and the MCP server must share one virtual environment. A system
    interpreter would load the client from one site-packages while the server and
    the code tools used another, which is the mismatch the unified setup exists to
    prevent, so it is refused instead of silently repaired from ``PATH`` or from a
    different checkout's ``.venv``.
    """
    if sys.prefix != sys.base_prefix:
        return None
    return (
        "the harness is not running inside a virtual environment "
        f"(interpreter: {sys.executable}, prefix: {sys.prefix}). "
        "Start it with the environment that has requirements.txt installed: "
        f"\"{project_python()}\" -m src.main"
    )


def require_supported_interpreter() -> None:
    """Raises a clear error when the harness was started from a global Python."""
    problem = interpreter_problem()
    if problem:
        raise RuntimeError(problem)


def mcp_server_candidates() -> list[Path]:
    """The console-script path inside the active environment.

    Only ``sys.prefix`` is considered. Falling back to the project's ``.venv``
    used to launch a server from one environment while ``--python-path`` named
    another interpreter; if the active environment lacks the server, the bridge
    now says how to install it instead of starting a mismatched one.

    A bare command name is deliberately never used: resolving through PATH would
    launch a globally installed server whose site-packages differ from the ones
    run_python_code imports from.
    """
    name = f"{SERVER_SCRIPT_NAME}.exe" if os.name == "nt" else SERVER_SCRIPT_NAME
    return [environment_bin_dir(sys.prefix) / name]


def resolve_mcp_server_command() -> str:
    """The server executable inside an existing environment, or "" if none."""
    for candidate in mcp_server_candidates():
        if candidate.is_file():
            return str(candidate)
    return ""


MCP_SERVER_COMMAND = os.getenv("MCP_SERVER_COMMAND") or resolve_mcp_server_command()
MCP_SERVER_ARGS = json.loads(os.getenv("MCP_SERVER_ARGS", "null")) or [
    "--dir", str(WORKSPACE_DIR),
    "--python-path", MCP_PYTHON,
]
MCP_CALL_TIMEOUT = float(os.getenv("MCP_CALL_TIMEOUT", "300"))
MCP_STARTUP_TIMEOUT = float(os.getenv("MCP_STARTUP_TIMEOUT", "60"))
# Optional OS sandbox: this launcher and its arguments are prepended to the MCP
# server command, so bwrap/firejail/nsjail/container-exec can confine the tools.
MCP_SANDBOX_COMMAND = os.getenv("MCP_SANDBOX_COMMAND", "").strip()
MCP_SANDBOX_ARGS = json.loads(os.getenv("MCP_SANDBOX_ARGS") or "null") or []
# Package installation mutates the environment, so the harness itself asks before
# it runs: "ask" prompts on the terminal that started the agent, "deny" refuses
# every request. The model cannot approve its own install by writing about it.
MCP_INSTALL_APPROVAL = os.getenv("MCP_INSTALL_APPROVAL", "ask").strip().lower() or "ask"

# NOTE: every value in this module is read once, when the harness starts. Editing
# .env while the agent is running has no effect until it is restarted (the REPL's
# /provider and /model commands change the connection, not these limits).

# ------------------------------------------------------- Token budget / LLM
# Groq's free tier for openai/gpt-oss-120b is ~8K tokens per rolling minute,
# counting input AND output. A request whose input does not fit is rejected with
# 413; a drained window throttles the rest with 429. These caps exist so a whole
# coding turn stays inside one bucket.
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "8"))
# MAX_ITERATIONS bounds model calls; one response can request many tool calls, so
# executions get their own per-turn ceiling.
MAX_TOOL_CALLS_PER_TURN = int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "20"))
# Identical arguments do not guarantee identical results (a clock, a random seed,
# a file another call just wrote), so repeats execute by default. Set to 1 only
# for providers that loop on the same call.
DEDUPE_TOOL_CALLS = os.getenv("DEDUPE_TOOL_CALLS", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# The generation limit asked of the model. It is a *token* request and is clamped
# per model/provider by src/models.py (context window, TPM bucket), so a generic
# 4000 never overrides a lower model-specific maximum.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "4000"))
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "6500"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
# GET /models is a control-plane call, not a completion: a slow or blocked
# discovery endpoint should not hold up the REPL for the completion timeout.
MODEL_DISCOVERY_TIMEOUT = float(os.getenv("MODEL_DISCOVERY_TIMEOUT", "20"))

# Enforced before a request is built: one tool result, the memory block, and how
# many recent tool exchanges are replayed verbatim. TOOL_OUTPUT_MAX_CHARS is a
# *character* cap applied where tool output enters history; it is not a token
# limit and does not have to equal MAX_OUTPUT_TOKENS.
TOOL_OUTPUT_MAX_CHARS = int(os.getenv("TOOL_OUTPUT_MAX_CHARS", "4000"))
MEMORY_MAX_CHARS = int(os.getenv("MEMORY_MAX_CHARS", "1200"))
HISTORY_KEEP_EXCHANGES = int(os.getenv("HISTORY_KEEP_EXCHANGES", "2"))
# Auto-compaction is evaluated by the agent against the complete request (system
# prompt + tool schemas + history), not just the history text. It is eligible only
# from AUTO_COMPACT_MIN_TURNS user turns onward; it fires when the measured
# request exceeds AUTO_COMPACT_THRESHOLD and collapses older tool exchanges until
# it reaches AUTO_COMPACT_TARGET. A value <= 1 is a fraction of the input budget;
# a larger value is an absolute token count. The old AUTOCOMPACT_* spellings are
# still accepted as fallbacks.
def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "disabled")


AUTO_COMPACT = _env_flag("AUTO_COMPACT", True)
AUTO_COMPACT_MIN_TURNS = int(
    os.getenv("AUTO_COMPACT_MIN_TURNS")
    or os.getenv("AUTOCOMPACT_AFTER_TURNS")
    or "2"
)
AUTO_COMPACT_THRESHOLD = float(
    os.getenv("AUTO_COMPACT_THRESHOLD")
    or os.getenv("AUTOCOMPACT_TOKEN_FRACTION")
    or "0.6"
)
AUTO_COMPACT_TARGET = float(os.getenv("AUTO_COMPACT_TARGET", "0.5"))
# The chars/4 fallback self-calibrates from provider-reported prompt tokens so a
# model with an exotic tokenizer does not silently undercount the request. The
# learned ratios live in .agent/tokenizer.json; set this to 0 to keep it fixed.
TOKENIZER_CALIBRATION = _env_flag("TOKENIZER_CALIBRATION", True)
# Backwards-compatible names for any external reader; the new code uses the four
# settings above. Do not introduce new references to these.
AUTOCOMPACT_AFTER_TURNS = AUTO_COMPACT_MIN_TURNS
AUTOCOMPACT_TOKEN_FRACTION = AUTO_COMPACT_THRESHOLD
DEBUG_TOKENS = os.getenv("DEBUG_TOKENS", "0") == "1"

# Ollama has no tokens-per-minute ceiling, only a context window, so local runs
# use a much larger budget (qwen2.5-coder is commonly served with 32K context).
OLLAMA_MAX_INPUT_TOKENS = int(os.getenv("OLLAMA_MAX_INPUT_TOKENS", "32000"))