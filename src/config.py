import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root directories
ROOT_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = ROOT_DIR / "workspace"
MEMORY_FILE = ROOT_DIR / "memory.md"

# Ensure workspace and memory exist
WORKSPACE_DIR.mkdir(exist_ok=True)
if not MEMORY_FILE.exists():
    MEMORY_FILE.write_text("# Agent Memory\n\n- User preferences and notes will go here.\n")

# LLM Configurations
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:coder")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")