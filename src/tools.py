import os, subprocess
from pathlib import Path
from src.config import WORKSPACE_DIR

def _get_safe_path(file_path: str) -> Path:
    """
    Ensures that any file path requested by the agent stays strictly 
    within the WORKSPACE_DIR sandbox, preventing directory traversal.
    """
    target_path = (WORKSPACE_DIR / file_path).resolve()
    if not str(target_path).startswith(str(WORKSPACE_DIR.resolve())):
        raise ValueError(f"Security Error: Access denied. Path '{file_path}' is outside the workspace sandbox.")
    return target_path

def list_files() -> str:
    """Lists all files inside the workspace directory."""
    try:
        files = []
        for item in WORKSPACE_DIR.rglob("*"):
            if item.is_file():
                files.append(str(item.relative_to(WORKSPACE_DIR)))
        
        if not files:
            return "The workspace directory is currently empty."
        
        return "Files present in workspace:\n" + "\n".join(f"- {f}" for f in files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

def read_file(file_path: str) -> str:
    """Reads and returns the text content of a file within the workspace."""
    try:
        safe_path = _get_safe_path(file_path)
        
        if not safe_path.exists():
            return f"Error: File '{file_path}' does not exist in the workspace."
        if not safe_path.is_file():
            return f"Error: '{file_path}' is a directory, not a file."
            
        content = safe_path.read_text(encoding="utf-8")
        
        # If the file is too big, truncate the preview sent to the LLM
        if len(content) > 2000:
            return f"[File truncated for size. Total length: {len(content)} chars]\nPreview:\n{content[:1500]}\n..."
        return content
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"

def write_file(file_path: str, content: str) -> str:
    """Creates or overwrites a file inside the workspace with provided content."""
    try:
        safe_path = _get_safe_path(file_path)
        
        # Automatically create nested subfolders if the agent specifies them
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        
        safe_path.write_text(content, encoding="utf-8")
        return f"Success: File '{file_path}' was successfully written."
    except Exception as e:
        return f"Error writing file '{file_path}': {str(e)}"

def run_python_script(script_path: str) -> str:
    """Execute a python script in the workspace and return its stdout/stderr with path fallback."""
    try:
        # Check if file exists directly, otherwise check inside 'workspace/' folder
        target_path = script_path
        if not os.path.exists(target_path):
            alt_path = os.path.join("workspace", script_path)
            if os.path.exists(alt_path):
                target_path = alt_path

        result = subprocess.run(
            ["python", target_path], 
            capture_output=True, 
            text=True, 
            timeout=15
        )
        output = f"STDOUT:\n{result.stdout}"
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output
    except Exception as e:
        return f"Execution error: {str(e)}"