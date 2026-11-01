"""
Defines the OpenAI-compatible JSON function schemas for our sandboxed tools.
These schemas tell the LLM what tools exist, what they do, and what parameters they expect.
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Lists all files currently present inside the sandboxed workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads and returns the text content of a specified file inside the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative path of the file to read (e.g., 'notes.txt' or 'subfolder/script.py').",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Creates or overwrites a file inside the workspace directory with the provided text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative path where the file should be written (e.g., 'output.txt').",
                    },
                    "content": {
                        "type": "string",
                        "description": "The exact text content to write into the file.",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
    "type": "function",
    "function": {
        "name": "save_memory",
        "description": "Saves an important note, user preference, or insight to persistent memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory": {
                    "type": "string",
                    "description": "The specific detail or insight to remember."
                }
            },
            "required": ["memory"]
        }
    }
},
]