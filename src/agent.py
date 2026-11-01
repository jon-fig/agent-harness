"""
Core agentic loop module. Handles message history, tool execution, 
robust error handling for argument parsing, and iterative model reasoning.
"""

import json
from src.llm_client import get_llm_client
from src.tool_schemas import TOOL_SCHEMAS
from src.tools import list_files, read_file, write_file, run_python_script
from src.memory import load_memory, save_memory

# Map function name strings to actual Python callables
TOOL_MAP = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "save_memory": save_memory,
    "run_python_script": run_python_script,
}

def run_agent(user_prompt: str, messages: list) -> str:
    """
    Takes a user prompt and the conversation message history, then executes 
    the agentic loop (calling tools iteratively) until the model provides a final text response.
    """
    client, model_name = get_llm_client()
    
    # Append the new user message to the history
    messages.append({"role": "user", "content": user_prompt})

    # ---------------------------------------------------------
    # 1. DEFINE SAFETY LIMITS HERE
    # ---------------------------------------------------------
    max_iterations = 6
    current_iteration = 0
    
    # Agentic loop (allows multiple iterations of tool calls before finishing)
    while current_iteration < max_iterations:
        current_iteration += 1
        
        if current_iteration >= max_iterations:
            print("\n[Agent Warning] Reached maximum iteration limit (6 turns). Forcing exit.")
            return "I got stuck in a reasoning loop and had to stop execution."
        
        # Dynamically inject the latest persistent memory into the system prompt
        memory_content = load_memory()
        system_prompt = (
            f"""You are an autonomous, expert Data Science AI Agent operating inside a local workspace.
Your primary objective is to assist with data exploration, analysis, machine learning scripting, data cleaning, and reporting.

GUIDELINES & BEHAVIOR:
1. Workspace Scope: You operate inside a sandboxed workspace directory. Use your tools (`list_files`, `read_file`, `write_file`) to inspect, load, and edit datasets or scripts.
2. Tool Discipline: Always check the contents of a directory or file before making assumptions about schemas or columns.
3. Code Execution & Data Analysis: Write clean, vectorized Python code using standard data science libraries (pandas, numpy, scikit-learn).
4. Memory Management: Utilize `save_memory` to store crucial insight highlights, column definitions, or data stats so they persist across user turns.
5. Precision: Be explicit, accurate, and concise in your explanations. Never invent or summarize columns you haven't explicitly verified via file inspection.
6. Never use read_file to read large CSV datasets directly into the chat. Always write a Python script using pandas to inspect, query, or process CSV files locally.

--- Persistent Memory ---
{memory_content}"""
        )

        # Construct the payload message stack (System prompt + full conversation history)
        payload_messages = [{"role": "system", "content": system_prompt}] + messages
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=payload_messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto"
            )
        except Exception as e:
            error_msg = f"Error communicating with LLM provider: {str(e)}"
            print(f"[Agent Error] {error_msg}")
            return error_msg
        
        response_message = response.choices[0].message
        content = response_message.content or ""
        
        # Check native tool_calls first
        detected_tool_calls = response_message.tool_calls
        
        # Fallback: Parse raw JSON text if model didn't use native tool calls
        if not detected_tool_calls:
            cleaned_content = content.strip()
            
            # Strip markdown code blocks (e.g., ```json ... ```)
            if cleaned_content.startswith("```"):
                lines = cleaned_content.splitlines()
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned_content = "\n".join(lines).strip()
            
            if cleaned_content.startswith("{"):
                try:
                    parsed_json = json.loads(cleaned_content)
                    func_name = parsed_json.get("name", "")
                    
                    # Ignore template placeholders or empty names
                    if func_name and not func_name.startswith("<"):
                        print(f"[Agent Info] Captured text-based tool call -> '{func_name}'")
                        
                        class MockFunction:
                            def __init__(self, name, arguments):
                                self.name = name
                                self.arguments = json.dumps(arguments) if isinstance(arguments, dict) else arguments
                        
                        class MockToolCall:
                            def __init__(self, name, arguments):
                                self.id = "call_text_fallback_1"
                                self.function = MockFunction(name, arguments.get("arguments", arguments))

                        detected_tool_calls = [MockToolCall(func_name, parsed_json.get("arguments", {}))]
                except json.JSONDecodeError:
                    pass

        # Execute tools if detected (native or fallback)
        if detected_tool_calls:
            # Append the assistant's response to history so the model remembers its action
            messages.append(response_message)
            
            for tool_call in detected_tool_calls:
                function_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments
                
                print(f"\n[Agent Action] Executing tool -> '{function_name}'")
                
                try:
                    arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else (raw_arguments or {})
                except json.JSONDecodeError as jde:
                    error_feedback = f"Error: Failed to parse tool arguments as valid JSON: {str(jde)}."
                    print(f"[Agent Warning] {error_feedback}")
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": error_feedback})
                    continue
                
                # Execute the corresponding Python tool function safely
                if function_name in TOOL_MAP:
                    try:
                        tool_result = TOOL_MAP[function_name](**arguments)
                    except Exception as tool_exc:
                        tool_result = f"Error executing Python function '{function_name}': {str(tool_exc)}"
                else:
                    tool_result = f"Error: Tool '{function_name}' does not exist in the environment."
                
                preview = str(tool_result)[:150].replace('\n', ' ')
                print(f"[Agent Result] Output -> {preview}...")
                
                # Feed the tool execution result back into the loop
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })
        else:
            # No tool calls required; model returned a final text answer to the user
            messages.append({"role": "assistant", "content": content})
            return content