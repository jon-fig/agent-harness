"""
CLI entrypoint for the custom agentic harness. 
Handles the user input loop, conversation history, and terminal output.
"""

from src.agent import run_agent

def main():
    print("==================================================")
    print("🤖 Custom Python AI Agent Harness Initialized")
    print("Type your request below, or type 'exit' / 'quit' to close.")
    print("==================================================\n")
    
    # Maintain conversation history across user prompts
    messages = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting agent harness. Goodbye!")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ["exit", "quit"]:
            print("\nShutting down agent harness. See you later!")
            break
            
        print("\nAgent is thinking...")
        
        # Run the agentic loop
        response = run_agent(user_input, messages)
        
        print(f"\nAgent:\n{response}")

if __name__ == "__main__":
    main()