"""
CLI entrypoint for the custom agentic harness.

Keeps the conversation across prompts, persists it to .agent/session.json so long
work survives a restart, and provides /clear and /compact because the history is
the main input-token cost on a metered provider.
"""

import argparse

from src import config, models, usage
from src.agent import AgentError, run_agent
from src.budget import compact_history, estimate_tokens
from src.session import clear_session_file, load_session, save_session

HELP = """Commands:
  /model            show the active model and the bundled options
  /model <id>       switch model (/model 20b, /model qwen, or a full Groq id)
  /clear            forget the conversation (and delete the saved session)
  /compact          collapse older tool exchanges now, keeping the latest ones verbatim
  /help             show this help
  exit              quit (history is saved automatically)"""


def _describe(messages: list) -> str:
    tokens = estimate_tokens(messages) if messages else 0
    return f"{len(messages)} messages, ~{tokens} tokens"


def _auto_compact(messages: list, provider: str) -> None:
    """Collapses old tool exchanges once a session gets expensive.

    Two triggers, whichever comes first: the configured turn count (default 3) or
    the history passing a fraction of the input budget. On an 8K TPM free tier the
    token trigger usually fires first, because a single turn of tool output can
    cost more than several ordinary turns.
    """
    if not messages:
        return

    turns = sum(1 for message in messages if message.get("role") == "user") + 1
    budget = (
        config.OLLAMA_MAX_INPUT_TOKENS
        if provider == "ollama"
        else models.input_budget(models.active())
    )
    before_tokens = estimate_tokens(messages)
    threshold = budget * config.AUTOCOMPACT_TOKEN_FRACTION
    by_turns = turns >= config.AUTOCOMPACT_AFTER_TURNS
    if not by_turns and before_tokens <= threshold:
        return

    messages[:] = compact_history(messages)
    after_tokens = estimate_tokens(messages)
    if after_tokens >= before_tokens:
        return  # nothing old enough to collapse

    reason = f"turn {turns}" if by_turns else f"~{before_tokens} history tokens > {threshold:.0f}"
    print(
        f"[Session] Auto-compacted ({reason}): ~{before_tokens} -> ~{after_tokens} tokens."
    )


def _handle_model(argument: str, provider: str) -> None:
    """`/model` with no argument reports; with an argument it switches."""
    if provider == "ollama":
        if not argument:
            print(f"[Model] Ollama model: {config.OLLAMA_MODEL}")
            print("        Switch with /model <name>; Groq models need --provider groq.")
            return
        config.OLLAMA_MODEL = argument
        print(f"[Model] Ollama model -> {argument} (applies from the next turn)")
        return

    if not argument:
        print(models.describe_active())
        print(f"        {usage.describe(models.active())}")
        print(models.table())
        return

    try:
        model_id = models.set_active(argument)
    except ValueError as exc:
        print(f"[Model] {exc}. Type /model to list options.")
        return

    spec = models.spec_for(model_id)
    if spec is None:
        print(
            f"[Model] Active -> {model_id} (not in the bundled catalog; "
            f"budget falls back to {config.MAX_INPUT_TOKENS} tokens/request)."
        )
        return

    print(
        f"[Model] Active -> {spec.id} | input budget {models.input_budget(model_id)} "
        f"tokens/request | {spec.tpm:,} TPM"
    )
    print(f"        {usage.describe(model_id)}")
    print("        Pacing state is per model, so switching does not inherit a used window.")


def main():
    parser = argparse.ArgumentParser(description="Custom Python AI agent harness")
    parser.add_argument(
        "--debug-tokens",
        action="store_true",
        help="log approximate vs. reported token usage and rate-limit headroom",
    )
    parser.add_argument(
        "--provider",
        choices=("groq", "ollama"),
        help="override LLM_PROVIDER for this run (ollama has no tokens-per-minute ceiling)",
    )
    parser.add_argument("--no-session", action="store_true", help="start from an empty history")
    parser.add_argument(
        "--model",
        help="Groq model id or alias for this and later runs (overrides .env)",
    )
    args = parser.parse_args()

    if args.debug_tokens:
        config.DEBUG_TOKENS = True

    provider = (args.provider or config.LLM_PROVIDER).lower()
    if args.model:
        try:
            models.set_active(args.model)
        except ValueError as exc:
            raise SystemExit(f"--model: {exc}")

    messages = [] if args.no_session else load_session()
    history = _describe(messages) if messages else "empty"
    active_model = config.OLLAMA_MODEL if provider == "ollama" else models.active()

    print("==================================================")
    print("🤖 Custom Python AI Agent Harness Initialized")
    print(f"Provider: {provider} | model: {active_model} | history: {history}")
    print("Type /help for commands, or 'exit' / 'quit' to close.")
    print("==================================================\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting agent harness. Goodbye!")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in ("exit", "quit"):
            print("\nShutting down agent harness. See you later!")
            break

        if command == "/help":
            print(HELP)
            continue

        if command == "/clear":
            before = _describe(messages)
            messages.clear()
            clear_session_file()
            print(f"[Session] Cleared ({before} -> empty).")
            continue

        if command == "/compact":
            before = _describe(messages)
            messages[:] = compact_history(messages)
            save_session(messages)
            print(f"[Session] Compacted ({before} -> {_describe(messages)}).")
            continue

        if command == "/model" or command.startswith("/model "):
            parts = user_input.split(maxsplit=1)
            _handle_model(parts[1].strip() if len(parts) > 1 else "", provider)
            continue

        if command.startswith("/"):
            print(f"Unknown command '{user_input}'. Type /help.")
            continue

        _auto_compact(messages, provider)
        print("\nAgent is thinking...")

        try:
            response = run_agent(user_input, messages, provider=provider)
        except AgentError as exc:
            # The turn did not finish; do not print an error as if it were the answer.
            print(f"\n[Agent Error] {exc}")
            continue
        except Exception as exc:
            print(f"\n[Harness Error] {exc}")
            continue
        finally:
            save_session(messages)

        print(f"\nAgent:\n{response}")

    save_session(messages)


if __name__ == "__main__":
    main()
