"""
CLI entrypoint for the custom agentic harness.

Keeps the conversation across prompts, persists it to .agent/session.json so long
work survives a restart, and provides /clear and /compact because the history is
the main input-token cost on a metered provider.

Provider selection is deliberately not a fixed list: /provider switches between
any endpoints configured in .env, and /model list asks the active provider for its
real catalogue over GET /models.
"""

import argparse

# CPython enables line editing for input() only when the readline module is
# imported, and it does that automatically only for the interactive REPL. A
# module run ("python -m src.main") starts without it, so arrow keys, Home/End,
# Ctrl+A/E and backspace across wrapped lines are echoed as raw escape
# sequences and never reach the prompt. POSIX: import it explicitly. Windows:
# the console host already provides its own editing, and the missing module is
# ignored so native behavior is preserved.
try:
    import readline  # noqa: F401
except ImportError:  # pragma: no cover - Windows console editing stays native
    pass

from src import config, models, providers, usage
from src.agent import AgentError, run_agent
from src.budget import compact_history, estimate_tokens
from src.llm_client import list_models
from src.session import (
    clear_session_file,
    load_session,
    save_session,
    set_session_enabled,
)

HELP = """Commands:
  /model            show the active model, provider and bundled options
  /model list       ask the provider for its catalogue (GET /models)
  /model <id|#>     switch model (id, alias, or a number from /model list)
  /provider         show the active provider and the other configured ones
  /provider <name>  switch provider for this session
  /clear            forget the conversation (and delete the saved session unless --no-session)
  /compact          collapse older tool exchanges now, keeping the latest ones verbatim
  /help             show this help
  exit              quit (history is saved unless --no-session was given)"""

# Model ids fetched during this session, used to resolve "/model 7".
_MODEL_CACHE: dict[str, list[str]] = {}


def _describe(messages: list, provider: str) -> str:
    if not messages:
        return "empty"
    model_id = models.active(provider)
    estimate = estimate_tokens(messages, model_id=model_id, provider=provider)
    return f"{len(messages)} messages, ~{estimate:,} tokens"


def _budget_line(provider: str) -> str:
    """One-line request budget for the active model."""
    model_id = models.active(provider)
    budget = models.request_budget(model_id, provider)
    return budget.describe()


def _fetch_models(provider: str) -> list[str]:
    """Fetches the provider's catalogue and prints a numbered chooser."""
    try:
        endpoint = providers.resolve(provider)
    except ValueError as exc:
        print(f"[Model] {exc}")
        return []

    print(f"[Model] Querying {endpoint.base_url}/models ...")
    try:
        ids = list_models(provider)
    except RuntimeError as exc:
        print(f"[Model] {exc}")
        return []

    _MODEL_CACHE[provider] = ids
    if not ids:
        print("[Model] The provider returned an empty catalogue.")
        return []

    current = models.active(provider)
    print(f"[Model] {len(ids)} models available from {endpoint.label}:")
    for index, model_id in enumerate(ids, start=1):
        marker = "*" if model_id == current else " "
        print(f" {marker} {index:>3}. {model_id}")
    print("Switch with: /model <id> or /model <number>.")
    return ids


def _handle_model(argument: str, provider: str) -> None:
    """`/model` reports, `/model list` discovers, anything else switches."""
    provider = providers.normalize(provider)

    if argument.lower() in ("list", "ls", "refresh"):
        _fetch_models(provider)
        return

    if not argument:
        print(models.describe_active(provider))
        print(f"        {usage.describe(models.active(provider), provider)}")
        print(models.table(provider))
        cached = _MODEL_CACHE.get(provider)
        if cached:
            print(f"        last fetched list: {len(cached)} models (use /model <number>)")
        return

    # A number refers to the last /model list for this provider.
    cached = _MODEL_CACHE.get(provider, [])
    if argument.isdigit() and cached:
        index = int(argument)
        if not 1 <= index <= len(cached):
            print(f"[Model] No entry #{index}; the last list has {len(cached)} models.")
            return
        argument = cached[index - 1]

    try:
        model_id = models.set_active(argument, provider)
    except ValueError as exc:
        print(f"[Model] {exc}. Type /model to list options.")
        return

    endpoint = providers.resolve(provider)
    budget = models.request_budget(model_id, provider)
    if models.spec_for(model_id, provider) is None:
        print(
            f"[Model] Active -> {model_id} on {endpoint.label} "
            f"({budget.describe()})."
        )
        print(f"        {usage.describe(model_id, provider)}")
        return

    print(
        f"[Model] Active -> {model_id} | {budget.describe()}"
    )
    print(f"        {usage.describe(model_id, provider)}")
    print("        Pacing state is per provider/model, so switching does not inherit a used window.")


def _handle_provider(argument: str, provider: str) -> str:
    """`/provider` reports, `/provider <name>` switches for this session."""
    if not argument:
        try:
            endpoint = providers.resolve(provider)
            print(f"[Provider] Active -> {endpoint.describe()}")
        except ValueError as exc:
            print(f"[Provider] {exc}")
        print(providers.describe_all(provider))
        print(f"        Known presets: {', '.join(sorted(providers.PRESETS))}")
        return provider

    try:
        endpoint = providers.resolve(argument)
    except ValueError as exc:
        print(f"[Provider] {exc}")
        return provider

    active = models.active(endpoint.name)
    budget = models.request_budget(active, endpoint.name)
    print(f"[Provider] Active -> {endpoint.describe()}")
    print(f"        model: {active} | {budget.describe()}")
    print("        The conversation is kept; the next turn uses this endpoint.")
    return endpoint.name


def main():
    parser = argparse.ArgumentParser(description="Custom Python AI agent harness")
    parser.add_argument(
        "--debug-tokens",
        action="store_true",
        help="log approximate vs. reported token usage and rate-limit headroom",
    )
    parser.add_argument(
        "--provider",
        help=(
            "OpenAI-compatible provider to use for this run (groq, nvidia, openai, "
            "openrouter, together, deepseek, mistral, cerebras, fireworks, "
            "perplexity, ollama, or any custom name)"
        ),
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="start from an empty history and do not load or save the session file",
    )
    parser.add_argument(
        "--model",
        help="model id or alias for this provider for this and later runs (overrides .env)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="print the provider's GET /models catalogue and exit",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="print the providers configured in this environment and exit",
    )
    args = parser.parse_args()

    # Disable both directions of persistence before anything can touch the file:
    # history loading below, and the save/clear paths in the REPL and at exit.
    set_session_enabled(not args.no_session)

    try:
        config.require_supported_interpreter()
    except RuntimeError as exc:
        raise SystemExit(f"[Environment Error] {exc}")

    if args.debug_tokens:
        config.DEBUG_TOKENS = True

    provider = providers.normalize(args.provider or config.LLM_PROVIDER)
    try:
        endpoint = providers.resolve(provider)
    except ValueError as exc:
        raise SystemExit(f"[Provider Error] {exc}")

    if args.list_providers:
        print(f"[Provider] Active -> {endpoint.describe()}")
        print(providers.describe_all(provider))
        return

    if args.list_models:
        try:
            for model_id in list_models(provider):
                print(model_id)
        except RuntimeError as exc:
            raise SystemExit(f"[Model Error] {exc}")
        return

    if args.model:
        try:
            models.set_active(args.model, provider)
        except ValueError as exc:
            raise SystemExit(f"--model: {exc}")

    messages = [] if args.no_session else load_session()
    history = _describe(messages, provider)
    active_model = models.active(provider)
    key_source = endpoint.api_key_source or ("not required" if endpoint.local else "missing")

    print("==================================================")
    print("🤖 Custom Python AI Agent Harness Initialized")
    print(f"Provider: {endpoint.label} ({endpoint.name}) | {endpoint.base_url}")
    print(f"Auth: {key_source} | model: {active_model}")
    print(f"Budget: {_budget_line(provider)} | history: {history}")
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
            before = _describe(messages, provider)
            messages.clear()
            clear_session_file()
            print(f"[Session] Cleared ({before} -> empty).")
            continue

        if command == "/compact":
            before = _describe(messages, provider)
            messages[:] = compact_history(messages)
            save_session(messages)
            print(f"[Session] Compacted ({before} -> {_describe(messages, provider)}).")
            continue

        if (
            command in ("/model", "/models")
            or command.startswith("/model ")
            or command.startswith("/models ")
        ):
            parts = user_input.split(maxsplit=1)
            _handle_model(parts[1].strip() if len(parts) > 1 else "", provider)
            continue

        if command == "/provider" or command.startswith("/provider ") or command == "/providers":
            parts = user_input.split(maxsplit=1)
            provider = _handle_provider(
                parts[1].strip() if len(parts) > 1 else "", provider
            )
            continue

        if command.startswith("/"):
            print(f"Unknown command '{user_input}'. Type /help.")
            continue

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
