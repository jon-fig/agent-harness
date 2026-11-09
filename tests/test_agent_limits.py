"""Per-turn tool-execution ceiling and duplicate-call semantics."""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from src import agent, config


def tool_response(count: int, name: str = "run_python_code", code: str = "print(1)"):
    calls = [
        SimpleNamespace(
            id=f"call_{index}",
            function=SimpleNamespace(name=name, arguments=json.dumps({"code": code})),
        )
        for index in range(count)
    ]
    message = SimpleNamespace(content="", tool_calls=calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")]
    )


def final_response(text: str = "done"):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def run_turn(responses, max_tool_calls: int, dedupe: bool):
    executed: list[tuple[str, dict]] = []

    def dispatch(name, arguments):
        if name == "clear_session":
            return "session cleared"
        executed.append((name, arguments))
        return f"result {len(executed)}"

    messages: list[dict] = []
    with mock.patch.object(
        agent, "get_llm_client", return_value=(object(), "test-model")
    ), mock.patch.object(
        agent, "get_tooling", return_value=([], dispatch)
    ), mock.patch.object(
        agent, "_request_completion", side_effect=list(responses)
    ), mock.patch.object(
        agent, "build_system_prompt", return_value="system"
    ), mock.patch.object(
        config, "MAX_TOOL_CALLS_PER_TURN", max_tool_calls
    ), mock.patch.object(
        config, "DEDUPE_TOOL_CALLS", dedupe
    ):
        result = agent.run_agent("do the thing", messages, provider="test")

    return result, messages, executed


def tool_messages(messages: list[dict]) -> list[dict]:
    return [message for message in messages if message.get("role") == "tool"]


class ToolCallLimitTestCase(unittest.TestCase):
    def test_executions_are_capped_per_turn(self):
        result, messages, executed = run_turn(
            [tool_response(5), final_response()], max_tool_calls=2, dedupe=False
        )
        self.assertEqual(result, "done")
        self.assertEqual(len(executed), 2)

        # Every requested call still receives a tool message, so the provider
        # never sees an unanswered tool_call id.
        results = tool_messages(messages)
        self.assertEqual(len(results), 5)
        skipped = [item for item in results if "limit" in item["content"]]
        self.assertEqual(len(skipped), 3)

    def test_limit_is_a_floor_of_one(self):
        result, messages, executed = run_turn(
            [tool_response(2), final_response()], max_tool_calls=0, dedupe=False
        )
        self.assertEqual(result, "done")
        self.assertEqual(len(executed), 1)


class DuplicateToolCallTestCase(unittest.TestCase):
    def test_identical_calls_execute_by_default(self):
        result, messages, executed = run_turn(
            [tool_response(2), final_response()], max_tool_calls=10, dedupe=False
        )
        self.assertEqual(result, "done")
        # Identical arguments can still produce different results, so the second
        # call must actually run.
        self.assertEqual(len(executed), 2)
        self.assertEqual(tool_messages(messages)[1]["content"], "result 2")

    def test_opt_in_dedupe_replays_the_earlier_result(self):
        result, messages, executed = run_turn(
            [tool_response(2), final_response()], max_tool_calls=10, dedupe=True
        )
        self.assertEqual(result, "done")
        self.assertEqual(len(executed), 1)
        self.assertTrue(
            tool_messages(messages)[1]["content"].startswith("[Not executed again:")
        )


if __name__ == "__main__":
    unittest.main()
