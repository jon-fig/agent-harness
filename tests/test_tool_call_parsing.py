"""Regression tests for tool calls that arrive as text instead of tool_calls.

NVIDIA NIM serving MiniMax returns the model's tool calls serialized inside
message.content in the model's own XML-ish format, with the model's degraded
tool-call special tokens left as literal noise in front of every structural
tag. Before the fix that text fell through to the "final answer" path: it was
printed to the terminal, saved as history, and never executed.
"""

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from src import agent, config

# The literal noise the provider leaves where the model emitted its tool-call
# special tokens. Safe as a plain string here: unlike the XML tags below, it is
# not a sequence other tool-call parsers key on.
NOISE = "]<]minimax[>["

LT, GT = chr(60), chr(62)  # the raw angle brackets, kept out of the fixtures
OPEN = f"{LT}tool_call{GT}"
CLOSE = f"{LT}/tool_call{GT}"


def invoke_block(name: str, params: dict) -> str:
    """One MiniMax invoke, serialized the way the provider returns it."""
    text = f'{LT}invoke name="{name}"{GT}'
    for key, value in params.items():
        text += f"{LT}{key}{GT}{value}{LT}/{key}{GT}"
    return text + f"{LT}/invoke{GT}"


def minimax_content(*invokes: str, open_tag: bool = True, close_tag: bool = True) -> str:
    """A full tool_call block with the noise in front of every tag."""
    body = "\n".join(invokes)
    text = f"{NOISE}{OPEN}\n{body}\n"
    if close_tag:
        text += f"{NOISE}{CLOSE}"
    if open_tag is False:  # simulate a degraded open marker
        text = text.replace(f"{NOISE}{OPEN}", f"{NOISE}{OPEN[:-1]} name{GT}")
    return text


class MiniMaxParsingTestCase(unittest.TestCase):
    def test_complete_block_is_parsed_with_arguments(self):
        content = minimax_content(
            invoke_block("read_file", {"file_path": "notes.txt"}),
        )
        calls = agent._parse_minimax_tool_calls(content)
        self.assertEqual(calls, [("read_file", json.dumps({"file_path": "notes.txt"}))])

    def test_multiple_invokes_in_one_block_are_kept(self):
        content = minimax_content(
            invoke_block("read_file", {"file_path": "a.md"}),
            invoke_block("list_directory", {"directory_path": "."}),
        )
        calls = agent._parse_minimax_tool_calls(content)
        self.assertEqual([name for name, _ in calls], ["read_file", "list_directory"])

    def test_unambiguous_json_parameters_are_converted(self):
        content = minimax_content(
            invoke_block(
                "write_file",
                {"file_path": "data.json", "overwrite": "true", "rows": '{"a": 1}'},
            )
        )
        (_, arguments), = agent._parse_minimax_tool_calls(content)
        self.assertEqual(json.loads(arguments)["overwrite"], True)
        self.assertEqual(json.loads(arguments)["rows"], {"a": 1})

    def test_numbers_stay_strings_so_paths_are_not_retyped(self):
        content = minimax_content(
            invoke_block("write_file", {"file_path": "2024", "overwrite": "false"}),
        )
        (_, arguments), = agent._parse_minimax_tool_calls(content)
        self.assertEqual(json.loads(arguments)["file_path"], "2024")

    def test_noise_in_ordinary_prose_is_not_a_tool_call(self):
        # The degraded tokens also appear when the model merely talks about the
        # format; without an invoke attempt that text is an ordinary answer.
        self.assertIsNone(
            agent._parse_minimax_tool_calls(f"The provider uses {NOISE} markers.")
        )
        self.assertIsNone(agent._parse_minimax_tool_calls("Just answer prose."))

    def test_incomplete_invoke_is_reported_as_malformed(self):
        content = (
            f"{NOISE}{OPEN}\n"
            f'{NOISE}{LT}invoke name="read_file"{GT}'
            f'{NOISE}{LT}file_path{GT}notes.txt'
            # </file_path>, </invoke> and the tool_call close are all missing:
            # an output cut mid-call.
        )
        self.assertEqual(agent._parse_minimax_tool_calls(content), [])

    def test_degraded_open_marker_with_no_invoke_is_malformed(self):
        # Seen in the wild: an open tag with the tag name cut off. There is a
        # serialization attempt, but nothing executable inside.
        content = minimax_content("", open_tag=False, close_tag=False)
        self.assertEqual(agent._parse_minimax_tool_calls(content), [])


class JSONFallbackStillWorksTestCase(unittest.TestCase):
    def test_bare_json_tool_call_is_still_parsed(self):
        content = json.dumps({"name": "read_file", "arguments": {"file_path": "a.md"}})
        self.assertEqual(
            agent._parse_text_tool_call(content),
            ("read_file", json.dumps({"file_path": "a.md"})),
        )


def run_turn(responses, executed=None):
    """Runs one agent turn against canned responses, recording tool dispatches."""
    recorded = executed if executed is not None else []

    def dispatch(name, arguments):
        if name == "clear_session":
            return "session cleared"
        recorded.append((name, dict(arguments)))
        return f"result {len(recorded)}"

    messages: list[dict] = []
    with mock.patch.object(
        agent, "get_llm_client", return_value=(object(), "test-model")
    ), mock.patch.object(
        agent, "get_tooling", return_value=([], dispatch)
    ), mock.patch.object(
        agent, "_request_completion", side_effect=list(responses)
    ), mock.patch.object(
        agent, "build_system_prompt", return_value="system"
    ):
        result = agent.run_agent("do the thing", messages, provider="test")

    return result, messages, recorded


def text_response(content: str, finish_reason: str = "stop"):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def native_response(calls, finish_reason: str = "tool_calls"):
    """A response whose tool calls arrived in the native tool_calls field."""
    tool_calls = [
        SimpleNamespace(
            id=f"call_{index}",
            function=SimpleNamespace(name=name, arguments=arguments),
        )
        for index, (name, arguments) in enumerate(calls, start=1)
    ]
    message = SimpleNamespace(content="", tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


class MalformedNativeToolCallTestCase(unittest.TestCase):
    def test_a_tool_call_cut_off_by_the_output_limit_is_not_executed(self):
        executed: list[tuple[str, dict]] = []
        response = native_response(
            [("write_file", json.dumps({"file_path": "notes.md", "content": "half"}))],
            finish_reason="length",
        )
        with self.assertRaises(agent.AgentError) as ctx:
            run_turn([response], executed)
        self.assertIn("output limit", str(ctx.exception))
        self.assertEqual(executed, [])

    def test_malformed_native_arguments_are_reported_without_executing(self):
        executed: list[tuple[str, dict]] = []
        response = native_response([("read_file", "{not valid json")])
        result, messages, executed = run_turn(
            [response, text_response("done")], executed
        )

        self.assertEqual(result, "done")
        self.assertEqual(executed, [])
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        self.assertTrue(tool_messages)
        self.assertIn("Failed to parse tool arguments", tool_messages[-1]["content"])

    def test_a_valid_native_call_reaches_dispatch(self):
        executed: list[tuple[str, dict]] = []
        response = native_response(
            [("read_file", json.dumps({"file_path": "notes.txt"}))]
        )
        result, messages, executed = run_turn(
            [response, text_response("done")], executed
        )

        self.assertEqual(result, "done")
        self.assertEqual(executed, [("read_file", {"file_path": "notes.txt"})])
        self.assertEqual(messages[2]["tool_call_id"], "call_1")


class MiniMaxTurnTestCase(unittest.TestCase):
    def test_text_serialization_is_replayed_as_a_real_tool_call(self):
        response = text_response(
            minimax_content(invoke_block("read_file", {"file_path": "notes.txt"}))
        )
        result, messages, executed = run_turn([response, text_response("done")])

        self.assertEqual(result, "done")
        self.assertEqual(executed, [("read_file", {"file_path": "notes.txt"})])

        # The history holds a proper assistant.tool_calls entry answered by a
        # tool message, so the provider never sees a dangling id.
        assistant = messages[1]
        self.assertEqual(assistant["role"], "assistant")
        call = assistant["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "read_file")
        self.assertEqual(call["id"], messages[2]["tool_call_id"])

        # The raw serialization never reaches the visible answer or the history
        # as plain text.
        self.assertNotIn(NOISE, result)
        for message in messages:
            self.assertNotIn(NOISE, message.get("content") or "")

    def test_malformed_text_call_is_neither_executed_nor_shown(self):
        malformed = (
            f"{NOISE}{OPEN}\n"
            f'{NOISE}{LT}invoke name="read_file"{GT}'
            f'{NOISE}{LT}file_path{GT}notes.txt'
        )
        with self.assertRaises(agent.AgentError) as ctx:
            run_turn([text_response(malformed)])

        self.assertIn("malformed", str(ctx.exception).lower())
        self.assertNotIn(NOISE, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
