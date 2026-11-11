"""Tool output caps: predictable truncation without losing the useful ends.

`_truncate_middle` is the single point where MCP output enters chat history.
The head carries column names, the tail carries a traceback or the last rows,
and structured output must never look like valid JSON after being cut.
"""

import json
import unittest
from types import SimpleNamespace

from src.mcp_client import _format_result, _truncate_middle


class MiddleTruncationTestCase(unittest.TestCase):
    def test_output_within_the_limit_is_unchanged(self):
        text = "short output"
        self.assertEqual(_truncate_middle(text, limit=100), text)

    def test_oversized_output_keeps_the_head_and_tail_and_says_what_was_lost(self):
        text = "BEGIN " + ("x" * 2000) + " END"
        result = _truncate_middle(text, limit=300)

        self.assertTrue(result.startswith("BEGIN "))
        self.assertTrue(result.endswith(" END"))
        self.assertIn("chars omitted", result)
        self.assertIn("truncated", result)
        # The middle is what disappears, not the useful beginning or ending.
        self.assertNotIn("x" * 400, result)

    def test_a_traceback_ending_is_preserved(self):
        log = ("starting\n" * 1000) + (
            "Traceback (most recent call last):\n"
            '  File "analysis.py", line 3, in <module>\n'
            "ValueError: the last row is malformed\n"
        )
        result = _truncate_middle(log, limit=400)

        self.assertTrue(result.startswith("starting"))
        self.assertIn("ValueError: the last row is malformed", result)
        self.assertIn("chars omitted", result)

    def test_truncated_json_is_not_presented_as_valid_json(self):
        document = json.dumps({"rows": list(range(5000))})
        result = _truncate_middle(document, limit=400)

        self.assertIn("NOT valid JSON", result)
        self.assertIn("chars omitted", result)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(result)

    def test_structured_content_uses_the_same_clear_marker(self):
        result = SimpleNamespace(
            content=[],
            structured_content={"rows": list(range(5000))},
        )
        text = _format_result(result)

        self.assertIn("NOT valid JSON", text)
        self.assertIn("chars omitted", text)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(text)

    def test_small_structured_content_stays_valid_json(self):
        result = SimpleNamespace(content=[], structured_content={"rows": [1, 2, 3]})
        text = _format_result(result)

        self.assertEqual(json.loads(text), {"rows": [1, 2, 3]})


if __name__ == "__main__":
    unittest.main()
