"""Automatic compaction eligibility, thresholds, targets and preservation."""

import copy
import json
import unittest
from unittest import mock

from src import budget, config
from src.models import RateLimits, RequestBudget


def request_budget(limit: int) -> RequestBudget:
    return RequestBudget(
        model_id="test-model",
        provider="test",
        context_window=None,
        max_input_tokens=limit,
        max_output_tokens=1000,
        rate_limits=RateLimits(),
    )


def exchange(call_id: str, code: str, output: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "run_python_code",
                        "arguments": json.dumps({"code": code}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": output},
    ]


def conversation(turns: int, output_chars: int = 2000, code_chars: int = 100) -> list[dict]:
    """A session with ``turns`` user turns, one tool exchange and answer each."""
    messages: list[dict] = []
    for turn in range(1, turns + 1):
        messages.append({"role": "user", "content": f"User instruction {turn}"})
        messages.extend(
            exchange(
                f"call_{turn}",
                "print(1)" + "x" * code_chars,
                "result\n" * max(1, output_chars // 7),
            )
        )
        messages.append({"role": "assistant", "content": f"Finished turn {turn}"})
    return messages


SYSTEM = "You are a test agent."


def compact(messages, limit, threshold, target, keep=1, enabled=True):
    with mock.patch.object(config, "AUTO_COMPACT", enabled), mock.patch.object(
        config, "AUTO_COMPACT_MIN_TURNS", 2
    ), mock.patch.object(
        config, "AUTO_COMPACT_THRESHOLD", threshold
    ), mock.patch.object(
        config, "AUTO_COMPACT_TARGET", target
    ), mock.patch.object(
        budget, "HISTORY_KEEP_EXCHANGES", keep
    ):
        return budget.maybe_auto_compact(SYSTEM, messages, [], request_budget(limit))


class CompactionTestCase(unittest.TestCase):
    def test_not_eligible_on_first_turn(self):
        messages = conversation(turns=1)
        result = compact(messages, limit=10_000, threshold=0.0001, target=0.0001)
        self.assertFalse(result.eligible)
        self.assertFalse(result.changed)
        self.assertEqual(result.turns, 1)
        self.assertIn("minimum", result.reason)
        self.assertEqual(result.reduction, 0)

    def test_eligible_from_second_turn(self):
        messages = conversation(turns=2)
        result = compact(messages, limit=10_000, threshold=0.0001, target=0.0001)
        self.assertEqual(result.turns, 2)
        self.assertTrue(result.eligible)
        self.assertTrue(result.changed)
        self.assertGreater(result.reduction, 0)
        self.assertLess(result.after_tokens, result.before_tokens)

    def test_stays_untouched_below_threshold(self):
        messages = conversation(turns=2)
        result = compact(messages, limit=10_000_000, threshold=0.6, target=0.5)
        self.assertTrue(result.eligible)
        self.assertFalse(result.changed)
        self.assertIn("threshold", result.reason)

    def test_absolute_threshold_is_respected(self):
        messages = conversation(turns=2, output_chars=40_000)
        # An absolute count (> 1) is a token budget, not a fraction.
        result = compact(messages, limit=100_000, threshold=500, target=400)
        self.assertTrue(result.changed)
        self.assertEqual(result.threshold_tokens, 500)

    def test_compaction_reaches_target(self):
        messages = conversation(turns=2, output_chars=20_000)
        result = compact(messages, limit=20_000, threshold=0.3, target=0.7)
        self.assertTrue(result.changed)
        self.assertGreater(result.before_tokens, result.threshold_tokens)
        self.assertLessEqual(result.after_tokens, result.target_tokens)
        self.assertEqual(result.after_tokens, result.after_estimate.tokens)

    def test_preserves_instructions_and_newest_exchange(self):
        messages = conversation(turns=2, output_chars=8_000, code_chars=4_000)
        original = copy.deepcopy(messages)
        result = compact(messages, limit=20_000, threshold=0.0001, target=0.0001)
        self.assertTrue(result.changed)

        # User instructions and assistant prose are never rewritten.
        self.assertEqual(result.messages[0], original[0])
        self.assertEqual(result.messages[3], original[3])
        self.assertEqual(result.messages[4], original[4])
        self.assertEqual(result.messages[7], original[7])

        # The newest exchange is replayed verbatim.
        self.assertEqual(result.messages[5], original[5])
        self.assertEqual(result.messages[6], original[6])

        # The older exchange is collapsed: head of arguments, one-line outcome.
        collapsed_arguments = result.messages[1]["tool_calls"][0]["function"]["arguments"]
        self.assertIn("chars collapsed", collapsed_arguments)
        self.assertTrue(result.messages[2]["content"].startswith("run_python_code(...) -> ok"))

        # The caller's history must not be mutated in place.
        self.assertEqual(messages, original)

    def test_no_stale_exchanges_to_collapse(self):
        # Two turns (so compaction is eligible) but only one exchange, which is
        # the one the automatic path always keeps verbatim.
        messages = [
            {"role": "user", "content": "User instruction 1"},
            {"role": "assistant", "content": "Finished turn 1"},
            {"role": "user", "content": "User instruction 2"},
            *exchange("call_2", "print(2)", "result\n" * 2000),
            {"role": "assistant", "content": "Finished turn 2"},
        ]
        result = compact(messages, limit=10_000, threshold=0.0001, target=0.0001)
        self.assertFalse(result.changed)
        self.assertIn("no stale", result.reason)

    def test_disabled_compaction(self):
        messages = conversation(turns=3, output_chars=20_000)
        result = compact(messages, limit=10_000, threshold=0.0001, target=0.0001, enabled=False)
        self.assertFalse(result.eligible)
        self.assertFalse(result.changed)
        self.assertEqual(result.reason, "disabled")

    def test_request_still_oversized_after_compaction(self):
        messages = [{"role": "user", "content": "x" * 40_000}]
        payload, tokens, fits = budget.build_payload(
            SYSTEM, messages, [], request_budget(500)
        )
        self.assertFalse(fits)
        self.assertGreater(tokens, 500)
        self.assertEqual(payload[0], {"role": "system", "content": SYSTEM})

    def test_oversized_conversation_reports_unchanged_with_reason(self):
        messages = [{"role": "user", "content": "x" * 40_000}]
        result = compact(messages, limit=500, threshold=0.0001, target=0.0001)
        self.assertFalse(result.changed)
        self.assertFalse(result.changed and result.after_tokens <= result.target_tokens)

    def test_manual_compact_collapses_every_exchange(self):
        messages = conversation(turns=2, code_chars=4_000)
        compacted = budget.compact_history(messages, keep_exchanges=0)
        self.assertEqual(compacted[0]["content"], "User instruction 1")
        for message in compacted:
            if message.get("role") == "tool":
                self.assertTrue(message["content"].startswith("run_python_code(...) -> ok"))
            if message.get("role") == "assistant" and message.get("tool_calls"):
                arguments = message["tool_calls"][0]["function"]["arguments"]
                self.assertIn("chars collapsed", arguments)


if __name__ == "__main__":
    unittest.main()
