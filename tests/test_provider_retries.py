"""Provider retry policy: 413, 429 (with retry-after) and bounded 5xx.

The OpenAI SDK client is built with max_retries=0, so these decisions are made
explicitly in `_request_completion` and can be tested without a live provider.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from src import agent


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, headers=None, message: str | None = None):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code
        self.response = SimpleNamespace(headers=headers or {})


def raw_response(response, headers=None):
    return SimpleNamespace(parse=lambda: response, headers=headers or {})


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        completions = SimpleNamespace(
            with_raw_response=SimpleNamespace(create=self._create)
        )
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class RetryPolicyTestCase(unittest.TestCase):
    def setUp(self):
        self.response = SimpleNamespace(usage=None, marker="success")
        self.budget = SimpleNamespace(
            max_input_tokens=10_000,
            max_output_tokens=1_000,
            context_window=20_000,
        )

    def _run(self, outcomes, max_retries=2):
        client = FakeClient(outcomes)
        messages = [{"role": "user", "content": "hello"}]
        compaction = SimpleNamespace(
            changed=False,
            eligible=False,
            before_tokens=0,
            threshold_tokens=0,
        )
        sleeper = mock.Mock()
        with mock.patch.object(
            agent, "maybe_auto_compact", return_value=compaction
        ), mock.patch.object(
            agent,
            "_prepare_payload",
            return_value=({"messages": messages}, 100, True, True),
        ) as prepare, mock.patch.object(
            agent, "LLM_MAX_RETRIES", max_retries
        ), mock.patch.object(
            agent, "compact_history", side_effect=lambda history, *_: []
        ), mock.patch.object(
            agent.usage, "guard", return_value=""
        ), mock.patch.object(
            agent.usage, "record"
        ), mock.patch.object(
            agent.time, "sleep", sleeper
        ):
            result = agent._request_completion(
                client,
                "test",
                "test-model",
                "system",
                messages,
                [],
                self.budget,
            )
        return result, client, sleeper, prepare, messages

    def test_413_compacts_the_history_and_retries_once(self):
        result, client, sleeper, prepare, messages = self._run(
            [FakeHTTPError(413, message="request too large"), raw_response(self.response)]
        )

        self.assertIs(result, self.response)
        self.assertEqual(client.calls, 2)
        self.assertEqual(prepare.call_count, 2)
        self.assertEqual(messages, [])
        sleeper.assert_not_called()

    def test_a_second_413_is_not_retried_forever(self):
        with self.assertRaises(FakeHTTPError):
            self._run(
                [
                    FakeHTTPError(413, message="request too large"),
                    FakeHTTPError(413, message="still too large"),
                ]
            )

    def test_429_waits_for_the_retry_after_header(self):
        result, client, sleeper, _, _ = self._run(
            [
                FakeHTTPError(429, {"retry-after": "3"}),
                raw_response(self.response),
            ]
        )

        self.assertIs(result, self.response)
        self.assertEqual(client.calls, 2)
        sleeper.assert_called_once_with(3.0)

    def test_429_accepts_a_duration_style_reset_header(self):
        _, _, sleeper, _, _ = self._run(
            [
                FakeHTTPError(429, {"x-ratelimit-reset-tokens": "7.66s"}),
                raw_response(self.response),
            ]
        )

        sleeper.assert_called_once_with(7.66)

    def test_5xx_retries_are_bounded(self):
        with self.assertRaises(RuntimeError) as ctx:
            self._run(
                [
                    FakeHTTPError(500),
                    FakeHTTPError(502),
                    FakeHTTPError(503),
                ]
            )

        self.assertIn("kept failing", str(ctx.exception))
        # Two retries means at most three calls, not an unbounded loop.
        self.assertIn("2 retries", str(ctx.exception))

    def test_5xx_recovers_within_the_bound(self):
        result, client, sleeper, _, _ = self._run(
            [FakeHTTPError(500), raw_response(self.response)]
        )

        self.assertIs(result, self.response)
        self.assertEqual(client.calls, 2)
        sleeper.assert_called_once()


if __name__ == "__main__":
    unittest.main()
