"""Daily accounting and the projected usage guard."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from src import usage
from src.usage_report import TokenUsage

MODEL = "openai/gpt-oss-120b"  # bundled: 1,000 RPD / 200,000 TPD on Groq


class UsageGuardTestCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.usage_file = Path(self._directory.name) / "usage.json"
        patcher = mock.patch.object(usage, "USAGE_FILE", self.usage_file)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_reported_usage_is_recorded_as_reported(self):
        usage.record(
            MODEL,
            TokenUsage(input=100, output=50, total=150, cached=20),
            provider="groq",
            estimated_tokens=999,
        )
        summary = usage.totals(MODEL, "groq")
        self.assertEqual(summary.reported_requests, 1)
        self.assertEqual(summary.estimated_requests, 0)
        self.assertEqual(summary.total_tokens, 150)
        self.assertEqual(summary.input_tokens, 100)
        self.assertEqual(summary.output_tokens, 50)
        self.assertEqual(summary.cached_tokens, 20)
        # An estimate never replaces or doubles a reported total.
        self.assertEqual(summary.estimated_tokens, 0)
        self.assertEqual(summary.tokens, 150)

    def test_missing_reported_total_uses_estimate_separately(self):
        usage.record(MODEL, TokenUsage(), provider="groq", estimated_tokens=400)
        summary = usage.totals(MODEL, "groq")
        self.assertEqual(summary.reported_requests, 0)
        self.assertEqual(summary.estimated_requests, 1)
        self.assertEqual(summary.total_tokens, 0)
        self.assertEqual(summary.estimated_tokens, 400)
        self.assertEqual(summary.tokens, 400)

    def test_guard_allows_a_projected_request_within_the_limit(self):
        usage.record(MODEL, TokenUsage(total=100_000), provider="groq")
        self.assertIsNone(usage.guard(MODEL, "groq", projected_tokens=10_000))

    def test_guard_blocks_a_projected_request_over_the_token_limit(self):
        usage.record(MODEL, TokenUsage(total=199_900), provider="groq")
        reason = usage.guard(MODEL, "groq", projected_tokens=200)
        self.assertIsNotNone(reason)
        self.assertIn("daily token limit", reason)

    def test_guard_projection_catches_the_last_fitting_request(self):
        # Past usage alone is under the limit, so only the projected request can
        # reveal that sending it would cross the line.
        usage.record(MODEL, TokenUsage(total=199_950), provider="groq")
        self.assertIsNone(usage.guard(MODEL, "groq", projected_tokens=0))
        self.assertIsNotNone(usage.guard(MODEL, "groq", projected_tokens=100))

    def test_guard_blocks_a_projected_request_over_the_daily_limit(self):
        reason = usage.guard(MODEL, "groq", projected_requests=1001)
        self.assertIsNotNone(reason)
        self.assertIn("daily request limit", reason)

    def test_cached_tokens_do_not_count_toward_the_daily_token_limit(self):
        # 199,000 total but only 9,000 billable after the 190,000 cached tokens,
        # which the provider does not charge against the daily bucket.
        usage.record(MODEL, TokenUsage(total=199_000, cached=190_000), provider="groq")
        self.assertEqual(usage.totals(MODEL, "groq").billable_tokens, 9_000)
        self.assertIsNone(usage.guard(MODEL, "groq", projected_tokens=1_000))
        self.assertIsNotNone(usage.guard(MODEL, "groq", projected_tokens=191_001))

    def test_guard_is_silent_without_published_limits(self):
        self.assertIsNone(usage.guard("some/model", "nvidia", projected_tokens=10**9))

    def test_counters_reset_on_a_new_day(self):
        self.usage_file.write_text(
            json.dumps(
                {
                    "date": "2020-01-01",
                    "models": {f"groq:{MODEL}": {"requests": 500, "tokens": 100_000}},
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(usage.totals(MODEL, "groq").tokens, 0)

    def test_legacy_entry_shape_is_still_read(self):
        self.usage_file.write_text(
            json.dumps(
                {
                    "date": time.strftime("%Y-%m-%d", time.gmtime()),
                    "models": {
                        f"groq:{MODEL}": {
                            "requests": 3,
                            "tokens": 900,
                            "input_tokens": 600,
                            "output_tokens": 300,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        summary = usage.totals(MODEL, "groq")
        self.assertEqual(summary.requests, 3)
        self.assertEqual(summary.tokens, 900)

    def test_describe_shows_usage_against_published_limits(self):
        usage.record(MODEL, TokenUsage(total=150), provider="groq")
        text = usage.describe(MODEL, "groq")
        self.assertIn("/1,000 requests", text)
        self.assertIn("/200,000 tokens", text)

    def test_describe_reports_billable_tokens(self):
        usage.record(MODEL, TokenUsage(total=1_000, cached=400), provider="groq")
        self.assertIn("600/200,000 tokens", usage.describe(MODEL, "groq"))


if __name__ == "__main__":
    unittest.main()
