"""Normalisation of provider-reported usage into TokenUsage."""

import unittest
from types import SimpleNamespace

from src.usage_report import TokenUsage


class TokenUsageTestCase(unittest.TestCase):
    def test_openai_style_usage(self):
        usage = TokenUsage.from_mapping(
            {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 10},
            }
        )
        self.assertEqual(usage.input, 120)
        self.assertEqual(usage.output, 30)
        self.assertEqual(usage.total, 150)
        self.assertEqual(usage.cached, 40)
        self.assertEqual(usage.reasoning, 10)
        self.assertTrue(usage.reported)
        self.assertEqual(usage.effective_total, 150)

    def test_gateway_style_usage(self):
        usage = TokenUsage.from_mapping({"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
        self.assertEqual(usage.input, 7)
        self.assertEqual(usage.output, 3)
        self.assertEqual(usage.total, 10)

    def test_missing_usage_is_not_invented(self):
        usage = TokenUsage.from_mapping(None)
        self.assertFalse(usage.reported)
        self.assertIsNone(usage.effective_total)
        self.assertIsNone(usage.billable_total)
        self.assertEqual(usage.describe(), "not reported")
        self.assertEqual(usage.as_dict(), {})

    def test_partial_usage_keeps_unknown_fields_unknown(self):
        usage = TokenUsage.from_mapping({"prompt_tokens": 50})
        self.assertEqual(usage.input, 50)
        self.assertIsNone(usage.output)
        self.assertIsNone(usage.total)
        # A lone input count is not a total: assuming output=0 would undercount.
        self.assertIsNone(usage.effective_total)
        self.assertTrue(usage.reported)

    def test_total_is_derived_only_when_both_halves_exist(self):
        usage = TokenUsage(input=10, output=5)
        self.assertEqual(usage.effective_total, 15)
        self.assertEqual(usage.as_dict()["total"], 15)

    def test_cached_tokens_are_excluded_from_billable_total(self):
        usage = TokenUsage(total=100, cached=30)
        self.assertEqual(usage.billable_total, 70)

    def test_invalid_and_negative_values(self):
        self.assertIsNone(TokenUsage.from_mapping({"total_tokens": "n/a"}).total)
        self.assertIsNone(TokenUsage.from_mapping({"total_tokens": True}).total)
        self.assertEqual(TokenUsage.from_mapping({"total_tokens": -5}).total, 0)

    def test_sdk_objects_are_read_by_attribute(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4, total_tokens=15)
        )
        usage = TokenUsage.from_response(response)
        self.assertEqual((usage.input, usage.output, usage.total), (11, 4, 15))

    def test_response_without_usage(self):
        usage = TokenUsage.from_response(SimpleNamespace(choices=[]))
        self.assertFalse(usage.reported)

    def test_describe_lists_reported_fields_only(self):
        text = TokenUsage(input=10, output=5).describe()
        self.assertIn("input=10", text)
        self.assertIn("output=5", text)
        self.assertIn("total=15", text)
        self.assertNotIn("cached", text)


if __name__ == "__main__":
    unittest.main()
