"""Provider- and model-specific request budgets."""

import os
import unittest
from unittest import mock

from src import config, models

# Prefixes whose values could leak in from the real environment or .env.
_BUDGET_PREFIXES = ("NVIDIA_", "GROQ_", "OLLAMA_", "MODEL_", "CUSTOM_", "OPENAI_")


class BudgetTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = {
            key: value
            for key, value in os.environ.items()
            if key.startswith(_BUDGET_PREFIXES)
        }
        for key in list(os.environ):
            if key.startswith(_BUDGET_PREFIXES):
                del os.environ[key]

        self._patches = [
            mock.patch.object(config, "MAX_OUTPUT_TOKENS", 1200),
            mock.patch.object(config, "MAX_INPUT_TOKENS", 6500),
            mock.patch.object(config, "OLLAMA_MAX_INPUT_TOKENS", 32000),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()
        for key in list(os.environ):
            if key.startswith(_BUDGET_PREFIXES):
                del os.environ[key]
        os.environ.update(self._saved)

    def test_bundled_groq_limits(self):
        budget = models.request_budget("openai/gpt-oss-120b", "groq")
        self.assertEqual(budget.context_window, 131_072)
        self.assertEqual(budget.max_output_tokens, 1_200)
        # min(MAX_INPUT_TOKENS, context - output, TPM - output - framing margin).
        self.assertEqual(budget.max_input_tokens, 6500)
        self.assertEqual(budget.rate_limits.tpm, 8_000)
        self.assertEqual(budget.rate_limits.rpm, 30)
        self.assertEqual(budget.rate_limits.rpd, 1_000)
        self.assertEqual(budget.rate_limits.tpd, 200_000)
        self.assertEqual(budget.source, "bundled")

    def test_unknown_provider_has_no_rate_limits(self):
        budget = models.request_budget("some/model", "nvidia")
        self.assertIsNone(budget.context_window)
        self.assertFalse(budget.rate_limits.any)
        self.assertEqual(budget.max_input_tokens, 6500)
        self.assertEqual(budget.source, "default")

    def test_provider_environment_overrides(self):
        os.environ["NVIDIA_CONTEXT_WINDOW"] = "131072"
        os.environ["NVIDIA_MAX_OUTPUT_TOKENS"] = "4096"
        os.environ["NVIDIA_MAX_INPUT_TOKENS"] = "30000"
        budget = models.request_budget("minimaxai/minimax-m3.1-flash", "nvidia")
        self.assertEqual(budget.context_window, 131_072)
        self.assertEqual(budget.max_output_tokens, 4096)
        self.assertEqual(budget.max_input_tokens, 30_000)
        self.assertEqual(budget.source, "environment")

    def test_model_specific_override_beats_provider(self):
        os.environ["NVIDIA_MAX_OUTPUT_TOKENS"] = "2048"
        os.environ["MODEL_MINIMAXAI_MINIMAX_M3_1_FLASH_MAX_OUTPUT_TOKENS"] = "8192"
        budget = models.request_budget("minimaxai/minimax-m3.1-flash", "nvidia")
        self.assertEqual(budget.max_output_tokens, 8192)

    def test_per_minute_bucket_constrains_input(self):
        os.environ["GROQ_TPM"] = "4000"
        budget = models.request_budget("openai/gpt-oss-120b", "groq")
        self.assertEqual(budget.rate_limits.tpm, 4000)
        self.assertEqual(budget.max_input_tokens, 4000 - 1200 - 200)

    def test_context_window_reserves_output(self):
        os.environ["NVIDIA_CONTEXT_WINDOW"] = "2000"
        os.environ["NVIDIA_MAX_INPUT_TOKENS"] = "30000"
        budget = models.request_budget("some/model", "nvidia")
        self.assertEqual(budget.max_input_tokens, 2000 - 1200)

    def test_output_cap_never_exceeds_tpm_bucket(self):
        os.environ["NVIDIA_TPM"] = "1000"
        budget = models.request_budget("some/model", "nvidia")
        # tpm - output - 200 would be negative, so the floor of 400 applies.
        self.assertEqual(budget.max_input_tokens, 400)

    def test_local_provider_uses_its_larger_budget(self):
        budget = models.request_budget("qwen2.5-coder", "ollama")
        self.assertEqual(budget.max_input_tokens, 32_000)
        self.assertEqual(budget.source, "local")
        self.assertFalse(budget.rate_limits.any)


if __name__ == "__main__":
    unittest.main()
