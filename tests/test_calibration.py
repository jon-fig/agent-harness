"""Self-calibration of the approximate tokenizer from reported usage."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import config, tokenization

MODEL = "minimaxai/minimax-m3.1-flash"
OTHER = "deepseek-ai/deepseek-v4.1-flash"


class CalibrationTestCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._file = Path(self._directory.name) / "tokenizer.json"

        patcher = mock.patch.object(tokenization, "CALIBRATION_FILE", self._file)
        patcher.start()
        self.addCleanup(patcher.stop)

        self._saved_env = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("TOKENIZER_")
        }
        for key in [key for key in os.environ if key.startswith("TOKENIZER_")]:
            del os.environ[key]
        tokenization.clear_cache()

    def tearDown(self):
        tokenization.clear_cache()
        for key in [key for key in os.environ if key.startswith("TOKENIZER_")]:
            del os.environ[key]
        os.environ.update(self._saved_env)

    def test_record_learns_the_ratio(self):
        value = tokenization.record_calibration(MODEL, "nvidia", 2790, 1000)
        self.assertAlmostEqual(value, 2.79, places=4)

        tokenizer = tokenization.resolve_tokenizer(MODEL, "nvidia")
        self.assertFalse(tokenizer.exact)
        self.assertIn("calibrated", tokenizer.name)
        self.assertEqual(tokenizer.count("x" * 2790), 1000)

    def test_observations_are_clamped(self):
        dense = tokenization.record_calibration(MODEL, "nvidia", 10_000_000, 1)
        sparse = tokenization.record_calibration(OTHER, "nvidia", 1, 1_000_000)
        self.assertEqual(dense, tokenization.MAX_CHARS_PER_TOKEN)
        self.assertEqual(sparse, tokenization.MIN_CHARS_PER_TOKEN)

    def test_repeated_observations_are_smoothed(self):
        first = tokenization.record_calibration(MODEL, "nvidia", 4000, 1000)
        second = tokenization.record_calibration(MODEL, "nvidia", 3000, 1000)
        self.assertAlmostEqual(first, 4.0, places=4)
        # 0.5 * 4.0 + 0.5 * 3.0, so one unusual request cannot dominate.
        self.assertAlmostEqual(second, 3.5, places=4)

    def test_invalid_metadata_is_ignored(self):
        self.assertIsNone(tokenization.record_calibration(MODEL, "nvidia", 1000, 0))
        self.assertIsNone(tokenization.record_calibration(MODEL, "nvidia", 1000, None))
        self.assertIsNone(tokenization.record_calibration(MODEL, "nvidia", 0, 100))
        self.assertIsNone(tokenization.record_calibration("", "nvidia", 1000, 100))
        self.assertIsNone(tokenization.calibrated_chars_per_token(MODEL, "nvidia"))
        self.assertFalse(self._file.exists())

    def test_state_is_persisted_per_provider_and_model(self):
        tokenization.record_calibration(MODEL, "nvidia", 2790, 1000)
        tokenization.record_calibration(OTHER, "openrouter", 4000, 1000)

        data = json.loads(self._file.read_text(encoding="utf-8"))
        self.assertEqual(data["version"], 1)
        self.assertIn(f"nvidia/{MODEL}", data["models"])
        self.assertIn(f"openrouter/{OTHER}", data["models"])
        self.assertAlmostEqual(
            tokenization.calibrated_chars_per_token(OTHER, "openrouter"), 4.0
        )

    def test_estimate_uses_the_calibrated_ratio(self):
        self.assertEqual(tokenization.estimate_tokens("x" * 1000, MODEL, "nvidia"), 250)
        tokenization.record_calibration(MODEL, "nvidia", 2500, 1000)
        # The cached tokenizer is dropped, so the new ratio applies immediately.
        self.assertEqual(tokenization.estimate_tokens("x" * 1000, MODEL, "nvidia"), 400)

    def test_exact_tokenizer_is_not_replaced_by_calibration(self):
        tokenization.record_calibration("gpt-4o", "openai", 1000, 500)
        tokenizer = tokenization.resolve_tokenizer("gpt-4o", "openai")
        if tokenization.tiktoken is None:
            self.assertFalse(tokenizer.exact)
            return
        self.assertTrue(tokenizer.exact)
        self.assertEqual(tokenizer.name, "tiktoken:o200k_base")

    def test_calibration_can_be_disabled(self):
        with mock.patch.object(config, "TOKENIZER_CALIBRATION", False):
            self.assertIsNone(tokenization.record_calibration(MODEL, "nvidia", 2000, 1000))
        self.assertIsNone(tokenization.calibrated_chars_per_token(MODEL, "nvidia"))

    def test_request_characters_counts_payload_and_schemas(self):
        payload = [{"role": "user", "content": "hi"}]
        schemas = [{"type": "function", "function": {"name": "read_file"}}]
        self.assertEqual(
            tokenization.request_characters(payload, schemas),
            len(tokenization._serialize(payload)[0])
            + len(tokenization._serialize(schemas)[0]),
        )
        self.assertEqual(
            tokenization.request_characters(payload),
            len(tokenization._serialize(payload)[0]),
        )


if __name__ == "__main__":
    unittest.main()
