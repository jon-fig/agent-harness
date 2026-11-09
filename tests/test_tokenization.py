"""Tokenizer resolution and approximate-fallback behaviour."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import tokenization


def _clear_tokenizer_env() -> None:
    for key in [key for key in os.environ if key.startswith("TOKENIZER_")]:
        del os.environ[key]


class TokenizerTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("TOKENIZER_")
        }
        _clear_tokenizer_env()
        tokenization.clear_cache()

        # Keep a real .agent/tokenizer.json from influencing resolution here.
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        patcher = mock.patch.object(
            tokenization,
            "CALIBRATION_FILE",
            Path(self._directory.name) / "tokenizer.json",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        _clear_tokenizer_env()
        os.environ.update(self._saved)
        tokenization.clear_cache()

    def test_openai_family_resolves_tiktoken(self):
        tokenizer = tokenization.resolve_tokenizer("gpt-4o", "openai")
        if tokenization.tiktoken is None:
            # Without the optional package the resolver must degrade, not fail.
            self.assertFalse(tokenizer.exact)
            return
        self.assertTrue(tokenizer.exact)
        self.assertEqual(tokenizer.name, "tiktoken:o200k_base")

    def test_gpt_oss_resolves_harmony_encoding(self):
        tokenizer = tokenization.resolve_tokenizer("openai/gpt-oss-120b", "groq")
        if tokenization.tiktoken is None:
            self.assertFalse(tokenizer.exact)
            return
        self.assertTrue(tokenizer.exact)
        self.assertEqual(tokenizer.name, "tiktoken:o200k_harmony")

    def test_unknown_family_falls_back_to_approximation(self):
        tokenizer = tokenization.resolve_tokenizer("qwen/qwen3-8b", "groq")
        self.assertFalse(tokenizer.exact)
        self.assertEqual(tokenizer.name, "approx")
        # The estimate rounds up so the guard rail never undercounts.
        self.assertEqual(tokenizer.count("a" * 5), 2)

    def test_per_model_env_spec_wins(self):
        os.environ["TOKENIZER_QWEN_QWEN3_8B"] = "tiktoken:cl100k_base"
        tokenizer = tokenization.resolve_tokenizer("qwen/qwen3-8b", "groq")
        if tokenization.tiktoken is None:
            self.assertFalse(tokenizer.exact)
            return
        self.assertTrue(tokenizer.exact)
        self.assertEqual(tokenizer.name, "tiktoken:cl100k_base")

    def test_map_spec_matches_by_pattern(self):
        os.environ["TOKENIZER_MAP"] = '{"qwen/*": "tiktoken:cl100k_base"}'
        tokenizer = tokenization.resolve_tokenizer("qwen/qwen3-8b", "groq")
        if tokenization.tiktoken is None:
            self.assertFalse(tokenizer.exact)
            return
        self.assertEqual(tokenizer.name, "tiktoken:cl100k_base")

    def test_default_spec_applies_to_every_model(self):
        os.environ["TOKENIZER_DEFAULT"] = "approx:6"
        tokenizer = tokenization.resolve_tokenizer("qwen/qwen3-8b", "groq")
        self.assertFalse(tokenizer.exact)
        self.assertEqual(tokenizer.chars_per_token, 6)
        self.assertEqual(tokenizer.count("a" * 7), 2)

    def test_unknown_tiktoken_encoding_degrades_to_approximation(self):
        os.environ["TOKENIZER_DEFAULT"] = "tiktoken:not-a-real-encoding"
        tokenizer = tokenization.resolve_tokenizer("some/model", "custom")
        self.assertFalse(tokenizer.exact)
        self.assertIn("approx", tokenizer.name)

    def test_missing_tiktoken_package_degrades(self):
        with mock.patch.object(tokenization, "tiktoken", None):
            tokenization.clear_cache()
            tokenizer = tokenization.resolve_tokenizer("gpt-4o", "openai")
        self.assertFalse(tokenizer.exact)
        self.assertEqual(tokenizer.name, "approx")

    def test_string_estimate_is_exact_but_payload_is_not(self):
        string_estimate = tokenization.estimate_value("hello world", "gpt-4o", "openai")
        payload_estimate = tokenization.estimate_value(
            [{"role": "user", "content": "hello world"}], "gpt-4o", "openai"
        )
        if tokenization.tiktoken is None:
            self.assertFalse(string_estimate.exact)
        else:
            self.assertTrue(string_estimate.exact)
            self.assertEqual(string_estimate.note, "")
        # Chat framing is never represented by a raw tokenizer, so a structured
        # payload must stay approximate even with an exact tokenizer.
        self.assertFalse(payload_estimate.exact)
        self.assertIn("chat framing", payload_estimate.note)

    def test_estimate_tokens_matches_estimate_value(self):
        value = {"role": "user", "content": "count me"}
        self.assertEqual(
            tokenization.estimate_tokens(value, "some/model", "custom"),
            tokenization.estimate_value(value, "some/model", "custom").tokens,
        )

    def test_describe_labels_approximation(self):
        estimate = tokenization.TokenEstimate(
            tokens=42, exact=False, tokenizer="approx", note="character-based fallback"
        )
        text = estimate.describe()
        self.assertIn("approximate", text)
        self.assertIn("42", text)


if __name__ == "__main__":
    unittest.main()
