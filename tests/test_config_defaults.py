"""Defaults in src/config.py, and the documentation that advertises them.

The defaults are reloaded with .env and the shell environment scrubbed, so a
developer's local .env cannot make this test pass or fail.
"""

import importlib
import os
import re
import unittest
from unittest import mock

from src import config

EXPECTED_INT = {
    "MAX_ITERATIONS": 8,
    "MAX_TOOL_CALLS_PER_TURN": 20,
    "MAX_OUTPUT_TOKENS": 4000,
    "MAX_INPUT_TOKENS": 6500,
    "TOOL_OUTPUT_MAX_CHARS": 4000,
    "MEMORY_MAX_CHARS": 1200,
    "HISTORY_KEEP_EXCHANGES": 2,
    "AUTO_COMPACT_MIN_TURNS": 2,
    "OLLAMA_MAX_INPUT_TOKENS": 32000,
}
EXPECTED_FLOAT = {
    "LLM_TIMEOUT": 120.0,
    "LLM_MAX_RETRIES": 2.0,
    "AUTO_COMPACT_THRESHOLD": 0.6,
    "AUTO_COMPACT_TARGET": 0.5,
}

_TRACKED = (
    tuple(EXPECTED_INT)
    + tuple(EXPECTED_FLOAT)
    + (
        "AUTO_COMPACT",
        "DEBUG_TOKENS",
        # Legacy fallbacks read by config.py. They can be set in the developer's
        # shell (this environment had AUTOCOMPACT_AFTER_TURNS=3), so a defaults
        # test must scrub them as well or it tests the review environment.
        "AUTOCOMPACT_AFTER_TURNS",
        "AUTOCOMPACT_TOKEN_FRACTION",
    )
)


class ConfigDefaultsTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = {key: os.environ.pop(key, None) for key in _TRACKED}
        # Patching _load_env itself would not survive the reload (the def statement
        # rebinds it), so the .env reader is neutralised at its source instead.
        self._patcher = mock.patch("dotenv.dotenv_values", return_value={})
        self._patcher.start()
        importlib.reload(config)

    def tearDown(self):
        self._patcher.stop()
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(config)

    def test_integer_defaults(self):
        for name, expected in EXPECTED_INT.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(config, name), expected)

    def test_float_defaults(self):
        for name, expected in EXPECTED_FLOAT.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(config, name), expected)

    def test_character_limits_are_separate_from_token_limits(self):
        # A char cap is not a token cap: they are different settings and the
        # tool-output cap is not derived from the generation limit.
        self.assertEqual(config.TOOL_OUTPUT_MAX_CHARS, 4000)
        self.assertEqual(config.MAX_OUTPUT_TOKENS, 4000)
        self.assertIsInstance(config.TOOL_OUTPUT_MAX_CHARS, int)

    def test_environment_values_control_the_settings(self):
        with mock.patch.dict(
            os.environ,
            {"MAX_OUTPUT_TOKENS": "777", "TOOL_OUTPUT_MAX_CHARS": "888"},
            clear=False,
        ):
            importlib.reload(config)
            self.assertEqual(config.MAX_OUTPUT_TOKENS, 777)
            self.assertEqual(config.TOOL_OUTPUT_MAX_CHARS, 888)

    def test_documented_defaults_match_the_code(self):
        text = (config.ROOT_DIR / ".env.example").read_text(encoding="utf-8")
        documented = {}
        for line in text.splitlines():
            match = re.fullmatch(r"([A-Z0-9_]+)=(\S+)", line.strip())
            if match:
                documented[match.group(1)] = match.group(2)
        for name, expected in {**EXPECTED_INT, **EXPECTED_FLOAT}.items():
            with self.subTest(name=name):
                self.assertIn(name, documented, f"{name} is not documented in .env.example")
                self.assertEqual(float(documented[name]), float(expected))


if __name__ == "__main__":
    unittest.main()
