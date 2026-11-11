"""`--no-session` must not touch the user's real conversation file.

The incident: the harness was started with `--no-session`, but the exit path
still called save_session([]) and replaced `.agent/session.json` with an empty
conversation. The flag now disables loading, saving and clearing at the
persistence boundary, so no caller can bypass it.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src import main as main_module
from src import session


class SessionSafetyTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "session.json"
        self.saved = {"version": 1, "messages": [{"role": "user", "content": "keep me"}]}
        self.path.write_text(json.dumps(self.saved), encoding="utf-8")
        self.original = self.path.read_bytes()

        patcher = mock.patch.object(session, "SESSION_FILE", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        session.set_session_enabled(True)
        self.addCleanup(session.set_session_enabled, True)

    def _endpoint(self):
        return SimpleNamespace(
            label="Test endpoint",
            name="ollama",
            base_url="http://localhost:11434/v1",
            api_key_source="env",
            local=True,
        )

    def _run_main(self, argv, inputs, run_agent=None):
        """Runs the REPL with canned input, without touching a real provider."""
        output = io.StringIO()
        agent = run_agent or mock.Mock(return_value="ok")
        with mock.patch.object(sys, "argv", ["agent", *argv]), mock.patch.object(
            main_module.providers, "resolve", return_value=self._endpoint()
        ), mock.patch.object(
            main_module.models, "active", return_value="test-model"
        ), mock.patch.object(
            main_module, "_describe", return_value="1 messages"
        ), mock.patch.object(
            main_module, "_budget_line", return_value="test budget"
        ), mock.patch.object(
            main_module, "run_agent", agent
        ), mock.patch(
            "builtins.input", side_effect=list(inputs)
        ), contextlib.redirect_stdout(output):
            main_module.main()
        return agent

    def test_no_session_disables_load_save_and_clear(self):
        session.set_session_enabled(False)

        self.assertEqual(session.load_session(), [])
        session.save_session([{"role": "user", "content": "do not persist"}])
        session.clear_session_file()

        self.assertTrue(self.path.exists())
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_a_no_session_launch_and_exit_leave_the_file_untouched(self):
        self._run_main(["--no-session"], [EOFError()])

        self.assertTrue(self.path.exists())
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_an_interrupted_no_session_run_also_leaves_the_file_untouched(self):
        self._run_main(
            ["--no-session"],
            ["a prompt that fails", EOFError()],
            run_agent=mock.Mock(side_effect=main_module.AgentError("provider down")),
        )

        self.assertTrue(self.path.exists())
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_normal_saving_still_works(self):
        messages = [{"role": "user", "content": "new conversation"}]
        session.save_session(messages)

        self.assertEqual(session.load_session(), messages)
        self.assertNotEqual(self.path.read_bytes(), self.original)

    def test_a_failed_normal_run_keeps_the_existing_conversation(self):
        self._run_main(
            [],
            ["a prompt that fails", EOFError()],
            run_agent=mock.Mock(side_effect=main_module.AgentError("provider down")),
        )

        # Normal mode rewrites the file (saved_at changes), but the conversation
        # must still be there rather than replaced by an empty list.
        self.assertEqual(
            session.load_session(), self.saved["messages"]
        )
        self.assertTrue(self.path.exists())


if __name__ == "__main__":
    unittest.main()
