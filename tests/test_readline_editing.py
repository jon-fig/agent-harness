"""The CLI prompt must have line editing.

CPython enables line editing for input() only once the readline module is
loaded, and it does that automatically only in the interactive REPL — a module
run ("python -m src.main") starts without it. Before the fix, arrow keys, Home/
End and Ctrl+A/E were echoed as raw escape sequences at the "You:" prompt and
backspace could not cross a wrapped line.
"""

import sys
import unittest


@unittest.skipIf(sys.platform == "win32", "the Windows console edits lines natively")
class ReadlineEditingTestCase(unittest.TestCase):
    def test_importing_the_cli_loads_readline(self):
        import src.main  # noqa: F401

        self.assertIn("readline", sys.modules)

    def test_approval_prompt_module_loads_readline(self):
        import src.approval  # noqa: F401

        self.assertIn("readline", sys.modules)


if __name__ == "__main__":
    unittest.main()
