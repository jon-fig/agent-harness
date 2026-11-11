"""One virtual environment serves the harness, the server and the code tools.

The unit tests pin the resolution rules; the runtime tests start the real MCP
server and prove that both Python tools report the interpreter that runs the
harness. The runtime class is skipped when the server is not installed.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import config


def _json_from(text: str) -> dict:
    for line in reversed(str(text).splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    raise AssertionError(f"no JSON report in tool output: {text!r}")


def _script_name() -> str:
    return "mcp-python-interpreter.exe" if os.name == "nt" else "mcp-python-interpreter"


class ServerResolutionTestCase(unittest.TestCase):
    def test_console_script_resolves_from_the_active_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "some-venv"
            script = config.environment_bin_dir(prefix) / _script_name()
            script.parent.mkdir(parents=True)
            script.write_text("")
            with mock.patch.object(sys, "prefix", str(prefix)), mock.patch.object(
                config, "ROOT_DIR", Path(tmp) / "repo"
            ):
                self.assertEqual(config.resolve_mcp_server_command(), str(script))

    def test_no_fallback_to_the_project_venv_from_another_environment(self):
        # A system interpreter must not launch a server from the checkout's .venv
        # while passing its own executable as --python-path.
        with tempfile.TemporaryDirectory() as tmp:
            active = Path(tmp) / "active"
            project = Path(tmp) / "repo" / ".venv"
            script = config.environment_bin_dir(project) / _script_name()
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("")
            with mock.patch.object(sys, "prefix", str(active)), mock.patch.object(
                config, "ROOT_DIR", Path(tmp) / "repo"
            ):
                resolved = config.resolve_mcp_server_command()
            self.assertEqual(resolved, "")

    def test_a_global_interpreter_is_rejected_clearly(self):
        with mock.patch.object(sys, "prefix", "/usr"), mock.patch.object(
            sys, "base_prefix", "/usr"
        ), mock.patch.object(sys, "executable", "/usr/bin/python3"):
            problem = config.interpreter_problem()
            self.assertIn("not running inside a virtual environment", problem)
            self.assertIn(".venv", problem)
            self.assertIn("-m src.main", problem)
            with self.assertRaises(RuntimeError):
                config.require_supported_interpreter()

    def test_a_virtual_environment_is_accepted(self):
        with mock.patch.object(sys, "prefix", "/repo/.venv"), mock.patch.object(
            sys, "base_prefix", "/usr"
        ):
            self.assertIsNone(config.interpreter_problem())
            config.require_supported_interpreter()

    def test_no_bare_command_fallback_to_path(self):
        # A missing install must not silently resolve a global server: "" makes
        # the bridge raise an install instruction instead.
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            sys, "prefix", str(Path(tmp) / "empty")
        ), mock.patch.object(config, "ROOT_DIR", Path(tmp) / "repo"), mock.patch(
            "shutil.which", return_value="/usr/local/bin/mcp-python-interpreter"
        ):
            self.assertEqual(config.resolve_mcp_server_command(), "")

    def test_script_directory_is_platform_specific(self):
        # The mapping itself is asserted rather than both branches at once: on a
        # POSIX host, os.name="nt" would make pathlib instantiate WindowsPath.
        expected_dir = "Scripts" if os.name == "nt" else "bin"
        self.assertEqual(config.BIN_DIR_NAME, expected_dir)
        self.assertEqual(
            config.environment_bin_dir(Path("env")), Path("env") / expected_dir
        )
        expected_name = (
            "mcp-python-interpreter.exe"
            if os.name == "nt"
            else "mcp-python-interpreter"
        )
        self.assertEqual(config.mcp_server_candidates()[0].name, expected_name)

    def test_server_is_launched_with_the_harness_interpreter(self):
        self.assertEqual(config.MCP_PYTHON, sys.executable)
        args = list(config.MCP_SERVER_ARGS)
        self.assertIn("--python-path", args)
        self.assertEqual(args[args.index("--python-path") + 1], sys.executable)
        self.assertEqual(args[args.index("--dir") + 1], str(config.WORKSPACE_DIR))

    def test_every_path_points_into_the_active_environment(self):
        bin_dir = config.environment_bin_dir(sys.prefix)
        self.assertEqual(Path(config.MCP_PYTHON).parent, bin_dir)
        if config.MCP_SERVER_COMMAND:
            self.assertEqual(Path(config.MCP_SERVER_COMMAND).parent, bin_dir)


@unittest.skipUnless(
    config.MCP_SERVER_COMMAND, "the MCP server is not installed in this environment"
)
class UnifiedRuntimeTestCase(unittest.TestCase):
    """Starts the real server once and checks the environment it executes in."""

    @classmethod
    def setUpClass(cls):
        from src.mcp_client import MCPToolBridge

        cls.bridge = MCPToolBridge(
            command=config.MCP_SERVER_COMMAND,
            args=list(config.MCP_SERVER_ARGS),
            cwd=str(config.WORKSPACE_DIR),
            startup_timeout=config.MCP_STARTUP_TIMEOUT,
            call_timeout=120.0,
        )
        cls.bridge.start()

    @classmethod
    def tearDownClass(cls):
        cls.bridge.close()

    def _probe(self) -> str:
        return (
            "import json, os, sys\n"
            "print(json.dumps({'exe': sys.executable, 'prefix': sys.prefix,"
            " 'base': sys.base_prefix, 'cwd': os.getcwd()}))\n"
        )

    def test_server_starts_and_lists_the_expected_tools(self):
        names = {schema["function"]["name"] for schema in self.bridge.openai_schemas()}
        for expected in (
            "run_python_code",
            "run_python_file",
            "read_file",
            "write_file",
            "list_directory",
            "list_installed_packages",
            "list_python_environments",
            "install_package",
        ):
            self.assertIn(expected, names)

    def test_inline_and_file_execution_share_one_interpreter(self):
        inline = _json_from(
            self.bridge.call(
                "run_python_code", {"code": self._probe(), "execution_mode": "inline"}
            )
        )
        script = config.WORKSPACE_DIR / "_env_probe_test.py"
        script.write_text(self._probe(), encoding="utf-8")
        self.addCleanup(lambda: script.unlink(missing_ok=True))
        on_file = _json_from(
            self.bridge.call("run_python_file", {"file_path": script.name})
        )

        harness = Path(sys.executable).resolve()
        self.assertEqual(Path(inline["exe"]).resolve(), harness)
        self.assertEqual(Path(on_file["exe"]).resolve(), harness)
        # sys.executable, sys.prefix and the environment path are consistent.
        self.assertEqual(Path(inline["prefix"]).resolve(), Path(sys.prefix).resolve())
        self.assertEqual(inline["prefix"], on_file["prefix"])
        self.assertNotEqual(inline["prefix"], inline["base"])
        # Generated scripts can use workspace-relative paths in both tools.
        self.assertEqual(Path(inline["cwd"]).resolve(), config.WORKSPACE_DIR.resolve())
        self.assertEqual(Path(on_file["cwd"]).resolve(), config.WORKSPACE_DIR.resolve())

    def test_package_discovery_reports_the_same_environment(self):
        listing = str(
            self.bridge.call("list_installed_packages", {"environment": "default"})
        )
        self.assertIn("Installed Packages", listing)
        self.assertIn("mcp", listing.lower())

        environments = str(self.bridge.call("list_python_environments", {}))
        self.assertIn(str(Path(sys.executable)), environments)

    @unittest.skipUnless(
        importlib.util.find_spec("pandas") is not None, "pandas is not installed"
    )
    def test_a_package_in_the_main_environment_is_importable_from_both_tools(self):
        code = "import pandas as pd\nprint('PANDAS=' + pd.__version__)\n"
        inline = str(
            self.bridge.call(
                "run_python_code", {"code": code, "execution_mode": "inline"}
            )
        )
        script = config.WORKSPACE_DIR / "_pandas_probe_test.py"
        script.write_text(code, encoding="utf-8")
        self.addCleanup(lambda: script.unlink(missing_ok=True))
        on_file = str(self.bridge.call("run_python_file", {"file_path": script.name}))

        def version(output: str) -> str:
            self.assertNotIn("Traceback", output)
            self.assertNotIn("ModuleNotFoundError", output)
            line = next(
                (item.strip() for item in output.splitlines() if "PANDAS=" in item), ""
            )
            return line.split("PANDAS=", 1)[1].strip()

        self.assertEqual(version(inline), version(on_file))
        self.assertTrue(version(inline))


if __name__ == "__main__":
    unittest.main()
