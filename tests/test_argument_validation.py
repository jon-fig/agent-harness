"""Argument validation against advertised tool schemas.

The MCP server advertises every tool with an empty properties object, so the
harness must not treat that as a closed set: it rejected even the arguments the
server itself expects, which blocked a model from recovering after the server
named the required field in its own validation error.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import agent


class EmptySchemaTestCase(unittest.TestCase):
    def test_empty_properties_does_not_reject_arguments(self):
        self.assertIsNone(
            agent._validate_arguments(
                "read_file", {"type": "object", "properties": {}, "required": []},
                {"file_path": "notes.txt"},
            )
        )

    def test_arguments_are_forwarded_when_nothing_is_declared(self):
        self.assertIsNone(agent._validate_arguments("run_python_code", None, {"code": "print(1)"}))

    def test_declared_properties_still_reject_unknown_keys(self):
        problem = agent._validate_arguments(
            "save_memory",
            {
                "type": "object",
                "properties": {"memory": {"type": "string"}},
                "required": ["memory"],
                "additionalProperties": False,
            },
            {"memories": "note"},
        )
        self.assertIn("does not accept argument(s) 'memories'", problem)
        self.assertIn("Allowed: memory", problem)

    def test_declared_types_are_still_checked(self):
        problem = agent._validate_arguments(
            "save_memory",
            {
                "type": "object",
                "properties": {"memory": {"type": "string"}},
                "required": ["memory"],
                "additionalProperties": False,
            },
            {"memory": 3},
        )
        self.assertIn("must be string, got int", problem)

    def test_missing_required_arguments_are_still_reported(self):
        problem = agent._validate_arguments(
            "save_memory",
            {
                "type": "object",
                "properties": {"memory": {"type": "string"}},
                "required": ["memory"],
                "additionalProperties": False,
            },
            {},
        )
        self.assertIn("missing required argument(s) 'memory'", problem)


class WorkspacePathBoundaryTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.workspace = base / "workspace"
        self.workspace.mkdir()
        self.outside = base / "outside"
        self.outside.mkdir()
        self.outside_file = self.outside / "note.txt"
        self.outside_file.write_text("fixture only", encoding="utf-8")
        patcher = mock.patch.object(agent.config, "WORKSPACE_DIR", self.workspace)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_rejects_traversal_for_all_filesystem_tools(self):
        for name, args in (
            ("read_file", {"file_path": "../outside/note.txt"}),
            ("write_file", {"file_path": "../outside/new.txt"}),
            ("list_directory", {"directory_path": "../outside"}),
            ("run_python_file", {"file_path": "../outside/job.py"}),
        ):
            with self.subTest(name=name):
                error = agent._workspace_path_error(name, args)
                self.assertIn("outside the workspace", error)

    def test_rejects_absolute_paths_outside_workspace(self):
        error = agent._workspace_path_error(
            "read_file", {"file_path": str(self.outside_file)}
        )
        self.assertIn("outside the workspace", error)

    def test_rejects_symlink_escape_for_existing_and_new_paths(self):
        link = self.workspace / "external"
        try:
            link.symlink_to(self.outside, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")

        for tool, key, value in (
            ("read_file", "file_path", "external/note.txt"),
            ("write_file", "file_path", "external/new.txt"),
            ("list_directory", "directory_path", "external"),
            ("run_python_file", "file_path", "external/job.py"),
        ):
            with self.subTest(tool=tool):
                self.assertIn(
                    "outside the workspace",
                    agent._workspace_path_error(tool, {key: value}),
                )

    def test_checks_save_as_after_server_python_suffix_normalization(self):
        self.assertIn(
            "outside the workspace",
            agent._workspace_path_error(
                "run_python_code", {"save_as": "../outside/generated"}
            ),
        )

    def test_allows_workspace_paths_and_empty_directory_root(self):
        self.assertIsNone(
            agent._workspace_path_error("write_file", {"file_path": "new.txt"})
        )
        self.assertIsNone(
            agent._workspace_path_error("list_directory", {"directory_path": ""})
        )
        self.assertIsNone(
            agent._workspace_path_error(
                "run_python_code", {"save_as": "generated.py"}
            )
        )


if __name__ == "__main__":
    unittest.main()
