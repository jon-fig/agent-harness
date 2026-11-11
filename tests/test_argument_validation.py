"""Argument validation against advertised tool schemas.

The MCP server advertises every tool with an empty properties object, so the
harness must not treat that as a closed set: it rejected even the arguments the
server itself expects, which blocked a model from recovering after the server
named the required field in its own validation error.
"""

import unittest

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


if __name__ == "__main__":
    unittest.main()
