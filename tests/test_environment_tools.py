"""Package installation approval, validation and environment targeting.

The important assertion in this file is that the *harness* asks: dispatch() routes
install_package through src.environment, so a model that never asks still cannot
install anything.
"""

import json
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from src import agent, approval, config, environment


class FakeBridge:
    """Records calls and answers with canned tool output."""

    def __init__(self, responses=None, error=None):
        self.calls: list[tuple[str, dict]] = []
        self.responses = responses or {}
        self.error = error

    def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if self.error is not None:
            raise self.error
        return self.responses.get(name, "ok")

    def openai_schemas(self):
        return []

    def names(self):
        return [name for name, _ in self.calls]


class RequirementValidationTestCase(unittest.TestCase):
    def test_accepts_a_name_and_a_pin(self):
        self.assertEqual(environment.parse_requirement("pandas"), ("pandas", "pandas"))
        self.assertEqual(
            environment.parse_requirement(" pandas==2.2.0 "), ("pandas", "pandas==2.2.0")
        )
        self.assertEqual(
            environment.parse_requirement("scikit-learn>=1.4"),
            ("scikit-learn", "scikit-learn>=1.4"),
        )

    def test_rejects_options_urls_paths_and_shell_syntax(self):
        for bad in (
            "",
            "   ",
            "--index-url=http://example.invalid/simple",
            "-e .",
            "pandas; rm -rf /",
            "pandas && echo hi",
            "git+https://example.invalid/x.git",
            "pandas[extra]",
            "../evil",
            "pandas @ https://example.invalid/pandas.whl",
            "pandas/../../x",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                environment.parse_requirement(bad)

    def test_module_name_maps_distributions_to_imports(self):
        self.assertEqual(environment.module_name("pandas"), "pandas")
        self.assertEqual(environment.module_name("scikit-learn"), "sklearn")
        self.assertEqual(environment.module_name("Pillow"), "PIL")
        self.assertEqual(environment.module_name("PyYAML"), "yaml")
        # The regression: normalising "-" to "_" imports python_dotenv, which does
        # not exist; the real module is dotenv.
        self.assertEqual(environment.module_name("python-dotenv"), "dotenv")

    def test_installed_metadata_wins_over_name_normalisation(self):
        mapping = {
            "custom_import": ["some-dist"],
            "dotenv": ["python-dotenv"],
            "pandas": ["pandas"],
        }
        with mock.patch.object(
            environment.importlib_metadata,
            "packages_distributions",
            return_value=mapping,
        ):
            self.assertEqual(environment.module_name("some-dist"), "custom_import")
            self.assertEqual(environment.module_names("some-dist"), ["custom_import"])
            self.assertEqual(environment.module_name("python-dotenv"), "dotenv")

    def test_normalisation_is_only_the_fallback_when_metadata_is_unavailable(self):
        with mock.patch.object(
            environment.importlib_metadata,
            "packages_distributions",
            return_value={},
        ):
            self.assertEqual(environment.module_name("pandas"), "pandas")
            self.assertEqual(environment.module_name("some-package"), "some_package")


class InstallApprovalTestCase(unittest.TestCase):
    def setUp(self):
        approval.set_approver(None)
        self.addCleanup(approval.set_approver, None)

    def test_refuses_to_install_into_a_global_interpreter(self):
        with mock.patch.object(sys, "prefix", "/usr"), mock.patch.object(
            sys, "base_prefix", "/usr"
        ):
            result = environment.install_package({"package_name": "pandas"})
        self.assertIn("refusing to install", result)

    def test_no_approval_means_no_install(self):
        approval.set_approver(lambda prompt: False)
        bridge = FakeBridge()
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            result = environment.install_package({"package_name": "pandas==2.2.0"})
        self.assertIn("not approved", result)
        self.assertEqual(bridge.calls, [])

    def test_validation_runs_before_the_user_is_asked(self):
        asked: list[str] = []
        approval.set_approver(lambda prompt: asked.append(prompt) or True)
        bridge = FakeBridge()
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            result = environment.install_package({"package_name": "--index-url=http://x"})
        self.assertTrue(result.startswith("Error"))
        self.assertEqual(asked, [])
        self.assertEqual(bridge.calls, [])

    def test_upgrade_is_refused(self):
        approval.set_approver(lambda prompt: True)
        bridge = FakeBridge()
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            result = environment.install_package(
                {"package_name": "pandas", "upgrade": True}
            )
        self.assertIn("upgrade is not allowed", result)
        self.assertEqual(bridge.calls, [])

    def test_approval_targets_the_unified_environment_and_verifies_the_import(self):
        prompts: list[str] = []
        approval.set_approver(lambda prompt: prompts.append(prompt) or True)
        report = {
            "package": "pandas",
            "installed": True,
            "version": "2.2.0",
            "modules": ["pandas"],
            "source": "metadata",
            "imports": {"pandas": None},
        }
        bridge = FakeBridge(
            {
                "install_package": "Successfully installed pandas==2.2.0 in system.",
                "run_python_code": json.dumps(report),
            }
        )
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            result = environment.install_package(
                {"package_name": "pandas==2.2.0", "environment": "system"}
            )

        self.assertEqual(len(prompts), 1)
        self.assertIn("pandas==2.2.0", prompts[0])
        self.assertIn(config.MCP_PYTHON, prompts[0])

        install_name, install_args = bridge.calls[0]
        self.assertEqual(install_name, "install_package")
        self.assertEqual(install_args["package_name"], "pandas==2.2.0")
        # Never "default"/"system" chosen by the model, never another conda env.
        self.assertEqual(install_args["environment"], "system")

        verify_name, verify_args = bridge.calls[1]
        self.assertEqual(verify_name, "run_python_code")
        self.assertIn('"pandas"', verify_args["code"])
        self.assertIn("Verified", result)
        self.assertIn("2.2.0", result)
        self.assertIn("import pandas", result)

    def test_python_dotenv_is_verified_through_its_real_import_name(self):
        approval.set_approver(lambda prompt: True)
        report = {
            "package": "python-dotenv",
            "installed": True,
            "version": "1.0.1",
            "modules": ["dotenv"],
            "source": "metadata",
            "imports": {"dotenv": None},
        }
        bridge = FakeBridge(
            {
                "install_package": "Successfully installed python-dotenv",
                "run_python_code": json.dumps(report),
            }
        )
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            result = environment.install_package({"package_name": "python-dotenv"})

        self.assertIn("import dotenv", result)
        self.assertNotIn("python_dotenv", result)

    def test_a_failed_install_is_reported(self):
        approval.set_approver(lambda prompt: True)
        bridge = FakeBridge({"install_package": "Error installing pandas: no network"})
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            result = environment.install_package({"package_name": "pandas"})
        self.assertIn("failed", result)
        self.assertIn("no network", result)
        self.assertEqual(bridge.names(), ["install_package"])

    def test_an_unimportable_package_is_reported_after_install(self):
        approval.set_approver(lambda prompt: True)
        report = {
            "package": "pandas",
            "installed": True,
            "version": "2.2.0",
            "modules": ["pandas"],
            "source": "metadata",
            "imports": {"pandas": "ModuleNotFoundError: pandas"},
        }
        bridge = FakeBridge(
            {
                "install_package": "Successfully installed pandas",
                "run_python_code": json.dumps(report),
            }
        )
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            result = environment.install_package({"package_name": "pandas"})
        self.assertIn("still fails", result)
        self.assertIn("ModuleNotFoundError", result)

    def test_a_wrong_fallback_import_name_is_not_an_install_failure(self):
        # The distribution is installed; only the guessed import name was wrong.
        # That is an unverified import, not a failed installation.
        approval.set_approver(lambda prompt: True)
        report = {
            "package": "some-dist",
            "installed": True,
            "version": "1.2.3",
            "modules": ["some_dist"],
            "source": "fallback",
            "imports": {"some_dist": "ModuleNotFoundError: some_dist"},
        }
        bridge = FakeBridge(
            {
                "install_package": "Successfully installed some-dist",
                "run_python_code": json.dumps(report),
            }
        )
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            result = environment.install_package({"package_name": "some-dist"})
        self.assertNotIn("Error", result)
        self.assertNotIn("still fails", result)
        self.assertIn("could not be confirmed", result)
        self.assertIn("Installed some-dist", result)


class DispatchApprovalTestCase(unittest.TestCase):
    """The gate lives in the tool-dispatch path, not in the model's prose."""

    def setUp(self):
        approval.set_approver(None)
        self.addCleanup(approval.set_approver, None)

    def test_dispatch_never_installs_without_approval(self):
        approval.set_approver(lambda prompt: False)
        bridge = FakeBridge()
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            _, dispatch = agent.get_tooling()
            result = dispatch("install_package", {"package_name": "pandas"})
        self.assertIn("not approved", result)
        self.assertEqual(bridge.calls, [])

    def test_dispatch_installs_only_after_approval(self):
        approval.set_approver(lambda prompt: True)
        bridge = FakeBridge(
            {
                "install_package": "Successfully installed pandas",
                "run_python_code": json.dumps(
                    {
                        "package": "pandas",
                        "installed": True,
                        "version": "2.2.0",
                        "modules": ["pandas"],
                        "source": "metadata",
                        "imports": {"pandas": None},
                    }
                ),
            }
        )
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            _, dispatch = agent.get_tooling()
            result = dispatch("install_package", {"package_name": "pandas"})
        self.assertEqual(bridge.names()[0], "install_package")
        self.assertIn("Installed", result)

    def test_deny_mode_blocks_even_when_no_prompt_is_possible(self):
        bridge = FakeBridge()
        with mock.patch.object(config, "MCP_INSTALL_APPROVAL", "deny"), mock.patch(
            "src.mcp_client.get_bridge", return_value=bridge
        ):
            _, dispatch = agent.get_tooling()
            result = dispatch("install_package", {"package_name": "pandas"})
        self.assertIn("not approved", result)
        self.assertEqual(bridge.calls, [])


class ToolCallRepresentationTestCase(unittest.TestCase):
    """Native, JSON-text and MiniMax-text calls share one dispatch path."""

    def setUp(self):
        approval.set_approver(None)
        self.addCleanup(approval.set_approver, None)
        self._ask = mock.patch.object(config, "MCP_INSTALL_APPROVAL", "ask")
        self._ask.start()
        self.addCleanup(self._ask.stop)

    def _run(self, response, bridge=None):
        bridge = bridge or FakeBridge(
            {
                "install_package": "Successfully installed pandas",
                "run_python_code": json.dumps(
                    {
                        "package": "pandas",
                        "installed": True,
                        "version": "2.2.0",
                        "modules": ["pandas"],
                        "source": "metadata",
                        "imports": {"pandas": None},
                    }
                ),
            }
        )
        final = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ]
        )
        messages: list[dict] = []
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge), mock.patch.object(
            agent, "get_llm_client", return_value=(object(), "test-model")
        ), mock.patch.object(
            agent, "_request_completion", side_effect=[response, final]
        ), mock.patch.object(
            agent, "build_system_prompt", return_value="system"
        ):
            result = agent.run_agent("install pandas", messages, provider="test")
        return result, messages, bridge

    def _tool_text(self, messages) -> str:
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        self.assertTrue(tool_messages)
        return str(tool_messages[-1].get("content"))

    @staticmethod
    def _native(name: str, arguments: str):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_native",
                                function=SimpleNamespace(name=name, arguments=arguments),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ]
        )

    @staticmethod
    def _json_text(name: str, arguments: dict):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"name": name, "arguments": arguments}),
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ]
        )

    @staticmethod
    def _minimax_text(name: str, arguments: dict):
        noise = "]<]minimax[>["
        lt, gt = chr(60), chr(62)
        params = "".join(
            f"{lt}{key}{gt}{value}{lt}/{key}{gt}" for key, value in arguments.items()
        )
        content = (
            f"{noise}{lt}tool_call{gt}\n"
            f'{lt}invoke name="{name}"{gt}{params}{lt}/invoke{gt}\n'
            f"{noise}{lt}/tool_call{gt}"
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, tool_calls=None),
                    finish_reason="stop",
                )
            ]
        )

    def test_install_requires_approval_for_every_representation(self):
        representations = {
            "native": self._native(
                "install_package", json.dumps({"package_name": "pandas"})
            ),
            "json-text": self._json_text(
                "install_package", {"package_name": "pandas"}
            ),
            "minimax-text": self._minimax_text(
                "install_package", {"package_name": "pandas"}
            ),
        }
        for label, response in representations.items():
            with self.subTest(representation=label):
                prompts: list[str] = []
                approval.set_approver(lambda prompt: prompts.append(prompt) or False)
                result, messages, bridge = self._run(response)
                self.assertEqual(result, "done")
                self.assertEqual(len(prompts), 1)
                self.assertIn("not approved", self._tool_text(messages))
                self.assertNotIn("install_package", bridge.names())

    def test_invalid_arguments_are_rejected_before_approval_for_every_representation(self):
        schema = {
            "type": "function",
            "function": {
                "name": "install_package",
                "parameters": {
                    "type": "object",
                    "properties": {"package_name": {"type": "string"}},
                    "required": ["package_name"],
                    "additionalProperties": False,
                },
            },
        }
        representations = {
            "native": self._native(
                "install_package", json.dumps({"package": "pandas"})
            ),
            "json-text": self._json_text("install_package", {"package": "pandas"}),
            "minimax-text": self._minimax_text(
                "install_package", {"package": "pandas"}
            ),
        }
        for label, response in representations.items():
            with self.subTest(representation=label):
                prompts: list[str] = []
                approval.set_approver(lambda prompt: prompts.append(prompt) or True)
                bridge = FakeBridge()
                bridge.openai_schemas = lambda: [schema]
                result, messages, bridge = self._run(response, bridge=bridge)
                self.assertEqual(result, "done")
                self.assertEqual(prompts, [])
                self.assertIn(
                    "does not accept argument(s) 'package'", self._tool_text(messages)
                )
                self.assertNotIn("install_package", bridge.names())


class EnvironmentInfoTestCase(unittest.TestCase):
    def test_reports_the_interpreter_that_runs_tool_code(self):
        report = {
            "executable": config.MCP_PYTHON,
            "prefix": "/opt/agent/.venv",
            "base_prefix": "/usr",
            "version": "3.12.16",
            "cwd": str(config.WORKSPACE_DIR),
            "workspace": str(config.WORKSPACE_DIR),
            "packages": {"pandas": "3.1.1"},
        }
        bridge = FakeBridge({"run_python_code": json.dumps(report)})
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            text = environment.python_environment_info()
        self.assertIn(config.MCP_PYTHON, text)
        self.assertIn("Virtualenv: yes", text)
        self.assertIn("pandas 3.1.1", text)
        self.assertIn(str(config.WORKSPACE_DIR), text)
        self.assertNotIn("WARNING", text)
        # The report is produced by the tool server, not by the harness process.
        self.assertEqual(bridge.calls[0][0], "run_python_code")

    def test_flags_a_mismatched_tool_interpreter(self):
        report = {
            "executable": "/somewhere/else/bin/python",
            "prefix": "/somewhere/else",
            "base_prefix": "/usr",
            "version": "3.12.16",
            "cwd": "/tmp",
            "workspace": "/tmp",
            "packages": {},
        }
        bridge = FakeBridge({"run_python_code": json.dumps(report)})
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            text = environment.python_environment_info()
        self.assertIn("WARNING", text)

    def test_is_advertised_to_the_model(self):
        names = {schema["function"]["name"] for schema in agent.HARNESS_TOOL_SCHEMAS}
        self.assertIn("python_environment_info", names)


class MissingPathTestCase(unittest.TestCase):
    def test_a_missing_workspace_file_names_the_workspace_and_its_contents(self):
        bridge = FakeBridge()
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            _, dispatch = agent.get_tooling()
            result = dispatch("read_file", {"file_path": "definitely-missing.csv"})
        self.assertIn("does not exist in the workspace", result)
        self.assertIn(str(config.WORKSPACE_DIR), result)
        self.assertIn("available:", result)
        self.assertEqual(bridge.calls, [])

    def test_an_existing_file_reaches_the_server(self):
        target = config.WORKSPACE_DIR / "_presence_test.txt"
        target.write_text("hello", encoding="utf-8")
        self.addCleanup(lambda: target.unlink(missing_ok=True))
        bridge = FakeBridge({"read_file": "hello"})
        with mock.patch("src.mcp_client.get_bridge", return_value=bridge):
            _, dispatch = agent.get_tooling()
            self.assertEqual(dispatch("read_file", {"file_path": target.name}), "hello")
        self.assertEqual(bridge.names(), ["read_file"])


if __name__ == "__main__":
    unittest.main()
