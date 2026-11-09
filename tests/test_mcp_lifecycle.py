"""MCP bridge liveness, restart-on-next-call and sandbox wrapping."""

import unittest
from unittest import mock

from src import config, mcp_client


class FakeBridge:
    """Records lifecycle calls without spawning a server process."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.alive = True
        self.started = False
        self.closed = False

    def start(self):
        self.started = True
        return self

    def close(self):
        self.closed = True
        self.alive = False

    @property
    def is_alive(self):
        return self.alive


class BridgeLifecycleTestCase(unittest.TestCase):
    def test_a_live_bridge_is_reused(self):
        with mock.patch.object(mcp_client, "MCPToolBridge", FakeBridge), mock.patch.object(
            mcp_client, "_bridge", None
        ):
            first = mcp_client.get_bridge()
            second = mcp_client.get_bridge()
        self.assertIs(second, first)
        self.assertTrue(first.started)
        self.assertFalse(first.closed)

    def test_a_dead_bridge_is_replaced(self):
        with mock.patch.object(mcp_client, "MCPToolBridge", FakeBridge), mock.patch.object(
            mcp_client, "_bridge", None
        ):
            first = mcp_client.get_bridge()
            first.alive = False
            second = mcp_client.get_bridge()

        self.assertIsNot(second, first)
        self.assertTrue(first.closed)
        self.assertTrue(second.started)

    def test_calling_a_lost_connection_reports_and_detaches(self):
        bridge = mcp_client.MCPToolBridge("server-not-started", [])
        self.addCleanup(bridge.close)
        with self.assertRaises(RuntimeError) as caught:
            bridge.call("read_file", {"file_path": "notes.txt"})
        self.assertIn("connection was lost", str(caught.exception))
        self.assertIn("did not run", str(caught.exception))
        self.assertFalse(bridge.is_alive)

    def test_sandbox_wrapper_is_prepended(self):
        with mock.patch.object(config, "MCP_SANDBOX_COMMAND", "bwrap"), mock.patch.object(
            config, "MCP_SANDBOX_ARGS", ["--ro-bind", "/", "/", "--unshare-net", "--"]
        ):
            command, args = mcp_client._wrap_sandbox("mcp-server", ["--dir", "ws"])
        self.assertEqual(command, "bwrap")
        self.assertEqual(
            args[:5], ["--ro-bind", "/", "/", "--unshare-net", "--"]
        )
        self.assertEqual(args[5:], ["mcp-server", "--dir", "ws"])

    def test_no_sandbox_leaves_the_command_alone(self):
        with mock.patch.object(config, "MCP_SANDBOX_COMMAND", ""):
            command, args = mcp_client._wrap_sandbox("mcp-server", ["--dir", "ws"])
        self.assertEqual(command, "mcp-server")
        self.assertEqual(args, ["--dir", "ws"])

    def test_get_bridge_passes_the_wrapped_command(self):
        with mock.patch.object(mcp_client, "MCPToolBridge", FakeBridge), mock.patch.object(
            mcp_client, "_bridge", None
        ), mock.patch.object(
            mcp_client, "MCP_SERVER_COMMAND", "mcp-server"
        ), mock.patch.object(
            mcp_client, "MCP_SERVER_ARGS", ["--dir", "ws"]
        ), mock.patch.object(
            config, "MCP_SANDBOX_COMMAND", "sandbox"
        ), mock.patch.object(
            config, "MCP_SANDBOX_ARGS", ["--strict"]
        ):
            bridge = mcp_client.get_bridge()

        self.assertEqual(bridge.kwargs["command"], "sandbox")
        self.assertEqual(bridge.kwargs["args"], ["--strict", "mcp-server", "--dir", "ws"])


if __name__ == "__main__":
    unittest.main()
