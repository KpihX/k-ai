"""Tests for the MCP runtime foundation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch
import shutil

import pytest

from k_ai import ConfigManager
from k_ai.commands import CommandHandler
from k_ai.models import ToolCall
from k_ai.session import ChatSession


def write_fake_mcp_server(path: Path) -> Path:
    path.write_text(
        "from mcp import types\n"
        "from mcp.server.fastmcp import FastMCP\n"
        "\n"
        "app = FastMCP('Fake Filesystem')\n"
        "\n"
        "@app.resource('memo://status')\n"
        "def status_resource() -> str:\n"
        "    return 'RESOURCE:status'\n"
        "\n"
        "@app.prompt()\n"
        "def explain_status(topic: str) -> str:\n"
        "    return f'PROMPT:{topic}'\n"
        "\n"
        "@app.tool(annotations=types.ToolAnnotations(readOnlyHint=True))\n"
        "def read_text_file(path: str) -> str:\n"
        "    return f'READ:{path}'\n"
        "\n"
        "@app.tool(annotations=types.ToolAnnotations(destructiveHint=True))\n"
        "def write_file(path: str, content: str) -> str:\n"
        "    return f'WRITE:{path}:{content}'\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    app.run('stdio')\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def mcp_cm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    server_path = write_fake_mcp_server(tmp_path / "fake_mcp_server.py")
    cm = ConfigManager()
    cm.set("tools.mcp.enabled", True)
    cm.set("mcp.enabled", True)
    cm.set("mcp.servers.filesystem.enabled", False)
    cm.set("mcp.servers.fake.enabled", True)
    cm.set("mcp.servers.fake.transport", "stdio")
    cm.set("mcp.servers.fake.command", "python3")
    cm.set("mcp.servers.fake.args", [str(server_path)])
    cm.set("mcp.servers.fake.cwd", str(tmp_path))
    cm.set("mcp.servers.fake.roots.enabled", True)
    cm.set("mcp.servers.fake.roots.include_workspace_root", True)
    return cm


class TestMCPManager:
    def test_manager_disables_itself_cleanly_when_sdk_missing(self, mcp_cm):
        with patch("k_ai.mcp.manager.mcp_sdk_available", return_value=False):
            session = ChatSession(mcp_cm)
            assert session.mcp_manager.enabled() is False
            assert "missing Python MCP SDK" in session.mcp_manager.runtime_summary_cached()

    @pytest.mark.asyncio
    async def test_manager_discovers_stdio_server_tools(self, mcp_cm, tmp_path):
        session = ChatSession(mcp_cm)
        catalog = await session.mcp_manager.catalog(force_refresh=True)

        assert len(catalog.servers) == 1
        assert catalog.servers[0].server_title == "Fake Filesystem"
        assert {tool.name for tool in catalog.tools} == {"read_text_file", "write_file"}
        assert {item.uri for item in catalog.resources} == {"memo://status"}
        assert {item.name for item in catalog.prompts} == {"explain_status"}
        assert {tool.qualified_name for tool in catalog.tools} == {
            "mcp__fake__read_text_file",
            "mcp__fake__write_file",
        }

    @pytest.mark.asyncio
    async def test_session_loads_dynamic_mcp_tools_into_registry(self, mcp_cm):
        session = ChatSession(mcp_cm)

        await session._ensure_mcp_catalog_loaded(force_refresh=True)

        assert session.tool_registry.get("mcp__fake__read_text_file") is not None
        assert session.tool_registry.get("mcp__fake__write_file") is not None

    @pytest.mark.asyncio
    async def test_read_only_mcp_tool_executes_without_manual_approval(self, mcp_cm):
        session = ChatSession(mcp_cm)
        await session._ensure_mcp_catalog_loaded(force_refresh=True)

        result = await session._execute_internal_tool(
            ToolCall(
                id="m1",
                function_name="mcp__fake__read_text_file",
                arguments={"path": "/tmp/demo.txt"},
            )
        )

        assert result.success is True
        assert "READ:/tmp/demo.txt" in result.message

    @pytest.mark.asyncio
    async def test_manager_can_read_resources_and_prompts(self, mcp_cm):
        session = ChatSession(mcp_cm)
        await session._ensure_mcp_catalog_loaded(force_refresh=True)

        resource = await session.read_mcp_resource("fake", "memo://status")
        prompt = await session.get_mcp_prompt("fake", "explain_status", {"topic": "demo"})

        assert "RESOURCE:status" in resource.message
        assert "PROMPT:demo" in prompt.message

    @pytest.mark.asyncio
    async def test_admin_tool_can_disable_server(self, mcp_cm):
        session = ChatSession(mcp_cm)
        await session._ensure_mcp_catalog_loaded(force_refresh=True)

        with patch("k_ai.session.Confirm.ask", return_value=True):
            result = await session._execute_internal_tool(
                ToolCall(
                    id="m2",
                    function_name="mcp_server_set_enabled",
                    arguments={"server_name": "fake", "enabled": False},
                )
            )

        assert result.success is True
        assert session.cm.get_path("mcp.servers.fake.enabled") is False


class TestMCPCommands:
    @pytest.mark.asyncio
    async def test_mcp_tools_command_renders_catalog(self, mcp_cm):
        session = ChatSession(mcp_cm)
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        handler = CommandHandler(session)

        await handler.handle("/mcp tools")

        assert session.console.print.called

    @pytest.mark.asyncio
    async def test_mcp_resources_and_prompts_commands_render(self, mcp_cm):
        session = ChatSession(mcp_cm)
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        handler = CommandHandler(session)

        await handler.handle("/mcp resources")
        await handler.handle("/mcp prompts")

        assert session.console.print.called


@pytest.mark.skipif(shutil.which("mcp-server-filesystem") is None, reason="official filesystem MCP server is not installed")
class TestLiveFilesystemServer:
    @pytest.fixture
    def live_fs_cm(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cm = ConfigManager()
        cm.set("mcp.enabled", True)
        cm.set("tools.mcp.enabled", True)
        cm.set("mcp.servers.filesystem.enabled", True)
        cm.set("mcp.servers.filesystem.transport", "stdio")
        cm.set("mcp.servers.filesystem.command", "mcp-server-filesystem")
        cm.set("mcp.servers.filesystem.args", [str(tmp_path)])
        cm.set("mcp.servers.filesystem.cwd", str(tmp_path))
        cm.set("mcp.servers.filesystem.roots.enabled", True)
        cm.set("mcp.servers.filesystem.roots.include_workspace_root", True)
        return cm

    @pytest.mark.asyncio
    async def test_official_filesystem_server_discovers_and_executes(self, live_fs_cm, tmp_path):
        session = ChatSession(live_fs_cm)
        await session._ensure_mcp_catalog_loaded(force_refresh=True)

        catalog = session.mcp_manager.current_catalog()
        assert not catalog.issues
        assert any(tool.qualified_name == "mcp__filesystem__read_text_file" for tool in catalog.tools)
        assert any(tool.qualified_name == "mcp__filesystem__write_file" for tool in catalog.tools)

        target = tmp_path / "live.txt"
        with patch("k_ai.session.Confirm.ask", return_value=True):
            write_result = await session._execute_internal_tool(
                ToolCall(
                    id="live-write",
                    function_name="mcp__filesystem__write_file",
                    arguments={"path": str(target), "content": "hello live mcp"},
                )
            )
        assert write_result.success is True

        read_result = await session._execute_internal_tool(
            ToolCall(
                id="live-read",
                function_name="mcp__filesystem__read_text_file",
                arguments={"path": str(target)},
            )
        )
        assert read_result.success is True
        assert "hello live mcp" in read_result.message
