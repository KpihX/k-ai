"""Tests for the MCP runtime foundation."""

from __future__ import annotations

from pathlib import Path
import shutil
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from k_ai import ConfigManager
from k_ai.commands import CommandHandler
from k_ai.mcp.client import MCPClient
from k_ai.mcp.models import MCPServerSpec, MCPToolSpec
from k_ai.models import ToolCall, ToolResult
from k_ai.session import ChatSession
from k_ai.tools.mcp import MCPToolAdapter
from k_ai.tools.mcp_admin import MCPServerUpsertTool


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

        with patch.object(session, "_prompt_tool_approval", return_value=(True, None)):
            result = await session._execute_internal_tool(
                ToolCall(
                    id="m2",
                    function_name="mcp_server_set_enabled",
                    arguments={"server_name": "fake", "enabled": False},
                )
            )

        assert result.success is True
        assert session.cm.get_path("mcp.servers.fake.enabled") is False

    @pytest.mark.asyncio
    async def test_manager_uses_current_session_cwd_for_runtime_server_spec(self, mcp_cm, tmp_path):
        mcp_cm.set("mcp.servers.fake.cwd", "{session_cwd}")
        session = ChatSession(mcp_cm)
        await session._ensure_mcp_catalog_loaded(force_refresh=True)

        new_cwd = tmp_path / "shifted"
        new_cwd.mkdir()
        session.set_current_cwd(new_cwd)

        captured: dict[str, object] = {}

        async def fake_call_tool(*, spec, tool_name, qualified_name, arguments):
            captured["spec"] = spec
            captured["tool_name"] = tool_name
            captured["qualified_name"] = qualified_name
            captured["arguments"] = arguments
            return "ok"

        with patch.object(session.mcp_manager._client, "call_tool", side_effect=fake_call_tool):
            result = await session.mcp_manager.call_tool(
                "mcp__fake__write_file",
                {"path": "./test.md", "content": "demo"},
            )

        spec = captured["spec"]
        assert isinstance(spec, MCPServerSpec)
        assert spec.cwd == new_cwd
        assert spec.roots
        assert spec.roots[0].path == tmp_path
        assert captured["tool_name"] == "write_file"
        assert captured["qualified_name"] == "mcp__fake__write_file"
        assert captured["arguments"] == {"path": "./test.md", "content": "demo"}
        assert result == "ok"

    def test_filesystem_server_injects_root_paths_into_empty_args(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cm = ConfigManager()
        cm.set("mcp.enabled", True)
        cm.set("mcp.servers.filesystem.enabled", True)
        cm.set("mcp.servers.filesystem.command", "mcp-server-filesystem")
        cm.set("mcp.servers.filesystem.args", [])
        cm.set("mcp.servers.filesystem.cwd", "{workspace_root}")
        cm.set("mcp.servers.filesystem.roots.enabled", True)
        cm.set("mcp.servers.filesystem.roots.include_workspace_root", True)

        session = ChatSession(cm, workspace_root=str(tmp_path))
        spec = session.mcp_manager._server_spec_by_name("filesystem")

        assert spec is not None
        assert spec.cwd == tmp_path
        assert spec.args == (str(tmp_path), "/tmp")

    @pytest.mark.asyncio
    async def test_filesystem_allow_dir_tool_updates_session_scope_without_persisting(self, cm, tmp_path):
        cm.set("mcp.enabled", False)
        session = ChatSession(cm, workspace_root=str(tmp_path))

        with patch.object(session, "_prompt_tool_approval", return_value=(True, None)):
            result = await session._execute_internal_tool(
                ToolCall(
                    id="allow-dir-session",
                    function_name="mcp_filesystem_allow_dir",
                    arguments={"path": "./sandbox", "scope": "session"},
                )
            )

        assert result.success is True
        paths = session.cm.get_path("mcp.servers.filesystem.roots.additional_paths")
        assert str((tmp_path / "sandbox").resolve()) in paths
        assert "/tmp" in paths

    @pytest.mark.asyncio
    async def test_filesystem_allow_dir_tool_persists_when_requested(self, cm, tmp_path):
        cm.set("mcp.enabled", False)
        session = ChatSession(cm, workspace_root=str(tmp_path))

        with patch.object(session, "_prompt_tool_approval", return_value=(True, None)):
            result = await session._execute_internal_tool(
                ToolCall(
                    id="allow-dir-persistent",
                    function_name="mcp_filesystem_allow_dir",
                    arguments={"path": str(tmp_path / "persisted"), "scope": "persistent"},
                )
            )

        assert result.success is True
        saved_to = result.data["config_change"]["saved_to"]
        assert saved_to

    def test_filesystem_write_file_proposal_renders_preview_section(self, cm, tmp_path):
        session = ChatSession(cm, workspace_root=str(tmp_path))
        spec = MCPToolSpec(
            server_name="filesystem",
            server_title="Filesystem",
            name="write_file",
            qualified_name="mcp__filesystem__write_file",
            description="Write a file.",
            input_schema={"type": "object"},
            annotations={"destructiveHint": True},
        )
        tool = MCPToolAdapter(manager=session.mcp_manager, spec=spec, ctx=session._tool_ctx)

        sections = tool.proposal_sections(
            {"path": "./demo.py", "content": "print('a')\nprint('b')\n"},
            session._tool_ctx,
        )

        assert [title for title, _content in sections] == ["New File"]

    def test_filesystem_edit_file_proposal_renders_diff_section(self, cm, tmp_path):
        session = ChatSession(cm, workspace_root=str(tmp_path))
        spec = MCPToolSpec(
            server_name="filesystem",
            server_title="Filesystem",
            name="edit_file",
            qualified_name="mcp__filesystem__edit_file",
            description="Edit a file.",
            input_schema={"type": "object"},
            annotations={"destructiveHint": True},
        )
        tool = MCPToolAdapter(manager=session.mcp_manager, spec=spec, ctx=session._tool_ctx)

        sections = tool.proposal_sections(
            {
                "path": "./demo.py",
                "edits": [
                    {
                        "oldText": "def hello():\n    return 1",
                        "newText": "def hello():\n    return 2",
                    }
                ],
            },
            session._tool_ctx,
        )

        assert [title for title, _content in sections] == ["Planned Diff"]

    def test_filesystem_edit_result_renderable_is_short_confirmation(self, cm, tmp_path):
        session = ChatSession(cm, workspace_root=str(tmp_path))
        spec = MCPToolSpec(
            server_name="filesystem",
            server_title="Filesystem",
            name="edit_file",
            qualified_name="mcp__filesystem__edit_file",
            description="Edit a file.",
            input_schema={"type": "object"},
            annotations={"destructiveHint": True},
        )
        tool = MCPToolAdapter(manager=session.mcp_manager, spec=spec, ctx=session._tool_ctx)

        renderable = tool.result_renderable(
            ToolResult(success=True, message="diff body", data={"__requested_path": "./demo.py"}),
            max_display_length=2000,
            ctx=session._tool_ctx,
        )

        assert "Confirmed edit to ./demo.py." in renderable.plain

    def test_filesystem_preview_uses_mcp_centralized_defaults(self, cm, tmp_path):
        cm.set("mcp.runtime.proposals.filesystem.preview_max_lines", 2)
        cm.set("mcp.runtime.proposals.filesystem.line_window_mode", "tail")
        session = ChatSession(cm, workspace_root=str(tmp_path))
        spec = MCPToolSpec(
            server_name="filesystem",
            server_title="Filesystem",
            name="write_file",
            qualified_name="mcp__filesystem__write_file",
            description="Write a file.",
            input_schema={"type": "object"},
            annotations={"destructiveHint": True},
        )
        tool = MCPToolAdapter(manager=session.mcp_manager, spec=spec, ctx=session._tool_ctx)

        sections = tool.proposal_sections(
            {"path": "./demo.py", "content": "a\nb\nc\nd\n"},
            session._tool_ctx,
        )

        assert [title for title, _content in sections] == ["New File"]

    def test_mcp_server_upsert_proposal_explicitly_targets_current_runtime(self, cm, tmp_path):
        session = ChatSession(cm, workspace_root=str(tmp_path))
        tool = MCPServerUpsertTool()

        sections = tool.proposal_sections(
            {
                "server_name": "tick_mcp_kai",
                "transport": "streamable_http",
                "url": "https://tick.kpihx-labs.com/mcp",
                "persist": True,
            },
            session._tool_ctx,
        )

        assert [title for title, _content in sections] == ["MCP Server Config"]


class TestMCPClient:
    @pytest.mark.asyncio
    async def test_stdio_transport_quiets_server_stderr_by_default(self, tmp_path):
        client = MCPClient(
            protocol_version="2025-06-18",
            client_name="k-ai",
            client_version="0.2.0",
        )
        spec = MCPServerSpec(
            name="fake",
            enabled=True,
            transport="stdio",
            command="python3",
            args=("-c", "print('ready')"),
            cwd=Path(tmp_path),
            env={},
            stderr_mode="quiet",
            roots=(),
        )

        captured = {}

        class _Dummy:
            async def __aenter__(self):
                return ("read", "write")
            async def __aexit__(self, exc_type, exc, tb):
                return False

        def fake_stdio(params, errlog):
            captured["errlog_name"] = getattr(errlog, "name", None)
            captured["command"] = params.command
            return _Dummy()

        with patch("k_ai.mcp.client.stdio_client", fake_stdio):
            async with client._transport_streams(spec) as streams:
                assert streams == ("read", "write")

        assert captured["command"] == "python3"
        assert captured["errlog_name"] == "/dev/null"


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
        with patch.object(session, "_prompt_tool_approval", return_value=(True, None)):
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
