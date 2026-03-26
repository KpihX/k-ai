import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from k_ai.commands import CommandHandler
from k_ai.models import ToolResult
from k_ai.session import ChatSession


@pytest.fixture
def session_for_commands(cm, tmp_path):
    cm.set("sessions.directory", str(tmp_path / "sessions"))
    cm.set("memory.internal_file", str(tmp_path / "MEMORY.json"))
    sess = ChatSession(cm)
    sess.console = MagicMock()
    sess._tool_ctx.console = sess.console
    return sess


class TestCommandHandler:
    @pytest.mark.asyncio
    async def test_qmd_command_routes_through_internal_tool_executor(self, session_for_commands):
        handler = CommandHandler(session_for_commands)
        session_for_commands._execute_internal_tool = AsyncMock(
            return_value=ToolResult(success=True, message="ok")
        )

        await handler.handle("/qmd search session 753d3002")

        session_for_commands._execute_internal_tool.assert_awaited_once()
        tool_call = session_for_commands._execute_internal_tool.await_args.args[0]
        assert tool_call.function_name == "qmd_search"
        assert tool_call.arguments["query"] == "session 753d3002"

    @pytest.mark.asyncio
    async def test_memory_add_routes_through_internal_tool_executor(self, session_for_commands):
        handler = CommandHandler(session_for_commands)
        session_for_commands._execute_internal_tool = AsyncMock(
            return_value=ToolResult(success=True, message="ok")
        )

        await handler.handle("/memory add retenir ceci")

        session_for_commands._execute_internal_tool.assert_awaited_once()
        tool_call = session_for_commands._execute_internal_tool.await_args.args[0]
        assert tool_call.function_name == "memory_add"
        assert tool_call.arguments["text"] == "retenir ceci"

    @pytest.mark.asyncio
    async def test_config_get_uses_confirm_prompt(self, session_for_commands, tmp_path):
        handler = CommandHandler(session_for_commands)
        target = tmp_path / "config.yaml"
        target.write_text("old", encoding="utf-8")

        with patch("k_ai.commands.Confirm.ask", return_value=False):
            await handler.handle(f"/config get {target}")

        assert target.read_text(encoding="utf-8") == "old"

    @pytest.mark.asyncio
    async def test_digest_routes_through_internal_tool_executor(self, session_for_commands):
        handler = CommandHandler(session_for_commands)
        session_for_commands._execute_internal_tool = AsyncMock(
            return_value=ToolResult(success=True, message="ok")
        )

        await handler.handle("/digest abc123")

        tool_call = session_for_commands._execute_internal_tool.await_args.args[0]
        assert tool_call.function_name == "session_digest"
        assert tool_call.arguments["session_id"] == "abc123"

    @pytest.mark.asyncio
    async def test_load_accepts_last_n(self, session_for_commands):
        handler = CommandHandler(session_for_commands)
        session_for_commands._do_load_session = MagicMock()

        await handler.handle("/load abc123 20")

        session_for_commands._do_load_session.assert_called_once_with("abc123", last_n=20)

    @pytest.mark.asyncio
    async def test_set_routes_through_internal_tool_executor(self, session_for_commands):
        handler = CommandHandler(session_for_commands)
        session_for_commands._execute_internal_tool = AsyncMock(
            return_value=ToolResult(success=True, message="ok")
        )

        await handler.handle("/set cli.tool_result_max_display 800")

        tool_call = session_for_commands._execute_internal_tool.await_args.args[0]
        assert tool_call.function_name == "set_config"
        assert tool_call.arguments["key"] == "cli.tool_result_max_display"
        assert tool_call.arguments["value"] == "800"

    @pytest.mark.asyncio
    async def test_status_routes_through_runtime_tool(self, session_for_commands):
        handler = CommandHandler(session_for_commands)
        session_for_commands._execute_internal_tool = AsyncMock(
            return_value=ToolResult(success=True, message="ok")
        )

        await handler.handle("/status")

        tool_call = session_for_commands._execute_internal_tool.await_args.args[0]
        assert tool_call.function_name == "runtime_status"
        assert tool_call.arguments["mode"] == "full"
