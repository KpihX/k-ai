# test/test_tools.py
"""
Tests for the internal tool system: registry, tool execution, tool definitions.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

from k_ai.tools.base import InternalTool, ToolContext, ToolRegistry
from k_ai.tools.meta import (
    NewSessionTool, LoadSessionTool, ExitSessionTool, RenameSessionTool,
    ListSessionsTool, DeleteSessionTool, ClearScreenTool, SetConfigTool,
    register_meta_tools,
)
from k_ai.tools.memory_tools import MemoryAddTool, MemoryListTool, MemoryRemoveTool
from k_ai.tools.external import PythonExecTool, ShellExecTool
from k_ai.tools.qmd import QmdSearchTool
from k_ai.models import ToolResult
from k_ai.memory import MemoryStore
from k_ai.session_store import SessionStore
from k_ai.config import ConfigManager


@pytest.fixture
def ctx(tmp_path):
    """Build a ToolContext with real stores but mocked callbacks."""
    cm = ConfigManager()
    mem = MemoryStore(tmp_path / "MEMORY.json")
    mem.load()
    ss = SessionStore(tmp_path / "sessions")
    ss.init()

    return ToolContext(
        config=cm,
        memory=mem,
        session_store=ss,
        console=MagicMock(),
        get_session_id=MagicMock(return_value="test-session"),
        request_exit=MagicMock(),
        request_new_session=MagicMock(),
        request_load_session=MagicMock(),
        request_compact=MagicMock(),
    )


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = ClearScreenTool()
        registry.register(tool)
        assert registry.get("clear_screen") is tool

    def test_get_unknown_returns_none(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_is_internal(self):
        registry = ToolRegistry()
        registry.register(ClearScreenTool())
        assert registry.is_internal("clear_screen") is True
        assert registry.is_internal("unknown") is False

    def test_to_openai_tools(self):
        registry = ToolRegistry()
        registry.register(ClearScreenTool())
        registry.register(NewSessionTool())
        tools = registry.to_openai_tools()
        assert len(tools) == 2
        assert all(t["type"] == "function" for t in tools)
        names = {t["function"]["name"] for t in tools}
        assert "clear_screen" in names
        assert "new_session" in names

    def test_get_names_sorted(self):
        registry = ToolRegistry()
        registry.register(NewSessionTool())
        registry.register(ClearScreenTool())
        names = registry.get_names()
        assert names == sorted(names)

    def test_register_meta_tools(self, ctx):
        registry = ToolRegistry()
        register_meta_tools(registry, ctx)
        assert len(registry.list_tools()) == 9  # all meta tools


# ---------------------------------------------------------------------------
# Meta tools
# ---------------------------------------------------------------------------

class TestMetaTools:
    @pytest.mark.asyncio
    async def test_new_session(self, ctx):
        tool = NewSessionTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        ctx.request_new_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_exit_session(self, ctx):
        tool = ExitSessionTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        ctx.request_exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_session_not_found(self, ctx):
        tool = LoadSessionTool()
        result = await tool.execute({"session_id": "nonexistent"}, ctx)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_load_session_found(self, ctx):
        meta = ctx.session_store.create_session()
        tool = LoadSessionTool()
        result = await tool.execute({"session_id": meta.id}, ctx)
        assert result.success is True
        ctx.request_load_session.assert_called_once_with(meta.id)

    @pytest.mark.asyncio
    async def test_rename_session(self, ctx):
        ctx.session_store.create_session()
        tool = RenameSessionTool()
        result = await tool.execute({"title": "New Title"}, ctx)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_rename_empty_title(self, ctx):
        tool = RenameSessionTool()
        result = await tool.execute({"title": ""}, ctx)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_list_sessions(self, ctx):
        ctx.session_store.create_session()
        ctx.session_store.create_session()
        tool = ListSessionsTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        assert result.data is not None
        assert len(result.data) == 2

    @pytest.mark.asyncio
    async def test_delete_session(self, ctx):
        meta = ctx.session_store.create_session()
        tool = DeleteSessionTool()
        result = await tool.execute({"session_id": meta.id}, ctx)
        assert result.success is True
        assert ctx.session_store.get_session(meta.id) is None

    @pytest.mark.asyncio
    async def test_clear_screen(self, ctx):
        tool = ClearScreenTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        ctx.console.clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_config(self, ctx):
        tool = SetConfigTool()
        result = await tool.execute({"key": "temperature", "value": "0.5"}, ctx)
        assert result.success is True
        assert ctx.config.get("temperature") == 0.5


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------

class TestMemoryTools:
    @pytest.mark.asyncio
    async def test_memory_add(self, ctx):
        tool = MemoryAddTool()
        result = await tool.execute({"text": "test fact"}, ctx)
        assert result.success is True
        assert len(ctx.memory.entries) == 1

    @pytest.mark.asyncio
    async def test_memory_add_empty(self, ctx):
        tool = MemoryAddTool()
        result = await tool.execute({"text": ""}, ctx)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_memory_list_empty(self, ctx):
        tool = MemoryListTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        assert result.data == []

    @pytest.mark.asyncio
    async def test_memory_list_with_entries(self, ctx):
        ctx.memory.add("fact 1")
        ctx.memory.add("fact 2")
        tool = MemoryListTool()
        result = await tool.execute({}, ctx)
        assert len(result.data) == 2

    @pytest.mark.asyncio
    async def test_memory_remove(self, ctx):
        ctx.memory.add("removable")
        tool = MemoryRemoveTool()
        result = await tool.execute({"entry_id": 1}, ctx)
        assert result.success is True
        assert len(ctx.memory.entries) == 0

    @pytest.mark.asyncio
    async def test_memory_remove_not_found(self, ctx):
        tool = MemoryRemoveTool()
        result = await tool.execute({"entry_id": 999}, ctx)
        assert result.success is False


# ---------------------------------------------------------------------------
# External tools
# ---------------------------------------------------------------------------

class TestExternalTools:
    @pytest.mark.asyncio
    async def test_python_exec_disabled(self, ctx):
        ctx.config.set("tools.python_exec.enabled", "false")
        tool = PythonExecTool()
        result = await tool.execute({"code": "print(1)"}, ctx)
        assert result.success is False
        assert "disabled" in result.message

    @pytest.mark.asyncio
    async def test_python_exec_empty_code(self, ctx):
        tool = PythonExecTool()
        result = await tool.execute({"code": ""}, ctx)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_python_exec_success(self, ctx):
        tool = PythonExecTool()
        result = await tool.execute({"code": "print('hello')"}, ctx)
        assert result.success is True
        assert "hello" in result.message

    @pytest.mark.asyncio
    async def test_python_exec_error(self, ctx):
        tool = PythonExecTool()
        result = await tool.execute({"code": "raise ValueError('boom')"}, ctx)
        assert result.success is False
        assert "boom" in result.message

    @pytest.mark.asyncio
    async def test_shell_exec_success(self, ctx):
        tool = ShellExecTool()
        result = await tool.execute({"command": "echo ok"}, ctx)
        assert result.success is True
        assert "ok" in result.message

    @pytest.mark.asyncio
    async def test_shell_exec_empty_command(self, ctx):
        tool = ShellExecTool()
        result = await tool.execute({"command": ""}, ctx)
        assert result.success is False


# ---------------------------------------------------------------------------
# OpenAI tool schema generation
# ---------------------------------------------------------------------------

class TestToolSchema:
    def test_tool_has_required_fields(self):
        tool = NewSessionTool()
        schema = tool.to_openai_tool()
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

    def test_tool_with_parameters(self):
        tool = RenameSessionTool()
        schema = tool.to_openai_tool()
        params = schema["function"]["parameters"]
        assert "title" in params["properties"]
        assert "title" in params["required"]
