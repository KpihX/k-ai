# test/test_tools.py
"""
Tests for the internal tool system: registry, tool execution, tool definitions.
"""
import pytest
import sys
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

from k_ai.tools.base import InternalTool, ToolContext, ToolRegistry
from k_ai.tools.meta import (
    NewSessionTool, LoadSessionTool, ExitSessionTool, RenameSessionTool,
    ListSessionsTool, SessionDigestTool, SessionExtractTool, DeleteSessionTool, ClearScreenTool, SetConfigTool,
    GetConfigTool, ListConfigTool, RuntimeStatusTool, SaveConfigTool,
    ToolPolicyListTool, ToolPolicySetTool, ToolPolicyResetTool, ToolCapabilityListTool, ToolCapabilitySetTool,
    SwitchSessionTool, InitSystemTool,
    register_meta_tools,
)
from k_ai.tools.memory_tools import MemoryAddTool, MemoryListTool, MemoryRemoveTool
from k_ai.tools.external import (
    PythonExecTool,
    ShellExecTool,
    PythonSandboxPackagesTool,
    PythonSandboxListPackagesTool,
    PythonSandboxInstallPackagesTool,
    PythonSandboxRemovePackagesTool,
)
from k_ai.tools.qmd import QmdGetTool, QmdQueryTool, QmdSearchTool
from k_ai.models import Message, MessageRole, ToolResult
from k_ai.memory import MemoryStore
from k_ai.session_store import SessionStore
from k_ai.config import ConfigManager


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """Build a ToolContext with real stores but mocked callbacks."""
    cm = ConfigManager()
    cm.set("tools.python.sandbox_dir", str(tmp_path / "sandbox"))
    mem = MemoryStore(tmp_path / "MEMORY.json")
    mem.load()
    ss = SessionStore(tmp_path / "sessions")
    ss.init()
    monkeypatch.setattr(PythonExecTool, "_ensure_sandbox", AsyncMock(return_value=sys.executable))

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
        request_init=MagicMock(),
        complete_init=MagicMock(),
        get_tool_policy_overview=MagicMock(return_value={"rows": [], "defaults_by_risk": {"low": "auto"}, "protected_tools": [], "counts": {"ask": 0, "auto": 0, "protected": 0, "session_overrides": 0, "global_overrides": 0}}),
        update_tool_policy=MagicMock(return_value={"target_kind": "tool", "target": "clear_screen", "policy": "auto", "scope": "session", "previous": None, "saved_to": None}),
        reset_tool_policy=MagicMock(return_value={"target_kind": "tool", "target": "clear_screen", "scope": "session", "previous": "auto", "removed": True, "saved_to": None}),
        get_tool_capability_overview=MagicMock(return_value={"rows": [{"capability": "python", "enabled": True, "mutable": True, "tools": ["python_exec"], "description": "Sandboxed Python"}], "counts": {"enabled": 1, "disabled": 0, "mutable": 1}}),
        update_tool_capability=MagicMock(return_value={"capability": "python", "enabled": False, "previous": True, "saved_to": None}),
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
        assert len(registry.list_tools()) == 22  # all meta/runtime/config/admin tools


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
    async def test_new_session_with_type_seed(self, ctx):
        tool = NewSessionTool()
        result = await tool.execute({"session_type": "meta", "summary": "Admin tools", "themes": ["config"]}, ctx)
        assert result.success is True
        ctx.request_new_session.assert_called_with(seed={"session_type": "meta", "summary": "Admin tools", "themes": ["config"]})

    @pytest.mark.asyncio
    async def test_init_system_without_names_starts_init_flow(self, ctx):
        tool = InitSystemTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        ctx.request_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_init_system_persists_assistant_and_user_names(self, ctx):
        tool = InitSystemTool()
        result = await tool.execute({"assistant_name": "K-Prime", "user_name": "Ivann"}, ctx)
        assert result.success is True
        assert ctx.config.get_nested("prompts", "assistant_name") == "K-Prime"
        assert any(entry.text == "Preferred user name: Ivann." for entry in ctx.memory.list_entries())
        ctx.complete_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_capability_list(self, ctx):
        tool = ToolCapabilityListTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        assert result.data["counts"]["enabled"] == 1

    @pytest.mark.asyncio
    async def test_tool_capability_set(self, ctx):
        tool = ToolCapabilitySetTool()
        result = await tool.execute({"capability": "python", "enabled": False}, ctx)
        assert result.success is True
        ctx.update_tool_capability.assert_called_once()

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
        ctx.request_load_session.assert_called_once_with(meta.id, None)

    @pytest.mark.asyncio
    async def test_load_session_with_last_n(self, ctx):
        meta = ctx.session_store.create_session()
        tool = LoadSessionTool()
        result = await tool.execute({"session_id": meta.id, "last_n": 20}, ctx)
        assert result.success is True
        ctx.request_load_session.assert_called_once_with(meta.id, 20)

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
        first = ctx.session_store.list_sessions(limit=1)[0]
        ctx.session_store.update_digest(first.id, "Chat sur l'algebre", ["algèbre", "matrices"], "classic")
        tool = ListSessionsTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        assert result.data is not None
        assert len(result.data["sessions"]) == 2
        assert "themes=" in result.message
        assert "type=" in result.message

    @pytest.mark.asyncio
    async def test_list_sessions_oldest_order(self, ctx):
        first = ctx.session_store.create_session()
        ctx.session_store.create_session()
        tool = ListSessionsTool()
        result = await tool.execute({"order": "oldest"}, ctx)
        assert result.success is True
        assert result.data["order"] == "oldest"
        assert result.data["sessions"][0]["id"] == first.id

    @pytest.mark.asyncio
    async def test_list_sessions_filters_by_type(self, ctx):
        classic = ctx.session_store.create_session(session_type="classic")
        ctx.session_store.create_session(session_type="meta")
        tool = ListSessionsTool()
        result = await tool.execute({"session_type": "classic"}, ctx)
        assert result.success is True
        assert len(result.data["sessions"]) == 1
        assert result.data["sessions"][0]["id"] == classic.id

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

    @pytest.mark.asyncio
    async def test_set_config_rejects_tool_approval_namespace(self, ctx):
        tool = SetConfigTool()
        result = await tool.execute({"key": "tool_approval.catalog.clear_screen.default_policy", "value": "ask"}, ctx)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_get_config(self, ctx):
        tool = GetConfigTool()
        result = await tool.execute({"key": "temperature"}, ctx)
        assert result.success is True
        assert result.data["key"] == "temperature"

    @pytest.mark.asyncio
    async def test_list_config(self, ctx):
        tool = ListConfigTool()
        result = await tool.execute({"prefix": "cli"}, ctx)
        assert result.success is True
        assert any(key.startswith("cli.") for key, _ in result.data["items"])

    @pytest.mark.asyncio
    async def test_runtime_status(self, ctx):
        ctx.get_runtime_snapshot = MagicMock(return_value={"provider": "ollama", "model": "phi4", "context_window": 32000, "estimated_context_tokens": 10, "context_percent": 0.0})
        tool = RuntimeStatusTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        assert result.data["snapshot"]["provider"] == "ollama"

    @pytest.mark.asyncio
    async def test_save_config(self, ctx, tmp_path):
        tool = SaveConfigTool()
        target = tmp_path / "active.yaml"
        result = await tool.execute({"path": str(target)}, ctx)
        assert result.success is True
        assert target.exists()

    @pytest.mark.asyncio
    async def test_session_extract_current(self, ctx):
        meta = ctx.session_store.create_session()
        ctx.get_session_id = MagicMock(return_value=meta.id)
        ctx.session_store.save_message(meta.id, Message(role=MessageRole.USER, content="hello"))
        tool = SessionExtractTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        assert result.data["messages"][0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_session_digest_calls_generator(self, ctx):
        meta = ctx.session_store.create_session()
        ctx.get_session_id = MagicMock(return_value=meta.id)
        ctx.generate_session_digest = AsyncMock(return_value={"summary": "diag matrix", "themes": ["python", "linear algebra"], "session_type": "classic"})
        tool = SessionDigestTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        assert "diag matrix" in result.message
        assert "Type: classic" in result.message

    @pytest.mark.asyncio
    async def test_switch_session_requests_seeded_new_session(self, ctx):
        ctx.get_history = MagicMock(return_value=[Message(role=MessageRole.USER, content="nouveau sujet")])
        tool = SwitchSessionTool()
        result = await tool.execute(
            {"summary": "Nouveau sujet", "themes": ["python"], "session_type": "classic", "reason": "intent shift"},
            ctx,
        )
        assert result.success is True
        ctx.request_new_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_policy_list(self, ctx):
        tool = ToolPolicyListTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        ctx.get_tool_policy_overview.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_policy_set(self, ctx):
        tool = ToolPolicySetTool()
        result = await tool.execute({"target": "clear_screen", "policy": "auto"}, ctx)
        assert result.success is True
        ctx.update_tool_policy.assert_called_once()

    @pytest.mark.asyncio
    async def test_tool_policy_reset(self, ctx):
        tool = ToolPolicyResetTool()
        result = await tool.execute({"target": "clear_screen"}, ctx)
        assert result.success is True
        ctx.reset_tool_policy.assert_called_once()


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
        ctx.config.set("tools.python.enabled", "false")
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
    async def test_python_exec_prints_last_expression_value(self, ctx):
        tool = PythonExecTool()
        result = await tool.execute({"code": "x = 21 * 2\nx"}, ctx)
        assert result.success is True
        assert "42" in result.message

    @pytest.mark.asyncio
    async def test_python_sandbox_list_packages(self, ctx, monkeypatch):
        async def fake_run_pip(ctx_arg, args, timeout=None):
            return 0, '[{"name":"numpy","version":"1.0"}]', ""

        monkeypatch.setattr(PythonSandboxPackagesTool, "_run_pip", staticmethod(fake_run_pip))
        tool = PythonSandboxListPackagesTool()
        result = await tool.execute({}, ctx)
        assert result.success is True
        assert "numpy==1.0" in result.message

    @pytest.mark.asyncio
    async def test_python_sandbox_install_packages(self, ctx, monkeypatch):
        async def fake_run_pip(ctx_arg, args, timeout=None):
            assert args == ["install", "polars", "pyarrow"]
            return 0, "", ""

        monkeypatch.setattr(PythonSandboxPackagesTool, "_run_pip", staticmethod(fake_run_pip))
        tool = PythonSandboxInstallPackagesTool()
        result = await tool.execute({"packages": ["polars", "pyarrow"]}, ctx)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_python_sandbox_remove_packages_rejects_core(self, ctx):
        tool = PythonSandboxRemovePackagesTool()
        result = await tool.execute({"packages": ["pip"]}, ctx)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_python_sandbox_remove_packages(self, ctx, monkeypatch):
        async def fake_run_pip(ctx_arg, args, timeout=None):
            assert args == ["uninstall", "-y", "polars"]
            return 0, "", ""

        monkeypatch.setattr(PythonSandboxPackagesTool, "_run_pip", staticmethod(fake_run_pip))
        tool = PythonSandboxRemovePackagesTool()
        result = await tool.execute({"packages": ["polars"]}, ctx)
        assert result.success is True


class TestQmdTools:
    @pytest.mark.asyncio
    async def test_qmd_query_uses_configured_timeout(self, ctx, monkeypatch):
        captured = {}

        async def fake_run_qmd(*args, timeout=60):
            captured["args"] = args
            captured["timeout"] = timeout
            return True, "[]"

        monkeypatch.setattr("k_ai.tools.qmd._run_qmd", fake_run_qmd)
        ctx.config.set("tools.qmd.query_timeout", 180)

        tool = QmdQueryTool()
        result = await tool.execute({"query": "miami open"}, ctx)

        assert result.success is True
        assert captured["args"][:2] == ("query", "miami open")
        assert captured["timeout"] == 180

    @pytest.mark.asyncio
    async def test_qmd_get_resolves_short_session_id_to_indexed_jsonl(self, ctx, monkeypatch):
        meta = ctx.session_store.create_session()
        captured = {}

        async def fake_run_qmd(*args, timeout=60):
            captured["args"] = args
            return True, "ok"

        monkeypatch.setattr("k_ai.tools.qmd._run_qmd", fake_run_qmd)

        tool = QmdGetTool()
        result = await tool.execute({"file": meta.id[:8]}, ctx)

        assert result.success is True
        assert captured["args"] == ("get", f"qmd://k-ai/{meta.id}.jsonl")
        assert result.message == "ok"

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
