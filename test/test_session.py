# test/test_session.py
"""
Tests for ChatSession: programmatic API (send, send_with_tools), history
management, provider reload, and system-prompt handling.
"""
from contextlib import contextmanager
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from k_ai import ConfigManager
from k_ai.session import ChatSession
from k_ai.models import Message, MessageRole, ToolCall, ToolResult, TokenUsage, CompletionChunk
from k_ai.exceptions import LLMError, ContextLengthExceededError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_text_chunks(*parts: str, finish="stop") -> list[CompletionChunk]:
    """Build a list of CompletionChunks with text deltas."""
    chunks = [CompletionChunk(delta_content=p) for p in parts]
    chunks[-1] = CompletionChunk(delta_content=parts[-1], finish_reason=finish)
    return chunks


def make_tool_call_chunk(tc: ToolCall) -> CompletionChunk:
    return CompletionChunk(tool_calls=[tc], finish_reason="tool_calls")


async def stream_from(*chunks: CompletionChunk):
    for c in chunks:
        yield c


def mock_llm(response_chunks: list[CompletionChunk], provider="ollama", model="phi4-mini:latest"):
    """
    Build a MagicMock LLMProvider whose chat_stream is an async generator
    yielding `response_chunks`.
    """
    llm = MagicMock()
    llm.provider_name = provider
    llm.model_name = model

    async def _chat_stream(messages, config=None, tools=None):
        for chunk in response_chunks:
            yield chunk

    llm.chat_stream = _chat_stream
    return llm


@pytest.fixture
def session(cm, tmp_path):
    """ChatSession with ollama (no key needed); uses tmp dir for isolation."""
    cm.set("provider", "ollama")
    cm.set("sessions.directory", str(tmp_path / "sessions"))
    cm.set("memory.internal_file", str(tmp_path / "MEMORY.json"))
    sess = ChatSession(cm)
    return sess


# ===========================================================================
# send() — basic text
# ===========================================================================

class TestSend:
    @pytest.mark.asyncio
    async def test_send_returns_full_text(self, session):
        session.llm = mock_llm(make_text_chunks("Hello", " world"))
        result = await session.send("hi")
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_send_single_chunk(self, session):
        session.llm = mock_llm([CompletionChunk(delta_content="42", finish_reason="stop")])
        result = await session.send("what is 6*7?")
        assert result == "42"

    @pytest.mark.asyncio
    async def test_send_empty_response(self, session):
        session.llm = mock_llm([CompletionChunk(delta_content="", finish_reason="stop")])
        result = await session.send("silence")
        assert result == ""

    @pytest.mark.asyncio
    async def test_send_appends_user_message_to_history(self, session):
        session.llm = mock_llm([CompletionChunk(delta_content="pong")])
        await session.send("ping")
        assert session.history[0].role == MessageRole.USER
        assert session.history[0].content == "ping"

    @pytest.mark.asyncio
    async def test_send_appends_assistant_message_to_history(self, session):
        session.llm = mock_llm([CompletionChunk(delta_content="pong")])
        await session.send("ping")
        assert session.history[1].role == MessageRole.ASSISTANT
        assert session.history[1].content == "pong"

    @pytest.mark.asyncio
    async def test_send_multi_turn_history_grows(self, session):
        session.llm = mock_llm([CompletionChunk(delta_content="A")])
        await session.send("turn 1")
        session.llm = mock_llm([CompletionChunk(delta_content="B")])
        await session.send("turn 2")
        assert len(session.history) == 4  # user, assistant, user, assistant

    @pytest.mark.asyncio
    async def test_send_creates_and_persists_session(self, session):
        session.llm = mock_llm([CompletionChunk(delta_content="pong")])
        await session.send("ping")
        assert session._session_id is not None
        saved = session.session_store.load_messages(session._session_id)
        assert [m.role for m in saved] == [MessageRole.USER, MessageRole.ASSISTANT]

    @pytest.mark.asyncio
    async def test_send_honors_queued_new_session(self, session):
        session.llm = mock_llm([CompletionChunk(delta_content="first")])
        await session.send("hello")
        original = session._session_id

        session._handle_new_session(seed={"session_type": "meta", "summary": "Admin"})
        session.llm = mock_llm([CompletionChunk(delta_content="second")])
        await session.send("adjuste le config")

        assert session._session_id != original
        old_meta = session.session_store.get_session(original)
        new_meta = session.session_store.get_session(session._session_id)
        assert old_meta is not None and old_meta.summary
        assert new_meta is not None
        assert new_meta.session_type == "meta"
        assert new_meta.summary == "Admin"

    @pytest.mark.asyncio
    async def test_send_rolls_back_on_llm_failure(self, session):
        async def failing_stream(messages, config=None, tools=None):
            raise RuntimeError("provider down")
            yield

        session.llm.chat_stream = failing_stream
        with pytest.raises(RuntimeError):
            await session.send("hello")
        assert session.history == []
        if session._session_id:
            assert session.session_store.load_messages(session._session_id) == []

    @pytest.mark.asyncio
    async def test_send_history_passed_to_llm(self, session):
        """The LLM must receive the full history on each call."""
        calls = []

        async def tracking_stream(messages, config=None, tools=None):
            calls.append(list(messages))
            yield CompletionChunk(delta_content="ok")

        session.llm.chat_stream = tracking_stream
        # Pre-populate history
        session._do_new_session()
        session.history.append(Message(role=MessageRole.USER, content="first"))
        session.history.append(Message(role=MessageRole.ASSISTANT, content="reply"))
        await session.send("second")
        # history: system (always) + 2 pre-populated + new user = 4
        first_call = calls[0]
        assert len(first_call) == 4
        assert first_call[0].role == MessageRole.SYSTEM
        assert first_call[-1].content == "second"


# ===========================================================================
# System prompt
# ===========================================================================

class TestSystemPrompt:
    @pytest.mark.asyncio
    async def test_custom_system_prompt_included(self, session):
        """Custom system prompt must appear inside the built-in system message."""
        received_msgs = []

        async def tracking_stream(messages, config=None, tools=None):
            received_msgs.extend(messages)
            yield CompletionChunk(delta_content="ok")

        session.llm.chat_stream = tracking_stream
        session.system_prompt = "You are a pirate."
        await session.send("hello")

        assert received_msgs[0].role == MessageRole.SYSTEM
        assert "You are a pirate." in received_msgs[0].content

    @pytest.mark.asyncio
    async def test_builtin_system_prompt_always_present(self, session):
        """Even without a custom prompt, a system message is always prepended."""
        received_msgs = []

        async def tracking_stream(messages, config=None, tools=None):
            received_msgs.extend(messages)
            yield CompletionChunk(delta_content="ok")

        session.llm.chat_stream = tracking_stream
        session.system_prompt = None
        await session.send("hello")

        assert received_msgs[0].role == MessageRole.SYSTEM
        assert "k-ai" in received_msgs[0].content

    def test_messages_with_system_returns_system_first(self, session):
        session.system_prompt = "Be concise."
        session.history.append(Message(role=MessageRole.USER, content="hi"))
        msgs = session._messages_with_system()
        assert msgs[0].role == MessageRole.SYSTEM
        assert "Be concise." in msgs[0].content
        assert msgs[1].role == MessageRole.USER

    def test_messages_with_system_always_has_system(self, session):
        """Built-in system prompt is always present, even without custom prompt."""
        session.system_prompt = None
        session.history.append(Message(role=MessageRole.USER, content="hi"))
        msgs = session._messages_with_system()
        assert msgs[0].role == MessageRole.SYSTEM
        assert len(msgs) == 2

    def test_prompt_template_inserts_assistant_name(self, session):
        session.cm.set("prompts.assistant_name", "K-Prime")
        prompt = session._build_system_prompt()
        assert "You are K-Prime" in prompt

    def test_prompt_template_inserts_user_name_from_memory(self, session):
        session.memory.add("Preferred user name: Ivann.")
        rendered = session._render_prompt_template("Hello {user_name} from {assistant_name}.")
        assert rendered == "Hello Ivann from k-ai."

    def test_internal_memory_has_priority_over_external_memory(self, session):
        session.external_memory = "Preferred user name: OldExternal"
        session.memory.add("Preferred user name: NewInternal.")
        prompt = session._build_system_prompt()
        assert "1. Internal remembered facts" in prompt
        assert "2. Built-in and session-specific internal prompts/instructions." in prompt
        assert "3. External user context loaded from a file." in prompt
        assert "Preferred user name: NewInternal." in prompt
        assert prompt.index("## Remembered Facts") < prompt.index("## User Context")

    def test_build_system_prompt_includes_init_guidance_when_active(self, session):
        session._init_mode = True
        prompt = session._build_system_prompt()
        assert "Initialization mode is active" in prompt
        assert "init_system" in prompt

    def test_build_system_prompt_includes_post_switch_continuation_guidance(self, session):
        session._continuation_after_switch = {
            "summary": "Présentation de l'assistant",
            "reason": "Le sujet a changé",
        }
        prompt = session._build_system_prompt()
        assert "Continuation After Session Switch" in prompt
        assert "do not propose another session-switch tool" in prompt
        assert "Présentation de l'assistant" in prompt


class TestRuntimeConfig:
    def test_runtime_snapshot_contains_context_stats(self, session):
        session.history.append(Message(role=MessageRole.USER, content="hello world"))
        snapshot = session.get_runtime_snapshot()
        assert snapshot["context_window"] > 0
        assert snapshot["estimated_context_tokens"] >= 0
        assert "context_percent" in snapshot

    def test_apply_config_change_updates_runtime_model_override(self, session):
        session.apply_config_change("model", "phi4-mini:latest")
        assert session.cm.get("model") == "phi4-mini:latest"
        assert session.llm.model_name == "phi4-mini:latest"

    def test_handle_prompt_interrupt_requires_two_ctrl_c(self, session):
        assert session._handle_prompt_interrupt() is False
        assert session._handle_prompt_interrupt() is True

    def test_token_snapshot_falls_back_to_estimate_when_provider_usage_missing(self, session):
        session.history.append(Message(role=MessageRole.USER, content="bonjour"))
        session.history.append(Message(role=MessageRole.ASSISTANT, content="salut"))
        tokens = session.get_token_snapshot()
        assert tokens["token_source"] == "estimated"
        assert tokens["total_tokens"] > 0

    def test_should_offer_init_when_no_sessions_and_memory_empty(self, session):
        assert session._should_offer_init() is True

    def test_should_not_offer_init_when_memory_exists(self, session):
        session.memory.add("Preferred user name: Ivann.")
        assert session._should_offer_init() is False


class TestToolPolicies:
    def test_tool_catalog_covers_all_registered_tools(self, session):
        catalog_names = set(session._tool_catalog_entries().keys())
        registry_names = set(session.tool_registry.get_names())
        assert catalog_names == registry_names

    def test_default_policy_follows_risk(self, session):
        clear_screen = session.tool_registry.get("clear_screen")
        delete_session = session.tool_registry.get("delete_session")
        assert clear_screen.danger_level == "low"
        assert delete_session.danger_level == "high"
        assert session._resolve_tool_policy(clear_screen)["policy"] == "auto"
        assert session._resolve_tool_policy(delete_session)["policy"] == "ask"

    def test_session_override_takes_precedence(self, session):
        change = session.update_tool_policy("tool", "delete_session", "auto", scope="session")
        assert change["policy"] == "auto"
        delete_session = session.tool_registry.get("delete_session")
        resolved = session._resolve_tool_policy(delete_session)
        assert resolved["policy"] == "auto"
        assert resolved["source"] == "session"

    def test_global_risk_override_applies(self, session):
        session.update_tool_policy("risk", "medium", "auto", scope="global", persist=False)
        load_session = session.tool_registry.get("load_session")
        resolved = session._resolve_tool_policy(load_session)
        assert resolved["policy"] == "auto"
        assert resolved["source"] == "global"

    def test_protected_tool_policy_cannot_be_changed(self, session):
        with pytest.raises(ValueError):
            session.update_tool_policy("tool", "tool_policy_set", "auto", scope="session")
        protected = session.tool_registry.get("tool_policy_set")
        resolved = session._resolve_tool_policy(protected)
        assert resolved["policy"] == "ask"
        assert resolved["source"] == "protected"

    def test_invalid_global_override_bucket_raises(self, session):
        session.cm.set("tool_approval.global_overrides.invalid", {"clear_screen": "auto"})
        with pytest.raises(ValueError):
            session._tool_policy_global_overrides()

    def test_invalid_global_override_target_raises(self, session):
        session.cm.set("tool_approval.global_overrides.tools.unknown_tool", "auto")
        with pytest.raises(ValueError):
            session._tool_policy_global_overrides()

    @pytest.mark.asyncio
    async def test_low_risk_tool_auto_executes_without_confirm(self, session):
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        tc = ToolCall(id="c1", function_name="clear_screen", arguments={})
        with patch("k_ai.session.Confirm.ask", side_effect=AssertionError("Confirm should not be called")):
            result = await session._execute_internal_tool(tc)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_tool_proposal_uses_fallback_rationale_when_model_provides_none(self, session):
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        tc = ToolCall(id="c1", function_name="clear_screen", arguments={})

        with patch("k_ai.session.render_tool_proposal") as render_mock:
            result = await session._execute_internal_tool(tc, rationale="")

        assert result.success is True
        assert render_mock.call_args.kwargs["show_rationale"] is True
        assert "Use clear_screen" in render_mock.call_args.kwargs["rationale"]

    @pytest.mark.asyncio
    async def test_tool_proposal_can_hide_rationale_panel_via_config(self, session):
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        session.cm.set("cli.show_tool_rationale", False)
        tc = ToolCall(id="c1", function_name="clear_screen", arguments={})

        with patch("k_ai.session.render_tool_proposal") as render_mock:
            result = await session._execute_internal_tool(tc, rationale="explicit")

        assert result.success is True
        assert render_mock.call_args.kwargs["show_rationale"] is False

    @pytest.mark.asyncio
    async def test_protected_admin_tool_still_requests_confirmation(self, session):
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        tc = ToolCall(id="c1", function_name="tool_policy_list", arguments={})
        with patch("k_ai.session.Confirm.ask", return_value=False) as confirm:
            result = await session._execute_internal_tool(tc)
        assert confirm.called
        assert result.success is False


# ===========================================================================
# send_with_tools()
# ===========================================================================

class TestSendWithTools:
    @pytest.mark.asyncio
    async def test_direct_answer_no_tool_calls(self, session):
        """Model answers directly — no tool calls, one round."""
        session.llm = mock_llm([CompletionChunk(delta_content="Paris", finish_reason="stop")])

        async def executor(tc: ToolCall) -> str:
            pytest.fail("executor should not be called")

        result = await session.send_with_tools("capital of France?", tools=[], tool_executor=executor)
        assert result == "Paris"

    @pytest.mark.asyncio
    async def test_single_tool_call_round(self, session):
        tc = ToolCall(id="call_1", function_name="get_weather", arguments={"location": "Paris"})
        round1_chunks = [make_tool_call_chunk(tc)]
        round2_chunks = [CompletionChunk(delta_content="22°C in Paris.", finish_reason="stop")]

        call_count = 0

        async def _stream(messages, config=None, tools=None):
            nonlocal call_count
            chunks = round1_chunks if call_count == 0 else round2_chunks
            call_count += 1
            for c in chunks:
                yield c

        session.llm.chat_stream = _stream
        executor_calls = []

        async def executor(tc: ToolCall) -> str:
            executor_calls.append(tc)
            return f"22°C in {tc.arguments['location']}"

        tools_def = [{"type": "function", "function": {"name": "get_weather", "description": "...",
                                                         "parameters": {"type": "object", "properties": {}}}}]
        result = await session.send_with_tools("weather in Paris?", tools=tools_def, tool_executor=executor)

        assert result == "22°C in Paris."
        assert len(executor_calls) == 1
        assert executor_calls[0].function_name == "get_weather"

    @pytest.mark.asyncio
    async def test_tool_result_added_to_history(self, session):
        tc = ToolCall(id="c1", function_name="fn", arguments={})
        call_count = 0

        async def _stream(messages, config=None, tools=None):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                yield make_tool_call_chunk(tc)
            else:
                yield CompletionChunk(delta_content="done", finish_reason="stop")

        session.llm.chat_stream = _stream

        async def executor(tc: ToolCall) -> str:
            return "result_value"

        await session.send_with_tools("go", tools=[], tool_executor=executor)

        # history: user, assistant+tools, tool_result, assistant
        tool_msgs = [m for m in session.history if m.role == MessageRole.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "result_value"
        assert tool_msgs[0].tool_call_id == "c1"

    @pytest.mark.asyncio
    async def test_assistant_message_with_tool_calls_in_history(self, session):
        tc = ToolCall(id="c1", function_name="fn", arguments={})
        call_count = 0

        async def _stream(messages, config=None, tools=None):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                yield CompletionChunk(delta_content="thinking...")
                yield make_tool_call_chunk(tc)
            else:
                yield CompletionChunk(delta_content="done", finish_reason="stop")

        session.llm.chat_stream = _stream

        async def executor(tc: ToolCall) -> str:
            return "ok"

        await session.send_with_tools("run", tools=[], tool_executor=executor)

        assistant_with_tools = [m for m in session.history
                                 if m.role == MessageRole.ASSISTANT and m.tool_calls]
        assert len(assistant_with_tools) == 1
        assert assistant_with_tools[0].tool_calls[0].function_name == "fn"

    @pytest.mark.asyncio
    async def test_max_tool_rounds_safety_limit(self, session):
        """When every round returns a tool call, loop must stop at max_tool_rounds."""
        tc = ToolCall(id="c1", function_name="infinite", arguments={})

        async def _stream(messages, config=None, tools=None):
            yield make_tool_call_chunk(tc)

        session.llm.chat_stream = _stream
        executor_calls = 0

        async def executor(tc: ToolCall) -> str:
            nonlocal executor_calls
            executor_calls += 1
            return "still going"

        await session.send_with_tools("go forever", tools=[], tool_executor=executor, max_tool_rounds=3)
        assert executor_calls == 3

    @pytest.mark.asyncio
    async def test_executor_exception_sent_as_error_string(self, session):
        """If executor raises, the error string is fed back to the model."""
        tc = ToolCall(id="c1", function_name="bad_fn", arguments={})
        call_count = 0

        async def _stream(messages, config=None, tools=None):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                yield make_tool_call_chunk(tc)
            else:
                # Capture what was sent to us
                tool_result_msg = next(
                    (m for m in messages if m.role == MessageRole.TOOL), None
                )
                content = tool_result_msg.content if tool_result_msg else ""
                yield CompletionChunk(delta_content=content, finish_reason="stop")

        session.llm.chat_stream = _stream

        async def executor(tc: ToolCall) -> str:
            raise ValueError("tool crashed")

        result = await session.send_with_tools("run bad fn", tools=[], tool_executor=executor)
        # The error string must appear in the message fed to the model on round 2
        assert "tool crashed" in result or "Error" in result

    @pytest.mark.asyncio
    async def test_multiple_tools_per_round(self, session):
        """Two tool calls in one round — both executors run."""
        tc1 = ToolCall(id="c1", function_name="fn1", arguments={})
        tc2 = ToolCall(id="c2", function_name="fn2", arguments={})
        call_count = 0

        async def _stream(messages, config=None, tools=None):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                yield CompletionChunk(tool_calls=[tc1, tc2], finish_reason="tool_calls")
            else:
                yield CompletionChunk(delta_content="done", finish_reason="stop")

        session.llm.chat_stream = _stream
        called_fns = []

        async def executor(tc: ToolCall) -> str:
            called_fns.append(tc.function_name)
            return "ok"

        await session.send_with_tools("run both", tools=[], tool_executor=executor)
        assert set(called_fns) == {"fn1", "fn2"}

    @pytest.mark.asyncio
    async def test_send_with_tools_rolls_back_on_llm_failure(self, session):
        async def failing_stream(messages, config=None, tools=None):
            raise RuntimeError("provider down")
            yield

        session.llm.chat_stream = failing_stream

        async def executor(tc: ToolCall) -> str:
            return "ok"

        with pytest.raises(RuntimeError):
            await session.send_with_tools("boom", tools=[], tool_executor=executor)
        assert session.history == []


# ===========================================================================
# reload_provider
# ===========================================================================

# ===========================================================================
# Boot flow & lazy session creation
# ===========================================================================

class TestBootFlow:
    @pytest.mark.asyncio
    async def test_no_session_at_init(self, session):
        """Bug fix: session must NOT be created at init time."""
        assert session._session_id is None

    @pytest.mark.asyncio
    async def test_boot_greeting_does_not_create_session(self, session):
        """Boot greeting is ephemeral — no session created."""
        session.llm = mock_llm([CompletionChunk(delta_content="Hello!", finish_reason="stop")])
        await session._boot_greeting([])
        assert session._session_id is None

    @pytest.mark.asyncio
    async def test_boot_greeting_with_sessions_does_not_create_session(self, session):
        """Even with recent sessions, boot greeting must not create a session."""
        from k_ai.models import SessionMetadata
        recent = [SessionMetadata(
            id="abc123", title="Old chat", summary="We discussed X.",
            created_at="2026-03-25", updated_at="2026-03-25",
            message_count=10,
        )]
        session.llm = mock_llm([CompletionChunk(delta_content="Welcome back!", finish_reason="stop")])
        await session._boot_greeting(recent)
        assert session._session_id is None

    @pytest.mark.asyncio
    async def test_boot_greeting_with_sessions_only_exposes_load_session_tool(self, session):
        from k_ai.models import SessionMetadata

        recent = [SessionMetadata(
            id="abc123", title="Old chat", summary="We discussed X.",
            created_at="2026-03-25", updated_at="2026-03-25",
            message_count=10,
        )]
        seen_tools = []

        async def tracking_stream(messages, config=None, tools=None):
            seen_tools.extend(tools or [])
            yield CompletionChunk(delta_content="Welcome back!", finish_reason="stop")

        session.llm.chat_stream = tracking_stream
        await session._boot_greeting(recent)
        assert seen_tools
        assert {tool["function"]["name"] for tool in seen_tools} == {"load_session"}

    @pytest.mark.asyncio
    async def test_boot_greeting_error_does_not_create_session(self, session):
        """If the LLM fails during boot greeting, no session is created."""
        async def failing_stream(messages, config=None, tools=None):
            raise RuntimeError("LLM down")
            yield  # make it a generator

        session.llm.chat_stream = failing_stream
        await session._boot_greeting([])
        assert session._session_id is None

    def test_lazy_session_creation_on_new(self, session):
        """_do_new_session creates a session in the store."""
        assert session._session_id is None
        session._do_new_session()
        assert session._session_id is not None
        meta = session.session_store.get_session(session._session_id)
        assert meta is not None
        assert meta.session_type == "classic"

    def test_lazy_session_creation_on_new_with_seed(self, session):
        session._do_new_session(seed={"session_type": "meta", "summary": "Admin runtime", "themes": ["config"]})
        meta = session.session_store.get_session(session._session_id)
        assert meta is not None
        assert meta.session_type == "meta"
        assert meta.summary == "Admin runtime"
        assert "config" in meta.themes

    def test_persist_message_skipped_when_no_session(self, session):
        """Messages are NOT persisted if no session is active."""
        msg = Message(role=MessageRole.USER, content="ephemeral")
        session._persist_message(msg)  # should not raise
        # No session file should exist
        assert session._session_id is None

    def test_persist_message_works_with_active_session(self, session):
        """Messages ARE persisted when a session is active."""
        session._do_new_session()
        msg = Message(role=MessageRole.USER, content="persistent")
        session._persist_message(msg)
        loaded = session.session_store.load_messages(session._session_id)
        assert len(loaded) == 1
        assert loaded[0].content == "persistent"

    @pytest.mark.asyncio
    async def test_close_without_response_saves_nothing(self, session):
        """If user opens k-ai and closes immediately, nothing is saved."""
        session.llm = mock_llm([CompletionChunk(delta_content="Hi!", finish_reason="stop")])
        await session._boot_greeting([])
        # User closes (Ctrl+C) — session_id is still None
        assert session._session_id is None
        assert session.session_store.session_count() == 0

    def test_prompts_loaded_from_config(self, session):
        """All prompts should come from config, not hardcoded."""
        identity = session.cm.get_nested("prompts", "identity")
        assert identity is not None
        assert len(identity) > 10
        assert "python_exec" in identity
        assert "shell_exec" in identity
        assert "exa_search" in identity

        boot_with = session.cm.get_nested("prompts", "boot_with_sessions")
        assert boot_with is not None

        boot_no = session.cm.get_nested("prompts", "boot_no_sessions")
        assert boot_no is not None

        init_intro = session.cm.get_nested("prompts", "init_intro")
        assert init_intro is not None

        init_active = session.cm.get_nested("prompts", "init_active")
        assert init_active is not None

        compact = session.cm.get_nested("prompts", "compact_summarize")
        assert compact is not None

        exit_title = session.cm.get_nested("prompts", "exit_title")
        assert exit_title is not None

        exit_summary = session.cm.get_nested("prompts", "exit_summary")
        assert exit_summary is not None


class TestDisabledToolFiltering:
    def test_disabled_tools_excluded_from_active(self, session):
        """Tools with enabled:false in config must NOT appear in LLM definitions."""
        session.cm.config.setdefault("tools", {})["exa"] = {"enabled": False}
        active = session._get_active_tools()
        names = {t["function"]["name"] for t in active}
        assert "exa_search" not in names

    def test_enabled_tools_included(self, session):
        """Tools with enabled:true must appear."""
        session.cm.config.setdefault("tools", {})["python"] = {"enabled": True}
        active = session._get_active_tools()
        names = {t["function"]["name"] for t in active}
        assert "python_exec" in names

    def test_all_tools_present_when_none_disabled(self, session):
        """With no disabled tools, all tools are in the list."""
        active = session._get_active_tools()
        assert len(active) == len(session.tool_registry.list_tools())

    @pytest.mark.asyncio
    async def test_disabled_capability_blocks_direct_tool_execution(self, session):
        session.cm.config.setdefault("tools", {})["qmd"] = {"enabled": False}
        tc = ToolCall(id="c1", function_name="qmd_search", arguments={"query": "abc"})
        result = await session._execute_internal_tool(tc)
        assert result.success is False
        assert "disabled" in result.message


class TestDebugMode:
    def test_debug_false_by_default(self, session):
        assert session._debug is False

    def test_debug_reads_from_config(self, session):
        session.cm.config.setdefault("cli", {})["debug"] = True
        assert session._debug is True


class TestTurnRollback:
    @pytest.mark.asyncio
    async def test_process_message_rolls_back_empty_assistant_turn(self, session):
        async def _stream(messages, config=None, tools=None):
            yield CompletionChunk(delta_content="", finish_reason="stop")

        session.llm.chat_stream = _stream
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        session._do_new_session()

        await session._process_message("requete silencieuse")

        assert session.history == []
        persisted = session.session_store.load_messages(session._session_id)
        assert persisted == []

    @pytest.mark.asyncio
    async def test_process_message_interrupt_after_partial_content_keeps_user_and_partial_assistant(self, session):
        async def _stream(messages, config=None, tools=None):
            yield CompletionChunk(delta_content="Premiere partie")
            raise KeyboardInterrupt
            yield  # pragma: no cover

        session.llm.chat_stream = _stream
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        session._do_new_session()

        await session._process_message("question interrompue")

        assert len(session.history) == 2
        assert session.history[0].role == MessageRole.USER
        assert session.history[0].content == "question interrompue"
        assert session.history[1].role == MessageRole.ASSISTANT
        assert "Premiere partie" in session.history[1].content
        assert "[Response interrupted by user]" in session.history[1].content

    @pytest.mark.asyncio
    async def test_process_message_interrupt_before_any_token_keeps_user_message(self, session):
        async def _stream(messages, config=None, tools=None):
            raise KeyboardInterrupt
            yield  # pragma: no cover

        session.llm.chat_stream = _stream
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        session._do_new_session()

        await session._process_message("question sans reponse")

        assert len(session.history) == 1
        assert session.history[0].role == MessageRole.USER
        assert session.history[0].content == "question sans reponse"

    @pytest.mark.asyncio
    async def test_llm_error_after_tool_execution_rolls_back_whole_turn(self, session):
        tc = ToolCall(id="c1", function_name="clear_screen", arguments={})
        call_count = 0

        async def _stream(messages, config=None, tools=None):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                yield CompletionChunk(tool_calls=[tc], finish_reason="tool_calls")
            else:
                raise LLMError("boom")
                yield  # pragma: no cover

        session.llm.chat_stream = _stream
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        session._do_new_session()

        with patch("k_ai.session.Confirm.ask", return_value=True):
            await session._process_message("clear")

        assert session.history == []
        persisted = session.session_store.load_messages(session._session_id)
        assert persisted == []

    @pytest.mark.asyncio
    async def test_tool_result_is_truncated_before_adding_to_history(self, session):
        huge_result = "x" * 5000
        normalized = session._normalize_tool_result_for_history(huge_result)
        assert len(normalized) < len(huge_result)
        assert normalized.endswith("...(truncated for history)")

    @pytest.mark.asyncio
    async def test_execute_internal_tool_ctrl_c_on_confirm_returns_interrupted(self, session):
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        tc = ToolCall(id="c1", function_name="delete_session", arguments={"session_id": "deadbeef"})

        with patch("k_ai.session.Confirm.ask", side_effect=KeyboardInterrupt):
            result = await session._execute_internal_tool(tc)

        assert result.data["interrupted"] is True

    @pytest.mark.asyncio
    async def test_execute_internal_tool_detects_interrupt_flag_after_confirm(self, session):
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        tc = ToolCall(id="c1", function_name="delete_session", arguments={"session_id": "deadbeef"})

        @contextmanager
        def interrupted_scope(*args, **kwargs):
            session._interrupt_requested = False
            try:
                yield
            finally:
                session._interrupt_requested = True

        with patch.object(session, "_interrupt_scope", interrupted_scope):
            with patch("k_ai.session.Confirm.ask", return_value=True):
                result = await session._execute_internal_tool(tc)

        assert result.data["interrupted"] is True

    @pytest.mark.asyncio
    async def test_process_message_stops_batch_after_interrupted_tool(self, session):
        tc1 = ToolCall(id="c1", function_name="clear_screen", arguments={})
        tc2 = ToolCall(id="c2", function_name="clear_screen", arguments={})

        async def _stream(messages, config=None, tools=None):
            yield CompletionChunk(tool_calls=[tc1, tc2], finish_reason="tool_calls")

        session.llm.chat_stream = _stream
        session._do_new_session()
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        session._execute_internal_tool = AsyncMock(
            return_value=ToolResult(success=False, message="Interrupted by user.", data={"interrupted": True})
        )

        await session._process_message("batch")

        session._execute_internal_tool.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_message_reuses_duplicate_tool_call_within_same_turn(self, session):
        tc1 = ToolCall(id="c1", function_name="clear_screen", arguments={})
        tc2 = ToolCall(id="c2", function_name="clear_screen", arguments={})
        call_count = 0

        async def _stream(messages, config=None, tools=None):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                yield CompletionChunk(tool_calls=[tc1], finish_reason="tool_calls")
            elif call_count == 1:
                call_count += 1
                yield CompletionChunk(tool_calls=[tc2], finish_reason="tool_calls")
            else:
                yield CompletionChunk(delta_content="done", finish_reason="stop")

        session.llm.chat_stream = _stream
        session._do_new_session()
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        session._execute_internal_tool = AsyncMock(
            return_value=ToolResult(success=True, message="cleared")
        )

        await session._process_message("double")

        session._execute_internal_tool.assert_awaited_once()
        tool_msgs = [m for m in session.history if m.role == MessageRole.TOOL]
        assert len(tool_msgs) == 2
        assert tool_msgs[0].content == "cleared"
        assert tool_msgs[1].content == "cleared"

    @pytest.mark.asyncio
    async def test_switch_session_rolls_back_current_turn_and_queues_new_session(self, session):
        session._do_new_session(seed={"session_type": "meta", "summary": "Admin"})
        old = Message(role=MessageRole.USER, content="ancien sujet")
        session.history.append(old)
        session._persist_message(old)
        turn_start = len(session.history)

        tc = ToolCall(
            id="c1",
            function_name="switch_session",
            arguments={"summary": "Nouveau sujet", "themes": ["python"], "session_type": "classic"},
        )

        async def _stream(messages, config=None, tools=None):
            yield CompletionChunk(tool_calls=[tc], finish_reason="tool_calls")

        async def _execute(tool_call, rationale=""):
            session._handle_new_session(
                seed={"summary": "Nouveau sujet", "themes": ["python"], "session_type": "classic"},
                carry_over_message="nouveau sujet",
            )
            return ToolResult(success=True, message="switch")

        session.llm.chat_stream = _stream
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        session._execute_internal_tool = AsyncMock(side_effect=_execute)
        session.generate_session_digest = AsyncMock(return_value={"summary": "Admin", "themes": ["config"], "session_type": "meta"})

        await session._process_message("nouveau sujet")

        assert session._new_session_requested is True
        assert session._queued_new_session_message == "nouveau sujet"
        assert len(session.history) == turn_start
        assert session.history[-1].content == "ancien sujet"

    @pytest.mark.asyncio
    async def test_new_session_rolls_back_current_turn_and_queues_carry_message(self, session):
        session._do_new_session(seed={"session_type": "classic", "summary": "Sujet initial"})
        old = Message(role=MessageRole.USER, content="ancien sujet")
        session.history.append(old)
        session._persist_message(old)
        turn_start = len(session.history)

        tc = ToolCall(
            id="c1",
            function_name="new_session",
            arguments={"summary": "Méca céleste", "themes": ["orbites"], "session_type": "classic"},
        )

        async def _stream(messages, config=None, tools=None):
            yield CompletionChunk(tool_calls=[tc], finish_reason="tool_calls")

        async def _execute(tool_call, rationale=""):
            session._handle_new_session(
                seed={"summary": "Méca céleste", "themes": ["orbites"], "session_type": "classic"},
                carry_over_message="parle moi de meca celeste",
            )
            return ToolResult(success=True, message="new session requested", data={"carry_over_message": True})

        session.llm.chat_stream = _stream
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        session._execute_internal_tool = AsyncMock(side_effect=_execute)
        session.generate_session_digest = AsyncMock(return_value={"summary": "Sujet initial", "themes": ["ancien"], "session_type": "classic"})

        await session._process_message("parle moi de meca celeste")

        assert session._new_session_requested is True
        assert session._queued_new_session_message == "parle moi de meca celeste"
        assert len(session.history) == turn_start
        assert session.history[-1].content == "ancien sujet"

    @pytest.mark.asyncio
    async def test_process_message_suppresses_switch_session_after_already_switched(self, session):
        tc = ToolCall(
            id="c1",
            function_name="switch_session",
            arguments={"summary": "Présentation", "session_type": "meta", "reason": "nouveau sujet"},
        )
        call_count = 0

        async def _stream(messages, config=None, tools=None):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                yield CompletionChunk(tool_calls=[tc], finish_reason="tool_calls")
            else:
                yield CompletionChunk(delta_content="Je suis k-ai.", finish_reason="stop")

        session.llm.chat_stream = _stream
        session._do_new_session(seed={"session_type": "meta", "summary": "Présentation"})
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        session._continuation_after_switch = {"summary": "Présentation", "reason": "nouveau sujet"}
        session._execute_internal_tool = AsyncMock()

        await session._process_message("switch la session et parle de toi", suppress_switch=True)

        session._execute_internal_tool.assert_not_awaited()
        assistant_msgs = [m for m in session.history if m.role == MessageRole.ASSISTANT]
        assert assistant_msgs[-1].content == "Je suis k-ai."

    def test_session_shift_guidance_is_prompt_driven_and_includes_user_input(self, session):
        session._do_new_session(seed={"session_type": "classic", "summary": "Sport"})
        guidance = session._session_shift_guidance_for_turn("bascule sur une nouvelle session et parle moi de meca celeste")
        assert "bascule sur une nouvelle session" in guidance
        assert "new_session" in guidance
        assert "switch_session" in guidance


class TestPersistenceConsistency:
    def test_reset_history_rewrites_session_file(self, session):
        session._do_new_session()
        session._persist_message(Message(role=MessageRole.USER, content="hello"))
        session._persist_message(Message(role=MessageRole.ASSISTANT, content="world"))

        session.reset_history()

        assert session.history == []
        assert session.session_store.load_messages(session._session_id) == []

    @pytest.mark.asyncio
    async def test_compaction_rewrites_persisted_history(self, session):
        session.cm.set("compaction.keep_last_n", "2")
        session._do_new_session()

        messages = [
            Message(role=MessageRole.USER, content="u1"),
            Message(role=MessageRole.ASSISTANT, content="a1"),
            Message(role=MessageRole.USER, content="u2"),
            Message(role=MessageRole.ASSISTANT, content="a2"),
        ]
        for message in messages:
            session.history.append(message)
            session._persist_message(message)

        async def compact_stream(messages, config=None, tools=None):
            yield CompletionChunk(delta_content="compact summary", finish_reason="stop")

        session.llm.chat_stream = compact_stream
        await session._do_compact()

        persisted = session.session_store.load_messages(session._session_id)
        assert len(persisted) == 3
        assert persisted[0].role == MessageRole.SYSTEM
        assert "compact summary" in persisted[0].content


class TestSessionLoadShowsHistory:
    def test_load_session_shows_recent_messages(self, session):
        """Loading a session should display recent messages."""
        session._do_new_session()
        sid = session._session_id
        # Add some messages
        for i in range(5):
            session.session_store.save_message(sid, Message(role=MessageRole.USER, content=f"msg {i}"))
            session.session_store.save_message(sid, Message(role=MessageRole.ASSISTANT, content=f"reply {i}"))
        session.session_store.update_meta(sid, message_count=10)

        # Now load it into a fresh session
        from unittest.mock import MagicMock
        session.console = MagicMock()
        session._do_load_session(sid)
        assert session._session_id == sid
        assert len(session.history) == 10
        # Console should have been called with the history preview
        assert session.console.print.call_count >= 3  # Resumed + header + messages

    def test_load_session_with_last_n_only_loads_window(self, session):
        session._do_new_session()
        sid = session._session_id
        for i in range(6):
            session.session_store.save_message(sid, Message(role=MessageRole.USER, content=f"msg {i}"))
        session.console = MagicMock()
        session._tool_ctx.console = session.console
        session._do_load_session(sid, last_n=2)
        assert [m.content for m in session.history] == ["msg 4", "msg 5"]


class TestSessionDigest:
    @pytest.mark.asyncio
    async def test_generate_session_digest_parses_json(self, session):
        session._do_new_session()
        sid = session._session_id
        session.session_store.save_message(sid, Message(role=MessageRole.USER, content="Parlons de diagonalisation et Python"))

        async def digest_stream(messages, config=None, tools=None):
            yield CompletionChunk(
                delta_content='{"summary":"Diagonalisation de matrices en Python","themes":["python","algebre lineaire"],"session_type":"classic"}',
                finish_reason="stop",
            )

        session.llm.chat_stream = digest_stream
        digest = await session.generate_session_digest(sid, persist=True)
        assert digest["summary"] == "Diagonalisation de matrices en Python"
        assert "python" in digest["themes"]
        assert digest["session_type"] == "classic"
        meta = session.session_store.get_session(sid)
        assert meta.summary == digest["summary"]
        assert meta.session_type == "classic"

    @pytest.mark.asyncio
    async def test_generate_session_digest_includes_session_summary_guidance(self, session):
        session._do_new_session()
        sid = session._session_id
        session.session_store.save_message(sid, Message(role=MessageRole.USER, content="Parlons de diagonalisation et Python"))
        seen_system = []

        async def digest_stream(messages, config=None, tools=None):
            seen_system.append(messages[0].content)
            yield CompletionChunk(
                delta_content='{"summary":"Diagonalisation de matrices en Python","themes":["python"],"session_type":"classic"}',
                finish_reason="stop",
            )

        session.llm.chat_stream = digest_stream
        await session.generate_session_digest(sid, persist=False)
        assert "For the summary field specifically" in seen_system[0]
        assert "Summarize the user's intent" in seen_system[0]

    @pytest.mark.asyncio
    async def test_auto_rename_on_exit_uses_exit_prompts(self, session):
        session._do_new_session()
        sid = session._session_id
        session.session_store.save_message(sid, Message(role=MessageRole.USER, content="Sujet test"))
        session.session_store.save_message(sid, Message(role=MessageRole.ASSISTANT, content="Réponse test"))
        session.history = session.session_store.load_messages(sid)

        call_idx = 0

        async def stream(messages, config=None, tools=None):
            nonlocal call_idx
            payloads = [
                '{"summary":"Résumé court","themes":["test"],"session_type":"classic"}',
                "Titre final",
                "Résumé final de sortie",
            ]
            yield CompletionChunk(delta_content=payloads[call_idx], finish_reason="stop")
            call_idx += 1

        session.llm.chat_stream = stream
        await session._auto_rename_on_exit()
        meta = session.session_store.get_session(sid)
        assert meta.title == "Titre final"
        assert meta.summary == "Résumé final de sortie"

    @pytest.mark.asyncio
    async def test_auto_commit_runtime_store_on_exit_uses_session_digest(self, session):
        session._do_new_session()
        sid = session._session_id
        session.session_store.save_message(sid, Message(role=MessageRole.USER, content="Sujet test"))
        session.session_store.save_message(sid, Message(role=MessageRole.ASSISTANT, content="Réponse test"))
        session.history = session.session_store.load_messages(sid)
        session.session_store.update_summary(sid, "Résumé de commit", ["test"], "classic")

        with patch("k_ai.session.commit_runtime_state", return_value={"ok": True, "reason": "committed", "subject": "chat: Résumé de commit"}) as mocked:
            await session._auto_commit_runtime_store_on_exit()

        mocked.assert_called_once()
        kwargs = mocked.call_args.kwargs
        assert kwargs["summary"] == "Résumé de commit"
        assert kwargs["session_id"] == sid
        assert kwargs["session_type"] == "classic"
        assert kwargs["themes"] == ["test"]

    @pytest.mark.asyncio
    async def test_auto_commit_runtime_store_on_exit_respects_disable_flag(self, session):
        session.cm.set("runtime_git.auto_commit_on_chat_exit", False)
        session._do_new_session()
        sid = session._session_id
        session.session_store.save_message(sid, Message(role=MessageRole.USER, content="Sujet test"))
        session.session_store.save_message(sid, Message(role=MessageRole.ASSISTANT, content="Réponse test"))
        session.history = session.session_store.load_messages(sid)

        with patch("k_ai.session.commit_runtime_state") as mocked:
            await session._auto_commit_runtime_store_on_exit()

        mocked.assert_not_called()


class TestReloadProvider:
    def test_reload_changes_provider(self, session, cm):
        original_name = session.llm.provider_name
        session.reload_provider(provider="ollama", model="llama3:latest")
        assert session.llm.model_name == "llama3:latest"

    def test_reload_keeps_current_provider_when_none(self, session):
        original_name = session.llm.provider_name
        with patch("k_ai.session.get_provider") as mock_gp:
            mock_llm_obj = MagicMock()
            mock_llm_obj.provider_name = original_name
            mock_gp.return_value = mock_llm_obj
            session.reload_provider()
        mock_gp.assert_called_once_with(session.cm, provider=original_name, model=None)

    def test_preserve_simple_history_only_removes_tool_related_messages(self, session):
        session._do_new_session()
        session.history = [
            Message(role=MessageRole.USER, content="u1"),
            Message(role=MessageRole.ASSISTANT, content="", tool_calls=[ToolCall(id="call_1", function_name="memory_list", arguments={})]),
            Message(role=MessageRole.TOOL, content="tool", tool_call_id="call_1", name="memory_list"),
            Message(role=MessageRole.ASSISTANT, content="plain answer"),
        ]

        removed = session.preserve_simple_history_only()

        assert removed == 2
        assert [m.role for m in session.history] == [MessageRole.USER, MessageRole.ASSISTANT]
        assert session.history[-1].content == "plain answer"

    def test_apply_config_change_provider_preserves_simple_history_before_reload(self, session):
        session._do_new_session()
        session.history = [
            Message(role=MessageRole.USER, content="u1"),
            Message(role=MessageRole.ASSISTANT, content="", tool_calls=[ToolCall(id="call_1", function_name="memory_list", arguments={})]),
            Message(role=MessageRole.TOOL, content="tool", tool_call_id="call_1", name="memory_list"),
            Message(role=MessageRole.ASSISTANT, content="plain answer"),
        ]
        session.cm.set("model", "")

        with patch.object(session, "reload_provider") as reload_mock:
            session.apply_config_change("provider", "mistral")

        reload_mock.assert_called_once()
        assert [m.role for m in session.history] == [MessageRole.USER, MessageRole.ASSISTANT]
