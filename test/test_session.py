# test/test_session.py
"""
Tests for ChatSession: programmatic API (send, send_with_tools), history
management, provider reload, and system-prompt handling.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from k_ai import ConfigManager
from k_ai.session import ChatSession
from k_ai.models import Message, MessageRole, ToolCall, TokenUsage, CompletionChunk
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
    async def test_send_history_passed_to_llm(self, session):
        """The LLM must receive the full history on each call."""
        received_msgs = []

        async def tracking_stream(messages, config=None, tools=None):
            received_msgs.extend(messages)
            yield CompletionChunk(delta_content="ok")

        session.llm.chat_stream = tracking_stream
        # Pre-populate history
        session.history.append(Message(role=MessageRole.USER, content="first"))
        session.history.append(Message(role=MessageRole.ASSISTANT, content="reply"))
        await session.send("second")
        # history: system (always) + 2 pre-populated + new user = 4
        assert len(received_msgs) == 4
        assert received_msgs[0].role == MessageRole.SYSTEM
        assert received_msgs[-1].content == "second"


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

        boot_with = session.cm.get_nested("prompts", "boot_with_sessions")
        assert boot_with is not None

        boot_no = session.cm.get_nested("prompts", "boot_no_sessions")
        assert boot_no is not None

        compact = session.cm.get_nested("prompts", "compact_summarize")
        assert compact is not None

        exit_title = session.cm.get_nested("prompts", "exit_title")
        assert exit_title is not None

        exit_summary = session.cm.get_nested("prompts", "exit_summary")
        assert exit_summary is not None


class TestDisabledToolFiltering:
    def test_disabled_tools_excluded_from_active(self, session):
        """Tools with enabled:false in config must NOT appear in LLM definitions."""
        session.cm.config.setdefault("tools", {})["exa_search"] = {"enabled": False}
        active = session._get_active_tools()
        names = {t["function"]["name"] for t in active}
        assert "exa_search" not in names

    def test_enabled_tools_included(self, session):
        """Tools with enabled:true must appear."""
        session.cm.config.setdefault("tools", {})["python_exec"] = {"enabled": True}
        active = session._get_active_tools()
        names = {t["function"]["name"] for t in active}
        assert "python_exec" in names

    def test_all_tools_present_when_none_disabled(self, session):
        """With no disabled tools, all tools are in the list."""
        active = session._get_active_tools()
        assert len(active) == len(session.tool_registry.list_tools())


class TestDebugMode:
    def test_debug_false_by_default(self, session):
        assert session._debug is False

    def test_debug_reads_from_config(self, session):
        session.cm.config.setdefault("cli", {})["debug"] = True
        assert session._debug is True


class TestTurnRollback:
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
                delta_content='{"summary":"Diagonalisation de matrices en Python","themes":["python","algebre lineaire"]}',
                finish_reason="stop",
            )

        session.llm.chat_stream = digest_stream
        digest = await session.generate_session_digest(sid, persist=True)
        assert digest["summary"] == "Diagonalisation de matrices en Python"
        assert "python" in digest["themes"]
        meta = session.session_store.get_session(sid)
        assert meta.summary == digest["summary"]


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
