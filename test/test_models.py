# test/test_models.py
"""
Tests for k-ai Pydantic models: Message, ToolCall, CompletionChunk, etc.
"""
import json
import pytest
from k_ai.models import (
    Message,
    MessageRole,
    ToolCall,
    CompletionChunk,
    TokenUsage,
    LLMConfig,
)


# ---------------------------------------------------------------------------
# MessageRole
# ---------------------------------------------------------------------------

class TestMessageRole:
    def test_values(self):
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.TOOL == "tool"

    def test_is_str(self):
        assert isinstance(MessageRole.USER, str)


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------

class TestToolCall:
    def test_basic(self):
        tc = ToolCall(id="call_1", function_name="get_weather", arguments={"location": "Paris"})
        assert tc.id == "call_1"
        assert tc.function_name == "get_weather"
        assert tc.arguments == {"location": "Paris"}

    def test_empty_arguments(self):
        tc = ToolCall(id="x", function_name="ping", arguments={})
        assert tc.arguments == {}


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------

class TestTokenUsage:
    def test_defaults_are_zero(self):
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_custom_values(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert u.total_tokens == 150


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.stream is True

    def test_override(self):
        cfg = LLMConfig(temperature=0.0, max_tokens=512, stream=False)
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 512
        assert cfg.stream is False


# ---------------------------------------------------------------------------
# CompletionChunk
# ---------------------------------------------------------------------------

class TestCompletionChunk:
    def test_defaults(self):
        chunk = CompletionChunk()
        assert chunk.delta_content == ""
        assert chunk.delta_thought is None
        assert chunk.tool_calls is None
        assert chunk.finish_reason is None
        assert chunk.usage is None

    def test_with_content(self):
        chunk = CompletionChunk(delta_content="Hello")
        assert chunk.delta_content == "Hello"

    def test_with_tool_calls(self):
        tc = ToolCall(id="c1", function_name="foo", arguments={"k": "v"})
        chunk = CompletionChunk(tool_calls=[tc], finish_reason="tool_calls")
        assert len(chunk.tool_calls) == 1
        assert chunk.finish_reason == "tool_calls"

    def test_with_usage(self):
        u = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk = CompletionChunk(usage=u)
        assert chunk.usage.total_tokens == 15


# ---------------------------------------------------------------------------
# Message.to_litellm()
# ---------------------------------------------------------------------------

class TestMessageToLiteLLM:
    def test_system_message(self):
        msg = Message(role=MessageRole.SYSTEM, content="Be helpful.")
        d = msg.to_litellm()
        assert d == {"role": "system", "content": "Be helpful."}

    def test_user_message(self):
        msg = Message(role=MessageRole.USER, content="Hello!")
        d = msg.to_litellm()
        assert d == {"role": "user", "content": "Hello!"}

    def test_assistant_message(self):
        msg = Message(role=MessageRole.ASSISTANT, content="Hi there.")
        d = msg.to_litellm()
        assert d == {"role": "assistant", "content": "Hi there."}

    def test_assistant_message_with_tool_calls(self):
        tc = ToolCall(id="call_42", function_name="search", arguments={"q": "Paris"})
        msg = Message(role=MessageRole.ASSISTANT, content="", tool_calls=[tc])
        d = msg.to_litellm()

        assert d["role"] == "assistant"
        assert d["content"] is None  # empty content → null per OpenAI spec
        assert len(d["tool_calls"]) == 1
        tc_d = d["tool_calls"][0]
        assert tc_d["id"] == "call_42"
        assert tc_d["type"] == "function"
        assert tc_d["function"]["name"] == "search"
        # arguments must be a JSON *string*, not a dict
        args = json.loads(tc_d["function"]["arguments"])
        assert args == {"q": "Paris"}

    def test_assistant_message_with_tool_calls_has_content(self):
        """Non-empty content is preserved even when tool_calls is set."""
        tc = ToolCall(id="c1", function_name="f", arguments={})
        msg = Message(role=MessageRole.ASSISTANT, content="Let me check.", tool_calls=[tc])
        d = msg.to_litellm()
        assert d["content"] == "Let me check."

    def test_tool_result_message(self):
        msg = Message(
            role=MessageRole.TOOL,
            content="22°C and sunny",
            tool_call_id="call_42",
            name="get_weather",
        )
        d = msg.to_litellm()
        assert d["role"] == "tool"
        assert d["content"] == "22°C and sunny"
        assert d["tool_call_id"] == "call_42"
        assert d["name"] == "get_weather"

    def test_tool_result_without_name(self):
        msg = Message(role=MessageRole.TOOL, content="ok", tool_call_id="c1")
        d = msg.to_litellm()
        assert "name" not in d
        assert d["tool_call_id"] == "c1"

    def test_none_values_stripped(self):
        """name=None and tool_call_id=None must not appear in the dict."""
        msg = Message(role=MessageRole.USER, content="hi")
        d = msg.to_litellm()
        assert "name" not in d
        assert "tool_call_id" not in d
        assert "tool_calls" not in d

    def test_multiple_tool_calls(self):
        tcs = [
            ToolCall(id=f"c{i}", function_name=f"fn{i}", arguments={"i": i})
            for i in range(3)
        ]
        msg = Message(role=MessageRole.ASSISTANT, content="", tool_calls=tcs)
        d = msg.to_litellm()
        assert len(d["tool_calls"]) == 3
        assert d["tool_calls"][2]["function"]["name"] == "fn2"
