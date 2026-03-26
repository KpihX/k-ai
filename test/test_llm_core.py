# test/test_llm_core.py
"""
Tests for LiteLLMDriver: init, model routing, chat_stream, list_models,
exception mapping, and the get_provider factory.
"""
import asyncio
import pytest
import litellm
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from k_ai import ConfigManager
from k_ai.llm_core import LiteLLMDriver, get_provider
from k_ai.models import LLMConfig
from k_ai.exceptions import (
    ConfigurationError,
    ContextLengthExceededError,
    LLMError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ServiceUnavailableError,
)
from conftest import (
    make_chunk, make_usage, make_tool_delta, stream_chunks, make_litellm_exc,
    make_non_streaming_response, make_non_streaming_tool_call,
)


# ===========================================================================
# Initialisation
# ===========================================================================

class TestLiteLLMDriverInit:
    def test_no_auth_provider_created(self, cm):
        driver = LiteLLMDriver(cm, provider_name="ollama")
        assert driver.provider_name == "ollama"
        assert driver.auth_mode == "no_auth"
        assert driver.api_key is None

    def test_default_provider_from_config(self, cm):
        with patch("k_ai.llm_core.resolve_secret", return_value=("sk-test", "mock")):
            driver = LiteLLMDriver(cm)
        assert driver.provider_name == cm.get("provider")  # matches config default

    def test_model_from_config_default(self, cm):
        driver = LiteLLMDriver(cm, provider_name="ollama")
        assert driver.model_name == "phi4-mini:latest"

    def test_model_override(self, cm):
        driver = LiteLLMDriver(cm, provider_name="ollama", model_name="llama3:latest")
        assert driver.model_name == "llama3:latest"

    def test_api_key_provider_resolves_key(self, cm):
        with patch("k_ai.llm_core.resolve_secret", return_value=("sk-test", ".env")) as mock_rs:
            driver = LiteLLMDriver(cm, provider_name="anthropic")
        mock_rs.assert_called_once_with("ANTHROPIC_API_KEY")
        assert driver.api_key == "sk-test"
        assert driver.auth_mode == "api_key"

    def test_api_key_missing_raises_auth_error(self, cm):
        with patch("k_ai.llm_core.resolve_secret", return_value=(None, None)):
            with pytest.raises(ProviderAuthenticationError, match="ANTHROPIC_API_KEY"):
                LiteLLMDriver(cm, provider_name="anthropic")

    def test_unknown_provider_raises_config_error(self, cm):
        with pytest.raises(ConfigurationError, match="nonexistent"):
            LiteLLMDriver(cm, provider_name="nonexistent")

    def test_forced_auth_mode(self, cm):
        from k_ai import auth
        fake_loader = MagicMock(return_value="oauth-token")
        with patch.dict(auth.TOKEN_LOADERS, {"google": fake_loader}):
            driver = LiteLLMDriver(cm, provider_name="gemini", auth_mode="oauth")
        assert driver.auth_mode == "oauth"
        assert driver.api_key == "oauth-token"

    def test_inline_api_key_value(self, cm):
        """api_key_value in config should be used when no env var is set."""
        cm2 = ConfigManager()
        cm2.config.setdefault("api_key", {})["custom"] = {
            "api_key_value": "hardcoded-key",
            "default_model": "custom-model",
            "context_window": 4096,
        }
        with patch("k_ai.llm_core.resolve_secret", return_value=(None, None)):
            driver = LiteLLMDriver(cm2, provider_name="custom")
        assert driver.api_key == "hardcoded-key"

    def test_google_oauth_loader_reads_valid_token_file(self, tmp_path, cm):
        token_file = tmp_path / "google.json"
        token_file.write_text(
            '{"access_token":"oauth-token","expires_at":"2999-01-01T00:00:00+00:00","scopes":["https://www.googleapis.com/auth/cloud-platform"]}',
            encoding="utf-8",
        )
        cm2 = ConfigManager()
        cm2.config.setdefault("oauth", {})["gemini"] = {
            "oauth_provider_name": "google",
            "oauth_scopes": ["https://www.googleapis.com/auth/cloud-platform"],
            "token_path": str(token_file),
            "default_model": "gemini-2.5-flash",
            "context_window": 1000000,
        }
        driver = LiteLLMDriver(cm2, provider_name="gemini", auth_mode="oauth")
        assert driver.api_key == "oauth-token"

    def test_google_oauth_loader_refreshes_expired_token(self, tmp_path, cm):
        token_file = tmp_path / "google.json"
        token_file.write_text(
            '{"access_token":"expired","refresh_token":"rt","client_id":"cid","client_secret":"secret","token_uri":"https://oauth2.googleapis.com/token","expires_at":"2000-01-01T00:00:00+00:00","scopes":["https://www.googleapis.com/auth/cloud-platform"]}',
            encoding="utf-8",
        )
        cm2 = ConfigManager()
        cm2.config.setdefault("oauth", {})["gemini"] = {
            "oauth_provider_name": "google",
            "oauth_scopes": ["https://www.googleapis.com/auth/cloud-platform"],
            "token_path": str(token_file),
            "default_model": "gemini-2.5-flash",
            "context_window": 1000000,
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"access_token": "fresh-token", "expires_in": 3600}
        with patch("k_ai.auth.httpx.post", return_value=mock_resp):
            driver = LiteLLMDriver(cm2, provider_name="gemini", auth_mode="oauth")
        assert driver.api_key == "fresh-token"
        assert "fresh-token" in token_file.read_text(encoding="utf-8")


# ===========================================================================
# Model string routing (the LiteLLM openai/ prefix fix)
# ===========================================================================

class TestModelStringRouting:
    """
    When base_url is set (local/proxy), model must be "openai/<model>".
    When no base_url, model must be "<provider>/<model>".
    This is critical: wrong routing sends to native /api/generate instead of /v1/chat/completions.
    """

    async def _capture_params(self, cm, provider_name, **extra_driver_kwargs):
        """
        Run chat_stream with a mocked acompletion and return the captured params dict.
        """
        captured = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            return stream_chunks(make_chunk(content="ok", finish_reason="stop"))

        driver = LiteLLMDriver(cm, provider_name=provider_name, **extra_driver_kwargs)
        with patch("k_ai.llm_core.litellm.acompletion", new=fake_acompletion):
            from k_ai.models import Message, MessageRole
            async for _ in driver.chat_stream([Message(role=MessageRole.USER, content="hi")]):
                pass
        return captured

    @pytest.mark.asyncio
    async def test_ollama_uses_openai_prefix(self, cm):
        """ollama has base_url → model_str must be openai/<model>."""
        params = await self._capture_params(cm, "ollama")
        assert params["model"].startswith("openai/")

    @pytest.mark.asyncio
    async def test_ollama_base_url_forwarded(self, cm):
        params = await self._capture_params(cm, "ollama")
        assert "base_url" in params
        assert "11434" in params["base_url"]

    @pytest.mark.asyncio
    async def test_no_auth_provider_gets_dummy_api_key(self, cm):
        """Bug fix: ollama (no_auth + base_url) needs a dummy api_key
        so LiteLLM's OpenAI client doesn't error with 'api_key must be set'."""
        params = await self._capture_params(cm, "ollama")
        assert "api_key" in params
        assert params["api_key"] == "no-key-required"

    @pytest.mark.asyncio
    async def test_no_auth_with_tools_gets_dummy_api_key(self, cm):
        """Regression: passing tools to ollama was failing because no api_key."""
        captured = {}

        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            return stream_chunks(make_chunk(content="ok", finish_reason="stop"))

        driver = LiteLLMDriver(cm, provider_name="ollama")
        tools = [{"type": "function", "function": {"name": "t", "description": "d",
                                                     "parameters": {"type": "object", "properties": {}}}}]
        with patch("k_ai.llm_core.litellm.acompletion", new=fake_acompletion):
            from k_ai.models import Message, MessageRole
            async for _ in driver.chat_stream(
                [Message(role=MessageRole.USER, content="hi")],
                tools=tools,
            ):
                pass
        assert captured["api_key"] == "no-key-required"
        assert "tools" in captured

    @pytest.mark.asyncio
    async def test_anthropic_uses_provider_prefix(self, cm):
        """anthropic has no base_url → model_str must be anthropic/<model>."""
        with patch("k_ai.llm_core.resolve_secret", return_value=("sk-test", "mock")):
            params = await self._capture_params(cm, "anthropic")
        assert params["model"].startswith("anthropic/")
        assert "base_url" not in params

    @pytest.mark.asyncio
    async def test_groq_uses_openai_prefix(self, cm):
        """groq has base_url → openai/ prefix."""
        with patch("k_ai.llm_core.resolve_secret", return_value=("gsk-test", "mock")):
            params = await self._capture_params(cm, "groq")
        assert params["model"].startswith("openai/")

    @pytest.mark.asyncio
    async def test_model_name_in_model_str(self, cm):
        params = await self._capture_params(cm, "ollama")
        assert "phi4-mini:latest" in params["model"]

    @pytest.mark.asyncio
    async def test_xai_uses_native_provider_prefix(self, cm):
        """xAI has no base_url → LiteLLM native routing: model_str = xai/<model>."""
        with patch("k_ai.llm_core.resolve_secret", return_value=("xai-test", "mock")):
            params = await self._capture_params(cm, "xai")
        assert params["model"].startswith("xai/")
        assert "base_url" not in params


# ===========================================================================
# chat_stream — content streaming
# ===========================================================================

class TestChatStreamContent:
    @pytest.mark.asyncio
    async def test_yields_text_chunks(self, ollama_driver):
        chunks_in = [
            make_chunk("Hello "),
            make_chunk("world"),
            make_chunk("!", finish_reason="stop"),
        ]

        async def fake(**kwargs):
            return stream_chunks(*chunks_in)

        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="hi")]
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            result = ""
            async for chunk in ollama_driver.chat_stream(msgs):
                result += chunk.delta_content
        assert result == "Hello world!"

    @pytest.mark.asyncio
    async def test_finish_reason_propagated(self, ollama_driver):
        async def fake(**kwargs):
            return stream_chunks(make_chunk("done", finish_reason="stop"))

        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="hi")]
        finish_reasons = []
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for chunk in ollama_driver.chat_stream(msgs):
                if chunk.finish_reason:
                    finish_reasons.append(chunk.finish_reason)
        assert "stop" in finish_reasons

    @pytest.mark.asyncio
    async def test_usage_extracted_from_last_chunk(self, ollama_driver):
        usage_ns = make_usage(prompt=100, completion=50, total=150)
        async def fake(**kwargs):
            return stream_chunks(
                make_chunk("text"),
                make_chunk("", finish_reason="stop", usage=usage_ns),
            )

        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="q")]
        last_usage = None
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for chunk in ollama_driver.chat_stream(msgs):
                if chunk.usage:
                    last_usage = chunk.usage
        assert last_usage is not None
        assert last_usage.prompt_tokens == 100
        assert last_usage.completion_tokens == 50
        assert last_usage.total_tokens == 150

    @pytest.mark.asyncio
    async def test_usage_only_chunk_no_choices(self, ollama_driver):
        """A chunk with no choices but with usage must not raise."""
        usage_ns = make_usage()
        async def fake(**kwargs):
            return stream_chunks(
                make_chunk("text"),
                make_chunk(has_choices=False, usage=usage_ns),
            )

        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="q")]
        chunks_out = []
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for chunk in ollama_driver.chat_stream(msgs):
                chunks_out.append(chunk)
        # Must yield a chunk for the usage-only packet
        usage_chunks = [c for c in chunks_out if c.usage is not None]
        assert len(usage_chunks) == 1

    @pytest.mark.asyncio
    async def test_thinking_blocks_extracted(self, ollama_driver):
        """<think> tags are separated into delta_thought, not delta_content."""
        async def fake(**kwargs):
            return stream_chunks(
                make_chunk("<think>reasoning</think>Answer"),
                make_chunk("", finish_reason="stop"),
            )

        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="think")]
        thoughts = []
        content = []
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for chunk in ollama_driver.chat_stream(msgs):
                if chunk.delta_thought:
                    thoughts.append(chunk.delta_thought)
                content.append(chunk.delta_content)
        assert "reasoning" in "".join(thoughts)
        assert "Answer" in "".join(content)
        # Thinking text must NOT bleed into content
        assert "think" not in "".join(content)

    @pytest.mark.asyncio
    async def test_params_include_temperature_and_max_tokens(self, ollama_driver):
        captured = {}

        async def fake(**kwargs):
            captured.update(kwargs)
            return stream_chunks(make_chunk("x", finish_reason="stop"))

        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="q")]
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for _ in ollama_driver.chat_stream(msgs):
                pass

        assert "temperature" in captured
        assert "max_tokens" in captured
        assert captured["stream"] is True

    @pytest.mark.asyncio
    async def test_per_call_config_overrides(self, ollama_driver):
        captured = {}

        async def fake(**kwargs):
            captured.update(kwargs)
            return stream_chunks(make_chunk("x", finish_reason="stop"))

        from k_ai.models import Message, MessageRole, LLMConfig
        msgs = [Message(role=MessageRole.USER, content="q")]
        cfg = LLMConfig(temperature=0.0, max_tokens=128)
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for _ in ollama_driver.chat_stream(msgs, config=cfg):
                pass

        assert captured["temperature"] == 0.0
        assert captured["max_tokens"] == 128

    @pytest.mark.asyncio
    async def test_tools_forwarded(self, ollama_driver):
        captured = {}

        async def fake(**kwargs):
            captured.update(kwargs)
            return stream_chunks(make_chunk("x", finish_reason="stop"))

        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="q")]
        tools = [{"type": "function", "function": {"name": "fn", "description": "d",
                                                    "parameters": {"type": "object", "properties": {}}}}]
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for _ in ollama_driver.chat_stream(msgs, tools=tools):
                pass

        assert captured["tools"] == tools

    @pytest.mark.asyncio
    async def test_no_tools_no_tools_key(self, ollama_driver):
        """When tools=None, the 'tools' key must not be in params."""
        captured = {}

        async def fake(**kwargs):
            captured.update(kwargs)
            return stream_chunks(make_chunk("x", finish_reason="stop"))

        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="q")]
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for _ in ollama_driver.chat_stream(msgs):
                pass

        assert "tools" not in captured


# ===========================================================================
# chat_stream — tool-call accumulation
# ===========================================================================

class TestChatStreamToolCalls:
    @pytest.mark.asyncio
    async def test_tool_call_emitted_on_finish(self, ollama_driver):
        # Simulate streamed tool call: args arrive in two chunks
        delta1 = make_tool_delta(index=0, tc_id="call_1", name="get_weather", arguments='{"location":')
        delta2 = SimpleNamespace(
            index=0, id=None,
            function=SimpleNamespace(name=None, arguments=' "Paris"}'),
        )
        finish_chunk = make_chunk("", finish_reason="tool_calls")

        async def fake(**kwargs):
            return stream_chunks(
                make_chunk("", tool_call_deltas=[delta1]),
                make_chunk("", tool_call_deltas=[delta2]),
                finish_chunk,
            )

        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="weather?")]
        tool_calls_received = []
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for chunk in ollama_driver.chat_stream(msgs, tools=[]):
                if chunk.tool_calls:
                    tool_calls_received.extend(chunk.tool_calls)

        assert len(tool_calls_received) == 1
        tc = tool_calls_received[0]
        assert tc.id == "call_1"
        assert tc.function_name == "get_weather"
        assert tc.arguments == {"location": "Paris"}

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_emitted(self, ollama_driver):
        d1 = make_tool_delta(index=0, tc_id="c0", name="fn0", arguments='{"k":"v"}')
        d2 = make_tool_delta(index=1, tc_id="c1", name="fn1", arguments='{}')

        async def fake(**kwargs):
            return stream_chunks(
                make_chunk("", tool_call_deltas=[d1, d2]),
                make_chunk("", finish_reason="tool_calls"),
            )

        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="call both")]
        received = []
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for chunk in ollama_driver.chat_stream(msgs, tools=[]):
                if chunk.tool_calls:
                    received.extend(chunk.tool_calls)

        assert len(received) == 2
        names = {tc.function_name for tc in received}
        assert names == {"fn0", "fn1"}

    @pytest.mark.asyncio
    async def test_invalid_json_in_tool_args(self, ollama_driver):
        """Malformed JSON → stored under _raw key instead of crashing."""
        d = make_tool_delta(index=0, tc_id="c", name="fn", arguments="NOT-JSON")

        async def fake(**kwargs):
            return stream_chunks(
                make_chunk("", tool_call_deltas=[d]),
                make_chunk("", finish_reason="tool_calls"),
            )

        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="q")]
        received = []
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for chunk in ollama_driver.chat_stream(msgs, tools=[]):
                if chunk.tool_calls:
                    received.extend(chunk.tool_calls)

        assert len(received) == 1
        assert "_raw" in received[0].arguments


# ===========================================================================
# chat_stream — non-streaming path (stream=False)
# ===========================================================================

class TestChatStreamNonStreaming:
    """
    Verify the stream=False code path: litellm returns a ModelResponse (not an
    async generator); chat_stream must yield exactly ONE CompletionChunk with
    the full content, usage, finish_reason, and tool_calls.
    """

    def _make_driver(self, cm):
        return LiteLLMDriver(cm, provider_name="ollama")

    def _make_cfg(self, **kwargs):
        return LLMConfig(stream=False, **kwargs)

    async def _collect(self, driver, msgs, cfg, fake_response):
        chunks = []
        with patch(
            "k_ai.llm_core.litellm.acompletion",
            new=lambda **kw: asyncio.coroutine(lambda: fake_response)(),
        ):
            async for chunk in driver.chat_stream(msgs, config=cfg):
                chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------------
    # helper that patches acompletion properly (coroutine returning value)
    # ------------------------------------------------------------------
    @staticmethod
    async def _run(driver, cfg, response_obj):
        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="hi")]
        chunks = []
        async def fake(**kwargs):
            return response_obj
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for chunk in driver.chat_stream(msgs, config=cfg):
                chunks.append(chunk)
        return chunks

    @pytest.mark.asyncio
    async def test_yields_exactly_one_chunk(self, ollama_driver):
        resp = make_non_streaming_response(content="Hello")
        chunks = await self._run(ollama_driver, self._make_cfg(), resp)
        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_content_is_full_response(self, ollama_driver):
        resp = make_non_streaming_response(content="Full answer here")
        chunks = await self._run(ollama_driver, self._make_cfg(), resp)
        assert chunks[0].delta_content == "Full answer here"

    @pytest.mark.asyncio
    async def test_finish_reason_propagated(self, ollama_driver):
        resp = make_non_streaming_response(content="ok", finish_reason="stop")
        chunks = await self._run(ollama_driver, self._make_cfg(), resp)
        assert chunks[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_usage_populated(self, ollama_driver):
        usage = make_usage(prompt=20, completion=10, total=30)
        resp = make_non_streaming_response(content="ok", usage=usage)
        chunks = await self._run(ollama_driver, self._make_cfg(), resp)
        u = chunks[0].usage
        assert u is not None
        assert u.prompt_tokens == 20
        assert u.completion_tokens == 10
        assert u.total_tokens == 30

    @pytest.mark.asyncio
    async def test_no_usage_when_none(self, ollama_driver):
        resp = make_non_streaming_response(content="ok", usage=None)
        chunks = await self._run(ollama_driver, self._make_cfg(), resp)
        assert chunks[0].usage is None

    @pytest.mark.asyncio
    async def test_tool_calls_extracted(self, ollama_driver):
        tc = make_non_streaming_tool_call(tc_id="call_1", name="get_weather",
                                          arguments='{"location": "Paris"}')
        resp = make_non_streaming_response(content="", finish_reason="tool_calls",
                                           tool_calls=[tc])
        chunks = await self._run(ollama_driver, self._make_cfg(), resp)
        assert chunks[0].tool_calls is not None
        assert len(chunks[0].tool_calls) == 1
        t = chunks[0].tool_calls[0]
        assert t.id == "call_1"
        assert t.function_name == "get_weather"
        assert t.arguments == {"location": "Paris"}

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_extracted(self, ollama_driver):
        tc1 = make_non_streaming_tool_call(tc_id="c0", name="fn0", arguments='{"k": "v"}')
        tc2 = make_non_streaming_tool_call(tc_id="c1", name="fn1", arguments='{}')
        resp = make_non_streaming_response(content="", finish_reason="tool_calls",
                                           tool_calls=[tc1, tc2])
        chunks = await self._run(ollama_driver, self._make_cfg(), resp)
        names = {t.function_name for t in chunks[0].tool_calls}
        assert names == {"fn0", "fn1"}

    @pytest.mark.asyncio
    async def test_invalid_json_tool_args_use_raw(self, ollama_driver):
        tc = make_non_streaming_tool_call(tc_id="c", name="fn", arguments="NOT-JSON")
        resp = make_non_streaming_response(content="", finish_reason="tool_calls",
                                           tool_calls=[tc])
        chunks = await self._run(ollama_driver, self._make_cfg(), resp)
        assert "_raw" in chunks[0].tool_calls[0].arguments

    @pytest.mark.asyncio
    async def test_no_tool_calls_when_none(self, ollama_driver):
        resp = make_non_streaming_response(content="plain answer", tool_calls=None)
        chunks = await self._run(ollama_driver, self._make_cfg(), resp)
        assert chunks[0].tool_calls is None

    @pytest.mark.asyncio
    async def test_thinking_blocks_extracted(self, ollama_driver):
        resp = make_non_streaming_response(content="<think>reasoning</think>Answer")
        chunks = await self._run(ollama_driver, self._make_cfg(), resp)
        assert chunks[0].delta_thought == "reasoning"
        assert chunks[0].delta_content == "Answer"

    @pytest.mark.asyncio
    async def test_stream_false_passed_to_litellm(self, ollama_driver):
        """stream=False in LLMConfig must be forwarded to acompletion."""
        captured = {}
        async def fake(**kwargs):
            captured.update(kwargs)
            return make_non_streaming_response(content="ok")
        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="hi")]
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for _ in ollama_driver.chat_stream(msgs, config=self._make_cfg()):
                pass
        assert captured["stream"] is False

    @pytest.mark.asyncio
    async def test_temperature_zero_not_overridden(self, ollama_driver):
        """temperature=0.0 (falsy) must NOT fall back to config default."""
        captured = {}
        async def fake(**kwargs):
            captured.update(kwargs)
            return make_non_streaming_response(content="ok")
        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="hi")]
        cfg = LLMConfig(temperature=0.0, max_tokens=64, stream=False)
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for _ in ollama_driver.chat_stream(msgs, config=cfg):
                pass
        assert captured["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_max_tokens_zero_not_overridden(self, ollama_driver):
        """max_tokens=0 must not silently revert to config default."""
        captured = {}
        async def fake(**kwargs):
            captured.update(kwargs)
            return make_non_streaming_response(content="ok")
        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="hi")]
        cfg = LLMConfig(temperature=0.7, max_tokens=0, stream=False)
        with patch("k_ai.llm_core.litellm.acompletion", new=fake):
            async for _ in ollama_driver.chat_stream(msgs, config=cfg):
                pass
        assert captured["max_tokens"] == 0


# ===========================================================================
# chat_stream — exception mapping
# ===========================================================================

class TestChatStreamExceptions:
    async def _run_with_exc(self, driver, exc_instance):
        from k_ai.models import Message, MessageRole
        msgs = [Message(role=MessageRole.USER, content="q")]
        with patch("k_ai.llm_core.litellm.acompletion", new=AsyncMock(side_effect=exc_instance)):
            async for _ in driver.chat_stream(msgs):
                pass

    @pytest.mark.asyncio
    async def test_auth_error_mapped(self, ollama_driver):
        exc = make_litellm_exc(litellm.AuthenticationError)
        with pytest.raises(ProviderAuthenticationError):
            await self._run_with_exc(ollama_driver, exc)

    @pytest.mark.asyncio
    async def test_rate_limit_mapped(self, ollama_driver):
        exc = make_litellm_exc(litellm.RateLimitError)
        with pytest.raises(ProviderRateLimitError):
            await self._run_with_exc(ollama_driver, exc)

    @pytest.mark.asyncio
    async def test_context_window_mapped(self, ollama_driver):
        exc = make_litellm_exc(litellm.ContextWindowExceededError)
        with pytest.raises(ContextLengthExceededError):
            await self._run_with_exc(ollama_driver, exc)

    @pytest.mark.asyncio
    async def test_service_unavailable_mapped(self, ollama_driver):
        exc = make_litellm_exc(litellm.ServiceUnavailableError)
        with pytest.raises(ServiceUnavailableError):
            await self._run_with_exc(ollama_driver, exc)

    @pytest.mark.asyncio
    async def test_timeout_mapped(self, ollama_driver):
        exc = make_litellm_exc(litellm.Timeout)
        with pytest.raises(ProviderTimeoutError):
            await self._run_with_exc(ollama_driver, exc)

    @pytest.mark.asyncio
    async def test_generic_exception_wrapped_in_llm_error(self, ollama_driver):
        with pytest.raises(LLMError):
            await self._run_with_exc(ollama_driver, RuntimeError("boom"))


# ===========================================================================
# list_models — three strategies
# ===========================================================================

class TestListModels:
    @pytest.mark.asyncio
    async def test_strategy1_live_endpoint(self, ollama_driver):
        """Strategy 1: GET base_url/models via httpx."""
        import httpx
        fake_resp_data = {"data": [{"id": "phi4-mini:latest"}, {"id": "llama3:latest"}]}
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = fake_resp_data

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            models = await ollama_driver.list_models()

        assert "phi4-mini:latest" in models
        assert "llama3:latest" in models
        assert models == sorted(models)

    @pytest.mark.asyncio
    async def test_strategy1_empty_data_falls_back(self, ollama_driver):
        """Strategy 1 returns [] → falls to strategy 2/3."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": []}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            models = await ollama_driver.list_models()

        # Strategy 3 fallback: at least the current model
        assert ollama_driver.model_name in models

    @pytest.mark.asyncio
    async def test_strategy1_network_failure_falls_back(self, ollama_driver):
        """Strategy 1 raises → Strategy 2/3 used."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            models = await ollama_driver.list_models()

        assert isinstance(models, list)
        assert len(models) >= 1

    @pytest.mark.asyncio
    async def test_strategy2_litellm_model_cost_anthropic(self, cm):
        """Strategy 2 via model_cost for anthropic — no base_url, no key needed."""
        with patch("k_ai.llm_core.resolve_secret", return_value=("sk", "mock")):
            driver = LiteLLMDriver(cm, provider_name="anthropic")
        models = await driver.list_models()
        assert len(models) > 5
        assert any("claude" in m for m in models)
        assert models == sorted(models)

    @pytest.mark.asyncio
    async def test_strategy2_litellm_model_cost_openai(self, cm):
        """Strategy 2 via model_cost for openai."""
        with patch("k_ai.llm_core.resolve_secret", return_value=("sk", "mock")):
            driver = LiteLLMDriver(cm, provider_name="openai")
        models = await driver.list_models()
        assert len(models) > 10
        assert any("gpt" in m for m in models)

    @pytest.mark.asyncio
    async def test_strategy3_fallback_returns_current_model(self, cm):
        """Strategy 3: when model_cost lookup fails, return [current_model]."""
        with patch("k_ai.llm_core.resolve_secret", return_value=("sk", "mock")):
            driver = LiteLLMDriver(cm, provider_name="anthropic")

        with patch("k_ai.llm_core.litellm.model_cost", {}):
            models = await driver.list_models()

        assert models == [driver.model_name]

    @pytest.mark.asyncio
    async def test_strategy2_litellm_model_cost_xai(self, cm):
        """Strategy 2 via model_cost for xAI — native routing, no base_url."""
        with patch("k_ai.llm_core.resolve_secret", return_value=("xai-test", "mock")):
            driver = LiteLLMDriver(cm, provider_name="xai")
        models = await driver.list_models()
        assert len(models) > 10
        assert any("grok" in m for m in models)
        assert models == sorted(models)

    @pytest.mark.asyncio
    async def test_strategy2_works_for_all_api_key_providers(self, cm):
        """Every api_key provider in config must return models via Strategy 2."""
        providers_with_models = {}
        for prov in cm.list_providers():
            with patch("k_ai.llm_core.resolve_secret", return_value=("sk-fake", "mock")):
                try:
                    driver = LiteLLMDriver(cm, provider_name=prov)
                except Exception:
                    continue
            models = await driver.list_models()
            providers_with_models[prov] = len(models)
        # Every provider should have at least its own default model
        for prov, count in providers_with_models.items():
            assert count >= 1, f"{prov} returned 0 models"

    @pytest.mark.asyncio
    async def test_strategy1_auth_header_included_for_api_key_provider(self, cm):
        """When api_key is set, Authorization: Bearer is sent in Strategy 1."""
        with patch("k_ai.llm_core.resolve_secret", return_value=("sk-123", "mock")):
            driver = LiteLLMDriver(cm, provider_name="mistral")

        captured_headers = {}
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "mistral-large-latest"}]}

        async def fake_get(url, headers=None):
            captured_headers.update(headers or {})
            return mock_resp

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = fake_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_cls.return_value = mock_client

            await driver.list_models()

        assert "Authorization" in captured_headers
        assert "sk-123" in captured_headers["Authorization"]


# ===========================================================================
# get_provider factory
# ===========================================================================

class TestGetProvider:
    def test_returns_litellm_driver(self, cm):
        driver = get_provider(cm)
        assert isinstance(driver, LiteLLMDriver)

    def test_provider_override(self, cm):
        driver = get_provider(cm, provider="ollama")
        assert driver.provider_name == "ollama"

    def test_model_override(self, cm):
        driver = get_provider(cm, provider="ollama", model="custom:v1")
        assert driver.model_name == "custom:v1"

    def test_reraises_kai_error_unmodified(self, cm):
        with pytest.raises(ConfigurationError):
            get_provider(cm, provider="does_not_exist")

    def test_generic_exception_wrapped_in_llm_error(self, cm):
        with patch("k_ai.llm_core.LiteLLMDriver.__init__", side_effect=RuntimeError("boom")):
            with pytest.raises(LLMError):
                get_provider(cm)
