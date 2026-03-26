# test/conftest.py
"""
Shared pytest fixtures and helpers for the k-ai test suite.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from k_ai import ConfigManager
from k_ai.llm_core import LiteLLMDriver


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cm():
    """Default ConfigManager loaded from built-in defaults."""
    return ConfigManager()


@pytest.fixture
def ollama_driver(cm):
    """LiteLLMDriver for ollama (no_auth — no key needed)."""
    return LiteLLMDriver(cm, provider_name="ollama")


@pytest.fixture
def api_key_driver(cm):
    """LiteLLMDriver for a cloud provider, with a mocked API key."""
    with patch("k_ai.llm_core.resolve_secret", return_value=("sk-test", "mock")):
        driver = LiteLLMDriver(cm, provider_name="anthropic")
    return driver


# ---------------------------------------------------------------------------
# Streaming chunk builder
# ---------------------------------------------------------------------------

def make_chunk(
    content="",
    tool_call_deltas=None,
    finish_reason=None,
    usage=None,
    has_choices=True,
):
    """
    Build a SimpleNamespace that looks like a LiteLLM streaming chunk.

    Args:
        content:           text delta for this chunk
        tool_call_deltas:  list of tool-call delta SimpleNamespaces (or None)
        finish_reason:     "stop", "tool_calls", "length", or None
        usage:             SimpleNamespace with prompt/completion/total fields
        has_choices:       set False to simulate a usage-only chunk (no choices)
    """
    if not has_choices:
        return SimpleNamespace(choices=[], usage=usage)

    delta = SimpleNamespace(content=content, tool_calls=tool_call_deltas)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def make_usage(prompt=10, completion=5, total=15):
    """Build a fake usage SimpleNamespace."""
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def make_tool_delta(index=0, tc_id="call_1", name="get_weather", arguments='{"location":'):
    """Build a fake tool-call delta SimpleNamespace (partial streaming)."""
    return SimpleNamespace(
        index=index,
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


async def stream_chunks(*chunks):
    """Async generator that yields pre-built chunks — use as a fake acompletion response."""
    for chunk in chunks:
        yield chunk


def make_non_streaming_response(
    content="",
    finish_reason="stop",
    usage=None,
    tool_calls=None,
):
    """
    Build a mock non-streaming ModelResponse (returned by litellm.acompletion when
    stream=False).  The shape mirrors what LiteLLM actually returns.

    Args:
        content:      Full text of the model response.
        finish_reason: "stop", "length", "tool_calls", etc.
        usage:        SimpleNamespace from make_usage(), or None.
        tool_calls:   List of SimpleNamespaces from make_non_streaming_tool_call(), or None.
    """
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def make_non_streaming_tool_call(tc_id="call_1", name="get_weather", arguments='{"location": "Paris"}'):
    """Build a fake complete tool call as returned in a non-streaming ModelResponse."""
    return SimpleNamespace(
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


# ---------------------------------------------------------------------------
# LiteLLM exception factory
# ---------------------------------------------------------------------------

def make_litellm_exc(cls, msg="test error"):
    """
    Instantiate a litellm exception, handling varying constructor signatures
    across litellm versions (some require llm_provider + model kwargs).
    """
    for args, kwargs in [
        ((msg,), {}),
        ((), {"message": msg, "llm_provider": "test", "model": "test-model"}),
        ((), {"message": msg}),
    ]:
        try:
            return cls(*args, **kwargs)
        except (TypeError, Exception):
            continue
    # Last resort: forge the __class__ (works for pure-Python exception hierarchies)
    e = Exception(msg)
    e.__class__ = cls
    return e
