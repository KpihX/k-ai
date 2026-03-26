# src/k_ai/__init__.py
"""
k-ai: The Unified LLM CLI and Library.

Library usage examples::

    from k_ai import ConfigManager, ChatSession, get_provider, ToolCall

    # --- Option 1: Defaults only ---
    cm = ConfigManager()

    # --- Option 2: Partial override file ---
    cm = ConfigManager(override_path="~/my_config.yaml")

    # --- Option 3: Inline params ---
    cm = ConfigManager(provider="openai", temperature=0.2, max_tokens=2048)

    # --- Option 4: Both ---
    cm = ConfigManager(override_path="base.yaml", model="gpt-4o")

    # --- Get the default config template as YAML text ---
    print(ConfigManager.get_default_yaml())

    # --- Interactive chat session (CLI) ---
    import asyncio
    session = ChatSession(cm)
    asyncio.run(session.start())

    # --- Single programmatic call (no tools) ---
    response = asyncio.run(session.send("What is 2 + 2?"))

    # --- Agentic call with tools (full loop handled automatically) ---
    weather_tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }

    async def my_executor(tc: ToolCall) -> str:
        if tc.function_name == "get_weather":
            return f"22°C and sunny in {tc.arguments['location']}"
        raise ValueError(f"Unknown tool: {tc.function_name}")

    result = asyncio.run(
        session.send_with_tools("What's the weather in Paris?", [weather_tool], my_executor)
    )

    # --- Low-level provider access ---
    provider = get_provider(cm)
    async for chunk in provider.chat_stream(messages):
        print(chunk.delta_content, end="")
"""

from .config import ConfigManager
from .secrets import resolve_secret
from .exceptions import (
    ConfigurationError,
    ContextLengthExceededError,
    KAIError,
    LLMError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ServiceUnavailableError,
)
from .llm_core import LiteLLMDriver, LLMProvider, get_provider
from .models import (
    CompletionChunk,
    LLMConfig,
    Message,
    MessageRole,
    TokenUsage,
    ToolCall,
)
from .session import ChatSession
from .memory import MemoryStore
from .session_store import SessionStore

__version__ = "0.2.0"

__all__ = [
    # Config
    "ConfigManager",
    # Secrets
    "resolve_secret",
    # Session
    "ChatSession",
    # Persistence
    "SessionStore",
    "MemoryStore",
    # LLM core
    "get_provider",
    "LLMProvider",
    "LiteLLMDriver",
    # Models
    "Message",
    "MessageRole",
    "CompletionChunk",
    "TokenUsage",
    "LLMConfig",
    "ToolCall",
    # Exceptions
    "KAIError",
    "LLMError",
    "ConfigurationError",
    "ProviderAuthenticationError",
    "ProviderRateLimitError",
    "ContextLengthExceededError",
    "ServiceUnavailableError",
    "ProviderTimeoutError",
]
