# src/k_ai/llm_core.py
"""
Core LLM interaction logic, powered by LiteLLM.
"""
import os
import json
import litellm
import asyncio
from typing import AsyncGenerator, List, Dict, Optional, Any
from abc import ABC, abstractmethod

from .models import Message, CompletionChunk, ToolCall, LLMConfig, TokenUsage
from .exceptions import (
    LLMError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ContextLengthExceededError,
    ServiceUnavailableError,
    ConfigurationError,
)
from .config import ConfigManager
from .utils import ThinkingParser


class LLMProvider(ABC):
    """Abstract Base Class for all LLM providers."""
    def __init__(self, config_manager: ConfigManager, provider_name: Optional[str] = None, model_name: Optional[str] = None, auth_mode: Optional[str] = None):
        self.config_manager = config_manager

        # Priority: explicit > config > default
        self.provider_name = provider_name or self.config_manager.get("provider")

        self.provider_config = self.config_manager.get_provider_config(self.provider_name, auth_mode=auth_mode)
        if not self.provider_config:
            raise ConfigurationError(f"Provider '{self.provider_name}' not found in configuration for auth mode '{auth_mode}'.")

        self.model_name = model_name or self.provider_config.get("default_model")
        if not self.model_name:
            raise ConfigurationError(f"No default model specified for provider '{self.provider_name}'.")

    @abstractmethod
    async def chat_stream(self, history: List[Message], config: Optional[LLMConfig] = None, tools: Optional[List[Dict]] = None) -> AsyncGenerator[CompletionChunk, None]:
        pass


class LiteLLMDriver(LLMProvider):
    """
    Universal Provider using the LiteLLM library.
    """
    def __init__(self, config_manager: ConfigManager, provider_name: Optional[str] = None, model_name: Optional[str] = None, auth_mode: Optional[str] = None):
        super().__init__(config_manager, provider_name, model_name, auth_mode)

        self.api_key: Optional[str] = None
        current_auth_mode = self.provider_config.get("auth_mode", "no_auth")

        if current_auth_mode == "api_key":
            api_key_var = self.provider_config.get("api_key_env_var")
            if api_key_var:
                self.api_key = os.getenv(api_key_var)
            elif "api_key_value" in self.provider_config:
                 self.api_key = self.provider_config.get("api_key_value")

            if not self.api_key and api_key_var:
                 raise ProviderAuthenticationError(f"API Key not found for provider '{self.provider_name}'. Expected env var: {api_key_var}")

        elif current_auth_mode == "oauth":
            # This logic will be triggered for providers like 'gemini' (in oauth mode)
            from . import auth
            oauth_provider_name = self.provider_config.get("oauth_provider_name")
            token_path = self.provider_config.get("token_path")
            scopes = self.provider_config.get("oauth_scopes")
            if not all([oauth_provider_name, token_path, scopes]):
                raise ConfigurationError(f"OAuth provider '{self.provider_name}' is missing required settings.")

            if oauth_provider_name not in auth.TOKEN_LOADERS:
                raise ConfigurationError(f"No token loader for '{oauth_provider_name}' in auth.py.")

            self.api_key = auth.TOKEN_LOADERS[oauth_provider_name](token_path, scopes)

    def _get_model_str(self) -> str:
        """Constructs the model string for LiteLLM."""
        prefix = self.provider_config.get("litellm_prefix")
        if prefix:
            return f"{prefix}/{self.model_name}"
        return self.model_name

    async def chat_stream(self, history: List[Message], config: Optional[LLMConfig] = None, tools: Optional[List[Dict]] = None) -> AsyncGenerator[CompletionChunk, None]:
        litellm_messages = [m.model_dump(exclude_none=True) for m in history]
        model_str = self._get_model_str()

        params = {
            "model": model_str,
            "messages": litellm_messages,
            "stream": True,
            "api_key": self.api_key,
            "base_url": self.provider_config.get("base_url"),
            "temperature": self.config_manager.get("temperature"),
            "max_tokens": self.config_manager.get("max_tokens"),
            "tools": tools,
        }

        try:
            response = await litellm.acompletion(**params)
            parser = ThinkingParser()

            async for chunk in response:
                usage = None
                if hasattr(chunk, "usage") and chunk.usage:
                     usage = TokenUsage(prompt_tokens=chunk.usage.prompt_tokens, completion_tokens=chunk.usage.completion_tokens, total_tokens=chunk.usage.total_tokens)

                if not chunk.choices:
                     if usage: yield CompletionChunk(usage=usage)
                     continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                raw_content = delta.content or ""
                thought, content = parser.parse(raw_content)

                tool_calls = []
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        tool_calls.append(ToolCall(id=tc.id or "", function_name=tc.function.name or "", arguments=json.loads(tc.function.arguments or "{}")))

                yield CompletionChunk(delta_content=content, delta_thought=thought, tool_calls=tool_calls, finish_reason=finish_reason, usage=usage)

        except litellm.AuthenticationError as e:
            raise ProviderAuthenticationError(f"LiteLLM AuthenticationError: {e}")
        except litellm.RateLimitError as e:
            raise ProviderRateLimitError(f"LiteLLM RateLimitError: {e}")
        except litellm.ContextWindowExceededError as e:
            raise ContextLengthExceededError(f"LiteLLM ContextWindowExceededError: {e}")
        except Exception as e:
            raise LLMError(f"LiteLLM unhandled error: {e}") from e

def get_provider(config_manager: "ConfigManager", provider: Optional[str] = None, model: Optional[str] = None, auth_mode: Optional[str] = None) -> LLMProvider:
    """Factory to instantiate the LiteLLM driver."""
    try:
        return LiteLLMDriver(config_manager=config_manager, provider_name=provider, model_name=model, auth_mode=auth_mode)
    except Exception as e:
        raise LLMError(f"Failed to get provider: {e}") from e

