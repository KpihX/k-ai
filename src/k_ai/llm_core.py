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
        self.provider_name = provider_name or self.config_manager.get("provider")
        
        self.provider_config, self.auth_mode = self.config_manager.get_provider_config_with_auth_mode(self.provider_name, auth_mode=auth_mode)
        if not self.provider_config:
            raise ConfigurationError(f"Provider '{self.provider_name}' not found in configuration for auth mode '{auth_mode}'.")

        self.model_name = model_name or self.provider_config.get("default_model")
        if not self.model_name:
            raise ConfigurationError(f"No default model specified for provider '{self.provider_name}'.")

    @abstractmethod
    async def chat_stream(self, history: List[Message], config: Optional[LLMConfig] = None, tools: Optional[List[Dict]] = None) -> AsyncGenerator[CompletionChunk, None]:
        pass
    
    @abstractmethod
    async def list_models(self) -> List[str]:
        pass


class LiteLLMDriver(LLMProvider):
    """
    Universal Provider using the LiteLLM library.
    """
    def __init__(self, config_manager: ConfigManager, provider_name: Optional[str] = None, model_name: Optional[str] = None, auth_mode: Optional[str] = None):
        super().__init__(config_manager, provider_name, model_name, auth_mode)
        
        self.api_key: Optional[str] = None

        if self.auth_mode == "api_key":
            api_key_var = self.provider_config.get("api_key_env_var")
            if api_key_var:
                self.api_key = os.getenv(api_key_var)
            elif "api_key_value" in self.provider_config:
                 self.api_key = self.provider_config.get("api_key_value")
                 
            if not self.api_key and api_key_var:
                 raise ProviderAuthenticationError(f"API Key not found for provider '{self.provider_name}'. Expected env var: {api_key_var}")
        
        elif self.auth_mode == "oauth":
            from . import auth
            oauth_provider_name = self.provider_config.get("oauth_provider_name")
            token_path = self.provider_config.get("token_path")
            scopes = self.provider_config.get("oauth_scopes")
            if not all([oauth_provider_name, token_path, scopes]):
                raise ConfigurationError(f"OAuth provider '{self.provider_name}' is missing required settings.")
            
            if oauth_provider_name not in auth.TOKEN_LOADERS:
                raise ConfigurationError(f"No token loader for '{oauth_provider_name}' in auth.py.")
            
            self.api_key = auth.TOKEN_LOADERS[oauth_provider_name](token_path, scopes)

    async def chat_stream(self, history: List[Message], config: Optional[LLMConfig] = None, tools: Optional[List[Dict]] = None) -> AsyncGenerator[CompletionChunk, None]:
        litellm_messages = [m.model_dump(exclude_none=True) for m in history]
        
        # The model string is now the provider name + model name
        model_str = f"{self.provider_name}/{self.model_name}"

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

    async def list_models(self) -> List[str]:
        """
        List available models for the current provider using litellm.
        """
        try:
            loop = asyncio.get_running_loop()
            all_models = await loop.run_in_executor(None, litellm.get_model_list)
            
            provider_prefix = f"{self.provider_name}/"
            provider_models = [m for m in all_models if m.startswith(provider_prefix)]
            return sorted(provider_models)
        except Exception as e:
            raise LLMError(f"Failed to list models for provider '{self.provider_name}' via litellm: {e}") from e


def get_provider(config_manager: "ConfigManager", provider: Optional[str] = None, model: Optional[str] = None, auth_mode: Optional[str] = None) -> LLMProvider:
    """Factory to instantiate the LiteLLM driver."""
    try:
        return LiteLLMDriver(config_manager=config_manager, provider_name=provider, model_name=model, auth_mode=auth_mode)
    except Exception as e:
        raise LLMError(f"Failed to get provider: {e}") from e

