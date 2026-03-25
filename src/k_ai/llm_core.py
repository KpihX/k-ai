# src/k_ai/llm_core.py
"""
Core LLM interaction logic, powered by LiteLLM.
"""
import os
import litellm
import asyncio
from typing import AsyncGenerator, List, Dict, Optional, Any

# Assuming base, models, and exceptions will be created in their own files
from .models import Message, CompletionChunk, ToolCall, LLMConfig, TokenUsage
from .exceptions import (
    LLMError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ContextLengthExceededError,
    ServiceUnavailableError,
    ConfigurationError,
)
# We will create a simple base class for now
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract Base Class for all LLM providers."""
    def __init__(self, config_manager, provider_name: Optional[str] = None, model_name: Optional[str] = None):
        self.config_manager = config_manager
        self.provider_name = provider_name or self.config_manager.get("provider")
        
        self.provider_config = self.config_manager.get_provider_config(self.provider_name)
        if not self.provider_config:
            raise ConfigurationError(f"Provider '{self.provider_name}' not found in configuration.")

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
    def __init__(self, config_manager, provider_name: Optional[str] = None, model_name: Optional[str] = None):
        super().__init__(config_manager, provider_name, model_name)
        
        self.api_key: Optional[str] = None
        auth_mode = self.provider_config.get("auth_mode", "api_key")

        if auth_mode == "api_key":
            api_key_var = self.provider_config.get("api_key_env_var")
            if api_key_var:
                self.api_key = os.getenv(api_key_var)
            elif "api_key_value" in self.provider_config:
                 self.api_key = self.provider_config.get("api_key_value")
                 
            if not self.api_key and api_key_var:
                 raise ValueError(f"API Key not found for provider '{self.provider_name}'. Expected env var: {api_key_var}")
        # OAuth logic will be added later
        
    async def chat_stream(self, history: List[Message], config: Optional[LLMConfig] = None, tools: Optional[List[Dict]] = None) -> AsyncGenerator[CompletionChunk, None]:
        # Simplified for now
        litellm_messages = [m.model_dump(exclude_none=True) for m in history]
        
        # Construct model string for litellm
        # For providers like openai, groq, mistral, it's just the model name.
        # For others, it might be `provider/model_name`. LiteLLM handles this well.
        model_str = f"{self.provider_name}/{self.model_name}" if self.provider_name not in ["openai", "mistral", "groq"] else self.model_name

        try:
            response = await litellm.acompletion(
                model=model_str,
                messages=litellm_messages,
                stream=True,
                api_key=self.api_key,
                base_url=self.provider_config.get("base_url"),
                temperature=self.config_manager.get("temperature"),
                max_tokens=self.config_manager.get("max_tokens")
            )
            async for chunk in response:
                yield CompletionChunk(delta_content=chunk.choices[0].delta.content or "")
        except Exception as e:
            raise LLMError(f"LiteLLM API call failed: {e}") from e

    async def list_models(self) -> List[str]:
        # Simplified
        return [self.model_name]


def get_provider(config_manager: "ConfigManager", provider: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
    """
    Factory to instantiate the LiteLLM driver.
    """
    try:
        return LiteLLMDriver(
            config_manager=config_manager,
            provider_name=provider,
            model_name=model,
        )
    except (ValueError, ConfigurationError) as e:
        raise ConfigurationError(str(e)) from e
    except Exception as e:
        raise LLMError(f"Unexpected error initializing provider: {e}") from e
