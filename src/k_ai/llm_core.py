# src/k_ai/llm_core.py
"""
Core LLM interaction logic, powered by LiteLLM.
"""
import json
import asyncio
import litellm
from typing import AsyncGenerator, List, Dict, Optional, Any
from abc import ABC, abstractmethod

from .models import Message, CompletionChunk, LLMConfig, ToolCall, TokenUsage
from .exceptions import (
    KAIError,
    LLMError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ContextLengthExceededError,
    ServiceUnavailableError,
    ConfigurationError,
)
from .config import ConfigManager
from .secrets import resolve_secret
from .utils import ThinkingParser


class LLMProvider(ABC):
    """
    Abstract Base Class for all LLM providers.

    Subclasses must implement `chat_stream` (streaming token generator) and
    `list_models` (enumerate models available to this provider/key combination).
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        auth_mode: Optional[str] = None,
    ):
        """
        Locate the provider block in config and resolve the active model name.

        Args:
            config_manager: The active ConfigManager instance.
            provider_name:  Override the provider key (defaults to config "provider").
            model_name:     Override the model (defaults to provider's default_model).
            auth_mode:      Force a specific auth section: "no_auth", "api_key", or "oauth".
        """
        self.config_manager = config_manager
        self.provider_name = provider_name or self.config_manager.get("provider")

        self.provider_config, self.auth_mode = (
            self.config_manager.get_provider_config_with_auth_mode(
                self.provider_name, auth_mode=auth_mode
            )
        )
        if not self.provider_config:
            raise ConfigurationError(
                f"Provider '{self.provider_name}' not found in configuration "
                f"(auth_mode='{auth_mode}')."
            )

        self.model_name = model_name or self.provider_config.get("default_model")
        if not self.model_name:
            raise ConfigurationError(
                f"No default model specified for provider '{self.provider_name}'."
            )

    @abstractmethod
    async def chat_stream(
        self,
        history: List[Message],
        config: Optional[LLMConfig] = None,
        tools: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[CompletionChunk, None]:
        """
        Stream a completion for the given message history.

        Args:
            history: Full conversation history (system message already prepended
                     by the caller when needed).
            config:  Per-call overrides for temperature / max_tokens / stream.
                     When None the ConfigManager values are used.
            tools:   Optional list of OpenAI-style tool definitions.  When
                     provided the model may emit tool_call chunks.
        """

    @abstractmethod
    async def list_models(self) -> List[str]:
        """Return a sorted list of model identifiers available for this provider."""


class LiteLLMDriver(LLMProvider):
    """
    Universal LLM driver backed by the LiteLLM library.

    Supports three authentication modes (resolved automatically from config):
      - no_auth  — local endpoints like Ollama (config section "no_auth").
      - api_key  — cloud providers with an API key (config section "api_key").
      - oauth    — browser-based OAuth flow (config section "oauth").
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        auth_mode: Optional[str] = None,
    ):
        """
        Resolve auth credentials after the base class locates the provider block.

        Raises:
            ProviderAuthenticationError: When an expected API key env var is unset.
            ConfigurationError:          When required OAuth settings are missing.
        """
        super().__init__(config_manager, provider_name, model_name, auth_mode)

        self.api_key: Optional[str] = None

        if self.auth_mode == "api_key":
            api_key_var = self.provider_config.get("api_key_env_var")
            if api_key_var:
                # Use 3-tier resolution: .env → os.environ → zsh -l -c
                self.api_key, _ = resolve_secret(api_key_var)
            elif "api_key_value" in self.provider_config:
                # Inline value in config (not recommended for production)
                self.api_key = self.provider_config.get("api_key_value")

            if not self.api_key and api_key_var:
                raise ProviderAuthenticationError(
                    f"API key not found for provider '{self.provider_name}'. "
                    f"Tried .env, os.environ, and zsh login shell for: {api_key_var}"
                )

        elif self.auth_mode == "oauth":
            from . import auth  # lazy: OAuth deps may be heavy/optional
            oauth_provider_name = self.provider_config.get("oauth_provider_name")
            token_path = self.provider_config.get("token_path")
            scopes = self.provider_config.get("oauth_scopes")
            if not all([oauth_provider_name, token_path, scopes]):
                raise ConfigurationError(
                    f"OAuth provider '{self.provider_name}' is missing required settings "
                    "(oauth_provider_name, token_path, oauth_scopes)."
                )

            if oauth_provider_name not in auth.TOKEN_LOADERS:
                raise ConfigurationError(
                    f"No token loader registered for '{oauth_provider_name}' in auth.py."
                )

            self.api_key = auth.TOKEN_LOADERS[oauth_provider_name](token_path, scopes)

    async def chat_stream(
        self,
        history: List[Message],
        config: Optional[LLMConfig] = None,
        tools: Optional[List[Dict]] = None,
    ) -> AsyncGenerator[CompletionChunk, None]:
        """
        Yield CompletionChunk objects as the model generates its response.

        Always returns an AsyncGenerator regardless of stream mode — callers
        iterate identically whether streaming (many small chunks) or not
        (one complete chunk).  The internal LiteLLM call is streaming or
        blocking according to the resolved ``stream`` setting.

        Resolution order for every param (last wins):
          1. Built-in default_config.yaml
          2. User override file
          3. Inline ConfigManager kwargs
          4. Per-call LLMConfig (this argument)

        Args:
            history: Full message list to send (system prompt already prepended).
            config:  Per-call overrides for temperature / max_tokens / stream.
                     ``None`` → all values come from ConfigManager.
            tools:   OpenAI-style tool definitions.  Pass None for plain chat.
        """
        # Use to_litellm() for correct OpenAI wire format (especially for tool
        # messages and assistant messages with tool_calls).
        litellm_messages = [m.to_litellm() for m in history]

        # When a base_url is set (local/proxied OpenAI-compatible endpoint),
        # LiteLLM must route via "openai/" prefix to hit /v1/chat/completions.
        # Without it, "ollama/..." would be sent to the native /api/generate endpoint.
        base_url = self.provider_config.get("base_url")
        if base_url:
            model_str = f"openai/{self.model_name}"
        else:
            model_str = f"{self.provider_name}/{self.model_name}"

        # Resolve every param from config/LLMConfig.
        # Explicit None-check required: 0.0, 0, and False are valid overrides
        # that must NOT be shadowed by the global default via a falsy `or`.
        if config is not None:
            cfg_temperature = config.temperature
            cfg_max_tokens  = config.max_tokens
            cfg_stream      = config.stream
        else:
            cfg_temperature = self.config_manager.get("temperature")
            cfg_max_tokens  = self.config_manager.get("max_tokens")
            cfg_stream      = self.config_manager.get("stream", True)

        # Build the params dict; only include optional keys when set to avoid
        # surprising providers with strict parameter validation.
        params: Dict[str, Any] = {
            "model":       model_str,
            "messages":    litellm_messages,
            "stream":      cfg_stream,
            "temperature": cfg_temperature,
            "max_tokens":  cfg_max_tokens,
        }
        if self.api_key is not None:
            params["api_key"] = self.api_key
        elif base_url is not None:
            # OpenAI-compatible endpoints (ollama, etc.) routed via openai/ prefix
            # need a non-None api_key to satisfy the SDK client constructor,
            # even though the server ignores it.
            params["api_key"] = "no-key-required"
        if base_url is not None:
            params["base_url"] = base_url
        if tools:
            params["tools"] = tools

        try:
            response = await litellm.acompletion(**params)
            parser = ThinkingParser()

            if cfg_stream:
                # -------------------------------------------------------
                # Streaming path — iterate chunk by chunk.
                # Accumulate partial tool-call argument strings across chunks.
                # LiteLLM sends the arguments JSON split over multiple deltas.
                # -------------------------------------------------------
                _tool_call_accum: Dict[int, Dict[str, Any]] = {}

                async for chunk in response:
                    usage = None
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = TokenUsage(
                            prompt_tokens=chunk.usage.prompt_tokens or 0,
                            completion_tokens=chunk.usage.completion_tokens or 0,
                            total_tokens=chunk.usage.total_tokens or 0,
                        )

                    if not chunk.choices:
                        if usage:
                            yield CompletionChunk(usage=usage)
                        continue

                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason

                    # --- Text / thinking content ---
                    raw_content = delta.content or ""
                    thought, content = parser.parse(raw_content)

                    # --- Tool call accumulation ---
                    completed_tool_calls: List[ToolCall] = []
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in _tool_call_accum:
                                _tool_call_accum[idx] = {
                                    "id":        tc_delta.id or "",
                                    "name":      (tc_delta.function.name or "") if tc_delta.function else "",
                                    "arguments": "",
                                }
                            acc = _tool_call_accum[idx]
                            if tc_delta.id:
                                acc["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    acc["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    acc["arguments"] += tc_delta.function.arguments

                    if finish_reason == "tool_calls":
                        for acc in _tool_call_accum.values():
                            try:
                                args = json.loads(acc["arguments"] or "{}")
                            except json.JSONDecodeError:
                                args = {"_raw": acc["arguments"]}
                            completed_tool_calls.append(
                                ToolCall(id=acc["id"], function_name=acc["name"], arguments=args)
                            )
                        _tool_call_accum.clear()

                    yield CompletionChunk(
                        delta_content=content,
                        delta_thought=thought,
                        tool_calls=completed_tool_calls if completed_tool_calls else None,
                        finish_reason=finish_reason,
                        usage=usage,
                    )

            else:
                # -------------------------------------------------------
                # Non-streaming path — single ModelResponse, yield one chunk.
                # Callers iterate identically: `async for chunk in chat_stream()`
                # -------------------------------------------------------
                choice  = response.choices[0]
                message = choice.message

                usage = None
                if response.usage:
                    usage = TokenUsage(
                        prompt_tokens=response.usage.prompt_tokens or 0,
                        completion_tokens=response.usage.completion_tokens or 0,
                        total_tokens=response.usage.total_tokens or 0,
                    )

                raw_content = message.content or ""
                thought, content = parser.parse(raw_content)

                tool_calls: List[ToolCall] = []
                if message.tool_calls:
                    for tc in message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {"_raw": tc.function.arguments}
                        tool_calls.append(
                            ToolCall(id=tc.id, function_name=tc.function.name, arguments=args)
                        )

                yield CompletionChunk(
                    delta_content=content,
                    delta_thought=thought,
                    tool_calls=tool_calls if tool_calls else None,
                    finish_reason=choice.finish_reason,
                    usage=usage,
                )

        except litellm.AuthenticationError as e:
            raise ProviderAuthenticationError(f"Authentication failed: {e}") from e
        except litellm.RateLimitError as e:
            raise ProviderRateLimitError(f"Rate limit exceeded: {e}") from e
        except litellm.ContextWindowExceededError as e:
            raise ContextLengthExceededError(f"Context window exceeded: {e}") from e
        except litellm.ServiceUnavailableError as e:
            raise ServiceUnavailableError(f"Service unavailable: {e}") from e
        except litellm.Timeout as e:
            raise ProviderTimeoutError(f"Request timed out: {e}") from e
        except Exception as e:
            raise LLMError(f"LiteLLM error: {e}") from e

    async def list_models(self) -> List[str]:
        """
        Return a sorted list of model identifiers available for this provider.

        Strategy (best-effort, provider-dependent):
          1. Ollama / OpenAI-compatible base_url  → GET /v1/models via httpx.
          2. Known cloud providers (openai, anthropic, mistral, groq, gemini,
             dashscope) → return the curated litellm model map for that prefix.
          3. Fallback → return [current_model] so the caller always gets at
             least one valid entry.
        """
        import httpx

        provider_prefix = f"{self.provider_name}/"
        base_url = self.provider_config.get("base_url")

        # --- Strategy 1: OpenAI-compatible /v1/models endpoint ---
        if base_url:
            try:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(f"{base_url.rstrip('/')}/models", headers=headers)
                    resp.raise_for_status()
                    data = resp.json().get("data", [])
                    models = sorted(item["id"] for item in data if "id" in item)
                    return models or [self.model_name]
            except Exception:
                # Fall through to other strategies silently
                pass

        # --- Strategy 2: litellm's model_cost dict (has litellm_provider metadata) ---
        # This works for all providers including openai/anthropic which have no
        # prefix in litellm.model_list but are correctly tagged in model_cost.
        try:
            loop = asyncio.get_running_loop()
            cost_entries: List[tuple] = await loop.run_in_executor(
                None,
                lambda: [
                    (model, data)
                    for model, data in litellm.model_cost.items()
                    if data.get("litellm_provider") == self.provider_name
                    and data.get("mode") in ("chat", "completion", None)
                ],
            )
            provider_models = sorted(
                model.removeprefix(provider_prefix) for model, _ in cost_entries
            )
            if provider_models:
                return provider_models
        except Exception:
            pass

        # --- Strategy 3: Guaranteed fallback ---
        return [self.model_name]


def get_provider(
    config_manager: "ConfigManager",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    auth_mode: Optional[str] = None,
) -> LLMProvider:
    """
    Factory that instantiates and returns a LiteLLMDriver.

    Re-raises KAIError subclasses (ConfigurationError, ProviderAuthenticationError,
    etc.) unchanged so they are never double-wrapped inside a generic LLMError.
    """
    try:
        return LiteLLMDriver(
            config_manager=config_manager,
            provider_name=provider,
            model_name=model,
            auth_mode=auth_mode,
        )
    except KAIError:
        raise  # already a typed error — do not re-wrap
    except Exception as e:
        raise LLMError(f"Failed to get provider: {e}") from e

