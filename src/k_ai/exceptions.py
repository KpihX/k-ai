# src/k_ai/exceptions.py
"""
Custom exceptions for k-ai.
"""

class KAIError(Exception):
    """Base exception for all k-ai errors."""
    pass

class LLMError(KAIError):
    """Raised for general LLM API errors."""
    pass

class ConfigurationError(KAIError):
    """Raised for configuration-related errors."""
    pass

class ProviderAuthenticationError(LLMError):
    """Raised for authentication failures with an LLM provider."""
    pass

class ProviderRateLimitError(LLMError):
    """Raised when a rate limit is exceeded."""
    pass

class ContextLengthExceededError(LLMError):
    """Raised when the model's context window is exceeded."""
    pass

class ServiceUnavailableError(LLMError):
    """Raised when an LLM provider's service is unavailable (5xx / overloaded)."""
    pass

class ProviderTimeoutError(LLMError):
    """Raised when a request to the LLM provider times out."""
    pass

class SessionStoreError(KAIError):
    """Raised for session persistence errors (save/load/index)."""
    pass

class MemoryStoreError(KAIError):
    """Raised for memory store errors (corrupt file, validation)."""
    pass

class ToolExecutionError(KAIError):
    """Raised when an internal tool fails to execute."""
    pass
