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
    """Raised when an LLM provider's service is unavailable."""
    pass
