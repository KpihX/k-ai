# test/test_exceptions.py
"""
Tests for the k-ai exception hierarchy.
"""
import pytest
from k_ai.exceptions import (
    KAIError,
    LLMError,
    ConfigurationError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ContextLengthExceededError,
    ServiceUnavailableError,
    ProviderTimeoutError,
)


# ---------------------------------------------------------------------------
# Existence and base-class relationships
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_kaiError_is_exception(self):
        assert issubclass(KAIError, Exception)

    def test_llm_error_is_kai_error(self):
        assert issubclass(LLMError, KAIError)

    def test_configuration_error_is_kai_error(self):
        assert issubclass(ConfigurationError, KAIError)

    def test_provider_auth_error_is_llm_error(self):
        assert issubclass(ProviderAuthenticationError, LLMError)
        assert issubclass(ProviderAuthenticationError, KAIError)

    def test_rate_limit_error_is_llm_error(self):
        assert issubclass(ProviderRateLimitError, LLMError)

    def test_context_length_error_is_llm_error(self):
        assert issubclass(ContextLengthExceededError, LLMError)

    def test_service_unavailable_is_llm_error(self):
        assert issubclass(ServiceUnavailableError, LLMError)

    def test_timeout_error_is_llm_error(self):
        assert issubclass(ProviderTimeoutError, LLMError)


# ---------------------------------------------------------------------------
# Raise / catch semantics
# ---------------------------------------------------------------------------

class TestExceptionRaise:
    def test_raise_and_catch_kai_error(self):
        with pytest.raises(KAIError, match="base"):
            raise KAIError("base")

    def test_auth_error_caught_as_llm_error(self):
        with pytest.raises(LLMError):
            raise ProviderAuthenticationError("bad key")

    def test_auth_error_caught_as_kai_error(self):
        with pytest.raises(KAIError):
            raise ProviderAuthenticationError("bad key")

    def test_configuration_error_not_llm_error(self):
        """ConfigurationError is under KAIError, not LLMError."""
        assert not issubclass(ConfigurationError, LLMError)
        try:
            raise ConfigurationError("bad config")
        except LLMError:
            pytest.fail("ConfigurationError should not be caught as LLMError")
        except ConfigurationError:
            pass  # expected

    def test_timeout_error_caught_as_llm_error(self):
        with pytest.raises(LLMError):
            raise ProviderTimeoutError("timed out")

    def test_service_unavailable_caught_as_kai_error(self):
        with pytest.raises(KAIError):
            raise ServiceUnavailableError("503")

    def test_all_llm_errors_caught_as_kai_error(self):
        for exc_cls in [
            ProviderAuthenticationError,
            ProviderRateLimitError,
            ContextLengthExceededError,
            ServiceUnavailableError,
            ProviderTimeoutError,
        ]:
            with pytest.raises(KAIError):
                raise exc_cls("test")

    def test_exception_message_preserved(self):
        msg = "something went wrong"
        exc = ProviderAuthenticationError(msg)
        assert str(exc) == msg
