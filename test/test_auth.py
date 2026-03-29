import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from k_ai.auth import (
    GoogleOAuthLoader,
    OAuthLoaderRegistry,
    OAuthProviderLoader,
    OAuthTokenState,
    TokenFileStore,
    build_google_oauth_authorization_url,
)
from k_ai.config import ConfigManager
from k_ai.exceptions import ConfigurationError, ProviderAuthenticationError
from k_ai.llm_core import LiteLLMDriver


def _write_token(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestTokenFileStore:
    def test_load_missing_file_raises(self, tmp_path):
        store = TokenFileStore.for_path(str(tmp_path / "missing.json"), "Google")
        with pytest.raises(ProviderAuthenticationError, match="token file not found"):
            store.load()

    def test_load_invalid_json_raises(self, tmp_path):
        path = tmp_path / "token.json"
        path.write_text("{invalid", encoding="utf-8")
        store = TokenFileStore.for_path(str(path), "Google")
        with pytest.raises(ConfigurationError, match="Invalid Google OAuth token file"):
            store.load()

    def test_load_non_object_json_raises(self, tmp_path):
        path = tmp_path / "token.json"
        path.write_text('["not-an-object"]', encoding="utf-8")
        store = TokenFileStore.for_path(str(path), "Google")
        with pytest.raises(ConfigurationError, match="must contain a JSON object"):
            store.load()

    def test_save_roundtrip(self, tmp_path):
        path = tmp_path / "nested" / "token.json"
        store = TokenFileStore.for_path(str(path), "Google")
        store.save({"access_token": "abc"})
        loaded = store.load()
        assert loaded.access_token == "abc"


class TestOAuthTokenState:
    def test_expiry_parses_unix_timestamp(self):
        state = OAuthTokenState(payload={"access_token": "tok", "expires_at": 4102444800})
        assert state.expiry is not None
        assert state.is_valid()

    def test_expiry_parses_zulu_string(self):
        state = OAuthTokenState(payload={"access_token": "tok", "expiry": "2999-01-01T00:00:00Z"})
        assert state.expiry is not None
        assert state.expiry.tzinfo is not None
        assert state.is_valid()

    def test_require_scopes_rejects_missing_scope(self, tmp_path):
        state = OAuthTokenState(payload={"access_token": "tok", "scopes": ["scope:a"]})
        with pytest.raises(ProviderAuthenticationError, match="missing required scopes"):
            state.require_scopes(["scope:a", "scope:b"], path=tmp_path / "token.json", provider_name="Google")


class TestOAuthProviderLoader:
    def test_base_loader_rejects_expired_token_without_refresh(self, tmp_path):
        path = _write_token(
            tmp_path / "token.json",
            {"access_token": "expired", "expires_at": "2000-01-01T00:00:00+00:00"},
        )
        loader = OAuthProviderLoader()
        with pytest.raises(ProviderAuthenticationError, match="has no refresh implementation"):
            loader.load_access_token(str(path), [])


class TestGoogleOAuthLoader:
    def test_load_valid_token_without_refresh(self, tmp_path):
        path = _write_token(
            tmp_path / "token.json",
            {
                "access_token": "oauth-token",
                "expires_at": "2999-01-01T00:00:00+00:00",
                "scopes": ["scope:a"],
            },
        )
        assert GoogleOAuthLoader().load_access_token(str(path), ["scope:a"]) == "oauth-token"

    def test_expired_token_without_refresh_token_raises(self, tmp_path):
        path = _write_token(
            tmp_path / "token.json",
            {"access_token": "expired", "expires_at": "2000-01-01T00:00:00+00:00"},
        )
        with pytest.raises(ProviderAuthenticationError, match="has no refresh_token"):
            GoogleOAuthLoader().load_access_token(str(path), [])

    def test_expired_token_without_client_credentials_raises(self, tmp_path):
        path = _write_token(
            tmp_path / "token.json",
            {
                "access_token": "expired",
                "refresh_token": "rt",
                "expires_at": "2000-01-01T00:00:00+00:00",
            },
        )
        with pytest.raises(ProviderAuthenticationError, match="missing client_id/client_secret"):
            GoogleOAuthLoader().load_access_token(str(path), [])

    def test_refresh_http_status_error_is_wrapped(self, tmp_path):
        path = _write_token(
            tmp_path / "token.json",
            {
                "access_token": "expired",
                "refresh_token": "rt",
                "client_id": "cid",
                "client_secret": "secret",
                "expires_at": "2000-01-01T00:00:00+00:00",
            },
        )
        request = httpx.Request("POST", "https://oauth2.googleapis.com/token")
        response = httpx.Response(401, request=request)
        with patch("k_ai.auth.httpx.post", side_effect=httpx.HTTPStatusError("bad", request=request, response=response)):
            with pytest.raises(ProviderAuthenticationError, match="HTTP 401"):
                GoogleOAuthLoader().load_access_token(str(path), [])

    def test_refresh_network_error_is_wrapped(self, tmp_path):
        path = _write_token(
            tmp_path / "token.json",
            {
                "access_token": "expired",
                "refresh_token": "rt",
                "client_id": "cid",
                "client_secret": "secret",
                "expires_at": "2000-01-01T00:00:00+00:00",
            },
        )
        with patch("k_ai.auth.httpx.post", side_effect=httpx.ConnectError("offline")):
            with pytest.raises(ProviderAuthenticationError, match="refresh failed"):
                GoogleOAuthLoader().load_access_token(str(path), [])

    def test_refresh_invalid_json_is_wrapped(self, tmp_path):
        path = _write_token(
            tmp_path / "token.json",
            {
                "access_token": "expired",
                "refresh_token": "rt",
                "client_id": "cid",
                "client_secret": "secret",
                "expires_at": "2000-01-01T00:00:00+00:00",
            },
        )
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.side_effect = ValueError("invalid json")
        with patch("k_ai.auth.httpx.post", return_value=response):
            with pytest.raises(ProviderAuthenticationError, match="invalid JSON"):
                GoogleOAuthLoader().load_access_token(str(path), [])

    def test_refresh_without_access_token_raises(self, tmp_path):
        path = _write_token(
            tmp_path / "token.json",
            {
                "access_token": "expired",
                "refresh_token": "rt",
                "client_id": "cid",
                "client_secret": "secret",
                "expires_at": "2000-01-01T00:00:00+00:00",
            },
        )
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"expires_in": 3600}
        with patch("k_ai.auth.httpx.post", return_value=response):
            with pytest.raises(ProviderAuthenticationError, match="returned no access_token"):
                GoogleOAuthLoader().load_access_token(str(path), [])

    def test_refresh_persists_updated_payload(self, tmp_path):
        path = _write_token(
            tmp_path / "token.json",
            {
                "access_token": "expired",
                "refresh_token": "rt",
                "client_id": "cid",
                "client_secret": "secret",
                "expires_at": "2000-01-01T00:00:00+00:00",
            },
        )
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"access_token": "fresh", "refresh_token": "rt2", "expires_in": 3600}
        with patch("k_ai.auth.httpx.post", return_value=response):
            token = GoogleOAuthLoader().load_access_token(str(path), [])
        assert token == "fresh"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["access_token"] == "fresh"
        assert payload["refresh_token"] == "rt2"
        assert payload["token_uri"] == "https://oauth2.googleapis.com/token"
        assert "expires_at" in payload


class TestOAuthLoaderRegistry:
    def test_register_requires_non_empty_key(self):
        registry = OAuthLoaderRegistry()
        with pytest.raises(ConfigurationError, match="cannot be empty"):
            registry.register("   ", GoogleOAuthLoader())

    def test_build_token_loaders_normalizes_keys(self, tmp_path):
        path = _write_token(
            tmp_path / "token.json",
            {"access_token": "tok", "expires_at": "2999-01-01T00:00:00+00:00"},
        )
        registry = OAuthLoaderRegistry()
        registry.register(" Google ", GoogleOAuthLoader())
        loaders = registry.build_token_loaders()
        assert "google" in loaders
        assert loaders["google"](str(path), []) == "tok"


class TestGoogleOAuthLoginHelpers:
    def test_build_google_oauth_authorization_url_contains_pkce_and_offline_access(self):
        url = build_google_oauth_authorization_url(
            client_id="cid",
            redirect_uri="http://127.0.0.1:8765/oauth/google/callback",
            scopes=["scope:a", "scope:b"],
            state="state123",
            code_challenge="challenge456",
        )
        assert "client_id=cid" in url
        assert "redirect_uri=http%3A%2F%2F127.0.0.1%3A8765%2Foauth%2Fgoogle%2Fcallback" in url
        assert "scope=scope%3Aa+scope%3Ab" in url
        assert "access_type=offline" in url
        assert "prompt=consent" in url
        assert "state=state123" in url
        assert "code_challenge=challenge456" in url


class TestLiteLLMDriverOAuthErrors:
    def test_unknown_oauth_provider_name_raises(self):
        cm = ConfigManager()
        cm.config.setdefault("oauth", {})["custom"] = {
            "oauth_provider_name": "unknown-provider",
            "oauth_scopes": ["scope:a"],
            "token_path": "/tmp/token.json",
            "default_model": "custom-model",
            "context_window": 1000,
        }
        with pytest.raises(ConfigurationError, match="No token loader registered"):
            LiteLLMDriver(cm, provider_name="custom", auth_mode="oauth")

    def test_missing_oauth_settings_raise(self):
        cm = ConfigManager()
        cm.config.setdefault("oauth", {})["custom"] = {
            "oauth_provider_name": "google",
            "default_model": "custom-model",
            "context_window": 1000,
        }
        with pytest.raises(ConfigurationError, match="missing required settings"):
            LiteLLMDriver(cm, provider_name="custom", auth_mode="oauth")
