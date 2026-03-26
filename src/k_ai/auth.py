# src/k_ai/auth.py
"""
OAuth token loading and refresh helpers.

Public contract:
    TOKEN_LOADERS[oauth_provider_name](token_path, scopes) -> bearer token

The module is intentionally structured so new OAuth providers can be added
without turning this file into provider-specific spaghetti:
    - TokenFileStore handles disk IO
    - OAuthTokenState handles validation/expiry logic
    - OAuthProviderLoader implements shared load/refresh orchestration
    - OAuthLoaderRegistry centralizes provider registration
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List

import httpx

from .exceptions import ConfigurationError, ProviderAuthenticationError


TokenLoader = Callable[[str, List[str]], str]


def _parse_expiry_value(value: object) -> datetime | None:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


@dataclass(frozen=True)
class OAuthTokenState:
    payload: dict

    @property
    def access_token(self) -> str:
        return str(self.payload.get("access_token", "") or "").strip()

    @property
    def refresh_token(self) -> str:
        return str(self.payload.get("refresh_token", "") or "").strip()

    @property
    def token_uri(self) -> str:
        return str(self.payload.get("token_uri", "") or "").strip()

    @property
    def client_id(self) -> str:
        return str(self.payload.get("client_id", "") or "").strip()

    @property
    def client_secret(self) -> str:
        return str(self.payload.get("client_secret", "") or "").strip()

    @property
    def expiry(self) -> datetime | None:
        expires_at = _parse_expiry_value(self.payload.get("expires_at"))
        if expires_at is not None:
            return expires_at
        expiry = _parse_expiry_value(self.payload.get("expiry"))
        if expiry is not None:
            return expiry

        expires_in = self.payload.get("expires_in")
        if isinstance(expires_in, (int, float)):
            return datetime.now(timezone.utc) + timedelta(seconds=float(expires_in))
        return None

    def is_valid(self, leeway_seconds: int = 60) -> bool:
        if not self.access_token:
            return False
        expiry = self.expiry
        if expiry is None:
            return True
        return expiry > (datetime.now(timezone.utc) + timedelta(seconds=leeway_seconds))

    def require_scopes(self, required_scopes: List[str], *, path: Path, provider_name: str) -> None:
        token_scopes = self.payload.get("scopes")
        if token_scopes and isinstance(token_scopes, list):
            missing_scopes = [scope for scope in required_scopes if scope not in token_scopes]
            if missing_scopes:
                raise ProviderAuthenticationError(
                    f"{provider_name} OAuth token at '{path}' is missing required scopes: "
                    + ", ".join(missing_scopes)
                )


@dataclass(frozen=True)
class TokenFileStore:
    path: Path
    provider_name: str

    @classmethod
    def for_path(cls, token_path: str, provider_name: str) -> "TokenFileStore":
        return cls(path=Path(token_path).expanduser(), provider_name=provider_name)

    def load(self) -> OAuthTokenState:
        if not self.path.exists():
            raise ProviderAuthenticationError(f"{self.provider_name} OAuth token file not found: {self.path}")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConfigurationError(
                f"Invalid {self.provider_name} OAuth token file at '{self.path}': {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ConfigurationError(
                f"{self.provider_name} OAuth token file at '{self.path}' must contain a JSON object."
            )
        return OAuthTokenState(payload=payload)

    def save(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class OAuthProviderLoader:
    provider_name = "oauth"

    def load_access_token(self, token_path: str, scopes: List[str]) -> str:
        store = TokenFileStore.for_path(token_path, self.provider_name)
        state = store.load()
        state.require_scopes(scopes, path=store.path, provider_name=self.provider_name)
        if state.is_valid():
            return state.access_token
        refreshed = self.refresh_access_token(state=state, store=store, scopes=scopes)
        refreshed.require_scopes(scopes, path=store.path, provider_name=self.provider_name)
        if refreshed.is_valid():
            return refreshed.access_token
        raise ProviderAuthenticationError(
            f"{self.provider_name} OAuth token at '{store.path}' is invalid after refresh."
        )

    def refresh_access_token(
        self,
        *,
        state: OAuthTokenState,
        store: TokenFileStore,
        scopes: List[str],
    ) -> OAuthTokenState:
        raise ProviderAuthenticationError(
            f"{self.provider_name} OAuth token at '{store.path}' is expired and this provider has no refresh implementation."
        )


class GoogleOAuthLoader(OAuthProviderLoader):
    provider_name = "Google"
    default_token_uri = "https://oauth2.googleapis.com/token"

    def refresh_access_token(
        self,
        *,
        state: OAuthTokenState,
        store: TokenFileStore,
        scopes: List[str],
    ) -> OAuthTokenState:
        if not state.refresh_token:
            raise ProviderAuthenticationError(
                f"{self.provider_name} OAuth token at '{store.path}' is expired and has no refresh_token."
            )
        if not state.client_id or not state.client_secret:
            raise ProviderAuthenticationError(
                f"{self.provider_name} OAuth token at '{store.path}' is expired and missing client_id/client_secret for refresh."
            )

        token_uri = state.token_uri or self.default_token_uri
        try:
            response = httpx.post(
                token_uri,
                data={
                    "client_id": state.client_id,
                    "client_secret": state.client_secret,
                    "refresh_token": state.refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=15.0,
            )
            response.raise_for_status()
            refreshed = response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderAuthenticationError(
                f"{self.provider_name} OAuth refresh failed with HTTP {exc.response.status_code} "
                f"for '{store.path}'."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderAuthenticationError(
                f"{self.provider_name} OAuth refresh failed for '{store.path}': {exc}"
            ) from exc
        except ValueError as exc:
            raise ProviderAuthenticationError(
                f"{self.provider_name} OAuth refresh returned invalid JSON for '{store.path}'."
            ) from exc
        access_token = str(refreshed.get("access_token", "") or "").strip()
        if not access_token:
            raise ProviderAuthenticationError(
                f"{self.provider_name} OAuth refresh succeeded but returned no access_token for '{store.path}'."
            )

        updated = dict(state.payload)
        updated["access_token"] = access_token
        updated.setdefault("token_uri", token_uri)
        if "expires_in" in refreshed:
            updated["expires_in"] = refreshed["expires_in"]
            updated["expires_at"] = (
                datetime.now(timezone.utc) + timedelta(seconds=float(refreshed["expires_in"]))
            ).isoformat()
        if refreshed.get("refresh_token"):
            updated["refresh_token"] = refreshed["refresh_token"]
        store.save(updated)
        return OAuthTokenState(payload=updated)


class OAuthLoaderRegistry:
    def __init__(self) -> None:
        self._providers: Dict[str, OAuthProviderLoader] = {}

    def register(self, key: str, provider_loader: OAuthProviderLoader) -> None:
        normalized = key.strip().lower()
        if not normalized:
            raise ConfigurationError("OAuth provider registry key cannot be empty.")
        self._providers[normalized] = provider_loader

    def build_token_loaders(self) -> Dict[str, TokenLoader]:
        return {
            key: provider_loader.load_access_token
            for key, provider_loader in self._providers.items()
        }


REGISTRY = OAuthLoaderRegistry()
REGISTRY.register("google", GoogleOAuthLoader())

TOKEN_LOADERS: Dict[str, TokenLoader] = REGISTRY.build_token_loaders()
