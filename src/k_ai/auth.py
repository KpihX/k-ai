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
import secrets
import threading
import webbrowser
from base64 import urlsafe_b64encode
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from hashlib import sha256
from pathlib import Path
from typing import Callable, Dict, List
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from .exceptions import ConfigurationError, ProviderAuthenticationError
from .secrets import resolve_secret


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


def _urlsafe_b64_no_pad(raw: bytes) -> str:
    return urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def build_google_oauth_authorization_url(
    *,
    client_id: str,
    redirect_uri: str,
    scopes: List[str],
    state: str,
    code_challenge: str,
) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)


def run_google_oauth_login(
    *,
    token_path: str,
    scopes: List[str],
    client_id_env_var: str = "GOOGLE_CLIENT_ID",
    client_secret_env_var: str = "GOOGLE_CLIENT_SECRET",
    callback_host: str = "127.0.0.1",
    callback_port: int = 0,
    timeout_seconds: int = 180,
) -> dict:
    client_id, client_id_source = resolve_secret(client_id_env_var)
    if not client_id:
        raise ProviderAuthenticationError(
            f"Google OAuth login requires {client_id_env_var} to be available."
        )
    client_secret, client_secret_source = resolve_secret(client_secret_env_var)
    if not client_secret:
        raise ProviderAuthenticationError(
            f"Google OAuth login requires {client_secret_env_var} to be available."
        )

    verifier = _urlsafe_b64_no_pad(secrets.token_bytes(48))
    challenge = _urlsafe_b64_no_pad(sha256(verifier.encode("ascii")).digest())
    state = secrets.token_urlsafe(24)
    token_uri = GoogleOAuthLoader.default_token_uri
    payload_holder: dict[str, str] = {}
    event = threading.Event()

    class OAuthCallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            payload_holder["path"] = parsed.path
            payload_holder["state"] = params.get("state", [""])[0]
            payload_holder["code"] = params.get("code", [""])[0]
            payload_holder["error"] = params.get("error", [""])[0]
            body = ""
            status = 200
            if payload_holder["error"]:
                status = 400
                body = "Google OAuth login failed. You can close this tab and return to k-ai."
            elif payload_holder["state"] != state or not payload_holder["code"]:
                status = 400
                body = "Google OAuth callback was invalid. You can close this tab and return to k-ai."
            else:
                body = "Google OAuth login completed. You can close this tab and return to k-ai."
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
            event.set()

        def log_message(self, format, *args):  # noqa: A003
            return

    try:
        server = ThreadingHTTPServer((callback_host, callback_port), OAuthCallbackHandler)
    except OSError as exc:
        raise ProviderAuthenticationError(
            f"Could not start local Google OAuth callback server on {callback_host}:{callback_port}: {exc}"
        ) from exc

    redirect_uri = f"http://{callback_host}:{server.server_port}/oauth/google/callback"
    auth_url = build_google_oauth_authorization_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scopes=scopes,
        state=state,
        code_challenge=challenge,
    )

    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    browser_opened = webbrowser.open(auth_url, new=1, autoraise=True)
    if not browser_opened:
        server.server_close()
        raise ProviderAuthenticationError(
            "Could not open the browser automatically for Google OAuth login. "
            f"Open this URL manually: {auth_url}"
        )
    if not event.wait(timeout_seconds):
        server.server_close()
        raise ProviderAuthenticationError(
            f"Google OAuth login timed out after {timeout_seconds} seconds waiting for the browser callback."
        )
    server.server_close()

    error = payload_holder.get("error", "").strip()
    if error:
        raise ProviderAuthenticationError(f"Google OAuth login failed: {error}")
    if payload_holder.get("path") != "/oauth/google/callback":
        raise ProviderAuthenticationError("Google OAuth callback arrived on an unexpected path.")
    code = payload_holder.get("code", "").strip()
    if not code:
        raise ProviderAuthenticationError("Google OAuth callback did not include an authorization code.")

    try:
        response = httpx.post(
            token_uri,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "code_verifier": verifier,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        exchanged = response.json()
    except httpx.HTTPStatusError as exc:
        raise ProviderAuthenticationError(
            f"Google OAuth token exchange failed with HTTP {exc.response.status_code}."
        ) from exc
    except httpx.HTTPError as exc:
        raise ProviderAuthenticationError(f"Google OAuth token exchange failed: {exc}") from exc
    except ValueError as exc:
        raise ProviderAuthenticationError("Google OAuth token exchange returned invalid JSON.") from exc

    access_token = str(exchanged.get("access_token", "") or "").strip()
    if not access_token:
        raise ProviderAuthenticationError("Google OAuth token exchange returned no access_token.")

    token_payload = {
        "access_token": access_token,
        "refresh_token": str(exchanged.get("refresh_token", "") or "").strip(),
        "client_id": client_id,
        "client_secret": client_secret,
        "token_uri": token_uri,
        "scopes": list(scopes),
    }
    if "expires_in" in exchanged:
        token_payload["expires_in"] = exchanged["expires_in"]
        token_payload["expires_at"] = (
            datetime.now(timezone.utc) + timedelta(seconds=float(exchanged["expires_in"]))
        ).isoformat()
    store = TokenFileStore.for_path(token_path, "Google")
    store.save(token_payload)
    return {
        "token_path": str(store.path),
        "browser_opened": browser_opened,
        "redirect_uri": redirect_uri,
        "auth_url": auth_url,
        "client_id_source": client_id_source,
        "client_secret_source": client_secret_source,
        "scopes": list(scopes),
    }


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
