# src/k_ai/auth.py
"""
OAuth token loaders registry.

TOKEN_LOADERS maps oauth_provider_name -> Callable[[token_path, scopes], str].
Each loader receives the token file path and requested scopes, and must return
a valid bearer token string, refreshing it when possible.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List

import httpx

from .exceptions import ConfigurationError, ProviderAuthenticationError


TokenLoader = Callable[[str, List[str]], str]


def _parse_expiry(payload: dict) -> datetime | None:
    expires_at = payload.get("expires_at")
    if isinstance(expires_at, (int, float)):
        return datetime.fromtimestamp(float(expires_at), tz=timezone.utc)
    if isinstance(expires_at, str) and expires_at.strip():
        text = expires_at.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = None
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

    expiry = payload.get("expiry")
    if isinstance(expiry, str) and expiry.strip():
        text = expiry.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = None
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

    expires_in = payload.get("expires_in")
    if isinstance(expires_in, (int, float)):
        return datetime.now(timezone.utc) + timedelta(seconds=float(expires_in))
    return None


def _token_is_valid(payload: dict) -> bool:
    access_token = str(payload.get("access_token", "") or "").strip()
    if not access_token:
        return False
    expiry = _parse_expiry(payload)
    if expiry is None:
        return True
    return expiry > (datetime.now(timezone.utc) + timedelta(seconds=60))


def _write_token_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _refresh_google_token(path: Path, payload: dict) -> dict:
    refresh_token = str(payload.get("refresh_token", "") or "").strip()
    client_id = str(payload.get("client_id", "") or "").strip()
    client_secret = str(payload.get("client_secret", "") or "").strip()
    token_uri = str(payload.get("token_uri", "") or "https://oauth2.googleapis.com/token").strip()

    if not refresh_token:
        raise ProviderAuthenticationError(
            f"Google OAuth token at '{path}' is expired and has no refresh_token."
        )
    if not client_id or not client_secret:
        raise ProviderAuthenticationError(
            f"Google OAuth token at '{path}' is expired and missing client_id/client_secret for refresh."
        )

    response = httpx.post(
        token_uri,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=15.0,
    )
    response.raise_for_status()
    refreshed = response.json()
    access_token = str(refreshed.get("access_token", "") or "").strip()
    if not access_token:
        raise ProviderAuthenticationError(
            f"Google OAuth refresh succeeded but returned no access_token for '{path}'."
        )

    updated = dict(payload)
    updated["access_token"] = access_token
    if "expires_in" in refreshed:
        updated["expires_in"] = refreshed["expires_in"]
        updated["expires_at"] = (
            datetime.now(timezone.utc) + timedelta(seconds=float(refreshed["expires_in"]))
        ).isoformat()
    if refreshed.get("refresh_token"):
        updated["refresh_token"] = refreshed["refresh_token"]
    updated.setdefault("token_uri", token_uri)
    _write_token_payload(path, updated)
    return updated


def _load_google_token(token_path: str, scopes: List[str]) -> str:
    path = Path(token_path).expanduser()
    if not path.exists():
        raise ProviderAuthenticationError(f"Google OAuth token file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigurationError(f"Invalid Google OAuth token file at '{path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise ConfigurationError(f"Google OAuth token file at '{path}' must contain a JSON object.")

    token_scopes = payload.get("scopes")
    if token_scopes and isinstance(token_scopes, list):
        missing_scopes = [scope for scope in scopes if scope not in token_scopes]
        if missing_scopes:
            raise ProviderAuthenticationError(
                f"Google OAuth token at '{path}' is missing required scopes: {', '.join(missing_scopes)}"
            )

    if _token_is_valid(payload):
        return str(payload["access_token"])

    refreshed = _refresh_google_token(path, payload)
    if _token_is_valid(refreshed):
        return str(refreshed["access_token"])
    raise ProviderAuthenticationError(f"Google OAuth token at '{path}' is invalid after refresh.")


TOKEN_LOADERS: Dict[str, TokenLoader] = {
    "google": _load_google_token,
}
