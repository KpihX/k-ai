# src/k_ai/secrets.py
"""
Multi-tier secret / API key resolution for k-ai.

Resolution order (first non-empty value wins):
  Tier 1 — .env file   : read from the project / user .env at startup.
                          Non-empty values only; blank placeholders are ignored.
  Tier 2 — os.environ  : inherited process environment (terminal exports, CI).
  Tier 3 — zsh -l -c   : login-shell injection (bw-env / ~/.zshrc exports).
                          Result is cached in os.environ for the process lifetime.

The active .env path is resolved in this order:
  1. K_AI_DOTENV_PATH env var (explicit override)
  2. Current working directory / .env
  3. ~/.k_ai/.env  (user-level config)

`resolve_secret(name)` returns (value, source) so callers can report exactly
where a credential came from — critical for the /status command.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from dotenv import dotenv_values

_log = logging.getLogger("k_ai.secrets")

# ---------------------------------------------------------------------------
# .env discovery
# ---------------------------------------------------------------------------

def _find_dotenv() -> Optional[Path]:
    """
    Locate the .env file to load.

    Checks (in order):
      1. K_AI_DOTENV_PATH env var (explicit path override).
      2. <cwd>/.env
      3. ~/.k_ai/.env
    """
    explicit = os.environ.get("K_AI_DOTENV_PATH")
    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            return p

    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env

    user_env = Path("~/.k_ai/.env").expanduser()
    if user_env.exists():
        return user_env

    return None


# Module-level state — populated once at import time.
_dotenv_path: Optional[Path] = None
_dotenv_values: dict[str, str] = {}


def _init() -> None:
    """Load the .env file into the module cache (called once at import)."""
    global _dotenv_path, _dotenv_values
    _dotenv_path = _find_dotenv()
    if _dotenv_path is None:
        _log.debug("No .env file found.")
        return
    # dotenv_values reads the file without mutating os.environ, giving us full
    # control over priority: we check our cache first, then os.environ.
    raw = dotenv_values(_dotenv_path)
    _dotenv_values = {k: v for k, v in raw.items() if v not in (None, "")}
    _log.debug("Loaded %d non-empty keys from %s", len(_dotenv_values), _dotenv_path)


_init()


# ---------------------------------------------------------------------------
# Login-shell fallback
# ---------------------------------------------------------------------------

def _shell_read_env(name: str) -> Optional[str]:
    """
    Spawn a zsh login shell to read a single environment variable.

    This is the canonical last-resort for variables injected by bw-env or
    ~/.zshrc that are not present in the current process environment.

    Returns None on any failure (timeout, zsh not found, variable unset).
    """
    try:
        result = subprocess.run(
            ["zsh", "-l", "-c", f'printf "%s" "${{{name}}}"'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        val = result.stdout.strip()
        return val if val else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        _log.debug("Login-shell read for %s failed: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------

def resolve_secret(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve an environment variable through the 3-tier chain.

    Args:
        name: The environment variable name (e.g. "OPENAI_API_KEY").

    Returns:
        A ``(value, source)`` tuple where ``source`` is one of:
          ``".env"``        — found in the .env file.
          ``"os.environ"``  — found in the process environment.
          ``"zsh -l -c"``   — found via login-shell read; cached into os.environ.
          ``None``          — not available anywhere (value is also None).
    """
    # Tier 1: .env file
    val = _dotenv_values.get(name)
    if val:
        return val, ".env"

    # Tier 2: inherited process environment
    val = os.environ.get(name)
    if val:
        return val, "os.environ"

    # Tier 3: login shell (expensive — run only when the other tiers fail)
    val = _shell_read_env(name)
    if val:
        os.environ[name] = val  # cache for subsequent lookups this session
        return val, "zsh -l -c"

    return None, None


# ---------------------------------------------------------------------------
# Status helpers (used by /status command)
# ---------------------------------------------------------------------------

def get_dotenv_path() -> Optional[Path]:
    """Return the .env file that was loaded at startup, or None."""
    return _dotenv_path


def get_all_key_status(config: dict) -> list[dict]:
    """
    Probe every api_providers entry in *config* and return their key status.

    Each result dict has:
      provider   (str)   — provider key from config
      env_var    (str)   — the environment variable name
      available  (bool)  — whether a value was found
      source     (str)   — where it came from (".env" / "os.environ" /
                           "zsh -l -c" / "not found")
    """
    results = []
    for provider_name, provider_cfg in config.get("api_providers", {}).items():
        env_var = provider_cfg.get("api_key_env_var", "")
        if not env_var:
            continue
        _, source = resolve_secret(env_var)
        results.append(
            {
                "provider": provider_name,
                "env_var": env_var,
                "available": source is not None,
                "source": source or "not found",
            }
        )
    return results
