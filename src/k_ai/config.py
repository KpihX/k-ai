# src/k_ai/config.py
"""
Configuration management for k-ai.
Handles loading, merging, and live editing of configuration with a clear override hierarchy:
  1. Built-in defaults  (src/k_ai/defaults/default_config.yaml)
  2. User config file   (--config path, or override_path=...)
  3. Inline kwargs      (ConfigManager(temperature=0.9, provider="openai"))
"""
import copy
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_FILENAME = "default_config.yaml"

# Sentinel used by get_nested() to distinguish "key absent" from "key present
# with a null value" — a plain None would be ambiguous in the latter case.
_SENTINEL = object()


@lru_cache(maxsize=8)
def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load and parse a YAML file from disk (no caching — callers deep-copy as needed)."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration at {path} must be a YAML mapping, got {type(config).__name__}.")
    return config


class ConfigManager:
    """
    Central configuration store for k-ai.

    Usage examples::

        # Defaults only
        cm = ConfigManager()

        # Override from file (partial file — only defined keys override defaults)
        cm = ConfigManager(override_path="~/my_config.yaml")

        # Override individual params inline
        cm = ConfigManager(provider="openai", temperature=0.2)

        # Both
        cm = ConfigManager(override_path="base.yaml", model="gpt-4o")

        # External library usage: get the default config template
        yaml_text = ConfigManager.get_default_yaml()
    """

    def __init__(self, override_path: Optional[str] = None, **kwargs: Any):
        self.default_config_path = Path(__file__).parent / "defaults" / DEFAULT_CONFIG_FILENAME
        self.override_path = Path(override_path).expanduser() if override_path else None
        self._kwargs: Dict[str, Any] = kwargs

        self.config: Dict[str, Any] = {}
        self._load_and_merge()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_merge(self) -> None:
        """Build self.config from all layers (deepcopy avoids mutating on-disk data)."""
        # Layer 1: package defaults
        self.config = copy.deepcopy(_load_yaml(self.default_config_path))

        # Layer 2: user override file
        if self.override_path and self.override_path.exists():
            self._deep_merge(self.config, _load_yaml(self.override_path))

        # Layer 3: inline kwargs (top-level only)
        for key, value in self._kwargs.items():
            self.config[key] = value

    @staticmethod
    def _deep_merge(base: Dict, overlay: Dict) -> Dict:
        """Recursively merge `overlay` into `base` in-place. `overlay` wins."""
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                ConfigManager._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return a top-level config value."""
        return self.config.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """
        Return a nested config value using a key path.

        Uses an internal sentinel so that a legitimately null config value is
        not confused with "key not found" (which would happen if None were used
        as the sentinel directly).

        Example::
            cm.get_nested("cli", "theme")  # equivalent to config["cli"]["theme"]
        """
        node = self.config
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, _SENTINEL)
            if node is _SENTINEL:
                return default
        return node

    def get_all(self) -> Dict[str, Any]:
        """Return a deep copy of the entire active configuration."""
        return copy.deepcopy(self.config)

    def flatten(self, prefix: str = "") -> Dict[str, Any]:
        """Return the active config as a flat dot-notation mapping."""
        result: Dict[str, Any] = {}

        def _walk(node: Any, path: List[str]) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    _walk(value, path + [key])
                return
            if not path:
                return
            flat_key = ".".join(path)
            if not prefix or flat_key == prefix or flat_key.startswith(prefix + "."):
                result[flat_key] = node

        _walk(self.config, [])
        return result

    def get_path(self, key: str, default: Any = None) -> Any:
        """Return a value from a dot-notation path."""
        if not key:
            return self.get_all()
        parts = key.split(".")
        return self.get_nested(*parts, default=default)

    # ------------------------------------------------------------------
    # Write API (live in-session edits)
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> Any:
        """
        Set a config value, supporting dot-notation for nested keys.
        Automatically coerces string values to the type of the existing value.

        Examples::
            cm.set("temperature", 0.9)
            cm.set("temperature", "0.9")        # auto-coerced to float
            cm.set("cli.show_token_usage", "false")  # auto-coerced to bool
            cm.set("max_tokens", "8192")        # auto-coerced to int
        """
        parts = key.split(".")
        node = self.config
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]

        leaf = parts[-1]
        existing = node.get(leaf)
        value = self._coerce(value, existing)
        node[leaf] = value
        return node[leaf]

    @staticmethod
    def _coerce(value: Any, existing: Any) -> Any:
        """Try to cast `value` to the type of `existing` when `value` is a string."""
        if existing is None or not isinstance(value, str):
            return value
        try:
            if isinstance(existing, bool):
                return value.lower() in ("true", "1", "yes", "on")
            if isinstance(existing, int):
                return int(value)
            if isinstance(existing, float):
                return float(value)
        except (ValueError, TypeError):
            pass
        return value

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def dump_yaml(self) -> str:
        """Serialise the full active configuration to a YAML string."""
        return yaml.dump(
            self.get_all(),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    @classmethod
    def get_default_yaml(cls) -> str:
        """Return the raw content of the built-in default config file."""
        path = Path(__file__).parent / "defaults" / DEFAULT_CONFIG_FILENAME
        return path.read_text(encoding="utf-8")

    def save_active_yaml(self, path: Optional[str] = None) -> Path:
        """Persist the active merged configuration to disk and return the written path."""
        target = Path(
            path
            or (
                str(self.override_path)
                if self.override_path
                else self.get_nested("config", "persist_path", default="~/.k-ai/config.yaml")
            )
        ).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.dump_yaml(), encoding="utf-8")
        return target

    # ------------------------------------------------------------------
    # Provider helpers (used by llm_core)
    # ------------------------------------------------------------------

    def list_providers(self) -> Dict[str, List[str]]:
        """
        Return all configured providers grouped by auth mode.

        The order follows ``provider_search_priority``.  Only sections that
        contain at least one provider entry are included.

        Example::

            {
                "no_auth":  ["ollama"],
                "api_key":  ["anthropic", "dashscope", "gemini", "groq", "mistral", "openai"],
                "oauth":    ["gemini"],
            }
        """
        result: Dict[str, List[str]] = {}
        for section in self.get("provider_search_priority", ["no_auth", "api_key", "oauth"]):
            providers = sorted(self.config.get(section, {}).keys())
            if providers:
                result[section] = providers
        return result

    def get_provider_config_with_auth_mode(
        self,
        provider_name: str,
        auth_mode: Optional[str] = None,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Locate a provider's configuration block and determine its auth mode.

        Searches sections in the order defined by `provider_search_priority`.
        Returns (provider_config_dict, auth_mode_string) or (None, None).
        """
        search_priority: list[str] = self.get(
            "provider_search_priority",
            ["no_auth", "api_key", "oauth"],
        )

        def _find(section_name: str) -> Optional[Dict[str, Any]]:
            return self.config.get(section_name, {}).get(provider_name)

        if auth_mode:
            # Section name equals auth_mode (e.g. "api_key" → config["api_key"][provider]).
            cfg = _find(auth_mode)
            if cfg:
                return cfg, auth_mode
        else:
            for section in search_priority:
                cfg = _find(section)
                if cfg:
                    return cfg, section  # section name IS the auth_mode

        return None, None
