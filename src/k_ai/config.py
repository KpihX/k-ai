# src/k_ai/config.py
"""
Configuration management for k-ai.
Handles loading, merging, and live editing of configuration with a clear override hierarchy:
  1. Built-in defaults  (src/k_ai/defaults/defaults.d/*.yaml)
  2. User config file   (--config path, or override_path=...)
  3. Inline kwargs      (ConfigManager(temperature=0.9, provider="openai"))
"""
import copy
import os
import shlex
import shutil
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CONFIG_DIRNAME = "defaults.d"
DEFAULT_CONFIG_SECTIONS: Tuple[Tuple[str, str, str], ...] = (
    ("models", "00-models.yaml", "Provider, model, temperature, auth-backed provider definitions."),
    ("ui", "10-ui-prompts.yaml", "CLI rendering, runtime transparency, and prompt templates."),
    ("sessions", "20-sessions-memory.yaml", "Session persistence, compaction, and memory defaults."),
    ("governance", "30-runtime-governance.yaml", "Config persistence, tool approval catalog, and tool runtime settings."),
)

# Sentinel used by get_nested() to distinguish "key absent" from "key present
# with a null value" — a plain None would be ambiguous in the latter case.
_SENTINEL = object()


@lru_cache(maxsize=8)
def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load and parse a YAML file from disk (cached by absolute path)."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration at {path} must be a YAML mapping, got {type(config).__name__}.")
    return config


def _deep_merge_dicts(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `overlay` into `base` in-place. `overlay` wins."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


@lru_cache(maxsize=4)
def _discover_yaml_fragments(root: Path) -> Tuple[Path, ...]:
    """Return ordered YAML fragments from a config root path."""
    if root.is_file():
        return (root,)
    if not root.exists():
        raise FileNotFoundError(f"Configuration defaults not found at: {root}")
    files = tuple(
        sorted(
            path for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        )
    )
    if not files:
        raise FileNotFoundError(f"No YAML defaults found in: {root}")
    return files


@lru_cache(maxsize=4)
def _load_yaml_tree(root: Path) -> Dict[str, Any]:
    """Load and merge a YAML file or an ordered directory of YAML fragments."""
    merged: Dict[str, Any] = {}
    for path in _discover_yaml_fragments(root):
        _deep_merge_dicts(merged, _load_yaml(path))
    return merged


@lru_cache(maxsize=4)
def _concat_yaml_tree(root: Path) -> str:
    """Concatenate ordered YAML fragment sources into one exportable config text."""
    chunks = [path.read_text(encoding="utf-8").rstrip() for path in _discover_yaml_fragments(root)]
    return "\n\n".join(chunk for chunk in chunks if chunk).rstrip() + "\n"


def _concat_yaml_files(paths: Tuple[Path, ...]) -> str:
    chunks = [path.read_text(encoding="utf-8").rstrip() for path in paths]
    return "\n\n".join(chunk for chunk in chunks if chunk).rstrip() + "\n"


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
        self.default_config_path = Path(__file__).parent / "defaults" / DEFAULT_CONFIG_DIRNAME
        self.default_config_files = list(_discover_yaml_fragments(self.default_config_path))
        self.default_config_sections = self._build_default_section_index(self.default_config_path)
        self.override_path = Path(override_path).expanduser() if override_path else None
        self._kwargs: Dict[str, Any] = kwargs

        self.config: Dict[str, Any] = {}
        self._load_and_merge()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_default_section_index(root: Path) -> Dict[str, Dict[str, Any]]:
        files_by_name = {path.name: path for path in _discover_yaml_fragments(root)}
        index: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for section_name, filename, description in DEFAULT_CONFIG_SECTIONS:
            path = files_by_name.get(filename)
            if path is None:
                missing.append(filename)
                continue
            index[section_name] = {
                "name": section_name,
                "file": path,
                "description": description,
            }
        if missing:
            raise FileNotFoundError(
                "Missing built-in config fragments: " + ", ".join(sorted(missing))
            )
        extra = sorted(set(files_by_name) - {filename for _, filename, _ in DEFAULT_CONFIG_SECTIONS})
        if extra:
            raise ValueError(
                "Unregistered built-in config fragments found in defaults.d: " + ", ".join(extra)
            )
        return index

    def _load_and_merge(self) -> None:
        """Build self.config from all layers (deepcopy avoids mutating on-disk data)."""
        # Layer 1: package defaults
        self.config = copy.deepcopy(_load_yaml_tree(self.default_config_path))

        # Layer 2: user override file
        if self.override_path and self.override_path.exists():
            self._deep_merge(self.config, _load_yaml(self.override_path))

        # Layer 3: inline kwargs (top-level only)
        for key, value in self._kwargs.items():
            self.config[key] = value

    @staticmethod
    def _deep_merge(base: Dict, overlay: Dict) -> Dict:
        """Recursively merge `overlay` into `base` in-place. `overlay` wins."""
        return _deep_merge_dicts(base, overlay)

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

    def delete_path(self, key: str) -> bool:
        """Delete a dot-notation path if it exists. Returns True when removed."""
        if not key:
            return False
        parts = key.split(".")
        node = self.config
        parents: List[tuple[Dict[str, Any], str]] = []
        for part in parts[:-1]:
            next_node = node.get(part)
            if not isinstance(next_node, dict):
                return False
            parents.append((node, part))
            node = next_node

        leaf = parts[-1]
        if leaf not in node:
            return False
        del node[leaf]

        # Prune empty nested mappings created only for this path.
        for parent, part in reversed(parents):
            child = parent.get(part)
            if isinstance(child, dict) and not child:
                del parent[part]
            else:
                break
        return True

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
    def list_default_sections(cls) -> List[Dict[str, str]]:
        """Return the available default config sections in load/export order."""
        root = Path(__file__).parent / "defaults" / DEFAULT_CONFIG_DIRNAME
        index = cls._build_default_section_index(root)
        return [
            {
                "name": name,
                "file": data["file"].name,
                "description": data["description"],
            }
            for name, _, _ in DEFAULT_CONFIG_SECTIONS
            for data in [index[name]]
        ]

    @classmethod
    def normalize_default_sections(cls, sections: Optional[List[str]] = None) -> List[str]:
        """Normalize a requested set of config sections."""
        available = cls.list_default_sections()
        ordered_names = [item["name"] for item in available]
        index = set(ordered_names)
        if not sections:
            return ordered_names
        normalized: List[str] = []
        seen: set[str] = set()
        for section in sections:
            name = str(section or "").strip().lower()
            if not name or name == "all":
                continue
            if name not in index:
                available = ", ".join(sorted(index))
                raise ValueError(f"Unknown config section: {section}. Available: {available}")
            if name not in seen:
                normalized.append(name)
                seen.add(name)
        return normalized or ordered_names

    @classmethod
    def get_default_yaml(cls, sections: Optional[List[str]] = None) -> str:
        """Return the built-in default config template assembled from ordered fragments."""
        root = Path(__file__).parent / "defaults" / DEFAULT_CONFIG_DIRNAME
        if not sections:
            return _concat_yaml_tree(root)
        index = cls._build_default_section_index(root)
        ordered = cls.normalize_default_sections(sections)
        paths = tuple(index[name]["file"] for name in ordered)
        return _concat_yaml_files(paths)

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

    def resolve_edit_target(self, target: Optional[str] = None) -> Path:
        """
        Resolve which YAML file should be opened by `config edit`.

        Targets:
          - None / all / active / user / override -> runtime override/persisted file
          - models / ui / sessions / governance   -> built-in split fragment
        """
        name = str(target or "all").strip().lower()
        if name in {"", "all", "active", "user", "override"}:
            return self.save_active_yaml()
        normalized = self.normalize_default_sections([name])[0]
        info = self.default_config_sections[normalized]
        return Path(info["file"])

    def resolve_editor_command(self) -> List[str]:
        """
        Return the editor command to use for `config edit`.

        Resolution order:
          1. config.editor
          2. $K_AI_EDITOR
          3. $VISUAL
          4. $EDITOR
          5. nano
        """
        candidates = [
            str(self.get_nested("config", "editor", default="") or "").strip(),
            os.environ.get("K_AI_EDITOR", "").strip(),
            os.environ.get("VISUAL", "").strip(),
            os.environ.get("EDITOR", "").strip(),
            "nano",
        ]
        first_nonempty = ""
        for raw in candidates:
            if not raw:
                continue
            first_nonempty = first_nonempty or raw
            command = shlex.split(raw)
            if not command:
                continue
            binary = shutil.which(command[0])
            if binary:
                command[0] = binary
                return command
        raise FileNotFoundError(
            f"Editor '{first_nonempty or 'nano'}' not found. Set config.editor, K_AI_EDITOR, VISUAL, or EDITOR to a valid editor."
        )

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
