# src/k_ai/config.py
"""
Configuration management for k-ai.
Handles loading default and user-provided configurations with a clear override hierarchy.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

# The name of the default configuration file included with the package
DEFAULT_CONFIG_FILENAME = "default_config.yaml"

@lru_cache(maxsize=None)
def load_config_from_path(path: Path) -> Dict[str, Any]:
    """
    Loads and parses a YAML configuration file from a given path.
    This function is cached to avoid repeated disk I/O.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {path}")
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise TypeError(f"Configuration at {path} is not a valid dictionary.")
        
    return config

class ConfigManager:
    def __init__(self, override_path: Optional[str] = None):
        """
        Initializes the ConfigManager.
        """
        self.default_config_path = Path(__file__).parent / "defaults" / DEFAULT_CONFIG_FILENAME
        self.override_path = Path(override_path).expanduser() if override_path else None
        
        self.config: Dict[str, Any] = {}
        self._load_and_merge()

    def _load_and_merge(self):
        """
        Loads the default configuration and merges the override config on top.
        """
        # 1. Load default config from cache
        self.config = load_config_from_path(self.default_config_path)

        # 2. Load override config if it exists
        if self.override_path and self.override_path.exists():
            override_config = load_config_from_path(self.override_path)
            # 3. Merge override into default (deep merge)
            self._deep_merge(self.config, override_config)

    def _deep_merge(self, base: Dict, new: Dict) -> Dict:
        """
        Recursively merges 'new' dictionary into 'base'.
        'new' values overwrite 'base' values.
        """
        for key, value in new.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a top-level configuration value.
        """
        return self.config.get(key, default)

    def get_provider_config_with_auth_mode(self, provider_name: str, auth_mode: Optional[str] = None) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Searches provider sections, finds the provider, and returns its config
        along with the determined authentication mode.
        """
        search_priority = self.get("provider_search_priority", ["no_auth_providers", "api_providers", "oauth_providers"])

        def find_in_section(section_name: str) -> Optional[Dict[str, Any]]:
            section = self.config.get(section_name, {})
            if provider_name in section:
                return section[provider_name]
            return None

        if auth_mode:
            section_name = f"{auth_mode}_providers"
            config = find_in_section(section_name)
            if config:
                return config, auth_mode
        else:
            for section_name in search_priority:
                config = find_in_section(section_name)
                if config:
                    derived_auth_mode = section_name.replace("_providers", "")
                    return config, derived_auth_mode
                    
        return None, None

# Example usage (for testing)
if __name__ == '__main__':
    # Test without override
    print("--- Loading Default Config ---")
    cm_default = ConfigManager()
    print(f"Default provider: {cm_default.get('provider')}")

    # Test with override
    # Create a dummy override file
    dummy_override_content = """
provider: "openai"
temperature: 0.9
"""
    dummy_path = Path("test_config.yaml")
    with open(dummy_path, "w") as f:
        f.write(dummy_override_content)

    print("\n--- Loading with Override Config ---")
    cm_override = ConfigManager(override_path=str(dummy_path))
    print(f"Overridden provider: {cm_override.get('provider')}")
    print(f"Overridden temperature: {cm_override.get('temperature')}")
    print(f"Default max_tokens (not overridden): {cm_override.get('max_tokens')}")

    # Clean up
    os.remove(dummy_path)
