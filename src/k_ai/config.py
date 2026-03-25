# src/k_ai/config.py
"""
Configuration management for k-ai.
Handles loading default and user-provided configurations with a clear override hierarchy.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# The name of the default configuration file included with the package
DEFAULT_CONFIG_FILENAME = "default_config.yaml"

class ConfigManager:
    def __init__(self, override_path: Optional[str] = None):
        """
        Initializes the ConfigManager.

        Args:
            override_path: Optional path to a user-provided config.yaml.
                           If provided, it will be merged on top of the default config.
        """
        self.default_config_path = Path(__file__).parent / "defaults" / DEFAULT_CONFIG_FILENAME
        self.override_path = Path(override_path).expanduser() if override_path else None
        
        self.config: Dict[str, Any] = {}
        self._load_and_merge()

    def _load_and_merge(self):
        """
        Loads the default configuration and merges the override config on top.
        """
        # 1. Load default config
        if not self.default_config_path.exists():
            raise FileNotFoundError(f"Default configuration file not found at {self.default_config_path}")
            
        with open(self.default_config_path, 'r') as f:
            default_config = yaml.safe_load(f)
        
        if not isinstance(default_config, dict):
            raise TypeError("Default configuration is not a valid dictionary.")
            
        self.config = default_config

        # 2. Load override config if it exists
        if self.override_path and self.override_path.exists():
            with open(self.override_path, 'r') as f:
                override_config = yaml.safe_load(f)
            
            if not isinstance(override_config, dict):
                print(f"Warning: Override config at {self.override_path} is not a valid dictionary. Skipping.")
                return

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

    def get_provider_config(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """
        Searches all '*_providers' sections and returns the config for a specific provider.
        """
        for key, section in self.config.items():
            if key.endswith("_providers") and isinstance(section, dict):
                if provider_name in section:
                    return section[provider_name]
        return None

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
