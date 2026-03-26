# test/test_config.py
"""
Tests for ConfigManager: loading, merging, get/set, providers, lru_cache.
"""
import copy
import pytest
import yaml
from pathlib import Path

from k_ai.config import ConfigManager, _load_yaml
from k_ai.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_yaml(tmp_path: Path, data: dict, name="override.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Default config loading
# ---------------------------------------------------------------------------

class TestDefaultLoad:
    def test_default_paths_resolve_inside_fake_home(self, cm):
        fake_home = Path("~").expanduser().resolve()
        sessions_dir = Path(str(cm.get_nested("sessions", "directory"))).expanduser().resolve()
        memory_file = Path(str(cm.get_nested("memory", "internal_file"))).expanduser().resolve()
        persist_path = Path(str(cm.get_nested("config", "persist_path"))).expanduser().resolve()
        oauth_token = Path(str(cm.get_nested("oauth", "gemini", "token_path"))).expanduser().resolve()

        assert str(sessions_dir).startswith(str(fake_home))
        assert str(memory_file).startswith(str(fake_home))
        assert str(persist_path).startswith(str(fake_home))
        assert str(oauth_token).startswith(str(fake_home))

    def test_default_provider_exists(self, cm):
        provider = cm.get("provider")
        assert isinstance(provider, str) and len(provider) > 0

    def test_default_temperature(self, cm):
        assert cm.get("temperature") == 0.7

    def test_default_max_tokens(self, cm):
        assert cm.get("max_tokens") == 4096

    def test_default_stream(self, cm):
        assert cm.get("stream") is True

    def test_default_cli_section(self, cm):
        assert cm.get_nested("cli", "show_token_usage") is True
        assert cm.get_nested("cli", "show_tool_rationale") is True
        assert cm.get_nested("cli", "show_welcome_panel") is True
        assert cm.get_nested("cli", "theme") == "default"

    def test_no_auth_section_has_ollama(self, cm):
        ollama = cm.get_nested("no_auth", "ollama")
        assert ollama is not None
        assert "base_url" in ollama
        assert "default_model" in ollama

    def test_api_key_section_has_anthropic(self, cm):
        cfg = cm.get_nested("api_key", "anthropic")
        assert cfg["api_key_env_var"] == "ANTHROPIC_API_KEY"
        assert "default_model" in cfg

    def test_oauth_section_has_gemini(self, cm):
        cfg = cm.get_nested("oauth", "gemini")
        assert cfg["oauth_provider_name"] == "google"

    def test_provider_search_priority(self, cm):
        psp = cm.get("provider_search_priority")
        assert "no_auth" in psp
        assert "api_key" in psp
        assert "oauth" in psp

    def test_missing_key_returns_default(self, cm):
        assert cm.get("does_not_exist") is None
        assert cm.get("does_not_exist", "fallback") == "fallback"

    def test_get_nested_missing(self, cm):
        assert cm.get_nested("cli", "nonexistent") is None
        assert cm.get_nested("cli", "nonexistent", default="x") == "x"

    def test_get_nested_deep_miss(self, cm):
        assert cm.get_nested("zzz", "yyy", "xxx") is None

    def test_get_nested_null_value_not_confused_with_missing(self, cm):
        """Explicit null in config is returned, not treated as missing."""
        cm.config["_test_null"] = None
        assert cm.get_nested("_test_null") is None
        del cm.config["_test_null"]


# ---------------------------------------------------------------------------
# Override file
# ---------------------------------------------------------------------------

class TestOverrideFile:
    def test_override_top_level_value(self, tmp_path):
        p = write_yaml(tmp_path, {"temperature": 0.1, "provider": "mistral"})
        cm = ConfigManager(override_path=str(p))
        assert cm.get("temperature") == 0.1
        assert cm.get("provider") is not None

    def test_override_nested_value(self, tmp_path):
        p = write_yaml(tmp_path, {"cli": {"theme": "fancy"}})
        cm = ConfigManager(override_path=str(p))
        assert cm.get_nested("cli", "theme") == "fancy"
        # Keys not in override keep their defaults
        assert cm.get_nested("cli", "show_token_usage") is True

    def test_override_adds_new_no_auth_provider(self, tmp_path):
        p = write_yaml(tmp_path, {
            "no_auth": {
                "lmstudio": {
                    "base_url": "http://localhost:1234/v1",
                    "default_model": "local-model",
                    "context_window": 4096,
                }
            }
        })
        cm = ConfigManager(override_path=str(p))
        cfg = cm.get_nested("no_auth", "lmstudio")
        assert cfg["default_model"] == "local-model"
        # ollama must still be there (deep merge, not replace)
        assert cm.get_nested("no_auth", "ollama") is not None

    def test_nonexistent_override_file_uses_defaults(self):
        cm = ConfigManager(override_path="/does/not/exist.yaml")
        assert cm.get("provider") is not None

    def test_legacy_tool_override_paths_are_normalized(self, tmp_path):
        p = write_yaml(
            tmp_path,
            {
                "tools": {
                    "python_exec": {
                        "enabled": False,
                        "sandbox_dir": "/tmp/legacy-sandbox",
                    }
                }
            },
        )
        cm = ConfigManager(override_path=str(p))
        assert cm.get_nested("tools", "python", "enabled") is False
        assert cm.get_nested("tools", "python", "sandbox_dir") == "/tmp/legacy-sandbox"
        assert cm.get_nested("tools", "python_exec", "enabled") is False
        assert any("tools.python_exec" in item for item in cm.normalization_notes())


# ---------------------------------------------------------------------------
# Inline kwargs
# ---------------------------------------------------------------------------

class TestInlineKwargs:
    def test_provider_kwarg(self):
        cm = ConfigManager(provider="groq")
        assert cm.get("provider") == "groq"

    def test_temperature_kwarg(self):
        cm = ConfigManager(temperature=0.0)
        assert cm.get("temperature") == 0.0

    def test_model_kwarg(self):
        cm = ConfigManager(model="gpt-4o")
        assert cm.get("model") == "gpt-4o"

    def test_kwarg_overrides_file_overrides_default(self, tmp_path):
        p = write_yaml(tmp_path, {"temperature": 0.5})
        cm = ConfigManager(override_path=str(p), temperature=0.99)
        assert cm.get("temperature") == 0.99


# ---------------------------------------------------------------------------
# set() and coercion
# ---------------------------------------------------------------------------

class TestSet:
    def test_set_top_level(self, cm):
        cm.set("temperature", 0.3)
        assert cm.get("temperature") == 0.3

    def test_set_nested_dot_notation(self, cm):
        cm.set("cli.theme", "fancy")
        assert cm.get_nested("cli", "theme") == "fancy"

    def test_set_creates_intermediate_dict(self, cm):
        cm.set("new_section.new_key", "hello")
        assert cm.get_nested("new_section", "new_key") == "hello"

    def test_coerce_string_to_int(self, cm):
        cm.set("max_tokens", "1024")
        assert cm.get("max_tokens") == 1024
        assert isinstance(cm.get("max_tokens"), int)

    def test_coerce_string_to_float(self, cm):
        cm.set("temperature", "0.5")
        assert cm.get("temperature") == 0.5
        assert isinstance(cm.get("temperature"), float)

    def test_coerce_string_true_to_bool(self, cm):
        for truthy in ("true", "True", "TRUE", "1", "yes", "on"):
            cm.set("stream", truthy)
            assert cm.get("stream") is True

    def test_coerce_string_false_to_bool(self, cm):
        for falsy in ("false", "False", "0", "no", "off"):
            cm.set("stream", falsy)
            assert cm.get("stream") is False

    def test_no_coerce_when_existing_is_none(self, cm):
        """When existing value is None, value is set as-is."""
        cm.config["new_key"] = None
        cm.set("new_key", "hello")
        assert cm.get("new_key") == "hello"

    def test_no_coerce_for_non_string_value(self, cm):
        cm.set("max_tokens", 2048)
        assert cm.get("max_tokens") == 2048

    def test_invalid_coerce_keeps_string(self, cm):
        """If coercion fails, keep the string."""
        cm.set("max_tokens", "not-a-number")
        assert cm.get("max_tokens") == "not-a-number"

    def test_set_accepts_legacy_tool_path_and_writes_canonical_location(self, cm):
        cm.set("tools.exa_search.enabled", "false")
        assert cm.get_nested("tools", "exa", "enabled") is False
        assert cm.get_path("tools.exa_search.enabled") is False


# ---------------------------------------------------------------------------
# get_all and dump_yaml
# ---------------------------------------------------------------------------

class TestGetAllDump:
    def test_get_all_returns_deep_copy(self, cm):
        result = cm.get_all()
        result["provider"] = "MODIFIED"
        assert cm.get("provider") is not None

    def test_dump_yaml_is_valid_yaml(self, cm):
        text = cm.dump_yaml()
        parsed = yaml.safe_load(text)
        assert "provider" in parsed

    def test_get_default_yaml_contains_sections(self):
        text = ConfigManager.get_default_yaml()
        assert "no_auth" in text
        assert "api_key" in text
        assert "oauth" in text
        assert "ollama" in text

    def test_get_default_yaml_can_export_single_section(self):
        text = ConfigManager.get_default_yaml(sections=["ui"])
        assert "cli:" in text
        assert "prompts:" in text
        assert "provider:" not in text

    def test_list_default_sections_returns_named_fragments(self):
        sections = ConfigManager.list_default_sections()
        assert [section["name"] for section in sections] == ["models", "ui", "sessions", "governance"]

    def test_flatten_returns_dot_notation(self, cm):
        flat = cm.flatten("cli")
        assert "cli.show_token_usage" in flat

    def test_save_active_yaml_writes_file(self, tmp_path, cm):
        target = tmp_path / "active.yaml"
        written = cm.save_active_yaml(str(target))
        assert written == target
        assert target.exists()

    def test_resolve_edit_target_all_uses_active_override(self, tmp_path):
        override_path = tmp_path / "user-config.yaml"
        cm = ConfigManager(override_path=str(override_path))
        target = cm.resolve_edit_target("all")
        assert target == override_path
        assert target.exists()

    def test_resolve_edit_target_section_returns_fragment_path(self, cm):
        target = cm.resolve_edit_target("governance")
        assert target.name == "30-runtime-governance.yaml"
        assert target.exists()

    def test_resolve_editor_command_prefers_configured_editor(self, cm, monkeypatch):
        cm.set("config.editor", "micro --config-dir /tmp/micro")
        monkeypatch.setattr("k_ai.config.shutil.which", lambda name: "/usr/bin/micro" if name == "micro" else None)
        command = cm.resolve_editor_command()
        assert command == ["/usr/bin/micro", "--config-dir", "/tmp/micro"]

    def test_resolve_editor_command_falls_through_envs_until_valid(self, cm, monkeypatch):
        monkeypatch.setenv("K_AI_EDITOR", "missing-editor --wait")
        monkeypatch.setenv("VISUAL", "vim -f")
        monkeypatch.setenv("EDITOR", "nano")
        monkeypatch.setattr(
            "k_ai.config.shutil.which",
            lambda name: "/usr/bin/vim" if name == "vim" else None,
        )
        command = cm.resolve_editor_command()
        assert command == ["/usr/bin/vim", "-f"]

    def test_resolve_editor_command_raises_if_no_editor_exists(self, cm, monkeypatch):
        cm.set("config.editor", "missing-editor")
        monkeypatch.delenv("K_AI_EDITOR", raising=False)
        monkeypatch.delenv("VISUAL", raising=False)
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.setattr("k_ai.config.shutil.which", lambda name: None)
        with pytest.raises(FileNotFoundError):
            cm.resolve_editor_command()

    def test_validate_runtime_coherence_rejects_non_boolean_capability_enabled(self, cm):
        cm.config.setdefault("tools", {}).setdefault("python", {})["enabled"] = "yes"
        report = cm.validate_runtime_coherence()
        assert any("tools.python.enabled must be boolean" in item for item in report["errors"])

    def test_validate_runtime_coherence_warns_when_runtime_git_paths_are_misaligned(self, cm, tmp_path):
        cm.set("config.persist_path", str(tmp_path / "cfg" / "config.yaml"))
        cm.set("memory.internal_file", str(tmp_path / "mem" / "MEMORY.json"))
        cm.set("sessions.directory", str(tmp_path / "sessions" / "store"))
        report = cm.validate_runtime_coherence()
        assert any("must share the same parent" in item for item in report["warnings"])


# ---------------------------------------------------------------------------
# lru_cache
# ---------------------------------------------------------------------------

class TestLruCache:
    def test_cache_hits_on_repeated_loads(self):
        _load_yaml.cache_clear()
        path = ConfigManager().default_config_files[0]
        _load_yaml(path)
        _load_yaml(path)
        _load_yaml(path)
        info = _load_yaml.cache_info()
        assert info.hits >= 2
        assert info.misses == 1

    def test_defaults_are_split_into_multiple_fragments(self):
        cm = ConfigManager()
        assert cm.default_config_path.name == "defaults.d"
        assert len(cm.default_config_files) >= 2

    def test_deepcopy_prevents_cache_mutation(self):
        """Mutating one ConfigManager must not affect a second one."""
        cm1 = ConfigManager()
        cm2 = ConfigManager()
        cm1.set("provider", "MUTATED")
        assert cm2.get("provider") != "MUTATED"

    def test_cache_clear_forces_reload(self, tmp_path):
        p = write_yaml(tmp_path, {"provider": "v1"})
        _load_yaml.cache_clear()
        from k_ai.config import _load_yaml as lru_fn
        lru_fn(p)
        # Overwrite the file
        p.write_text(yaml.dump({"provider": "v2"}))
        # Without clear, still v1
        result_cached = lru_fn(p)
        assert result_cached["provider"] == "v1"
        # After clear, v2
        lru_fn.cache_clear()
        result_fresh = lru_fn(p)
        assert result_fresh["provider"] == "v2"


# ---------------------------------------------------------------------------
# list_providers
# ---------------------------------------------------------------------------

class TestListProviders:
    def test_returns_dict_with_auth_sections(self, cm):
        providers = cm.list_providers()
        assert "no_auth" in providers
        assert "api_key" in providers

    def test_no_auth_has_ollama(self, cm):
        assert "ollama" in cm.list_providers()["no_auth"]

    def test_api_key_sorted_alphabetically(self, cm):
        api_key = cm.list_providers()["api_key"]
        assert api_key == sorted(api_key)

    def test_empty_sections_excluded(self, cm):
        cm2 = ConfigManager()
        cm2.config["no_auth"] = {}
        providers = cm2.list_providers()
        assert "no_auth" not in providers

    def test_oauth_section_present(self, cm):
        assert "oauth" in cm.list_providers()
        assert "gemini" in cm.list_providers()["oauth"]


# ---------------------------------------------------------------------------
# get_provider_config_with_auth_mode
# ---------------------------------------------------------------------------

class TestGetProviderConfig:
    def test_find_ollama_in_no_auth(self, cm):
        cfg, mode = cm.get_provider_config_with_auth_mode("ollama")
        assert mode == "no_auth"
        assert cfg["base_url"] == "http://localhost:11434/v1"

    def test_find_anthropic_in_api_key(self, cm):
        cfg, mode = cm.get_provider_config_with_auth_mode("anthropic")
        assert mode == "api_key"
        assert cfg["api_key_env_var"] == "ANTHROPIC_API_KEY"

    def test_find_gemini_in_api_key_by_priority(self, cm):
        """api_key appears before oauth in search priority, so api_key wins."""
        cfg, mode = cm.get_provider_config_with_auth_mode("gemini")
        assert mode == "api_key"

    def test_find_gemini_force_oauth(self, cm):
        cfg, mode = cm.get_provider_config_with_auth_mode("gemini", auth_mode="oauth")
        assert mode == "oauth"
        assert cfg["oauth_provider_name"] == "google"

    def test_unknown_provider_returns_none(self, cm):
        cfg, mode = cm.get_provider_config_with_auth_mode("nonexistent_provider")
        assert cfg is None
        assert mode is None

    def test_forced_auth_mode_not_found_returns_none(self, cm):
        cfg, mode = cm.get_provider_config_with_auth_mode("ollama", auth_mode="api_key")
        assert cfg is None
        assert mode is None

    def test_search_priority_order(self, cm):
        """Provider in no_auth is found before api_key even if it appeared in both."""
        cm2 = ConfigManager()
        cm2.config.setdefault("api_key", {})["ollama"] = {"default_model": "x", "api_key_env_var": "X"}
        cfg, mode = cm2.get_provider_config_with_auth_mode("ollama")
        # no_auth searched first
        assert mode == "no_auth"
