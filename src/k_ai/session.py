# src/k_ai/session.py
"""
Manages an interactive chat session with consciousness:
  - Session persistence (auto-save every message)
  - Boot flow (proactive greeting, session resume detection)
  - Agentic tool loop with human-in-the-loop confirmation
  - Context compaction when approaching the context window
  - Memory integration (external read-only + internal read-write)
"""
from pathlib import Path
from contextlib import contextmanager
import json
import os
import re
import select
import signal
import sys
import termios
import threading
import tty
from typing import Any, Awaitable, Callable, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel as RichPanel
from rich.prompt import Confirm
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import WordCompleter, FuzzyCompleter, ConditionalCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.styles import Style as PTStyle

from .config import ConfigManager
from .llm_core import get_provider, LLMProvider
from .models import (
    Message, MessageRole, ToolCall, TokenUsage, SessionMetadata, ToolResult,
)
from .memory import MemoryStore, load_external_memory
from .runtime_git import commit_runtime_state
from .session_store import SessionStore
from .tools import create_registry, ToolRegistry
from .tools.base import ToolContext
from .ui import (
    StreamingRenderer,
    render_assistant_panel,
    render_notice,
    render_runtime_panel,
    render_sessions_table,
    render_tool_proposal,
    render_tool_result,
    render_user_panel,
)
from .commands import CommandHandler, SLASH_COMMANDS
from .exceptions import LLMError, ProviderAuthenticationError, ContextLengthExceededError
from .tool_capabilities import capability_enabled, capability_for_tool, list_capabilities, normalize_capability_name


# Max rounds in the agentic tool loop per user message
_MAX_TOOL_ROUNDS = 10
_MAX_EMPTY_ASSISTANT_ROUNDS = 2
_INTERRUPTED_RESPONSE_MARKER = "[Response interrupted by user]"


class ChatSession:
    """
    Encapsulates a complete interactive chat session.

    Can be used as a standalone CLI (via ``await session.start()``) or driven
    programmatically by calling ``await session.send(message)``.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.cm = config_manager
        self.console = Console()

        initial_model = model if model is not None else (self.cm.get("model") or None)
        self.llm: LLMProvider = get_provider(self.cm, provider=provider, model=initial_model)
        self.history: List[Message] = []
        self.system_prompt: Optional[str] = None
        self.total_usage = TokenUsage()

        sessions_dir = self.cm.get_nested("sessions", "directory", default="~/.k-ai/sessions")
        self.session_store = SessionStore(sessions_dir)
        self.session_store.init()

        mem_path = self.cm.get_nested("memory", "internal_file", default="~/.k-ai/MEMORY.json")
        self.memory = MemoryStore(Path(mem_path))
        self.memory.load()

        ext_path = self.cm.get_nested("memory", "external_file", default="")
        self.external_memory = load_external_memory(ext_path)

        self._session_id: Optional[str] = None
        self._exit_requested: bool = False
        self._new_session_requested: bool = False
        self._load_session_id: Optional[str] = None
        self._load_session_last_n: Optional[int] = None
        self._compact_requested: bool = False
        self._init_requested: bool = False
        self._init_mode: bool = False
        self._interrupt_requested: bool = False
        self._prompt_interrupt_count: int = 0
        self._queued_new_session_seed: Optional[Dict[str, Any]] = None
        self._queued_new_session_message: Optional[str] = None
        self._continuation_after_switch: Optional[Dict[str, str]] = None
        self._turn_session_guidance: Optional[str] = None
        self._session_tool_policy_overrides: Dict[str, Dict[str, str]] = {
            "tools": {},
            "categories": {},
            "risks": {},
        }

        self._tool_ctx = ToolContext(
            config=self.cm,
            memory=self.memory,
            session_store=self.session_store,
            console=self.console,
            get_history=lambda: self.history,
            set_history=lambda h: setattr(self, "history", h),
            get_session_id=lambda: self._session_id,
            set_session_id=lambda sid: setattr(self, "_session_id", sid),
            get_system_prompt=lambda: self.system_prompt,
            reload_provider=lambda **kw: self.reload_provider(**kw),
            request_exit=lambda: setattr(self, "_exit_requested", True),
            request_new_session=self._handle_new_session,
            request_load_session=self._queue_load_session,
            request_compact=lambda: setattr(self, "_compact_requested", True),
            request_init=self.request_init,
            complete_init=self.complete_init,
            apply_config_change=self.apply_config_change,
            generate_session_digest=self.generate_session_digest,
            get_runtime_snapshot=self.get_runtime_snapshot,
            get_tool_policy_overview=self.get_tool_policy_overview,
            update_tool_policy=self.update_tool_policy,
            reset_tool_policy=self.reset_tool_policy,
            get_tool_capability_overview=self.get_tool_capability_overview,
            update_tool_capability=self.update_tool_capability,
            is_interrupt_requested=lambda: self._interrupt_requested,
        )
        self.tool_registry: ToolRegistry = create_registry(self._tool_ctx)
        self._apply_tool_catalog()
        self.command_handler = CommandHandler(self)

    # ------------------------------------------------------------------
    # Tool definitions (filter disabled tools)
    # ------------------------------------------------------------------

    def _get_active_tools(self, excluded_tools: Optional[set[str]] = None) -> list:
        """Return OpenAI tool definitions excluding disabled tools."""
        excluded = excluded_tools or set()
        return [
            t.to_openai_tool() for t in self.tool_registry.list_tools()
            if self.is_tool_enabled(t.name) and t.name not in excluded
        ]

    def is_tool_enabled(self, tool_name: str) -> bool:
        return capability_enabled(self.cm, capability_for_tool(tool_name) or tool_name) if capability_for_tool(tool_name) else True

    @property
    def _debug(self) -> bool:
        return bool(self.cm.get_nested("cli", "debug", default=False))

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def reload_provider(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        effective_provider = provider or self.llm.provider_name
        effective_model = model if model is not None else (self.cm.get("model") or None)
        self.llm = get_provider(self.cm, provider=effective_provider, model=effective_model)

    def provider_default_model(self, provider: Optional[str] = None) -> Optional[str]:
        effective_provider = provider or self.cm.get("provider") or self.llm.provider_name
        cfg, _auth_mode = self.cm.get_provider_config_with_auth_mode(str(effective_provider))
        if not cfg:
            return None
        default_model = cfg.get("default_model")
        return str(default_model) if default_model else None

    def history_has_nonportable_tool_state(self) -> bool:
        return any(
            message.role == MessageRole.TOOL or bool(message.tool_calls)
            for message in self.history
        )

    def preserve_simple_history_only(self) -> int:
        filtered = [
            message
            for message in self.history
            if message.role in {MessageRole.USER, MessageRole.ASSISTANT} and not message.tool_calls
        ]
        removed = len(self.history) - len(filtered)
        if removed <= 0:
            return 0
        self.history = filtered
        if self._session_id:
            self.session_store.rewrite_messages(self._session_id, self.history)
        return removed

    def _estimate_context_tokens(self, messages: Optional[List[Message]] = None) -> int:
        payload = messages if messages is not None else self.history
        return sum(len(m.content) for m in payload) // 4

    def _assistant_completion_estimate(self) -> int:
        return sum(len(m.content) for m in self.history if m.role == MessageRole.ASSISTANT) // 4

    def get_token_snapshot(self) -> Dict[str, Any]:
        actual_prompt = int(self.total_usage.prompt_tokens or 0)
        actual_completion = int(self.total_usage.completion_tokens or 0)
        actual_total = int(self.total_usage.total_tokens or 0)
        estimated_prompt = self._estimate_context_tokens(self._messages_with_system())
        estimated_completion = self._assistant_completion_estimate()
        estimated_total = self._estimate_context_tokens(self.history)

        if actual_total > 0:
            return {
                "prompt_tokens": actual_prompt,
                "completion_tokens": actual_completion,
                "total_tokens": actual_total,
                "token_source": "provider",
            }
        return {
            "prompt_tokens": estimated_prompt,
            "completion_tokens": estimated_completion,
            "total_tokens": estimated_total,
            "token_source": "estimated",
        }

    def _sync_session_totals(self) -> None:
        if not self._session_id:
            return
        snapshot = self.get_token_snapshot()
        self.session_store.update_meta(
            self._session_id,
            total_tokens=int(snapshot.get("total_tokens", 0) or 0),
        )

    def get_runtime_snapshot(self) -> Dict[str, Any]:
        ctx_window = int(self.llm.provider_config.get("context_window", 128000) or 128000)
        used = self._estimate_context_tokens()
        compaction_pct = int(self.cm.get_nested("compaction", "trigger_percent", default=80) or 0)
        threshold = int(ctx_window * compaction_pct / 100) if compaction_pct > 0 else 0
        remaining = max(ctx_window - used, 0)
        percent = round((used / ctx_window) * 100, 1) if ctx_window else 0.0
        session_meta = self.session_store.get_session(self._session_id) if self._session_id else None
        token_snapshot = self.get_token_snapshot()
        tool_policy = self.get_tool_policy_overview()
        persist_path = (
            str(self.cm.override_path.expanduser())
            if self.cm.override_path
            else str(Path(self.cm.get_nested("config", "persist_path", default="~/.k-ai/config.yaml")).expanduser())
        )
        return {
            "provider": self.llm.provider_name,
            "model": self.llm.model_name,
            "auth_mode": self.llm.auth_mode or "n/a",
            "temperature": self.cm.get("temperature"),
            "max_tokens": self.cm.get("max_tokens"),
            "stream": self.cm.get("stream"),
            "render_mode": self.cm.get_nested("cli", "render_mode", default="rich"),
            "session_id": self._session_id or "",
            "session_type": session_meta.session_type if session_meta else "",
            "session_summary": session_meta.summary if session_meta else "",
            "session_themes": list(session_meta.themes) if session_meta else [],
            "history_messages": len(self.history),
            "context_window": ctx_window,
            "estimated_context_tokens": used,
            "remaining_context_tokens": remaining,
            "context_percent": percent,
            "compaction_trigger_percent": compaction_pct,
            "compaction_trigger_tokens": threshold,
            "tool_result_max_display": self.cm.get_nested("cli", "tool_result_max_display", default=500),
            "tool_result_max_history": self.cm.get_nested("cli", "tool_result_max_history", default=4000),
            "confirm_all_tools": self.cm.get_nested("cli", "confirm_all_tools", default=True),
            "show_runtime_stats": self.cm.get_nested("cli", "show_runtime_stats", default=True),
            "runtime_stats_mode": self.cm.get_nested("cli", "runtime_stats_mode", default="compact"),
            "approval_defaults": tool_policy["defaults_by_risk"],
            "approval_counts": tool_policy["counts"],
            "tool_capabilities": self.get_tool_capability_overview()["rows"],
            "prompt_tokens": token_snapshot["prompt_tokens"],
            "completion_tokens": token_snapshot["completion_tokens"],
            "total_tokens": token_snapshot["total_tokens"],
            "token_source": token_snapshot["token_source"],
            "provider_prompt_tokens": self.total_usage.prompt_tokens,
            "provider_completion_tokens": self.total_usage.completion_tokens,
            "provider_total_tokens": self.total_usage.total_tokens,
            "persist_path": persist_path,
        }

    def apply_config_change(self, key: str, value: Any, persist: bool = False) -> Dict[str, Any]:
        if key == "tool_approval" or key.startswith("tool_approval."):
            raise ValueError("Use the dedicated tool approval admin tools to change tool_approval settings.")
        current = self.cm.get_path(key)
        applied = self.cm.set(key, value)
        saved_to: Optional[str] = None
        try:
            if key in {"provider", "model"}:
                if key == "provider" and current != applied and self.history_has_nonportable_tool_state():
                    self.preserve_simple_history_only()
                provider_name = self.cm.get("provider")
                model_name = self.cm.get("model") or None
                self.reload_provider(provider=provider_name, model=model_name)
            if persist:
                saved_to = str(self.cm.save_active_yaml())
        except Exception:
            self.cm.set(key, current)
            if key in {"provider", "model"}:
                self.reload_provider(
                    provider=self.cm.get("provider"),
                    model=self.cm.get("model") or None,
                )
            raise
        return {"key": key, "old_value": current, "value": applied, "saved_to": saved_to}

    def get_tool_capability_overview(self) -> Dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for item in list_capabilities():
            enabled = capability_enabled(self.cm, item["name"])
            rows.append(
                {
                    "capability": item["name"],
                    "label": item["label"],
                    "enabled": enabled,
                    "mutable": bool(item["mutable"]),
                    "tools": list(item["tools"]),
                    "description": item["description"],
                }
            )
        return {
            "rows": rows,
            "counts": {
                "enabled": sum(1 for row in rows if row["enabled"]),
                "disabled": sum(1 for row in rows if not row["enabled"]),
                "mutable": sum(1 for row in rows if row["mutable"]),
            },
        }

    def update_tool_capability(
        self,
        capability: str,
        enabled: bool,
        persist: bool | None = None,
    ) -> Dict[str, Any]:
        normalized = normalize_capability_name(capability)
        current = capability_enabled(self.cm, normalized)
        saved_to: Optional[str] = None
        should_persist = True if persist is None else bool(persist)
        try:
            self.cm.set(f"tools.{normalized}.enabled", bool(enabled))
            if should_persist:
                saved_to = str(self.cm.save_active_yaml())
        except Exception:
            self.cm.set(f"tools.{normalized}.enabled", current)
            raise
        return {
            "capability": normalized,
            "previous": current,
            "enabled": bool(enabled),
            "saved_to": saved_to,
        }

    def save_active_config(self, path: Optional[str] = None) -> str:
        return str(self.cm.save_active_yaml(path))

    @staticmethod
    def _normalize_session_type(value: Optional[str]) -> str:
        normalized = str(value or "classic").strip().lower()
        if normalized not in {"classic", "meta"}:
            return "classic"
        return normalized

    def _tool_catalog_entries(self) -> Dict[str, Dict[str, Any]]:
        raw_catalog = self.cm.get_nested("tool_approval", "catalog", default={}) or {}
        if not isinstance(raw_catalog, dict):
            raise ValueError("tool_approval.catalog must be a mapping of groups to tool entries.")

        catalog: Dict[str, Dict[str, Any]] = {}
        for group_name, group_entries in raw_catalog.items():
            if not isinstance(group_entries, dict):
                raise ValueError(f"tool_approval.catalog.{group_name} must be a mapping.")
            for tool_name, entry in group_entries.items():
                if tool_name in catalog:
                    raise ValueError(f"Duplicate tool catalog entry: {tool_name}")
                if not isinstance(entry, dict):
                    raise ValueError(f"tool_approval.catalog.{group_name}.{tool_name} must be a mapping.")
                category = str(entry.get("category", "") or "").strip()
                risk = str(entry.get("risk", "") or "").strip().lower()
                default_policy = str(entry.get("default_policy", "") or "").strip().lower()
                protected = entry.get("protected", False)
                description = str(entry.get("description", "") or "").strip()
                if not category:
                    raise ValueError(f"tool_approval.catalog.{group_name}.{tool_name}.category is required.")
                if risk not in {"low", "medium", "high", "critical"}:
                    raise ValueError(f"tool_approval.catalog.{group_name}.{tool_name}.risk must be low|medium|high|critical.")
                if default_policy not in {"ask", "auto"}:
                    raise ValueError(f"tool_approval.catalog.{group_name}.{tool_name}.default_policy must be ask|auto.")
                if not isinstance(protected, bool):
                    raise ValueError(f"tool_approval.catalog.{group_name}.{tool_name}.protected must be boolean.")
                catalog[tool_name] = {
                    "group": str(group_name),
                    "category": category,
                    "risk": risk,
                    "default_policy": default_policy,
                    "protected": protected,
                    "description": description,
                }
        return catalog

    def _apply_tool_catalog(self) -> None:
        catalog = self._tool_catalog_entries()
        registry_names = set(self.tool_registry.get_names())
        catalog_names = set(catalog.keys())
        missing = sorted(registry_names - catalog_names)
        extra = sorted(catalog_names - registry_names)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing from catalog: {', '.join(missing)}")
            if extra:
                details.append(f"unknown in catalog: {', '.join(extra)}")
            raise ValueError("tool_approval.catalog is out of sync with the registered tools: " + "; ".join(details))

        for tool_name, entry in catalog.items():
            tool = self.tool_registry.get(tool_name)
            if not tool:
                continue
            tool.category = entry["category"]
            tool.danger_level = entry["risk"]

    def _tool_approval_protected_names(self) -> set[str]:
        return {name for name, entry in self._tool_catalog_entries().items() if entry.get("protected")}

    def _tool_policy_defaults(self) -> Dict[str, str]:
        defaults: Dict[str, str] = {}
        for entry in self._tool_catalog_entries().values():
            risk = entry["risk"]
            policy = entry["default_policy"]
            existing = defaults.get(risk)
            if existing and existing != policy:
                raise ValueError(
                    f"tool_approval.catalog defines conflicting default_policy values for risk '{risk}': "
                    f"{existing!r} vs {policy!r}"
                )
            defaults[risk] = policy
        return defaults

    def _tool_policy_global_overrides(self) -> Dict[str, Dict[str, str]]:
        raw = self.cm.get_nested("tool_approval", "global_overrides", default={}) or {}
        if not isinstance(raw, dict):
            raise ValueError("tool_approval.global_overrides must be a mapping.")
        result: Dict[str, Dict[str, str]] = {"tools": {}, "categories": {}, "risks": {}}
        extra_buckets = sorted(set(raw) - set(result))
        if extra_buckets:
            raise ValueError(
                "tool_approval.global_overrides contains unknown buckets: "
                + ", ".join(extra_buckets)
            )
        catalog = self._tool_catalog_entries()
        valid_categories = {entry["category"] for entry in catalog.values()}
        valid_risks = {"low", "medium", "high", "critical"}
        protected = self._tool_approval_protected_names()
        for bucket in result:
            values = raw.get(bucket, {})
            if values in (None, {}):
                continue
            if not isinstance(values, dict):
                raise ValueError(f"tool_approval.global_overrides.{bucket} must be a mapping.")
            for key, value in values.items():
                normalized_key = str(key).strip().lower()
                if not normalized_key:
                    raise ValueError(f"tool_approval.global_overrides.{bucket} contains an empty key.")
                normalized = str(value).strip().lower()
                if normalized not in {"ask", "auto"}:
                    raise ValueError(
                        f"tool_approval.global_overrides.{bucket}.{normalized_key} must be ask|auto."
                    )
                if bucket == "tools":
                    if normalized_key not in catalog:
                        raise ValueError(f"Unknown tool override target: {normalized_key}")
                    if normalized_key in protected:
                        raise ValueError(
                            f"Protected tool '{normalized_key}' cannot be overridden globally."
                        )
                elif bucket == "categories" and normalized_key not in valid_categories:
                    raise ValueError(f"Unknown tool category override target: {normalized_key}")
                elif bucket == "risks" and normalized_key not in valid_risks:
                    raise ValueError(f"Unknown risk override target: {normalized_key}")
                result[bucket][normalized_key] = normalized
        return result

    def _tool_policy_bucket_name(self, target_kind: str) -> str:
        kind = str(target_kind or "tool").strip().lower()
        mapping = {"tool": "tools", "category": "categories", "risk": "risks"}
        if kind not in mapping:
            raise ValueError("target_kind must be one of: tool, category, risk")
        return mapping[kind]

    def _normalize_tool_policy_target(self, target_kind: str, target: str) -> str:
        normalized = str(target or "").strip().lower()
        if not normalized:
            raise ValueError("target cannot be empty")
        if target_kind == "tool":
            tool = self.tool_registry.get(normalized)
            if not tool:
                raise ValueError(f"Unknown tool: {target}")
            return tool.name
        if target_kind == "category":
            categories = {entry["category"] for entry in self._tool_catalog_entries().values()}
            if normalized not in categories:
                raise ValueError(f"Unknown tool category: {target}")
            return normalized
        if target_kind == "risk":
            if normalized not in {"low", "medium", "high", "critical"}:
                raise ValueError(f"Unknown risk level: {target}")
            return normalized
        raise ValueError("target_kind must be one of: tool, category, risk")

    def _resolve_tool_policy(self, tool) -> Dict[str, Any]:
        protected_names = self._tool_approval_protected_names()
        if tool.name in protected_names:
            return {
                "policy": "ask",
                "source": "protected",
                "scope": "protected",
                "target_kind": "tool",
                "target": tool.name,
                "protected": True,
                "default_policy": self._tool_policy_defaults().get(tool.danger_level, "ask"),
                "session_override": None,
                "global_override": None,
            }

        bucket_values = {
            "tools": tool.name,
            "categories": tool.category,
            "risks": tool.danger_level,
        }
        defaults = self._tool_policy_defaults()
        session_overrides = self._session_tool_policy_overrides
        global_overrides = self._tool_policy_global_overrides()

        session_override = (
            session_overrides["tools"].get(tool.name)
            or session_overrides["categories"].get(tool.category)
            or session_overrides["risks"].get(tool.danger_level)
        )
        global_override = (
            global_overrides["tools"].get(tool.name)
            or global_overrides["categories"].get(tool.category)
            or global_overrides["risks"].get(tool.danger_level)
        )
        default_policy = defaults.get(tool.danger_level, "ask")

        if session_overrides["tools"].get(tool.name):
            return {
                "policy": session_overrides["tools"][tool.name],
                "source": "session",
                "scope": "session",
                "target_kind": "tool",
                "target": tool.name,
                "protected": False,
                "default_policy": default_policy,
                "session_override": session_override,
                "global_override": global_override,
            }
        if session_overrides["categories"].get(tool.category):
            return {
                "policy": session_overrides["categories"][tool.category],
                "source": "session",
                "scope": "session",
                "target_kind": "category",
                "target": tool.category,
                "protected": False,
                "default_policy": default_policy,
                "session_override": session_override,
                "global_override": global_override,
            }
        if session_overrides["risks"].get(tool.danger_level):
            return {
                "policy": session_overrides["risks"][tool.danger_level],
                "source": "session",
                "scope": "session",
                "target_kind": "risk",
                "target": tool.danger_level,
                "protected": False,
                "default_policy": default_policy,
                "session_override": session_override,
                "global_override": global_override,
            }
        if global_overrides["tools"].get(tool.name):
            return {
                "policy": global_overrides["tools"][tool.name],
                "source": "global",
                "scope": "global",
                "target_kind": "tool",
                "target": tool.name,
                "protected": False,
                "default_policy": default_policy,
                "session_override": session_override,
                "global_override": global_override,
            }
        if global_overrides["categories"].get(tool.category):
            return {
                "policy": global_overrides["categories"][tool.category],
                "source": "global",
                "scope": "global",
                "target_kind": "category",
                "target": tool.category,
                "protected": False,
                "default_policy": default_policy,
                "session_override": session_override,
                "global_override": global_override,
            }
        if global_overrides["risks"].get(tool.danger_level):
            return {
                "policy": global_overrides["risks"][tool.danger_level],
                "source": "global",
                "scope": "global",
                "target_kind": "risk",
                "target": tool.danger_level,
                "protected": False,
                "default_policy": default_policy,
                "session_override": session_override,
                "global_override": global_override,
            }
        return {
            "policy": default_policy,
            "source": "default",
            "scope": "default",
            "target_kind": "risk",
            "target": tool.danger_level,
            "protected": False,
            "default_policy": default_policy,
            "session_override": session_override,
            "global_override": global_override,
        }

    def get_tool_policy_overview(
        self,
        filter_policy: Optional[str] = None,
        filter_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        defaults = self._tool_policy_defaults()
        protected = self._tool_approval_protected_names()
        global_overrides = self._tool_policy_global_overrides()
        for tool in sorted(self.tool_registry.list_tools(), key=lambda item: (item.category, item.name)):
            resolved = self._resolve_tool_policy(tool)
            rows.append({
                "tool": tool.name,
                "display_name": tool.display_name or tool.name,
                "category": tool.category,
                "risk": tool.danger_level,
                "default_policy": defaults.get(tool.danger_level, "ask"),
                "session_override": (
                    self._session_tool_policy_overrides["tools"].get(tool.name)
                    or self._session_tool_policy_overrides["categories"].get(tool.category)
                    or self._session_tool_policy_overrides["risks"].get(tool.danger_level)
                ),
                "global_override": (
                    global_overrides["tools"].get(tool.name)
                    or global_overrides["categories"].get(tool.category)
                    or global_overrides["risks"].get(tool.danger_level)
                ),
                "effective_policy": resolved["policy"],
                "source": resolved["source"],
                "protected": tool.name in protected,
            })

        if filter_policy:
            rows = [row for row in rows if row["effective_policy"] == filter_policy]
        if filter_source:
            rows = [row for row in rows if row["source"] == filter_source]

        return {
            "rows": rows,
            "defaults_by_risk": defaults,
            "protected_tools": sorted(protected),
            "counts": {
                "ask": sum(1 for row in rows if row["effective_policy"] == "ask"),
                "auto": sum(1 for row in rows if row["effective_policy"] == "auto"),
                "protected": sum(1 for row in rows if row["protected"]),
                "session_overrides": sum(1 for row in rows if row["source"] == "session"),
                "global_overrides": sum(1 for row in rows if row["source"] == "global"),
            },
        }

    def update_tool_policy(
        self,
        target_kind: str,
        target: str,
        policy: str,
        scope: str = "session",
        persist: Optional[bool] = None,
    ) -> Dict[str, Any]:
        kind = str(target_kind or "tool").strip().lower()
        scope = str(scope or "session").strip().lower()
        normalized_target = self._normalize_tool_policy_target(kind, target)
        normalized_policy = str(policy or "").strip().lower()
        if normalized_policy not in {"ask", "auto"}:
            raise ValueError("policy must be one of: ask, auto")
        if scope not in {"session", "global"}:
            raise ValueError("scope must be one of: session, global")
        if kind == "tool" and normalized_target in self._tool_approval_protected_names():
            raise ValueError(f"Tool policy for '{normalized_target}' is protected and always requires approval.")

        bucket = self._tool_policy_bucket_name(kind)
        previous = None
        saved_to = None
        if scope == "session":
            previous = self._session_tool_policy_overrides[bucket].get(normalized_target)
            self._session_tool_policy_overrides[bucket][normalized_target] = normalized_policy
        else:
            config_path = f"tool_approval.global_overrides.{bucket}.{normalized_target}"
            previous = self.cm.get_path(config_path, default=None)
            self.cm.set(config_path, normalized_policy)
            if persist is not False:
                saved_to = self.save_active_config()

        return {
            "target_kind": kind,
            "target": normalized_target,
            "policy": normalized_policy,
            "scope": scope,
            "previous": previous,
            "saved_to": saved_to,
        }

    def reset_tool_policy(
        self,
        target_kind: str,
        target: str,
        scope: str = "session",
        persist: Optional[bool] = None,
    ) -> Dict[str, Any]:
        kind = str(target_kind or "tool").strip().lower()
        scope = str(scope or "session").strip().lower()
        normalized_target = self._normalize_tool_policy_target(kind, target)
        if scope not in {"session", "global"}:
            raise ValueError("scope must be one of: session, global")
        if kind == "tool" and normalized_target in self._tool_approval_protected_names():
            raise ValueError(f"Tool policy for '{normalized_target}' is protected and cannot be reset.")

        bucket = self._tool_policy_bucket_name(kind)
        removed = False
        previous = None
        saved_to = None
        if scope == "session":
            previous = self._session_tool_policy_overrides[bucket].get(normalized_target)
            removed = normalized_target in self._session_tool_policy_overrides[bucket]
            self._session_tool_policy_overrides[bucket].pop(normalized_target, None)
        else:
            config_path = f"tool_approval.global_overrides.{bucket}.{normalized_target}"
            previous = self.cm.get_path(config_path, default=None)
            removed = self.cm.delete_path(config_path)
            if removed and persist is not False:
                saved_to = self.save_active_config()

        return {
            "target_kind": kind,
            "target": normalized_target,
            "scope": scope,
            "previous": previous,
            "removed": removed,
            "saved_to": saved_to,
        }

    def _print_runtime_snapshot(self, title: str = "Runtime Transparency") -> None:
        if not self.cm.get_nested("cli", "show_runtime_stats", default=True):
            return
        mode = self.cm.get_nested("cli", "runtime_stats_mode", default="compact")
        theme_name = self.cm.get_nested("cli", "theme", default="default")
        self.console.print(render_runtime_panel(self.get_runtime_snapshot(), title=title, mode=mode, theme_name=theme_name))

    @contextmanager
    def _interrupt_scope(self, allow_escape: bool = True):
        self._interrupt_requested = False
        stop_event = threading.Event()
        watcher: Optional[threading.Thread] = None
        fd: Optional[int] = None
        old_attrs = None
        old_handler = None

        def _trigger_interrupt() -> None:
            self._interrupt_requested = True

        if threading.current_thread() is threading.main_thread():
            old_handler = signal.getsignal(signal.SIGINT)

            def _handler(signum, frame):
                _trigger_interrupt()

            signal.signal(signal.SIGINT, _handler)

        if allow_escape and sys.stdin.isatty():
            try:
                fd = sys.stdin.fileno()
                old_attrs = termios.tcgetattr(fd)
                tty.setcbreak(fd)

                def _watch_input() -> None:
                    while not stop_event.is_set():
                        ready, _, _ = select.select([fd], [], [], 0.1)
                        if not ready:
                            continue
                        char = os.read(fd, 1)
                        if char in (b"\x1b", b"\x03"):
                            _trigger_interrupt()
                            break

                watcher = threading.Thread(target=_watch_input, daemon=True)
                watcher.start()
            except Exception:
                fd = None
                old_attrs = None

        try:
            yield
        finally:
            stop_event.set()
            if watcher and watcher.is_alive():
                watcher.join(timeout=0.2)
            if fd is not None and old_attrs is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
                except Exception:
                    pass
            if old_handler is not None:
                signal.signal(signal.SIGINT, old_handler)

    def _handle_prompt_interrupt(self) -> bool:
        self._prompt_interrupt_count += 1
        if self._prompt_interrupt_count >= 2:
            return True
        render_notice(
            self.console,
            self._notice_value(
                "prompt_interrupt_message",
                "Prompt input was interrupted. Press Ctrl+C a second time to exit, or keep typing.",
            ),
            level="warning",
            title=self._notice_value("prompt_interrupt_title", "Interrupt"),
        )
        return False

    def _reset_prompt_interrupts(self) -> None:
        self._prompt_interrupt_count = 0

    # ------------------------------------------------------------------
    # System prompt builder
    # ------------------------------------------------------------------

    def _build_system_prompt(self, include_sessions: bool = False) -> str:
        parts: List[str] = []

        parts.append(
            "## Instruction Priority Rule\n"
            "If instructions conflict, always apply this order of precedence:\n"
            "1. Internal remembered facts managed through memory_add / memory_remove.\n"
            "2. Built-in and session-specific internal prompts/instructions.\n"
            "3. External user context loaded from a file.\n"
            "When internal remembered facts conflict with any prompt instruction or external context, treat internal remembered facts as the source of truth."
        )

        if self.memory.entries:
            entries_text = "\n".join(f"- {e.text}" for e in self.memory.entries)
            parts.append(f"## Remembered Facts\n{entries_text}")

        identity = self._render_prompt_template(self.cm.get_nested("prompts", "identity", default=(
            "You are {assistant_name}, an intelligent CLI chat assistant."
        )))
        parts.append(identity)

        if self.external_memory:
            parts.append(f"## User Context\n{self.external_memory}")

        if self._session_id:
            active = self.session_store.get_session(self._session_id)
            if active:
                active_summary = active.summary or (active.title if active.title != active.id else "(untitled)")
                active_themes = ", ".join(active.themes[:6]) if active.themes else "-"
                parts.append(
                    "## Active Session Profile\n"
                    f"- id: {active.id[:8]}\n"
                    f"- type: {active.session_type}\n"
                    f"- summary: {active_summary}\n"
                    f"- themes: {active_themes}\n"
                    "Keep this session semantically homogeneous. If the user clearly changes to a different dominant intent, "
                    "propose switch_session instead of mixing unrelated topics into the current session."
                )

        if self._continuation_after_switch:
            source_summary = self._continuation_after_switch.get("summary", "").strip() or "(unspecified)"
            source_reason = self._continuation_after_switch.get("reason", "").strip() or "(no reason provided)"
            parts.append(
                "## Continuation After Session Switch\n"
                + self._render_prompt_template(
                    self.cm.get_nested(
                        "prompts",
                        "continuation_after_switch",
                        default=(
                            "A session switch has already been approved and completed for the current carried-over user request. "
                            "You are already inside the target session, so do not propose another session-switch tool just because "
                            "the carried message may mention switching topics or creating a new session. Continue the substantive "
                            "request directly here.\n"
                            "- target summary: {target_summary}\n"
                            "- switch reason: {switch_reason}"
                        ),
                    ),
                    target_summary=source_summary,
                    switch_reason=source_reason,
                )
            )

        if self._init_mode:
            parts.append("## Active Tools\n" + ", ".join(self._active_tool_names()))
            parts.append(
                "## Initialization Mode\n"
                + self._render_prompt_template(
                    self.cm.get_nested(
                        "prompts",
                        "init_active",
                        default=(
                            "Initialization mode is active. Ask for missing names, then call init_system "
                            "with assistant_name and user_name, then present the system capabilities."
                        ),
                    )
                )
            )

        if include_sessions:
            max_recent = self.cm.get_nested("sessions", "max_recent", default=10)
            sessions = self.session_store.list_sessions(limit=max_recent)
            if sessions:
                lines = []
                for s in sessions:
                    summary = s.summary or (s.title if s.title != s.id else "(untitled)")
                    themes = f" | themes: {', '.join(s.themes[:4])}" if s.themes else ""
                    lines.append(
                        f"- [{s.id[:8]}] type={s.session_type} \"{summary}\"{themes} "
                        f"({s.message_count} msgs, {s.updated_at[:10]})"
                    )
                parts.append(
                    "## Recent Sessions\n"
                    + "\n".join(lines)
                    + "\n\nThese sessions are ordered newest first from top to bottom. "
                    + "If the user refers to the first/top rows, they mean the newest shown sessions. "
                    + "If the user refers to the last/bottom rows, they mean the oldest shown sessions. "
                    + "In French, requests like 'les 3 derniers chats' refer to the last visible rows "
                    + "(the bottom of the table), not the top 3 most recent rows."
                    + " These lines already include summary, themes, message count, and date. "
                    + "Use that data directly. Do not call session_digest just to restate visible metadata "
                    + "unless the user explicitly asks to refresh or missing metadata blocks the answer."
                    + "\n\nYou may suggest resuming a session if relevant."
                )

        if self.system_prompt:
            parts.append(f"## Custom Instructions\n{self.system_prompt}")

        if self._turn_session_guidance:
            parts.append(f"## Current Turn Guidance\n{self._turn_session_guidance}")

        return "\n\n".join(parts)

    def _messages_with_system(self, include_sessions: bool = False) -> List[Message]:
        system_text = self._build_system_prompt(include_sessions=include_sessions)
        return [Message(role=MessageRole.SYSTEM, content=system_text)] + self.history

    def _prompt_template_vars(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        provider = str(getattr(self.llm, "provider_name", "") or "")
        model = str(getattr(self.llm, "model_name", "") or "")
        assistant_name = str(self.cm.get_nested("prompts", "assistant_name", default="k-ai") or "k-ai").strip()
        user_name = self._remembered_user_name()
        active_tools = ", ".join(self._active_tool_names())
        vars_map = {
            "assistant_name": assistant_name,
            "user_name": user_name or "the user",
            "provider": provider,
            "model": model,
            "active_tools": active_tools,
            "user_input": "",
            "target_summary": "",
            "switch_reason": "",
            "session_id": "",
            "session_title": "",
            "session_type": "",
            "message_count": "0",
        }
        if extra:
            vars_map.update({key: str(value) for key, value in extra.items()})
        return vars_map

    def _render_prompt_template(self, text: str, **extra: str) -> str:
        class _SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"

        return str(text or "").format_map(_SafeDict(self._prompt_template_vars(extra=extra)))

    def _notice_value(self, key: str, default: str = "", **extra: str) -> str:
        text = self.cm.get_nested("prompts", "notices", key, default=default)
        return self._render_prompt_template(text, **extra).strip()

    def _remembered_user_name(self) -> str:
        pattern = re.compile(r"^Preferred user name:\s*(.+?)\.?$", re.IGNORECASE)
        for entry in self.memory.list_entries():
            match = pattern.match(entry.text.strip())
            if match:
                return match.group(1).strip()
        return ""

    def _active_tool_names(self) -> List[str]:
        return sorted(
            tool["function"]["name"]
            for tool in self._get_active_tools()
            if tool.get("function", {}).get("name")
        )

    def _should_offer_init(self) -> bool:
        if self.history:
            return False
        if self.memory.list_entries():
            return False
        return not bool(self.session_store.list_sessions(limit=1))

    def _boot_tools(self) -> Optional[List[Dict[str, Any]]]:
        tools = self._get_active_tools()
        boot_tools = [
            tool for tool in tools
            if tool.get("function", {}).get("name") == "load_session"
        ]
        return boot_tools or None

    def _tool_rationale(self, assistant_content: str) -> str:
        """Extract a concise justification from assistant text for the tool proposal."""
        if not assistant_content.strip():
            return ""
        lines = []
        for raw_line in assistant_content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("```") or line.startswith("import ") or line.startswith("from "):
                break
            if line.startswith("#") or line.startswith("for ") or line.startswith("while "):
                break
            if line.startswith("Je vais exécuter"):
                lines.append(line)
                break
            if len(line) > 180:
                line = line[:177] + "..."
            lines.append(line)
            if len(lines) >= 2:
                break
        return " ".join(lines).strip()

    def _tool_call_signature(self, tc: ToolCall) -> str:
        try:
            args = json.dumps(tc.arguments or {}, sort_keys=True, ensure_ascii=True)
        except TypeError:
            args = json.dumps({"_raw": str(tc.arguments)}, sort_keys=True, ensure_ascii=True)
        return f"{tc.function_name}:{args}"

    def _session_shift_guidance_for_turn(self, user_input: str) -> str:
        if not self._session_id or not user_input.strip():
            return ""
        return self._render_prompt_template(
            self.cm.get_nested(
                "prompts",
                "turn_session_guidance",
                default=(
                    'Inspect the latest user message carefully:\n"{user_input}"\n'
                    "Decide whether it explicitly asks for a fresh session or clearly pivots to a different dominant topic "
                    "than the active session profile. If so, propose the appropriate session tool before answering the new topic directly: "
                    "use new_session when the user explicitly asks for a new chat/session, and use switch_session when the request mainly "
                    "signals topic drift and would be cleaner in a separate session. If the user rejects the proposal, answer the substantive "
                    "request naturally in the current session instead of insisting."
                ),
            ),
            user_input=user_input,
        )

    def _deduplicate_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolCall]:
        seen: set[str] = set()
        unique: List[ToolCall] = []
        for tc in tool_calls:
            signature = self._tool_call_signature(tc)
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(tc)
        return unique

    # ------------------------------------------------------------------
    # Interactive loop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the interactive REPL with consciousness."""
        if self.cm.get_nested("cli", "show_welcome_panel", default=True):
            self._print_welcome_panel()

        max_recent = self.cm.get_nested("sessions", "max_recent", default=10)
        recent = self.session_store.list_sessions(limit=max_recent)
        if recent:
            render_sessions_table(self.console, recent)

        if self._should_offer_init():
            self.request_init()

        if self._init_requested:
            self._init_requested = False
            await self._run_init_intro(trigger="first_launch")
        else:
            # Boot greeting (ephemeral)
            await self._boot_greeting(recent)

        # Build prompt session with styled prompt and slash autocompletion
        slash_completer = FuzzyCompleter(
            WordCompleter(SLASH_COMMANDS, sentence=True)
        )

        pt_style = PTStyle.from_dict({
            "prompt": "bold ansicyan",
        })

        prompt_session: PromptSession = PromptSession(
            history=InMemoryHistory(),
            completer=slash_completer,
            complete_while_typing=True,
            reserve_space_for_menu=4,
            style=pt_style,
        )

        @Condition
        def _is_slash():
            buf = prompt_session.app.current_buffer
            return buf.text.lstrip().startswith("/")

        prompt_session.completer = ConditionalCompleter(slash_completer, filter=_is_slash)

        while True:
            if self._exit_requested:
                break

            if self._new_session_requested:
                self._new_session_requested = False
                seed = self._queued_new_session_seed
                carry_message = self._queued_new_session_message
                self._queued_new_session_seed = None
                self._queued_new_session_message = None
                self._do_new_session(seed=seed)
                if carry_message:
                    render_notice(
                        self.console,
                        self._notice_value(
                            "session_switched_message",
                            "A fresh session is now active. Continue the carried request in this cleaner thread.",
                        ),
                        level="info",
                        title=self._notice_value("session_switched_title", "Session Switched"),
                    )
                    self._continuation_after_switch = {
                        "summary": str((seed or {}).get("summary", "") or ""),
                        "reason": str((seed or {}).get("reason", "") or ""),
                    }
                    try:
                        await self._process_message(carry_message, suppress_switch=True)
                        await self._maybe_compact()
                    finally:
                        self._continuation_after_switch = None
                continue

            if self._init_requested:
                self._init_requested = False
                await self._run_init_intro(trigger="manual")
                continue

            if self._load_session_id:
                sid = self._load_session_id
                last_n = self._load_session_last_n
                self._load_session_id = None
                self._load_session_last_n = None
                self._do_load_session(sid, last_n=last_n)
                continue

            if self._compact_requested:
                self._compact_requested = False
                await self._do_compact()
                continue

            try:
                user_input = await prompt_session.prompt_async(
                    [("class:prompt", "You: ")]
                )
                self._reset_prompt_interrupts()
                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    result = await self.command_handler.handle(user_input)
                    if result is False:
                        break
                    continue

                first_message = not self._session_id
                if first_message:
                    self._do_new_session()

                theme_name = self.cm.get_nested("cli", "theme", default="default")
                self.console.print(render_user_panel(user_input, theme_name=theme_name))
                await self._process_message(user_input)

                if first_message and self._session_id and len(self.history) >= 2:
                    await self._auto_generate_session_summary()

                await self._maybe_compact()

            except EOFError:
                break
            except KeyboardInterrupt:
                if self._handle_prompt_interrupt():
                    break
                continue

        if self._session_id:
            try:
                await self._auto_rename_on_exit()
                await self._auto_commit_runtime_store_on_exit()
            except KeyboardInterrupt:
                render_notice(
                    self.console,
                    self._notice_value(
                        "finalization_interrupted_message",
                        "Exit finalization was interrupted. The current session was kept as-is.",
                    ),
                    level="warning",
                    title=self._notice_value("finalization_interrupted_title", "Interrupt"),
                )
            except Exception as e:
                self.console.print(f"[dim]Exit finalization skipped: {e}[/dim]")
        self.console.print(f"\n[bold green]{self._notice_value('goodbye_message', 'Goodbye!')}[/bold green]")

    # ------------------------------------------------------------------
    # Boot greeting
    # ------------------------------------------------------------------

    async def _boot_greeting(self, recent: List[SessionMetadata]) -> None:
        if recent:
            boot_instruction = self._render_prompt_template(self.cm.get_nested(
                "prompts", "boot_with_sessions",
                default="[SESSION_BOOT] Greet the user and suggest resuming a session.",
            ))
        else:
            boot_instruction = self._render_prompt_template(self.cm.get_nested(
                "prompts", "boot_no_sessions",
                default="[SESSION_BOOT] Greet the user warmly.",
            ))

        boot_messages = [
            Message(role=MessageRole.SYSTEM, content=self._build_system_prompt(include_sessions=True)),
            Message(role=MessageRole.USER, content=boot_instruction),
        ]

        try:
            tools = self._boot_tools() if recent else None
            pending_tool_calls: List[ToolCall] = []

            render_mode = self.cm.get_nested("cli", "render_mode", default="rich")
            spinner_name = self.cm.get_nested("cli", "thinking_indicator", default="dots")
            theme_name = self.cm.get_nested("cli", "theme", default="default")
            with self._interrupt_scope(allow_escape=True):
                with StreamingRenderer(
                    self.console,
                    self.llm.model_name,
                    render_mode=render_mode,
                    spinner_name=spinner_name,
                    theme_name=theme_name,
                ) as renderer:
                    async for chunk in self.llm.chat_stream(boot_messages, tools=tools):
                        if self._interrupt_requested:
                            raise KeyboardInterrupt
                        renderer.update(chunk)
                        if chunk.tool_calls:
                            pending_tool_calls.extend(chunk.tool_calls)

            for tc in pending_tool_calls:
                if self.tool_registry.is_internal(tc.function_name):
                    await self._execute_internal_tool(tc)

        except KeyboardInterrupt:
            render_notice(
                self.console,
                self._notice_value(
                    "boot_interrupted_message",
                    "Boot generation was interrupted. Control has been returned to the prompt.",
                ),
                level="warning",
                title=self._notice_value("boot_interrupted_title", "Interrupt"),
            )
        except Exception as e:
            self.console.print(f"[dim]Boot greeting skipped: {e}[/dim]")

    async def _run_init_intro(self, trigger: str) -> None:
        self._init_mode = True
        prompt_key = "init_intro"
        init_instruction = self._render_prompt_template(
            self.cm.get_nested(
                "prompts",
                prompt_key,
                default=(
                    "[INIT] Ask what you should be called and how the user wants to be addressed. "
                    "Do not call tools yet."
                ),
            )
        )
        if trigger == "manual":
            init_instruction += "\nThe user explicitly asked to initialize or re-initialize the system."

        boot_messages = [
            Message(role=MessageRole.SYSTEM, content=self._build_system_prompt(include_sessions=True)),
            Message(role=MessageRole.USER, content=init_instruction),
        ]

        try:
            render_mode = self.cm.get_nested("cli", "render_mode", default="rich")
            spinner_name = self.cm.get_nested("cli", "thinking_indicator", default="dots")
            theme_name = self.cm.get_nested("cli", "theme", default="default")
            with self._interrupt_scope(allow_escape=True):
                with StreamingRenderer(
                    self.console,
                    self.llm.model_name,
                    render_mode=render_mode,
                    spinner_name=spinner_name,
                    theme_name=theme_name,
                ) as renderer:
                    async for chunk in self.llm.chat_stream(boot_messages):
                        if self._interrupt_requested:
                            raise KeyboardInterrupt
                        renderer.update(chunk)
        except KeyboardInterrupt:
            render_notice(
                self.console,
                self._notice_value(
                    "init_interrupted_message",
                    "Initialization was interrupted. Control has been returned to the prompt.",
                ),
                level="warning",
                title=self._notice_value("init_interrupted_title", "Interrupt"),
            )
        except Exception as e:
            self.console.print(f"[dim]Initialization intro skipped: {e}[/dim]")

    # ------------------------------------------------------------------
    # Agentic message processing loop
    # ------------------------------------------------------------------

    async def _process_message(self, user_input: str, suppress_switch: bool = False) -> None:
        """
        Full agentic loop: send user message, execute tools, loop back
        to the LLM until no more tool calls are pending.

        Flow:
          1. Append user message
          2. Call LLM with tools
          3. If tool_calls: execute each, append results, goto 2
          4. If no tool_calls: display final response, done
        """
        turn_start = len(self.history)
        self.history.append(Message(role=MessageRole.USER, content=user_input))
        self._persist_message(self.history[-1])

        render_mode = self.cm.get_nested("cli", "render_mode", default="rich")
        spinner_name = self.cm.get_nested("cli", "thinking_indicator", default="dots")
        theme_name = self.cm.get_nested("cli", "theme", default="default")
        blocked_tools = {"switch_session", "new_session"} if suppress_switch else set()
        tools = self._get_active_tools(excluded_tools=blocked_tools)
        empty_rounds = 0
        executed_tool_cache: Dict[str, str] = {}
        pending_tool_calls: List[ToolCall] = []
        full_content = ""
        self._turn_session_guidance = ""
        if not suppress_switch:
            self._turn_session_guidance = self._session_shift_guidance_for_turn(user_input)

        try:
            for _round in range(_MAX_TOOL_ROUNDS):
                messages = self._messages_with_system()

                # Debug: show raw prompt
                if self._debug:
                    self.console.print(RichPanel(
                        "\n".join(f"[{m.role.value}] {m.content[:200]}" for m in messages),
                        title="[dim]DEBUG: Prompt[/dim]",
                        border_style="dim",
                    ))

                pending_tool_calls = []
                full_content = ""
                with self._interrupt_scope(allow_escape=True):
                    with StreamingRenderer(
                        self.console,
                        self.llm.model_name,
                        render_mode=render_mode,
                        spinner_name=spinner_name,
                        theme_name=theme_name,
                    ) as renderer:
                        async for chunk in self.llm.chat_stream(messages, tools=tools):
                            if self._interrupt_requested:
                                raise KeyboardInterrupt
                            renderer.update(chunk)
                            full_content = renderer.full_content
                            if chunk.tool_calls:
                                pending_tool_calls.extend(chunk.tool_calls)
                        full_content = renderer.full_content

                    if renderer.last_usage:
                        u = renderer.last_usage
                        self.total_usage.prompt_tokens += u.prompt_tokens
                        self.total_usage.completion_tokens += u.completion_tokens
                        self.total_usage.total_tokens += u.total_tokens
                        if self.cm.get_nested("cli", "show_token_usage", default=True):
                            self.console.print(
                                f"[dim]  {u.prompt_tokens} in / {u.completion_tokens} out[/dim]",
                                justify="right",
                            )

                pending_tool_calls = self._deduplicate_tool_calls(pending_tool_calls)
                if suppress_switch:
                    blocked_session_calls = [
                        tc for tc in pending_tool_calls
                        if tc.function_name in {"switch_session", "new_session"}
                    ]
                    if blocked_session_calls:
                        pending_tool_calls = [
                            tc for tc in pending_tool_calls
                            if tc.function_name not in {"switch_session", "new_session"}
                        ]
                        render_notice(
                            self.console,
                            self._notice_value(
                                "switch_already_done_message",
                                "The session change has already been completed for this carried request. Any further session-split proposal for the same carried turn is ignored.",
                            ),
                            level="warning",
                            title=self._notice_value("switch_already_done_title", "Switch Already Completed"),
                        )

                if not full_content.strip() and not pending_tool_calls:
                    empty_rounds += 1
                    if empty_rounds < _MAX_EMPTY_ASSISTANT_ROUNDS:
                        render_notice(
                            self.console,
                            self._notice_value(
                                "empty_response_message",
                                "The model returned neither text nor tool calls. Retrying automatically.",
                            ),
                            level="warning",
                            title=self._notice_value("empty_response_title", "Empty Response"),
                        )
                        continue
                    self._rollback_turn(turn_start)
                    render_notice(
                        self.console,
                        self._notice_value(
                            "canceled_turn_message",
                            "The model produced no usable response or action, so nothing was recorded for this turn.",
                        ),
                        level="error",
                        title=self._notice_value("canceled_turn_title", "Turn Canceled"),
                    )
                    return

                empty_rounds = 0

                if full_content or pending_tool_calls:
                    msg = Message(
                        role=MessageRole.ASSISTANT,
                        content=full_content,
                        tool_calls=pending_tool_calls if pending_tool_calls else None,
                    )
                    self.history.append(msg)
                    self._persist_message(msg)

                if not pending_tool_calls:
                    self._sync_session_totals()
                    self._print_runtime_snapshot()
                    return

                rationale = self._tool_rationale(full_content)
                any_executed = False
                for tc in pending_tool_calls:
                    if self.tool_registry.is_internal(tc.function_name):
                        signature = self._tool_call_signature(tc)
                        if signature in executed_tool_cache:
                            tool_content = executed_tool_cache[signature]
                            tool_msg = Message(
                                role=MessageRole.TOOL,
                                content=tool_content,
                                tool_call_id=tc.id,
                                name=tc.function_name,
                            )
                            self.history.append(tool_msg)
                            self._persist_message(tool_msg)
                            any_executed = True
                            render_notice(
                                self.console,
                                f"Appel identique à {tc.function_name} déjà exécuté dans ce tour: résultat réutilisé.",
                                level="info",
                                title="Tool Call Dupliqué",
                            )
                            continue

                        result = await self._execute_internal_tool(tc, rationale=rationale)
                        if tc.function_name in {"switch_session", "new_session"} and result.success and self._new_session_requested:
                            current_session_id = self._session_id
                            self._rollback_turn(turn_start)
                            if current_session_id:
                                try:
                                    digest = await self.generate_session_digest(current_session_id, persist=True)
                                    if digest.get("summary"):
                                        self.session_store.update_digest(
                                            current_session_id,
                                            digest["summary"],
                                            digest.get("themes", []),
                                            digest.get("session_type"),
                                        )
                                except Exception:
                                    pass
                            render_notice(
                                self.console,
                                self._notice_value(
                                    "session_split_message",
                                    "The session change was approved. The carried request will continue inside a fresh session.",
                                ),
                                level="info",
                                title=self._notice_value("session_split_title", "Session Split"),
                            )
                            return
                        if result.data and isinstance(result.data, dict) and result.data.get("interrupted"):
                            self._rollback_turn(turn_start)
                            render_notice(
                                self.console,
                                self._notice_value(
                                    "action_interrupted_message",
                                    "The requested action was interrupted. Control has been returned to the prompt.",
                                ),
                                level="warning",
                                title=self._notice_value("action_interrupted_title", "Interrupt"),
                            )
                            return
                        tool_content = self._normalize_tool_result_for_history(result.message)
                        tool_msg = Message(
                            role=MessageRole.TOOL,
                            content=tool_content,
                            tool_call_id=tc.id,
                            name=tc.function_name,
                        )
                        self.history.append(tool_msg)
                        self._persist_message(tool_msg)
                        executed_tool_cache[signature] = tool_content
                        any_executed = True

                        if self._exit_requested or self._new_session_requested or self._load_session_id or self._init_requested:
                            return

                if not any_executed:
                    return

        except ContextLengthExceededError:
            self._rollback_turn(turn_start)
            self.console.print(
                "[bold red]Context length exceeded.[/bold red] "
                "Use [cyan]/compact[/cyan] or [cyan]/clear[/cyan]."
            )
        except KeyboardInterrupt:
            self._persist_interrupted_turn_state(
                partial_content=full_content,
                pending_tool_calls=pending_tool_calls,
            )
            render_notice(
                self.console,
                self._notice_value(
                    "generation_interrupted_message",
                    "Generation was interrupted by the user. Partial output, if any, was kept and control has been returned.",
                ),
                level="warning",
                title=self._notice_value("generation_interrupted_title", "Interrupt"),
            )
        except ProviderAuthenticationError as e:
            self._rollback_turn(turn_start)
            self.console.print(f"[bold red]Auth error:[/bold red] {e}")
        except LLMError as e:
            self._rollback_turn(turn_start)
            self.console.print(f"[bold red]LLM error:[/bold red] {e}")
        except Exception as e:
            self._rollback_turn(turn_start)
            self.console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        finally:
            self._turn_session_guidance = None

    # ------------------------------------------------------------------
    # Internal tool execution with inline confirmation
    # ------------------------------------------------------------------

    async def _execute_internal_tool(self, tc: ToolCall, rationale: str = "") -> ToolResult:
        """Execute an internal tool with inline human-in-the-loop."""
        tool = self.tool_registry.get(tc.function_name)
        if not tool:
            return ToolResult(success=False, message=f"Unknown tool: {tc.function_name}")
        if not self.is_tool_enabled(tool.name):
            capability = capability_for_tool(tool.name)
            if capability:
                return ToolResult(
                    success=False,
                    message=f"{tool.name} is unavailable because the '{capability}' capability is disabled.",
                )
            return ToolResult(success=False, message=f"{tool.name} is disabled in config.")

        legacy_confirm_all = bool(self.cm.get_nested("cli", "confirm_all_tools", default=False))
        approval = self._resolve_tool_policy(tool)
        needs_confirmation = legacy_confirm_all or approval["policy"] == "ask"
        approval_rows = [
            ("Effective", f"{approval['policy']} ({approval['source']})"),
            ("Default", approval["default_policy"]),
            ("Session override", approval["session_override"] or "-"),
            ("Global override", approval["global_override"] or "-"),
            ("Protected", str(bool(approval["protected"]))),
        ]
        proposal_sections = [("Approval Policy", approval_rows)] + tool.proposal_sections(tc.arguments or {}, self._tool_ctx)
        show_rationale = bool(self.cm.get_nested("cli", "show_tool_rationale", default=True))
        display_rationale = (rationale or "").strip()
        if show_rationale and not display_rationale:
            display_rationale = tool.proposal_rationale(tc.arguments or {})

        render_tool_proposal(
            self.console,
            tool.display_spec(),
            proposal_sections,
            rationale=display_rationale,
            show_rationale=show_rationale,
            requires_approval=needs_confirmation,
        )

        if needs_confirmation:
            try:
                with self._interrupt_scope(allow_escape=False):
                    approved = Confirm.ask("Approve tool execution?", console=self.console, default=True)
            except (KeyboardInterrupt, EOFError):
                return ToolResult(success=False, message="Interrupted by user.", data={"interrupted": True})
            if self._interrupt_requested:
                return ToolResult(success=False, message="Interrupted by user.", data={"interrupted": True})
            if not approved:
                self.console.print("  [dim]Skipped.[/dim]")
                return ToolResult(success=False, message="User rejected.")

        try:
            with self._interrupt_scope(allow_escape=True):
                result = await tool.execute(tc.arguments, self._tool_ctx)
                if self._interrupt_requested:
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            return ToolResult(success=False, message="Interrupted by user.", data={"interrupted": True})
        max_len = int(self.cm.get_nested("cli", "tool_result_max_display", default=500))
        render_tool_result(
            self.console,
            tool.display_spec(),
            result,
            tool.result_renderable(result, max_display_length=max_len, ctx=self._tool_ctx),
        )
        if result.success and tool.category == "config":
            self._print_runtime_snapshot(title="Runtime Updated")
        return result

    def _normalize_tool_result_for_history(self, content: str) -> str:
        max_len = int(self.cm.get_nested("cli", "tool_result_max_history", default=4000))
        if len(content) <= max_len:
            return content
        return content[:max_len] + "\n...(truncated for history)"

    def _build_interrupted_assistant_content(self, content: str) -> str:
        clean = content.rstrip()
        if not clean:
            return _INTERRUPTED_RESPONSE_MARKER
        if clean.endswith(_INTERRUPTED_RESPONSE_MARKER):
            return clean
        return f"{clean}\n\n{_INTERRUPTED_RESPONSE_MARKER}"

    def _persist_interrupted_turn_state(
        self,
        partial_content: str,
        pending_tool_calls: Optional[List[ToolCall]] = None,
    ) -> None:
        if pending_tool_calls:
            return
        if not partial_content.strip():
            return
        interrupted_msg = Message(
            role=MessageRole.ASSISTANT,
            content=self._build_interrupted_assistant_content(partial_content),
        )
        self.history.append(interrupted_msg)
        self._persist_message(interrupted_msg)
        self._sync_session_totals()

    def _rollback_turn(self, turn_start: int) -> None:
        """Rollback any in-memory and persisted messages added during the current turn."""
        if turn_start < 0 or turn_start > len(self.history):
            return
        self.history = self.history[:turn_start]
        if self._session_id:
            self.session_store.rewrite_messages(self._session_id, self.history)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _refresh_session_digest(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        sid = session_id or self._session_id
        if not sid:
            return {"summary": "", "themes": [], "session_type": "classic"}
        digest = await self.generate_session_digest(sid, persist=True)
        if digest.get("summary"):
            self.session_store.update_digest(
                sid,
                digest["summary"],
                digest.get("themes", []),
                digest.get("session_type"),
            )
        return digest

    async def _finalize_active_session(self) -> None:
        if not self._session_id or len(self.history) < 2:
            return
        try:
            await self._refresh_session_digest(self._session_id)
        except Exception:
            pass

    async def _prepare_programmatic_turn(self) -> None:
        if self._load_session_id:
            load_id = self._load_session_id
            load_last_n = self._load_session_last_n
            self._load_session_id = None
            self._load_session_last_n = None
            self._do_load_session(load_id, last_n=load_last_n)

        if self._new_session_requested:
            await self._finalize_active_session()
            seed = self._queued_new_session_seed or {}
            self._queued_new_session_seed = None
            self._queued_new_session_message = None
            self._new_session_requested = False
            self._do_new_session(seed=seed)

        if not self._session_id:
            self._do_new_session()

    async def _auto_generate_session_summary(self) -> None:
        """Generate a one-line summary after the first real exchange (best-effort)."""
        if not self._session_id or len(self.history) < 2:
            return
        meta = self.session_store.get_session(self._session_id)
        if not meta or meta.summary:
            return
        try:
            await self._refresh_session_digest(self._session_id)
        except Exception:
            pass  # Best effort

    def _do_new_session(self, seed: Optional[Dict[str, Any]] = None) -> None:
        seed = seed or {}
        session_type = self._normalize_session_type(seed.get("session_type"))
        summary = str(seed.get("summary", "") or "").strip()
        themes = [str(theme).strip()[:40] for theme in (seed.get("themes") or []) if str(theme).strip()][:8]
        meta = self.session_store.create_session(
            provider=self.llm.provider_name,
            model=self.llm.model_name,
            session_type=session_type,
            title=summary or "",
            summary=summary,
            themes=themes,
        )
        self._session_id = meta.id
        self.history = []
        self._session_tool_policy_overrides = {"tools": {}, "categories": {}, "risks": {}}

    def _do_load_session(self, session_id: str, last_n: Optional[int] = None) -> None:
        meta = self.session_store.get_session(session_id)
        if not meta:
            render_notice(
                self.console,
                self._notice_value(
                    "session_not_found_message",
                    "Session {session_id} was not found.",
                    session_id=session_id,
                ),
                level="error",
                title=self._notice_value("session_not_found_title", "Session Error"),
            )
            return
        load_default = int(self.cm.get_nested("sessions", "load_last_n", default=0))
        effective_last_n = last_n if last_n is not None else (load_default if load_default > 0 else None)
        messages = self.session_store.load_messages(meta.id, last_n=effective_last_n)
        self.history = messages
        self._session_id = meta.id
        theme_name = self.cm.get_nested("cli", "theme", default="default")
        render_notice(
            self.console,
            self._notice_value(
                "session_resumed_message",
                "Session [bold]{session_title}[/bold] ([dim]{session_type}[/dim]) was resumed with {message_count} loaded messages.",
                session_title=meta.summary or meta.title,
                session_type=meta.session_type,
                message_count=str(len(messages)),
            ),
            level="success",
            title=self._notice_value("session_resumed_title", "Session Resumed"),
        )

        # Show the last N messages with proper rendering (same as live chat)
        keep_n = int(self.cm.get_nested("sessions", "preview_last_n", default=10))
        recent = messages[-keep_n:] if len(messages) > keep_n else messages
        if recent:
            render_mode = self.cm.get_nested("cli", "render_mode", default="rich")
            from .ui.markdown import render_content
            skipped = len(messages) - len(recent)
            if skipped > 0:
                self.console.print(f"[dim]  ({skipped} older messages not shown)[/dim]")
            for m in recent:
                if m.role == MessageRole.SYSTEM:
                    continue
                elif m.role == MessageRole.USER:
                    self.console.print(render_user_panel(m.content, theme_name=theme_name))
                elif m.role == MessageRole.ASSISTANT:
                    content_renderable = render_content(m.content, render_mode) if m.content else Text("[dim](empty)[/dim]")
                    if m.content:
                        self.console.print(
                            render_assistant_panel(
                                m.content,
                                self.llm.model_name,
                                render_mode=render_mode,
                                theme_name=theme_name,
                            )
                        )
                    else:
                        self.console.print(RichPanel(
                            content_renderable,
                            title=f"[bold green]Assistant[/bold green] [dim]{self.llm.model_name}[/dim]",
                            border_style="green",
                        ))
                elif m.role == MessageRole.TOOL:
                    name = m.name or "tool"
                    border = "green"
                    self.console.print(RichPanel(
                        m.content[:300] + ("..." if len(m.content) > 300 else ""),
                        title=f"[bold {border}]Agent[/bold {border}] [dim]{name}[/dim]",
                        border_style=border,
                        expand=False,
                        padding=(0, 1),
                    ))

    def _persist_message(self, message: Message) -> None:
        if self._session_id:
            self.session_store.save_message(self._session_id, message)

    def _queue_load_session(self, session_id: str, last_n: Optional[int] = None) -> None:
        self._load_session_id = session_id
        self._load_session_last_n = last_n

    def reset_history(self) -> None:
        """Clear the active in-memory history and persist the cleared state."""
        self.history = []
        if self._session_id:
            self.session_store.rewrite_messages(self._session_id, [])

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    async def _maybe_compact(self) -> None:
        trigger_pct = self.cm.get_nested("compaction", "trigger_percent", default=80)
        if trigger_pct <= 0:
            return
        ctx_window = self.llm.provider_config.get("context_window", 128000)
        estimated_tokens = self._estimate_context_tokens()
        threshold = int(ctx_window * trigger_pct / 100)
        if estimated_tokens > threshold:
            self.console.print("[dim]Context approaching limit, auto-compacting...[/dim]")
            await self._do_compact()

    async def _do_compact(self) -> None:
        keep_n = self.cm.get_nested("compaction", "keep_last_n", default=10)
        if len(self.history) <= keep_n:
            self.console.print("[dim]History too short to compact.[/dim]")
            return

        old_messages = self.history[:-keep_n]
        recent_messages = self.history[-keep_n:]

        compact_instruction = self.cm.get_nested(
            "prompts", "compact_summarize",
            default="Summarize the following conversation concisely.",
        )
        compact_instruction = self._render_prompt_template(compact_instruction)
        summary_prompt = compact_instruction + "\n\n"
        for m in old_messages:
            summary_prompt += f"[{m.role.value}]: {m.content[:500]}\n"

        try:
            summary = ""
            summary_msgs = [
                Message(role=MessageRole.SYSTEM, content=compact_instruction),
                Message(role=MessageRole.USER, content=summary_prompt),
            ]
            async for chunk in self.llm.chat_stream(summary_msgs):
                summary += chunk.delta_content

            summary_msg = Message(
                role=MessageRole.SYSTEM,
                content=f"[Compacted context]\n{summary}",
            )
            self.history = [summary_msg] + recent_messages
            if self._session_id:
                self.session_store.rewrite_messages(self._session_id, self.history)
            self.console.print(
                f"[green]Compacted:[/green] {len(old_messages)} messages summarized, "
                f"{keep_n} recent kept."
            )

            if self.cm.get_nested("compaction", "auto_rename", default=True):
                if self._session_id:
                    await self._refresh_session_digest(self._session_id)

        except KeyboardInterrupt:
            self.console.print("[yellow]Compaction interrupted.[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Compaction failed:[/red] {e}")

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    async def _auto_rename_on_exit(self) -> None:
        if not self._session_id or len(self.history) < 2:
            return

        meta = self.session_store.get_session(self._session_id)
        if not meta:
            return

        try:
            await self._refresh_session_digest(self._session_id)
            messages = self.session_store.load_messages(self._session_id)
            transcript = "\n".join(
                f"[{m.role.value}] {m.content[:300]}"
                for m in messages
                if m.role != MessageRole.SYSTEM
            )[:6000]
            title_prompt = self.cm.get_nested("prompts", "exit_title", default="").strip()
            title_prompt = self._render_prompt_template(title_prompt)
            if title_prompt and transcript:
                raw_title = ""
                msgs = [
                    Message(role=MessageRole.SYSTEM, content=title_prompt),
                    Message(role=MessageRole.USER, content=transcript),
                ]
                async for chunk in self.llm.chat_stream(msgs):
                    raw_title += chunk.delta_content
                title = raw_title.strip().replace("\n", " ")[:60]
                if title:
                    self.session_store.rename_session(self._session_id, title)

            exit_summary_prompt = self.cm.get_nested("prompts", "exit_summary", default="").strip()
            exit_summary_prompt = self._render_prompt_template(exit_summary_prompt)
            if exit_summary_prompt and transcript:
                raw_summary = ""
                msgs = [
                    Message(role=MessageRole.SYSTEM, content=exit_summary_prompt),
                    Message(role=MessageRole.USER, content=transcript),
                ]
                async for chunk in self.llm.chat_stream(msgs):
                    raw_summary += chunk.delta_content
                summary = raw_summary.strip().replace("\n", " ")[:160]
                if summary:
                    self.session_store.update_summary(self._session_id, summary, meta.themes, meta.session_type)
        except Exception:
            pass

    async def _auto_commit_runtime_store_on_exit(self) -> None:
        if not self._session_id or len(self.history) < 2:
            return
        if not bool(self.cm.get_nested("runtime_git", "enabled", default=True)):
            return
        if not bool(self.cm.get_nested("runtime_git", "auto_commit_on_chat_exit", default=True)):
            return

        meta = self.session_store.get_session(self._session_id)
        if not meta:
            return

        result = commit_runtime_state(
            self.cm,
            summary=meta.summary or meta.title,
            session_id=meta.id,
            session_type=meta.session_type,
            themes=meta.themes,
            create_if_missing=True,
        )
        if result.get("ok") and result.get("reason") == "committed":
            self.console.print(f"[dim]Runtime state committed: {result.get('subject', '')}[/dim]")
        elif result.get("ok") and result.get("reason") == "clean":
            self.console.print("[dim]Runtime state unchanged; no git commit created.[/dim]")
        elif not result.get("ok"):
            reason = result.get("reason", "unknown")
            detail = result.get("stderr") or "; ".join(result.get("issues", [])) or ""
            suffix = f": {detail}" if detail else ""
            self.console.print(f"[dim]Runtime git auto-commit skipped ({reason}){suffix}[/dim]")

    async def generate_session_digest(
        self,
        session_id: Optional[str] = None,
        persist: bool = False,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a digest sentence and key themes for a session."""
        sid = session_id or self._session_id
        if not sid:
            return {"summary": "", "themes": [], "session_type": "classic"}

        messages = self.session_store.load_messages(sid, offset=offset, limit=limit)
        if not messages:
            return {"summary": "", "themes": [], "session_type": "classic"}

        digest_prompt = self.cm.get_nested(
            "prompts", "session_digest",
            default=(
                "Return strict JSON with keys summary, themes, and session_type. "
                "summary must be one short sentence describing the discussion. "
                "themes must be a short list of key topics. "
                "session_type must be either 'classic' for a real task/topic session "
                "or 'meta' for a session mainly about administration, settings, tooling, or chat management."
            ),
        )
        digest_prompt = self._render_prompt_template(digest_prompt)
        summary_guidance = self.cm.get_nested(
            "prompts", "session_summary",
            default="Summarize the user's intent and the discussion topic in one short sentence.",
        )
        summary_guidance = self._render_prompt_template(summary_guidance)
        transcript = "\n".join(
            f"[{m.role.value}] {m.content[:300]}"
            for m in messages
            if m.role != MessageRole.SYSTEM
        )
        raw = ""
        msgs = [
            Message(role=MessageRole.SYSTEM, content=f"{digest_prompt}\nFor the summary field specifically: {summary_guidance}"),
            Message(role=MessageRole.USER, content=transcript[:6000]),
        ]
        async for chunk in self.llm.chat_stream(msgs):
            raw += chunk.delta_content

        summary = ""
        themes: List[str] = []
        session_type = "classic"
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            payload = json.loads(raw[start:end + 1] if start != -1 and end != -1 else raw)
            summary = str(payload.get("summary", "")).strip().replace("\n", " ")[:120]
            raw_themes = payload.get("themes", [])
            if isinstance(raw_themes, list):
                themes = [str(theme).strip()[:40] for theme in raw_themes if str(theme).strip()][:8]
            session_type = self._normalize_session_type(payload.get("session_type"))
        except Exception:
            summary = raw.strip().replace("\n", " ")[:120]
            themes = []
            transcript_lower = transcript.lower()
            if any(marker in transcript_lower for marker in ["config", "session", "tool", "validation", "approval", "prompt", "runtime", "meta"]):
                session_type = "meta"

        digest = {"summary": summary, "themes": themes, "session_type": session_type}
        if persist and summary:
            self.session_store.update_digest(sid, summary, themes, session_type)
        return digest

    # ------------------------------------------------------------------
    # Programmatic API
    # ------------------------------------------------------------------

    async def send(self, message: str) -> str:
        await self._prepare_programmatic_turn()
        turn_start = len(self.history)
        try:
            user_msg = Message(role=MessageRole.USER, content=message)
            self.history.append(user_msg)
            self._persist_message(user_msg)
            messages = self._messages_with_system()
            full_content = ""
            async for chunk in self.llm.chat_stream(messages):
                full_content += chunk.delta_content
            assistant_msg = Message(role=MessageRole.ASSISTANT, content=full_content)
            self.history.append(assistant_msg)
            self._persist_message(assistant_msg)
            await self._auto_generate_session_summary()
            await self._maybe_compact()
            return full_content
        except Exception:
            self._rollback_turn(turn_start)
            raise

    async def send_with_tools(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        tool_executor: Callable[[ToolCall], Awaitable[str]],
        max_tool_rounds: int = 10,
    ) -> str:
        await self._prepare_programmatic_turn()
        turn_start = len(self.history)
        try:
            user_msg = Message(role=MessageRole.USER, content=message)
            self.history.append(user_msg)
            self._persist_message(user_msg)
            for _round in range(max_tool_rounds):
                messages = self._messages_with_system()
                full_content = ""
                pending_tool_calls: List[ToolCall] = []
                async for chunk in self.llm.chat_stream(messages, tools=tools):
                    full_content += chunk.delta_content
                    if chunk.tool_calls:
                        pending_tool_calls.extend(chunk.tool_calls)
                if not pending_tool_calls:
                    assistant_msg = Message(role=MessageRole.ASSISTANT, content=full_content)
                    self.history.append(assistant_msg)
                    self._persist_message(assistant_msg)
                    await self._auto_generate_session_summary()
                    await self._maybe_compact()
                    return full_content
                assistant_msg = Message(
                    role=MessageRole.ASSISTANT, content=full_content,
                    tool_calls=pending_tool_calls,
                )
                self.history.append(assistant_msg)
                self._persist_message(assistant_msg)
                for tc in pending_tool_calls:
                    try:
                        result = await tool_executor(tc)
                    except Exception as exc:
                        result = f"Error executing tool '{tc.function_name}': {exc}"
                    tool_msg = Message(
                        role=MessageRole.TOOL, content=result,
                        tool_call_id=tc.id, name=tc.function_name,
                    )
                    self.history.append(tool_msg)
                    self._persist_message(tool_msg)
            await self._auto_generate_session_summary()
            await self._maybe_compact()
            return full_content  # type: ignore[return-value]
        except Exception:
            self._rollback_turn(turn_start)
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _handle_new_session(
        self,
        seed: Optional[Dict[str, Any]] = None,
        carry_over_message: Optional[str] = None,
    ) -> None:
        self._queued_new_session_seed = seed or None
        self._queued_new_session_message = carry_over_message.strip() if carry_over_message else None
        self._new_session_requested = True

    def request_init(self) -> None:
        self._init_requested = True
        self._init_mode = True

    def complete_init(self) -> None:
        self._init_requested = False
        self._init_mode = False

    def _print_welcome_panel(self) -> None:
        theme_name = self.cm.get_nested("cli", "theme", default="default")
        assistant_name = str(self.cm.get_nested("prompts", "assistant_name", default="k-ai") or "k-ai")
        self.console.print(
            render_runtime_panel(
                self.get_runtime_snapshot(),
                title=f"{assistant_name}  |  Unified LLM Chat",
                mode="welcome",
                theme_name=theme_name,
            )
        )
        self.console.print("[dim]Type /help for commands, or just chat.[/dim]")
