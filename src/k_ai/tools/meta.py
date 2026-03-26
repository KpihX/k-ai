# src/k_ai/tools/meta.py
"""
Meta/session management tools.

These tools handle session lifecycle, runtime transparency, and config control.
"""
import json
from typing import Any, Dict

import yaml

from ..models import ToolResult
from .base import InternalTool, ToolContext, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_session_id(raw: str, ctx: ToolContext) -> str | None:
    """
    Resolve a session identifier: accepts an 8-char hex ID, a full ID,
    an ID prefix, or a position number (#N / N) from the most recent list.
    Returns the full session ID or None if not found.
    """
    raw = raw.strip().lstrip("#")

    # Try as a position number (1-based)
    try:
        pos = int(raw)
        if 1 <= pos <= 100:
            max_recent = ctx.config.get_nested("sessions", "max_recent", default=10)
            sessions = ctx.session_store.list_sessions(limit=max_recent)
            if 1 <= pos <= len(sessions):
                return sessions[pos - 1].id
    except ValueError:
        pass

    # Try as ID or prefix
    meta = ctx.session_store.get_session(raw)
    return meta.id if meta else None


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class NewSessionTool(InternalTool):
    name = "new_session"
    display_name = "Start New Session"
    category = "session"
    danger_level = "medium"
    accent_color = "blue"
    description = "Start a new chat session, clearing the current context."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.request_new_session:
            ctx.request_new_session()
        return ToolResult(success=True, message="New session started.")


class LoadSessionTool(InternalTool):
    name = "load_session"
    display_name = "Load Session"
    category = "session"
    danger_level = "medium"
    accent_color = "blue"
    description = (
        "Resume a previous chat session. Use the 8-char hex ID from list_sessions "
        "(e.g. 'd24a3534'), NOT the position number."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "The 8-char hex session ID (from list_sessions).",
            },
            "last_n": {
                "type": "integer",
                "description": "Optional: load only the last N messages (default from config sessions.load_last_n).",
            },
        },
        "required": ["session_id"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        session_id = _resolve_session_id(arguments.get("session_id", ""), ctx)
        if not session_id:
            return ToolResult(success=False, message=f"Session '{arguments.get('session_id')}' not found.")
        meta = ctx.session_store.get_session(session_id)
        if not meta:
            return ToolResult(success=False, message=f"Session '{session_id}' not found.")
        last_n = arguments.get("last_n")
        if ctx.request_load_session:
            ctx.request_load_session(meta.id, last_n)
        return ToolResult(
            success=True,
            message=f"Resumed session '{meta.summary or meta.title}' ({meta.message_count} messages).",
        )


class ExitSessionTool(InternalTool):
    name = "exit_session"
    display_name = "Exit Session"
    category = "session"
    danger_level = "high"
    accent_color = "red"
    description = "End the current chat session and exit k-ai."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.request_exit:
            ctx.request_exit()
        return ToolResult(success=True, message="Goodbye!")


class RenameSessionTool(InternalTool):
    name = "rename_session"
    display_name = "Rename Session"
    category = "session"
    danger_level = "medium"
    accent_color = "blue"
    description = "Set or update the title of the current session."
    parameters_schema = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "The new title for the session.",
            },
        },
        "required": ["title"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        title = arguments.get("title", "")
        if not title:
            return ToolResult(success=False, message="Title cannot be empty.")
        session_id = ctx.get_session_id() if ctx.get_session_id else None
        if not session_id:
            return ToolResult(success=False, message="No active session to rename.")
        ctx.session_store.rename_session(session_id, title)
        return ToolResult(success=True, message=f"Session renamed to '{title}'.")


class ListSessionsTool(InternalTool):
    name = "list_sessions"
    display_name = "List Sessions"
    category = "session"
    danger_level = "low"
    accent_color = "blue"
    description = (
        "List recent chat sessions. Each session shows its position number (#1, #2...) "
        "and its unique ID (8-char hex). IMPORTANT: when referring to a session in other "
        "tools (delete, load, rename), always use the 8-char hex ID, NOT the position number. "
        "If the user asks for the oldest sessions, set order='oldest'."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max sessions (default from config sessions.max_recent).",
            },
            "order": {
                "type": "string",
                "description": "Session order: 'recent' for newest first, 'oldest' for oldest first.",
                "enum": ["recent", "oldest"],
            },
        },
        "required": [],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        max_recent = ctx.config.get_nested("sessions", "max_recent", default=10)
        limit = arguments.get("limit", max_recent)
        order = str(arguments.get("order", "recent") or "recent").lower()
        if order not in {"recent", "oldest"}:
            order = "recent"
        sessions = ctx.session_store.list_sessions(limit=limit, order=order)
        if not sessions:
            return ToolResult(success=True, message="No sessions found.", data=[])
        lines = []
        for i, s in enumerate(sessions, 1):
            summary = s.summary or (s.title if s.title != s.id else "(untitled)")
            lines.append(f"#{i} ID={s.id[:8]} \"{summary}\" ({s.message_count} msgs)")
        return ToolResult(
            success=True,
            message="\n".join(lines),
            data={"order": order, "sessions": [s.model_dump() for s in sessions]},
        )

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.table import Table
        from rich.text import Text

        payload = result.data or {}
        sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
        if not sessions:
            return Text("No sessions found.")
        order = payload.get("order", "recent") if isinstance(payload, dict) else "recent"
        table = Table(show_header=True, header_style="bold", border_style="blue", title=f"Order: {order}")
        table.add_column("#", style="dim", width=3)
        table.add_column("ID", style="cyan", width=10)
        table.add_column("Summary")
        table.add_column("Msgs", justify="right", width=5)
        for i, session in enumerate(sessions, 1):
            summary = session["summary"] or (session["title"] if session["title"] != session["id"] else "(untitled)")
            table.add_row(str(i), session["id"][:8], summary, str(session["message_count"]))
        return table


class SessionExtractTool(InternalTool):
    name = "session_extract"
    display_name = "Extract Session Window"
    category = "session"
    danger_level = "low"
    accent_color = "blue"
    description = "Extract a window of messages from the current or a previous session."
    parameters_schema = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "Optional session ID or prefix. Defaults to current session."},
            "offset": {"type": "integer", "description": "Message offset to start from."},
            "limit": {"type": "integer", "description": "Number of messages to return."},
        },
        "required": [],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_session_id = arguments.get("session_id")
        if raw_session_id:
            session_id = _resolve_session_id(raw_session_id, ctx)
        else:
            session_id = ctx.get_session_id() if ctx.get_session_id else None
        if not session_id:
            return ToolResult(success=False, message="No target session found.")
        meta = ctx.session_store.get_session(session_id)
        if not meta:
            return ToolResult(success=False, message=f"Session '{session_id}' not found.")
        offset = int(arguments.get("offset", 0) or 0)
        default_limit = int(ctx.config.get_nested("sessions", "extract_limit", default=12))
        limit = int(arguments.get("limit", default_limit) or default_limit)
        messages = ctx.session_store.load_messages(meta.id, offset=offset, limit=limit)
        rows = [
            {"index": offset + i, "role": m.role.value, "content": m.content, "name": m.name}
            for i, m in enumerate(messages)
        ]
        body = "\n".join(
            f"[{row['index']}] {row['role']}: {row['content'][:180]}"
            for row in rows
        ) or "(empty window)"
        return ToolResult(success=True, message=body, data={"session": meta.model_dump(), "messages": rows})

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim")
        table.add_column("value")
        table.add_row("Session", str(arguments.get("session_id", "current session")))
        table.add_row("Offset", str(arguments.get("offset", 0)))
        table.add_row("Limit", str(arguments.get("limit", ctx.config.get_nested("sessions", "extract_limit", default=12))))
        return [("Extraction Window", table)]

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.table import Table
        from rich.text import Text

        data = result.data or {}
        messages = data.get("messages", [])
        if not messages:
            return Text("(empty window)")
        table = Table(show_header=True, header_style="bold", border_style="blue")
        table.add_column("#", style="dim", width=5)
        table.add_column("Role", style="cyan", width=10)
        table.add_column("Content")
        for row in messages:
            content = row["content"].replace("\n", " ")
            if len(content) > 120:
                content = content[:117] + "..."
            table.add_row(str(row["index"]), row["role"], content)
        return table


class SessionDigestTool(InternalTool):
    name = "session_digest"
    display_name = "Refresh Session Digest"
    category = "session"
    danger_level = "medium"
    accent_color = "blue"
    description = "Generate or refresh the summary sentence and key themes for the current or a previous session."
    parameters_schema = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "Optional session ID or prefix. Defaults to current session."},
            "persist": {"type": "boolean", "description": "Persist the generated digest into session metadata."},
        },
        "required": [],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_session_id = arguments.get("session_id")
        if raw_session_id:
            session_id = _resolve_session_id(raw_session_id, ctx)
        else:
            session_id = ctx.get_session_id() if ctx.get_session_id else None
        if not session_id:
            return ToolResult(success=False, message="No target session found.")
        if not ctx.generate_session_digest:
            return ToolResult(success=False, message="Digest generation is not available.")
        persist = bool(arguments.get("persist", True))
        digest = await ctx.generate_session_digest(session_id=session_id, persist=persist)
        summary = digest.get("summary", "")
        themes = digest.get("themes", [])
        msg = f"{summary}\nThemes: {', '.join(themes) if themes else '-'}"
        return ToolResult(success=True, message=msg, data=digest)

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.table import Table

        data = result.data or {}
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim", no_wrap=True)
        table.add_column("value")
        table.add_row("Summary", str(data.get("summary", "")))
        table.add_row("Themes", ", ".join(data.get("themes", [])) or "-")
        return table


class DeleteSessionTool(InternalTool):
    name = "delete_session"
    display_name = "Delete Session"
    category = "session"
    danger_level = "high"
    accent_color = "red"
    description = (
        "Permanently delete a chat session. Use the 8-char hex ID from list_sessions "
        "(e.g. 'd24a3534'), NOT the position number."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "The 8-char hex session ID (from list_sessions).",
            },
        },
        "required": ["session_id"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        session_id = _resolve_session_id(arguments.get("session_id", ""), ctx)
        if not session_id:
            return ToolResult(success=False, message=f"Session '{arguments.get('session_id')}' not found.")
        meta = ctx.session_store.get_session(session_id)
        title = meta.title if meta else session_id
        ok = ctx.session_store.delete_session(session_id)
        if ok:
            return ToolResult(success=True, message=f"Session '{title}' ({session_id[:8]}) deleted.")
        return ToolResult(success=False, message=f"Session '{session_id}' not found.")


class CompactSessionTool(InternalTool):
    name = "compact_session"
    display_name = "Compact Session"
    category = "session"
    danger_level = "high"
    accent_color = "yellow"
    description = (
        "Compress the conversation history by summarizing older messages, "
        "keeping only the most recent ones in full."
    )
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.request_compact:
            ctx.request_compact()
            return ToolResult(success=True, message="Compaction requested.")
        return ToolResult(success=False, message="Compaction not available.")


class ClearScreenTool(InternalTool):
    name = "clear_screen"
    display_name = "Clear Screen"
    category = "ui"
    danger_level = "low"
    accent_color = "cyan"
    description = "Clear the terminal screen without affecting the conversation."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx.console.clear()
        return ToolResult(success=True, message="Screen cleared.")


class SetConfigTool(InternalTool):
    name = "set_config"
    display_name = "Set Config"
    category = "config"
    danger_level = "high"
    accent_color = "yellow"
    description = (
        "Change a runtime configuration value. "
        "Supports dot-notation for nested keys (e.g. 'cli.theme')."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Config key (e.g. 'temperature')."},
            "value": {"type": "string", "description": "New value as a plain string or YAML/JSON scalar/list/object literal."},
            "persist": {"type": "boolean", "description": "Also save the active merged config to disk."},
        },
        "required": ["key", "value"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        key = arguments.get("key", "")
        raw_value = arguments.get("value", "")
        value = raw_value
        if isinstance(raw_value, str):
            parsed = yaml.safe_load(raw_value)
            if parsed is not None:
                value = parsed
        persist = bool(arguments.get("persist", False))
        try:
            if ctx.apply_config_change:
                change = ctx.apply_config_change(key, value, persist=persist)
                old_value = change.get("old_value")
                coerced = change["value"]
                saved_to = change.get("saved_to")
            else:
                old_value = ctx.config.get_path(key, default=None)
                coerced = ctx.config.set(key, value)
                saved_to = str(ctx.config.save_active_yaml()) if persist else None
            msg = f"{key} = {coerced!r}"
            if saved_to:
                msg += f"\nSaved to {saved_to}"
            return ToolResult(
                success=True,
                message=msg,
                data={"key": key, "old_value": old_value, "value": coerced, "saved_to": saved_to},
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to set {key}: {e}")

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim")
        table.add_column("value")
        table.add_row("Key", str(arguments.get("key", "")))
        table.add_row("Value", repr(arguments.get("value", "")))
        table.add_row("Persist", str(bool(arguments.get("persist", False))))
        return [("Config Change", table)]

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.table import Table

        data = result.data or {}
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim", no_wrap=True)
        table.add_column("value")
        table.add_row("Key", str(data.get("key", "")))
        table.add_row("Previous", repr(data.get("old_value", "")))
        table.add_row("Applied", repr(data.get("value", "")))
        table.add_row("Saved", str(data.get("saved_to") or "-"))
        return table


class GetConfigTool(InternalTool):
    name = "get_config"
    display_name = "Inspect Config"
    category = "config"
    danger_level = "low"
    accent_color = "cyan"
    description = "Read the current active configuration, either one key or the full merged config."
    parameters_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Optional dot-notation key such as 'cli.tool_result_max_display'."},
        },
        "required": [],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        key = str(arguments.get("key", "") or "").strip()
        if key:
            value = ctx.config.get_path(key, default=None)
            return ToolResult(success=True, message=f"{key} = {value!r}", data={"key": key, "value": value})
        payload = ctx.config.get_all()
        return ToolResult(
            success=True,
            message=yaml.dump(payload, allow_unicode=True, sort_keys=False),
            data={"key": "", "value": payload},
        )

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim")
        table.add_column("value")
        table.add_row("Key", str(arguments.get("key", "(full active config)")))
        return [("Config Read", table)]

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.syntax import Syntax
        from rich.table import Table

        data = result.data or {}
        if data.get("key"):
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("field", style="dim", no_wrap=True)
            table.add_column("value")
            table.add_row("Key", str(data.get("key", "")))
            table.add_row("Value", repr(data.get("value", None)))
            return table
        payload = yaml.dump(data.get("value", {}), allow_unicode=True, sort_keys=False)
        return Syntax(payload, "yaml", theme="monokai", line_numbers=False, word_wrap=True)


class ListConfigTool(InternalTool):
    name = "list_config"
    display_name = "List Config Keys"
    category = "config"
    danger_level = "low"
    accent_color = "cyan"
    description = "List active config keys and values, optionally filtered by a dot-notation prefix."
    parameters_schema = {
        "type": "object",
        "properties": {
            "prefix": {"type": "string", "description": "Optional prefix, e.g. 'cli' or 'tools.python_exec'."},
            "limit": {"type": "integer", "description": "Maximum number of entries to show."},
        },
        "required": [],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        prefix = str(arguments.get("prefix", "") or "").strip()
        limit = int(arguments.get("limit", 50) or 50)
        flat = ctx.config.flatten(prefix=prefix)
        items = list(flat.items())[:max(limit, 1)]
        lines = [f"{key} = {value!r}" for key, value in items]
        return ToolResult(
            success=True,
            message="\n".join(lines) or "(no matching keys)",
            data={"prefix": prefix, "items": items},
        )

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.table import Table

        table = Table(show_header=True, header_style="bold", border_style="cyan")
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        for key, value in (result.data or {}).get("items", []):
            table.add_row(str(key), repr(value))
        return table


class SaveConfigTool(InternalTool):
    name = "save_config"
    display_name = "Persist Active Config"
    category = "config"
    danger_level = "high"
    accent_color = "yellow"
    description = "Write the current active merged configuration to disk so live changes survive the session."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Optional destination path. Defaults to the active override file or config.persist_path."},
        },
        "required": [],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = str(arguments.get("path", "") or "").strip() or None
        try:
            saved_to = str(ctx.config.save_active_yaml(path))
            return ToolResult(success=True, message=f"Active config saved to {saved_to}", data={"path": saved_to})
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to save config: {e}")

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim")
        table.add_column("value")
        table.add_row("Destination", str(arguments.get("path") or ctx.config.override_path or ctx.config.get_nested("config", "persist_path", default="~/.k-ai/config.yaml")))
        return [("Config Save", table)]


class RuntimeStatusTool(InternalTool):
    name = "runtime_status"
    display_name = "Show Runtime Transparency"
    category = "runtime"
    danger_level = "low"
    accent_color = "cyan"
    description = "Show live runtime stats: provider, model, context window usage, token totals, compaction thresholds, and active UI limits."
    parameters_schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "description": "compact or full"},
        },
        "required": [],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not ctx.get_runtime_snapshot:
            return ToolResult(success=False, message="Runtime snapshot is unavailable.")
        snapshot = ctx.get_runtime_snapshot()
        mode = str(arguments.get("mode", "") or ctx.config.get_nested("cli", "runtime_stats_mode", default="compact"))
        return ToolResult(success=True, message=json.dumps(snapshot, ensure_ascii=False), data={"mode": mode, "snapshot": snapshot})

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from ..ui import render_runtime_panel

        data = result.data or {}
        return render_runtime_panel(data.get("snapshot", {}), title="Runtime Transparency", mode=data.get("mode", "compact"))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_meta_tools(registry: ToolRegistry, ctx: ToolContext) -> None:
    """Register all meta/session tools."""
    for tool_cls in [
        NewSessionTool,
        LoadSessionTool,
        ExitSessionTool,
        RenameSessionTool,
        ListSessionsTool,
        SessionExtractTool,
        SessionDigestTool,
        DeleteSessionTool,
        CompactSessionTool,
        ClearScreenTool,
        RuntimeStatusTool,
        GetConfigTool,
        ListConfigTool,
        SetConfigTool,
        SaveConfigTool,
    ]:
        registry.register(tool_cls())
