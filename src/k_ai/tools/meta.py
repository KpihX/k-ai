# src/k_ai/tools/meta.py
"""
Meta/session management tools.

These tools handle session lifecycle, runtime transparency, and config control.
"""
import json
from typing import Any, Dict

import yaml

from ..models import ToolResult
from ..ui_theme import resolve_syntax_theme
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


def _session_summary(meta) -> str:
    return meta.summary or (meta.title if meta.title != meta.id else "(untitled)")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

class NewSessionTool(InternalTool):
    name = "new_session"
    display_name = "Start New Session"
    category = "session"
    danger_level = "medium"
    accent_color = "blue"
    description = "Start a new chat session, clearing the current context. Optional seed metadata can define the new session type/summary/themes."
    parameters_schema = {
        "type": "object",
        "properties": {
            "session_type": {"type": "string", "enum": ["classic", "meta"], "description": "Optional session type for the new session."},
            "summary": {"type": "string", "description": "Optional one-line summary to seed the new session metadata."},
            "themes": {"type": "array", "items": {"type": "string"}, "description": "Optional seed themes for the new session."},
        },
        "required": [],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.request_new_session:
            ctx.request_new_session(seed={
                "session_type": arguments.get("session_type", "classic"),
                "summary": arguments.get("summary", ""),
                "themes": arguments.get("themes", []),
            })
        return ToolResult(success=True, message="New session started.")


class SwitchSessionTool(InternalTool):
    name = "switch_session"
    display_name = "Switch To New Session"
    category = "session"
    danger_level = "medium"
    accent_color = "blue"
    description = (
        "Finalize the current session and continue the current user request inside a new, semantically cleaner session. "
        "Use this proactively when the user clearly switches to a different dominant intent."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Short summary of the new session topic."},
            "themes": {"type": "array", "items": {"type": "string"}, "description": "Key themes for the new session."},
            "session_type": {"type": "string", "enum": ["classic", "meta"], "description": "Type for the new session."},
            "reason": {"type": "string", "description": "Brief reason why a new session would keep topics cleaner."},
        },
        "required": ["summary", "session_type"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        history = ctx.get_history() if ctx.get_history else []
        carry_message = ""
        for message in reversed(history or []):
            if message.role.value == "user" and message.content.strip():
                carry_message = message.content
                break

        seed = {
            "summary": str(arguments.get("summary", "") or "").strip(),
            "themes": arguments.get("themes", []) or [],
            "session_type": str(arguments.get("session_type", "classic") or "classic"),
        }
        reason = str(arguments.get("reason", "") or "").strip()
        if ctx.request_new_session:
            ctx.request_new_session(seed=seed, carry_over_message=carry_message)
        message = f"Switching to a new {seed['session_type']} session: {seed['summary']}"
        if reason:
            message += f"\nReason: {reason}"
        return ToolResult(success=True, message=message, data={"seed": seed, "reason": reason, "carry_over_message": bool(carry_message)})

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim")
        table.add_column("value")
        table.add_row("New type", str(arguments.get("session_type", "classic")))
        table.add_row("Summary", str(arguments.get("summary", "")))
        table.add_row("Themes", ", ".join(arguments.get("themes", []) or []) or "-")
        table.add_row("Reason", str(arguments.get("reason", "") or "-"))
        return [("Session Switch", table)]


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
        "List chat sessions. Each session shows its position number (#1, #2...) "
        "and its unique ID (8-char hex), summary, themes, and message count. IMPORTANT: when referring to a session in other "
        "tools (delete, load, rename), always use the 8-char hex ID, NOT the position number. "
        "The visible session table is ordered newest first from top to bottom. "
        "If the user asks for the oldest sessions, or refers to the last/bottom rows "
        "of the visible table, set order='oldest' instead of using limit=N on recent ordering. "
        "Example: in French, 'supprime les 3 derniers chats' means the 3 bottom rows of the "
        "visible table, not the top 3 most recent sessions."
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
            "session_type": {
                "type": "string",
                "description": "Optional filter by session type.",
                "enum": ["classic", "meta"],
            },
        },
        "required": [],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        max_recent = ctx.config.get_nested("sessions", "max_recent", default=10)
        limit = arguments.get("limit", max_recent)
        order = str(arguments.get("order", "recent") or "recent").lower()
        filter_type = str(arguments.get("session_type", "") or "").strip().lower() or None
        if order not in {"recent", "oldest"}:
            order = "recent"
        raw_limit = ctx.session_store.session_count() if filter_type in {"classic", "meta"} else limit
        sessions = ctx.session_store.list_sessions(limit=raw_limit, order=order)
        if filter_type in {"classic", "meta"}:
            sessions = [session for session in sessions if session.session_type == filter_type][:limit]
        if not sessions:
            return ToolResult(success=True, message="No sessions found.", data=[])
        lines = []
        for i, s in enumerate(sessions, 1):
            summary = _session_summary(s)
            themes = ", ".join(s.themes[:5]) if s.themes else "-"
            updated = s.updated_at[:10] if s.updated_at else "?"
            lines.append(
                f"#{i} session={s.id[:8]} type={s.session_type} summary={summary!r} themes={themes!r} "
                f"msgs={s.message_count} updated={updated} order={order}"
            )
        return ToolResult(
            success=True,
            message="\n".join(lines),
            data={"order": order, "session_type": filter_type or "", "sessions": [s.model_dump() for s in sessions]},
        )

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.table import Table
        from rich.text import Text

        payload = result.data or {}
        sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
        if not sessions:
            return Text("No sessions found.")
        order = payload.get("order", "recent") if isinstance(payload, dict) else "recent"
        title = f"Order: {order}"
        filter_type = payload.get("session_type", "") if isinstance(payload, dict) else ""
        if filter_type:
            title += f"  |  type: {filter_type}"
        table = Table(show_header=True, header_style="bold", border_style="blue", title=title)
        table.add_column("#", style="dim", width=3)
        table.add_column("Session", style="cyan", width=10)
        table.add_column("Type", width=8)
        table.add_column("Summary")
        table.add_column("Themes", min_width=24)
        table.add_column("Msgs", justify="right", width=5)
        table.add_column("Updated", width=12)
        for i, session in enumerate(sessions, 1):
            summary = session["summary"] or (session["title"] if session["title"] != session["id"] else "(untitled)")
            themes = ", ".join((session.get("themes") or [])[:4]) or "-"
            updated = (session.get("updated_at") or "")[:10] or "?"
            table.add_row(str(i), session["id"][:8], session.get("session_type", "classic"), summary, themes, str(session["message_count"]), updated)
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
    description = (
        "Generate or refresh the summary sentence and key themes for the current or a previous session. "
        "Use this when the user explicitly asks to summarize or refresh session metadata, or when summary/themes "
        "are genuinely missing and needed. Do not call it just to restate metadata already visible in list_sessions."
    )
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
        session_type = digest.get("session_type", "classic")
        msg = f"{summary}\nType: {session_type}\nThemes: {', '.join(themes) if themes else '-'}"
        return ToolResult(success=True, message=msg, data=digest)

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.table import Table

        data = result.data or {}
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim", no_wrap=True)
        table.add_column("value")
        table.add_row("Type", str(data.get("session_type", "classic")))
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
        if key == "tool_approval" or str(key).startswith("tool_approval."):
            return ToolResult(
                success=False,
                message="Use the dedicated tool policy admin tools instead of set_config for tool_approval.*",
            )
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
        syntax_theme = resolve_syntax_theme(ctx.config.get_nested("cli", "theme", default="default"))
        return Syntax(payload, "yaml", theme=syntax_theme, line_numbers=False, word_wrap=True)


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


class ToolPolicyListTool(InternalTool):
    name = "tool_policy_list"
    display_name = "Show Tool Policies"
    category = "tool-admin"
    danger_level = "critical"
    accent_color = "yellow"
    description = (
        "Inspect effective approval policies for every internal tool, including default risk policy, "
        "session overrides, global overrides, protected tools, and the final effective policy."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "policy": {"type": "string", "enum": ["ask", "auto"], "description": "Optional filter by effective policy."},
            "source": {
                "type": "string",
                "enum": ["default", "session", "global", "protected"],
                "description": "Optional filter by the source of the effective policy.",
            },
        },
        "required": [],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not ctx.get_tool_policy_overview:
            return ToolResult(success=False, message="Tool policy overview is unavailable.")
        policy = str(arguments.get("policy", "") or "").strip().lower() or None
        source = str(arguments.get("source", "") or "").strip().lower() or None
        overview = ctx.get_tool_policy_overview(filter_policy=policy, filter_source=source)
        counts = overview.get("counts", {})
        msg = (
            f"ask={counts.get('ask', 0)} auto={counts.get('auto', 0)} "
            f"session_overrides={counts.get('session_overrides', 0)} "
            f"global_overrides={counts.get('global_overrides', 0)} "
            f"protected={counts.get('protected', 0)}"
        )
        return ToolResult(success=True, message=msg, data=overview)

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.console import Group
        from rich.table import Table

        data = result.data or {}
        counts = data.get("counts", {})
        defaults = data.get("defaults_by_risk", {})
        summary = Table(show_header=False, box=None, padding=(0, 1))
        summary.add_column("field", style="dim", no_wrap=True)
        summary.add_column("value")
        summary.add_row("Ask", str(counts.get("ask", 0)))
        summary.add_row("Auto", str(counts.get("auto", 0)))
        summary.add_row("Session overrides", str(counts.get("session_overrides", 0)))
        summary.add_row("Global overrides", str(counts.get("global_overrides", 0)))
        summary.add_row("Protected", str(counts.get("protected", 0)))
        summary.add_row(
            "Risk defaults",
            ", ".join(f"{risk}={policy}" for risk, policy in defaults.items()) or "-",
        )

        table = Table(show_header=True, header_style="bold", border_style="yellow")
        table.add_column("Tool", style="cyan", width=18)
        table.add_column("Category", width=12)
        table.add_column("Risk", width=9)
        table.add_column("Default", width=8)
        table.add_column("Session", width=8)
        table.add_column("Global", width=8)
        table.add_column("Effective", width=9)
        table.add_column("Source", width=10)
        table.add_column("Protected", width=10)
        for row in data.get("rows", []):
            table.add_row(
                row["tool"],
                row["category"],
                row["risk"],
                row["default_policy"],
                row.get("session_override") or "-",
                row.get("global_override") or "-",
                row["effective_policy"],
                row["source"],
                "yes" if row["protected"] else "no",
            )
        return Group(summary, table)


class ToolPolicySetTool(InternalTool):
    name = "tool_policy_set"
    display_name = "Set Tool Policy"
    category = "tool-admin"
    danger_level = "critical"
    accent_color = "yellow"
    description = (
        "Change approval policy for a tool, category, or risk level. "
        "Policies are 'ask' or 'auto'. Scope can be 'session' or 'global'. "
        "Protected admin tools always require approval and cannot be changed."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "target_kind": {"type": "string", "enum": ["tool", "category", "risk"], "description": "What target to change."},
            "target": {"type": "string", "description": "Tool name, category name, or risk level."},
            "policy": {"type": "string", "enum": ["ask", "auto"], "description": "Desired approval policy."},
            "scope": {"type": "string", "enum": ["session", "global"], "description": "Where the override applies."},
            "persist": {"type": "boolean", "description": "For global scope, persist to config file immediately."},
        },
        "required": ["target", "policy"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not ctx.update_tool_policy:
            return ToolResult(success=False, message="Tool policy updates are unavailable.")
        target_kind = str(arguments.get("target_kind", "tool") or "tool")
        target = str(arguments.get("target", "") or "")
        policy = str(arguments.get("policy", "") or "")
        scope = str(arguments.get("scope", "session") or "session")
        persist = arguments.get("persist")
        try:
            change = ctx.update_tool_policy(target_kind=target_kind, target=target, policy=policy, scope=scope, persist=persist)
        except Exception as e:
            return ToolResult(success=False, message=str(e))
        msg = f"{change['scope']} {change['target_kind']} '{change['target']}' -> {change['policy']}"
        if change.get("saved_to"):
            msg += f"\nSaved to {change['saved_to']}"
        return ToolResult(success=True, message=msg, data=change)

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim")
        table.add_column("value")
        table.add_row("Target kind", str(arguments.get("target_kind", "tool")))
        table.add_row("Target", str(arguments.get("target", "")))
        table.add_row("Policy", str(arguments.get("policy", "")))
        table.add_row("Scope", str(arguments.get("scope", "session")))
        table.add_row("Persist", str(arguments.get("persist", True)))
        return [("Policy Change", table)]

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.table import Table

        data = result.data or {}
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim", no_wrap=True)
        table.add_column("value")
        table.add_row("Target", f"{data.get('target_kind', '')}:{data.get('target', '')}")
        table.add_row("Scope", str(data.get("scope", "")))
        table.add_row("Previous", str(data.get("previous") or "-"))
        table.add_row("Applied", str(data.get("policy") or "-"))
        table.add_row("Saved", str(data.get("saved_to") or "-"))
        return table


class ToolPolicyResetTool(InternalTool):
    name = "tool_policy_reset"
    display_name = "Reset Tool Policy"
    category = "tool-admin"
    danger_level = "critical"
    accent_color = "yellow"
    description = (
        "Remove a session or global override for a tool, category, or risk level, "
        "falling back to the next effective policy source. Protected admin tools cannot be changed."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "target_kind": {"type": "string", "enum": ["tool", "category", "risk"], "description": "What target to reset."},
            "target": {"type": "string", "description": "Tool name, category name, or risk level."},
            "scope": {"type": "string", "enum": ["session", "global"], "description": "Where the override should be removed."},
            "persist": {"type": "boolean", "description": "For global scope, persist to config file immediately."},
        },
        "required": ["target"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if not ctx.reset_tool_policy:
            return ToolResult(success=False, message="Tool policy reset is unavailable.")
        target_kind = str(arguments.get("target_kind", "tool") or "tool")
        target = str(arguments.get("target", "") or "")
        scope = str(arguments.get("scope", "session") or "session")
        persist = arguments.get("persist")
        try:
            change = ctx.reset_tool_policy(target_kind=target_kind, target=target, scope=scope, persist=persist)
        except Exception as e:
            return ToolResult(success=False, message=str(e))
        msg = f"Reset {change['scope']} {change['target_kind']} '{change['target']}'"
        if change.get("saved_to"):
            msg += f"\nSaved to {change['saved_to']}"
        return ToolResult(success=True, message=msg, data=change)

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim")
        table.add_column("value")
        table.add_row("Target kind", str(arguments.get("target_kind", "tool")))
        table.add_row("Target", str(arguments.get("target", "")))
        table.add_row("Scope", str(arguments.get("scope", "session")))
        table.add_row("Persist", str(arguments.get("persist", True)))
        return [("Policy Reset", table)]

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.table import Table

        data = result.data or {}
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("field", style="dim", no_wrap=True)
        table.add_column("value")
        table.add_row("Target", f"{data.get('target_kind', '')}:{data.get('target', '')}")
        table.add_row("Scope", str(data.get("scope", "")))
        table.add_row("Previous", str(data.get("previous") or "-"))
        table.add_row("Removed", str(bool(data.get("removed", False))))
        table.add_row("Saved", str(data.get("saved_to") or "-"))
        return table


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_meta_tools(registry: ToolRegistry, ctx: ToolContext) -> None:
    """Register all meta/session tools."""
    for tool_cls in [
        NewSessionTool,
        SwitchSessionTool,
        LoadSessionTool,
        ExitSessionTool,
        RenameSessionTool,
        ListSessionsTool,
        SessionExtractTool,
        SessionDigestTool,
        ToolPolicyListTool,
        ToolPolicySetTool,
        ToolPolicyResetTool,
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
