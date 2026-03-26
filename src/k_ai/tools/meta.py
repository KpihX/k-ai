# src/k_ai/tools/meta.py
"""
Meta/session management tools.

These tools handle session lifecycle: new, load, exit, rename, list, delete,
compact, clear screen, and set_config.
"""
from typing import Any, Dict

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
    description = "Start a new chat session, clearing the current context."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.request_new_session:
            ctx.request_new_session()
        return ToolResult(success=True, message="New session started.")


class LoadSessionTool(InternalTool):
    name = "load_session"
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
        if ctx.request_load_session:
            ctx.request_load_session(meta.id)
        return ToolResult(
            success=True,
            message=f"Resumed session '{meta.title}' ({meta.message_count} messages).",
        )


class ExitSessionTool(InternalTool):
    name = "exit_session"
    description = "End the current chat session and exit k-ai."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.request_exit:
            ctx.request_exit()
        return ToolResult(success=True, message="Goodbye!")


class RenameSessionTool(InternalTool):
    name = "rename_session"
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
    description = (
        "List recent chat sessions. Each session shows its position number (#1, #2...) "
        "and its unique ID (8-char hex). IMPORTANT: when referring to a session in other "
        "tools (delete, load, rename), always use the 8-char hex ID, NOT the position number."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max sessions (default from config sessions.max_recent).",
            },
        },
        "required": [],
    }
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        max_recent = ctx.config.get_nested("sessions", "max_recent", default=10)
        limit = arguments.get("limit", max_recent)
        sessions = ctx.session_store.list_sessions(limit=limit)
        if not sessions:
            return ToolResult(success=True, message="No sessions found.", data=[])
        lines = []
        for i, s in enumerate(sessions, 1):
            title = s.title or "(untitled)"
            summary = f" - {s.summary}" if s.summary else ""
            lines.append(f"#{i} ID={s.id[:8]} \"{title}\"{summary} ({s.message_count} msgs)")
        return ToolResult(
            success=True,
            message="\n".join(lines),
            data=[s.model_dump() for s in sessions],
        )


class DeleteSessionTool(InternalTool):
    name = "delete_session"
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
    description = "Clear the terminal screen without affecting the conversation."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx.console.clear()
        return ToolResult(success=True, message="Screen cleared.")


class SetConfigTool(InternalTool):
    name = "set_config"
    description = (
        "Change a runtime configuration value. "
        "Supports dot-notation for nested keys (e.g. 'cli.theme')."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "Config key (e.g. 'temperature')."},
            "value": {"type": "string", "description": "New value as string."},
        },
        "required": ["key", "value"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        key = arguments.get("key", "")
        value = arguments.get("value", "")
        try:
            ctx.config.set(key, value)
            return ToolResult(success=True, message=f"{key} = {value}")
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to set {key}: {e}")


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
        DeleteSessionTool,
        CompactSessionTool,
        ClearScreenTool,
        SetConfigTool,
    ]:
        registry.register(tool_cls())
