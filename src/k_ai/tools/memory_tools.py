# src/k_ai/tools/memory_tools.py
"""
Memory management tools: add, list, remove entries from the internal memory.
"""
from typing import Any, Dict

from ..models import ToolResult
from .base import InternalTool, ToolContext, ToolRegistry


class MemoryAddTool(InternalTool):
    name = "memory_add"
    description = "Add a new fact or preference to the persistent memory."
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to remember.",
            },
        },
        "required": ["text"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        text = arguments.get("text", "").strip()
        if not text:
            return ToolResult(success=False, message="Text cannot be empty.")
        entry = ctx.memory.add(text)
        return ToolResult(
            success=True,
            message=f"Remembered (#{entry.id}): {text}",
        )


class MemoryListTool(InternalTool):
    name = "memory_list"
    description = "List all entries in the persistent memory."
    parameters_schema = {"type": "object", "properties": {}, "required": []}
    requires_approval = False

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        entries = ctx.memory.list_entries()
        if not entries:
            return ToolResult(success=True, message="Memory is empty.", data=[])
        lines = [f"#{e.id}: {e.text}" for e in entries]
        return ToolResult(
            success=True,
            message="\n".join(lines),
            data=[e.model_dump() for e in entries],
        )


class MemoryRemoveTool(InternalTool):
    name = "memory_remove"
    description = "Remove a memory entry by its ID number."
    parameters_schema = {
        "type": "object",
        "properties": {
            "entry_id": {
                "type": "integer",
                "description": "The ID of the memory entry to remove.",
            },
        },
        "required": ["entry_id"],
    }
    requires_approval = True

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        entry_id = arguments.get("entry_id")
        if entry_id is None:
            return ToolResult(success=False, message="entry_id is required.")
        removed = ctx.memory.remove(int(entry_id))
        if removed:
            return ToolResult(success=True, message=f"Memory entry #{entry_id} removed.")
        return ToolResult(success=False, message=f"Entry #{entry_id} not found.")


def register_memory_tools(registry: ToolRegistry, ctx: ToolContext) -> None:
    for tool_cls in [MemoryAddTool, MemoryListTool, MemoryRemoveTool]:
        registry.register(tool_cls())
