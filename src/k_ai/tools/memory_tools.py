# src/k_ai/tools/memory_tools.py
"""
Memory management tools: add, list, remove entries from the internal memory.
"""
from typing import Any, Dict

from ..models import ToolResult
from .base import InternalTool, ToolContext, ToolRegistry


class MemoryAddTool(InternalTool):
    name = "memory_add"
    display_name = "Add Memory"
    category = "memory"
    danger_level = "medium"
    accent_color = "magenta"
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
    display_name = "List Memory"
    category = "memory"
    danger_level = "low"
    accent_color = "magenta"
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

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        from rich.table import Table
        from rich.text import Text

        entries = result.data or []
        if not entries:
            return Text("Memory is empty.")
        table = Table(show_header=True, header_style="bold", border_style="magenta")
        table.add_column("ID", style="magenta", width=6)
        table.add_column("Text")
        for entry in entries:
            table.add_row(str(entry["id"]), entry["text"])
        return table


class MemoryRemoveTool(InternalTool):
    name = "memory_remove"
    display_name = "Remove Memory"
    category = "memory"
    danger_level = "high"
    accent_color = "magenta"
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
