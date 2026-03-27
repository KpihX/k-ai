"""Dynamic MCP-backed tools."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from ..mcp import MCPManager, MCPToolSpec
from ..models import ToolResult
from .base import InternalTool, ToolContext, ToolRegistry


class MCPToolAdapter(InternalTool):
    category = "mcp"
    accent_color = "cyan"
    requires_catalog_entry = False

    def __init__(self, *, manager: MCPManager, spec: MCPToolSpec, ctx: ToolContext):
        self._manager = manager
        self._spec = spec
        self._ctx = ctx
        self.name = spec.qualified_name
        self.display_name = manager.config_text(
            "mcp",
            "tool_naming",
            "display_name_template",
            default="{server_title} / {tool_name}",
            server_title=spec.server_title,
            tool_name=spec.name,
        )
        self.description = spec.description or f"MCP tool {spec.name} exposed by server {spec.server_name}."
        self.parameters_schema = dict(spec.input_schema or {"type": "object"})
        self.danger_level, self.requires_approval = self._approval_policy(spec.annotations)

    def _approval_policy(self, annotations: Dict[str, Any]) -> tuple[str, bool]:
        read_only = bool(annotations.get("readOnlyHint", False))
        destructive = bool(annotations.get("destructiveHint", False))
        if read_only:
            return (
                self._ctx.config.get_nested("mcp", "approval", "read_only_risk", default="low"),
                bool(self._ctx.config.get_nested("mcp", "approval", "read_only_requires_approval", default=False)),
            )
        if destructive:
            return (
                self._ctx.config.get_nested("mcp", "approval", "destructive_risk", default="high"),
                bool(self._ctx.config.get_nested("mcp", "approval", "destructive_requires_approval", default=True)),
            )
        return (
            self._ctx.config.get_nested("mcp", "approval", "default_risk", default="medium"),
            bool(self._ctx.config.get_nested("mcp", "approval", "default_requires_approval", default=True)),
        )

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        result = await self._manager.call_tool(self.name, arguments or {})
        return ToolResult(
            success=result.success,
            message=result.message,
            data=dict(result.raw),
        )


def build_mcp_tools(manager: MCPManager, specs: Iterable[MCPToolSpec], ctx: ToolContext) -> List[InternalTool]:
    return [MCPToolAdapter(manager=manager, spec=spec, ctx=ctx) for spec in specs]


def register_mcp_tools(registry: ToolRegistry, tools: Iterable[InternalTool]) -> None:
    for tool in tools:
        registry.register(tool)
