# src/k_ai/tools/__init__.py
"""
Internal tool system for k-ai.

Every action (session management, memory, external tools) is a registered
tool that can be invoked by slash commands OR by LLM tool calls.
"""
from .base import InternalTool, ToolContext, ToolRegistry
from .meta import register_meta_tools
from .mcp_admin import register_mcp_admin_tools
from .external import register_external_tools
from .mcp import register_mcp_tools
from .qmd import register_qmd_tools
from .skills import register_skill_tools


def create_registry(context: ToolContext, dynamic_tools: list[InternalTool] | None = None) -> ToolRegistry:
    """Create and populate a ToolRegistry with all internal tools."""
    registry = ToolRegistry()
    register_meta_tools(registry, context)
    register_mcp_admin_tools(registry, context)
    register_external_tools(registry, context)
    register_qmd_tools(registry, context)
    register_skill_tools(registry, context)
    register_mcp_tools(registry, dynamic_tools or [])
    return registry


__all__ = [
    "InternalTool",
    "ToolContext",
    "ToolRegistry",
    "create_registry",
]
