# src/k_ai/tools/__init__.py
"""
Internal tool system for k-ai.

Every action (session management, memory, external tools) is a registered
tool that can be invoked by slash commands OR by LLM tool calls.
"""
from .base import InternalTool, ToolContext, ToolRegistry
from .meta import register_meta_tools
from .memory_tools import register_memory_tools
from .external import register_external_tools
from .qmd import register_qmd_tools


def create_registry(context: ToolContext) -> ToolRegistry:
    """Create and populate a ToolRegistry with all internal tools."""
    registry = ToolRegistry()
    register_meta_tools(registry, context)
    register_memory_tools(registry, context)
    register_external_tools(registry, context)
    register_qmd_tools(registry, context)
    return registry


__all__ = [
    "InternalTool",
    "ToolContext",
    "ToolRegistry",
    "create_registry",
]
