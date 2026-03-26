# src/k_ai/tools/base.py
"""
Base classes for the internal tool system.

Every tool follows the same pattern:
  1. The LLM (or a slash command) proposes a tool call.
  2. If requires_approval, the UI asks the user to validate.
  3. The tool executes and returns a ToolResult.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Awaitable, Dict, List, Optional

from ..models import ToolResult

if TYPE_CHECKING:
    from ..config import ConfigManager
    from ..memory import MemoryStore
    from ..session_store import SessionStore
    from rich.console import Console


@dataclass
class ToolContext:
    """
    Shared context passed to every tool at execution time.

    Avoids tools importing session directly (prevents circular deps).
    Fields are set by the ChatSession at boot time.
    """

    config: "ConfigManager"
    memory: "MemoryStore"
    session_store: "SessionStore"
    console: "Console"
    # Callbacks into the session (set after session init)
    get_history: Optional[Callable] = None
    set_history: Optional[Callable] = None
    get_session_id: Optional[Callable] = None
    set_session_id: Optional[Callable] = None
    get_system_prompt: Optional[Callable] = None
    reload_provider: Optional[Callable] = None
    request_exit: Optional[Callable] = None
    request_new_session: Optional[Callable] = None
    request_load_session: Optional[Callable] = None
    request_compact: Optional[Callable] = None


class InternalTool(ABC):
    """Abstract base class for an internal tool."""

    name: str
    description: str
    parameters_schema: Dict[str, Any]
    requires_approval: bool = True

    @abstractmethod
    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Execute the tool with the given arguments."""

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI-format tool definition for LLM tool calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


class ToolRegistry:
    """Registry of all available internal tools."""

    def __init__(self):
        self._tools: Dict[str, InternalTool] = {}

    def register(self, tool: InternalTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[InternalTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[InternalTool]:
        return list(self._tools.values())

    def get_names(self) -> List[str]:
        return sorted(self._tools.keys())

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Generate OpenAI-format tool definitions for all registered tools."""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def is_internal(self, tool_name: str) -> bool:
        """Check if a tool name corresponds to an internal tool."""
        return tool_name in self._tools
