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
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, Callable, Awaitable, Dict, List, Optional

from ..models import ToolResult
from ..ui_theme import resolve_syntax_theme

if TYPE_CHECKING:
    from ..config import ConfigManager
    from ..memory import MemoryStore
    from ..session_store import SessionStore
    from rich.console import Console


@dataclass(frozen=True)
class ToolDisplaySpec:
    display_name: str
    description: str = ""
    category: str = "general"
    danger_level: str = "low"
    accent_color: str = "cyan"


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
    request_init: Optional[Callable[[], None]] = None
    complete_init: Optional[Callable[[], None]] = None
    apply_config_change: Optional[Callable[..., Dict[str, Any]]] = None
    generate_session_digest: Optional[Callable[..., Awaitable[dict[str, Any]]]] = None
    get_runtime_snapshot: Optional[Callable[..., Dict[str, Any]]] = None
    get_tool_policy_overview: Optional[Callable[..., Dict[str, Any]]] = None
    update_tool_policy: Optional[Callable[..., Dict[str, Any]]] = None
    reset_tool_policy: Optional[Callable[..., Dict[str, Any]]] = None
    is_interrupt_requested: Optional[Callable[[], bool]] = None


class InternalTool(ABC):
    """Abstract base class for an internal tool."""

    name: str
    description: str
    parameters_schema: Dict[str, Any]
    requires_approval: bool = True
    display_name: Optional[str] = None
    category: str = "general"
    danger_level: str = "low"
    accent_color: str = "cyan"

    @abstractmethod
    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Execute the tool with the given arguments."""

    def display_spec(self) -> ToolDisplaySpec:
        return ToolDisplaySpec(
            display_name=self.display_name or self.name,
            description=self.description,
            category=self.category,
            danger_level=self.danger_level,
            accent_color=self.accent_color,
        )

    def proposal_rationale(self, arguments: Dict[str, Any]) -> str:
        """
        Fallback justification shown when the model did not provide one.

        Keep it short and grounded in the tool description plus one salient
        argument when available, so the approval panel remains transparent even
        when the assistant emits only a raw tool call.
        """
        description = (self.description or f"Use {self.name}.").strip()
        argument_hint = ""
        for key in ("query", "command", "code", "path", "key", "mode", "session_id"):
            value = arguments.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            if len(text) > 120:
                text = text[:117] + "..."
            argument_hint = f" Main input: {key}={text}"
            break
        return f"Use {self.name} to {description[0].lower() + description[1:]}{argument_hint}"

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> List[tuple[str, Any]]:
        """Renderable sections for the tool proposal panel."""
        from rich.syntax import Syntax

        payload = json.dumps(arguments or {}, indent=2, ensure_ascii=False, sort_keys=True)
        syntax_theme = resolve_syntax_theme(ctx.config.get_nested("cli", "theme", default="default"))
        return [("Arguments", Syntax(payload, "json", theme=syntax_theme, line_numbers=False, word_wrap=True))]

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        """Renderable body for the tool result panel."""
        from rich.syntax import Syntax
        from rich.text import Text

        msg = result.message
        if len(msg) > max_display_length:
            msg = msg[:max_display_length] + "\n...(truncated)"
        syntax_theme = resolve_syntax_theme(ctx.config.get_nested("cli", "theme", default="default"))

        if not result.success and ("Traceback" in msg or "Error" in msg):
            return Syntax(msg, "pytb", theme=syntax_theme, word_wrap=True)
        return Text(msg) if msg else Text("[dim](no output)[/dim]")

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
