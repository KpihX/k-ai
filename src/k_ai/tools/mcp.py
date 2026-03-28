"""Dynamic MCP-backed tools."""

from __future__ import annotations

from difflib import unified_diff
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rich.console import Group
from rich.syntax import Syntax
from rich.text import Text

from ..mcp import MCPManager, MCPToolSpec
from ..models import ToolResult
from ..ui_theme import resolve_syntax_theme
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

    @staticmethod
    def _window_lines(lines: list[str], max_lines: int, mode: str) -> tuple[list[str], bool]:
        if max_lines <= 0 or len(lines) <= max_lines:
            return lines, False
        normalized = str(mode or "split").strip().lower()
        if normalized == "head":
            return lines[:max_lines], True
        if normalized == "tail":
            return lines[-max_lines:], True
        head_count = max_lines // 2
        tail_count = max_lines - head_count
        if head_count <= 0:
            return lines[-tail_count:], True
        if tail_count <= 0:
            return lines[:head_count], True
        return lines[:head_count] + lines[-tail_count:], True

    def _filesystem_preview_setting(self, key: str, default: Any) -> Any:
        value = self._ctx.config.get_nested("mcp", "runtime", "proposals", "filesystem", key, default=None)
        if value is None:
            value = self._ctx.config.get_nested("cli", "tool_proposals", "filesystem", key, default=default)
        return default if value is None else value

    def _filesystem_preview_settings(self) -> tuple[int, int, str]:
        max_preview = int(self._filesystem_preview_setting("preview_max_lines", 96) or 96)
        max_diff = int(self._filesystem_preview_setting("diff_max_lines", 120) or 120)
        mode = str(self._filesystem_preview_setting("line_window_mode", "split") or "split").strip().lower()
        if mode not in {"head", "tail", "split"}:
            mode = "split"
        return max_preview, max_diff, mode

    @staticmethod
    def _lexer_for_path(path_text: str) -> str:
        suffix = Path(path_text).suffix.lower()
        if suffix == ".py":
            return "python"
        if suffix in {".md", ".markdown"}:
            return "markdown"
        if suffix in {".json"}:
            return "json"
        if suffix in {".yaml", ".yml"}:
            return "yaml"
        if suffix in {".toml"}:
            return "toml"
        if suffix in {".sh", ".bash", ".zsh"}:
            return "bash"
        if suffix in {".js", ".mjs", ".cjs"}:
            return "javascript"
        if suffix in {".ts"}:
            return "typescript"
        if suffix in {".html", ".htm"}:
            return "html"
        if suffix in {".css"}:
            return "css"
        return "text"

    def _filesystem_write_preview(self, arguments: Dict[str, Any]) -> list[tuple[str, Any]]:
        path_text = str(arguments.get("path", "") or "").strip()
        content = str(arguments.get("content", "") or "")
        max_preview, _max_diff, mode = self._filesystem_preview_settings()
        lines = content.splitlines()
        display_lines, truncated = self._window_lines(lines, max_preview, mode)
        preview_text = "\n".join(display_lines)
        if content.endswith("\n") and display_lines:
            preview_text += "\n"
        if truncated:
            hidden = max(0, len(lines) - len(display_lines))
            marker = f"\n# ... {hidden} line(s) hidden via {mode} window ..."
            preview_text = (preview_text + marker).strip("\n")
        syntax_theme = resolve_syntax_theme(self._ctx.config.get_nested("cli", "theme", default="default"))
        title = str(self._filesystem_preview_setting("create_title", "New File") or "New File")
        caption = str(
            self._filesystem_preview_setting("create_caption", "Create / overwrite preview") or "Create / overwrite preview"
        )
        body = Group(
            Text(path_text or "(no path)", style="bold #7dd3fc"),
            Text(caption, style="dim"),
            Syntax(preview_text or "", self._lexer_for_path(path_text), theme=syntax_theme, line_numbers=True, word_wrap=True),
        )
        return [(title, body)]

    def _filesystem_edit_preview(self, arguments: Dict[str, Any]) -> list[tuple[str, Any]]:
        path_text = str(arguments.get("path", "") or "").strip()
        edits = list(arguments.get("edits", []) or [])
        diff_lines: list[str] = []
        for index, edit in enumerate(edits, start=1):
            old_text = str(dict(edit).get("oldText", "") or "")
            new_text = str(dict(edit).get("newText", "") or "")
            old_lines = old_text.splitlines()
            new_lines = new_text.splitlines()
            label = f"edit-{index}"
            diff_lines.extend(
                unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"{path_text}:{label}:before",
                    tofile=f"{path_text}:{label}:after",
                    lineterm="",
                )
            )
            if index < len(edits):
                diff_lines.append("")
        _max_preview, max_diff, mode = self._filesystem_preview_settings()
        display_lines, truncated = self._window_lines(diff_lines, max_diff, mode)
        title = str(self._filesystem_preview_setting("diff_title", "Planned Diff") or "Planned Diff")
        caption = str(self._filesystem_preview_setting("diff_caption", "Planned diff") or "Planned diff")
        text = Text()
        for line in display_lines:
            style = None
            if line.startswith("+++ ") or line.startswith("--- "):
                style = "bold cyan"
            elif line.startswith("@@"):
                style = "bold yellow"
            elif line.startswith("+") and not line.startswith("+++"):
                style = "#86efac"
            elif line.startswith("-") and not line.startswith("---"):
                style = "#fca5a5"
            elif not line:
                style = "dim"
            text.append(line + "\n", style=style)
        if truncated:
            hidden = max(0, len(diff_lines) - len(display_lines))
            text.append(f"... {hidden} diff line(s) hidden via {mode} window ...", style="dim")
        body = Group(
            Text(path_text or "(no path)", style="bold white"),
            Text(caption, style="dim"),
            text,
        )
        return [(title, body)]

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            result = await self._manager.call_tool(self.name, arguments or {})
        except Exception as exc:
            return ToolResult(success=False, message=str(exc) or exc.__class__.__name__, data={"error": str(exc or "")})
        data = dict(result.raw)
        data["__tool_name"] = self._spec.name
        data["__requested_path"] = str((arguments or {}).get("path", "") or "")
        return ToolResult(
            success=result.success,
            message=result.message,
            data=data,
        )

    def proposal_sections(self, arguments: Dict[str, Any], ctx: ToolContext) -> list[tuple[str, Any]]:
        if self._spec.server_name == "filesystem":
            if self._spec.name == "write_file":
                return self._filesystem_write_preview(arguments)
            if self._spec.name == "edit_file":
                return self._filesystem_edit_preview(arguments)
        return super().proposal_sections(arguments, ctx)

    def result_renderable(self, result: ToolResult, max_display_length: int, ctx: ToolContext) -> Any:
        if self._spec.server_name == "filesystem" and result.success and self._spec.name in {"write_file", "edit_file"}:
            path_text = str(result.data.get("__requested_path", "") or "(unknown path)")
            if self._spec.name == "write_file":
                return Text(f"Confirmed write to {path_text}.", style="green")
            return Text(f"Confirmed edit to {path_text}.", style="green")
        return super().result_renderable(result, max_display_length, ctx)


def build_mcp_tools(manager: MCPManager, specs: Iterable[MCPToolSpec], ctx: ToolContext) -> List[InternalTool]:
    return [MCPToolAdapter(manager=manager, spec=spec, ctx=ctx) for spec in specs]


def register_mcp_tools(registry: ToolRegistry, tools: Iterable[InternalTool]) -> None:
    for tool in tools:
        registry.register(tool)
