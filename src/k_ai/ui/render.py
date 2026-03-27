# src/k_ai/ui/render.py
"""
Rich-based rendering for k-ai: streaming responses, session tables,
tool proposals, and tool result display.
"""
from typing import List, Optional, Sequence, Tuple

from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from ..models import CompletionChunk, SessionMetadata, ToolResult, TokenUsage
from ..tools.base import ToolDisplaySpec
from ..ui_theme import resolve_ui_theme
from .markdown import render_content

_STATUS_STYLES = {
    "info": ("cyan", "Info"),
    "success": ("green", "Success"),
    "warning": ("yellow", "Warning"),
    "error": ("red", "Error"),
}

_DANGER_STYLES = {
    "low": ("cyan", "Low risk"),
    "medium": ("yellow", "Requires attention"),
    "high": ("red", "High impact"),
    "critical": ("magenta", "Always ask"),
}


def render_assistant_panel(content: str, model_name: str, render_mode: str = "rich", usage: Optional[TokenUsage] = None, theme_name: str = "default") -> Panel:
    theme = resolve_ui_theme(theme_name)
    subtitle = None
    if usage and usage.total_tokens:
        subtitle = f"[dim]{usage.prompt_tokens} in / {usage.completion_tokens} out[/dim]"
    return Panel(
        render_content(content, render_mode),
        title=f"[bold green]Assistant[/bold green] [dim]{model_name}[/dim]",
        subtitle=subtitle,
        border_style=str(theme.get("assistant_border", "green")),
        expand=True,
        padding=(0, 1),
    )


def render_user_panel(content: str, theme_name: str = "default") -> Panel:
    theme = resolve_ui_theme(theme_name)
    return Panel(
        render_content(content, "rich"),
        title="[bold white]User[/bold white]",
        border_style=str(theme.get("user_border", "bright_black")),
        style=str(theme.get("user_style", "on #1f2329")),
        expand=True,
        padding=(0, 1),
    )


def render_runtime_panel(snapshot: dict, title: str = "Runtime Transparency", mode: str = "compact", theme_name: str = "default") -> Panel:
    theme = resolve_ui_theme(theme_name)
    context_used = int(snapshot.get("estimated_context_tokens", 0) or 0)
    context_total = int(snapshot.get("context_window", 0) or 0)
    context_pct = float(snapshot.get("context_percent", 0.0) or 0.0)
    threshold_tokens = int(snapshot.get("compaction_trigger_tokens", 0) or 0)
    threshold_pct = int(snapshot.get("compaction_trigger_percent", 0) or 0)
    themes = snapshot.get("session_themes", []) or []

    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(justify="right")
    header.add_row(
        Text.from_markup(
            f"[bold white]{snapshot.get('provider', '?')}[/bold white]"
            f"[dim] / {snapshot.get('model', '?')}[/dim]"
        ),
        Text.from_markup(
            f"[dim]temp[/dim] [bold]{snapshot.get('temperature')}[/bold]   "
            f"[dim]max[/dim] [bold]{snapshot.get('max_tokens')}[/bold]"
        ),
    )

    body = Table.grid(expand=True)
    body.add_column(style="dim", no_wrap=True)
    body.add_column()
    body.add_column(style="dim", no_wrap=True)
    body.add_column()
    body.add_row(
        "Session",
        snapshot.get("session_id")[:8] if snapshot.get("session_id") else "(none)",
        "Type",
        str(snapshot.get("session_type", "") or "-"),
    )
    body.add_row(
        "CWD",
        str(snapshot.get("cwd", "") or "-"),
        "",
        "",
    )
    body.add_row(
        "Context",
        f"{context_used:,} / {context_total:,} tok  ({context_pct:.1f}%)",
        "Auth",
        str(snapshot.get("auth_mode", "n/a")),
    )
    body.add_row(
        "Compaction",
        f"{threshold_tokens:,} tok  ({threshold_pct}%)" if threshold_tokens else "disabled",
        "Remaining",
        f"{int(snapshot.get('remaining_context_tokens', 0) or 0):,} tok",
    )
    body.add_row(
        "Tokens",
        f"{int(snapshot.get('prompt_tokens', 0) or 0):,} in / "
        f"{int(snapshot.get('completion_tokens', 0) or 0):,} out / "
        f"{int(snapshot.get('total_tokens', 0) or 0):,} total"
        + (" [dim](estimated)[/dim]" if snapshot.get("token_source") == "estimated" else ""),
        "History",
        f"{int(snapshot.get('history_messages', 0) or 0)} msgs",
    )
    body.add_row("Stream", str(snapshot.get("stream")), "", "")
    skills_summary = str(snapshot.get("skills_summary", "") or "").strip()
    skills_catalog_count = int(snapshot.get("skills_catalog_count", 0) or 0)
    hooks_summary = str(snapshot.get("hooks_summary", "") or "").strip()
    hooks_count = int(snapshot.get("hooks_count", 0) or 0)
    mcp_summary = str(snapshot.get("mcp_summary", "") or "").strip()
    mcp_server_count = int(snapshot.get("mcp_server_count", 0) or 0)
    mcp_tool_count = int(snapshot.get("mcp_tool_count", 0) or 0)
    mcp_resource_count = int(snapshot.get("mcp_resource_count", 0) or 0)
    mcp_prompt_count = int(snapshot.get("mcp_prompt_count", 0) or 0)
    if skills_summary:
        body.add_row("Skills", skills_summary, "Catalog", f"{skills_catalog_count} discovered")
    if hooks_summary:
        body.add_row("Hooks", hooks_summary, "Hook cfg", f"{hooks_count} matcher(s)")
    if mcp_summary:
        body.add_row(
            "MCP",
            mcp_summary,
            "Catalog",
            f"{mcp_server_count} server(s) / {mcp_tool_count} tool(s) / {mcp_resource_count} resource(s) / {mcp_prompt_count} prompt(s)",
        )

    if mode in {"full", "welcome"}:
        approval_counts = snapshot.get("approval_counts", {}) or {}
        defaults = snapshot.get("approval_defaults", {}) or {}
        body.add_row(
            "Render",
            str(snapshot.get("render_mode", "rich")),
            "Tools",
            f"display {snapshot.get('tool_result_max_display')} / history {snapshot.get('tool_result_max_history')}",
        )
        body.add_row(
            "Approvals",
            ", ".join(f"{risk}={policy}" for risk, policy in defaults.items()) or "-",
            "Persist path",
            str(snapshot.get("persist_path", "")),
        )
        body.add_row(
            "Token source",
            str(snapshot.get("token_source", "unknown")),
            "Provider usage",
            f"{int(snapshot.get('provider_total_tokens', 0) or 0):,} total",
        )
        body.add_row(
            "Policy counts",
            f"ask {approval_counts.get('ask', 0)} / auto {approval_counts.get('auto', 0)}",
            "Overrides",
            f"session {approval_counts.get('session_overrides', 0)} / global {approval_counts.get('global_overrides', 0)}",
        )
        if skills_summary:
            body.add_row(
                "Skill mode",
                str(snapshot.get("skills_visibility_mode", "announce")),
                "",
                "",
            )

    parts = [header, Rule(style="dim"), body]
    if snapshot.get("session_summary"):
        digest = Table.grid(expand=True)
        digest.add_column(style="dim", width=8)
        digest.add_column()
        digest.add_row("Summary", str(snapshot.get("session_summary")))
        digest.add_row("Themes", ", ".join(themes[:6]) if themes else "-")
        parts.extend([Rule(style="dim"), digest])

    return Panel(
        Group(*parts),
        title=f"[bold cyan]{title}[/bold cyan]",
        border_style=str(theme.get("runtime_border", "cyan")),
        expand=True,
        padding=(0, 1),
    )


def render_thinking_panel(content: str, render_mode: str = "rich", active: bool = False, theme_name: str = "default") -> Panel:
    theme = resolve_ui_theme(theme_name)
    return Panel(
        render_content(content, render_mode),
        title="[bold yellow]Reasoning[/bold yellow]",
        subtitle="[dim]streaming[/dim]" if active else None,
        border_style=str(theme.get("thinking_border", "yellow")),
        expand=True,
        padding=(0, 1),
    )


def render_assistant_stream_panel(
    content: str,
    model_name: str,
    *,
    render_mode: str = "rich",
    usage: Optional[TokenUsage] = None,
    theme_name: str = "default",
    initial: bool = False,
) -> Panel:
    theme = resolve_ui_theme(theme_name)
    subtitle = None
    if usage and usage.total_tokens:
        subtitle = f"[dim]{usage.prompt_tokens} in / {usage.completion_tokens} out[/dim]"
    return Panel(
        render_content(content, render_mode),
        title=f"[bold green]Assistant[/bold green] [dim]{model_name}[/dim]" if initial else None,
        subtitle=subtitle,
        border_style=str(theme.get("assistant_border", "green")),
        expand=True,
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# StreamingRenderer — handles the live streaming display
# ---------------------------------------------------------------------------

class StreamingRenderer:
    """
    Context-manager that renders a streaming LLM response in the terminal.

    Display states:
      1. Spinner       — waiting for the first token (transient)
      2. Thinking panel — <think>...</think> content (transient, then committed)
      3. Response stream — append-only header + incremental static chunks

    Content is streamed append-only after the first token instead of using one
    giant full-height Live panel. This avoids clipping, redraw blinking, and
    delayed visibility on very long answers.
    """

    def __init__(
        self,
        console: Console,
        model_name: str,
        render_mode: str = "rich",
        spinner_name: str = "dots",
        theme_name: str = "default",
        flush_min_chars: int = 600,
        tail_chars: int = 120,
        interrupt_hint: str = "",
    ):
        self.console = console
        self.model_name = model_name
        self.render_mode = render_mode
        self.spinner_name = spinner_name or "dots"
        self.theme_name = theme_name or "default"
        self.flush_min_chars = max(int(flush_min_chars or 0), 80)
        self.tail_chars = max(int(tail_chars or 0), 0)
        self.interrupt_hint = str(interrupt_hint or "").strip()

        self.full_content: str = ""
        self.full_thought: str = ""
        self.last_usage: Optional[TokenUsage] = None

        self._thinking_committed: bool = False
        self._content_started: bool = False
        self._stream_header_printed: bool = False
        self._flushed_chars: int = 0
        self._live: Optional[Live] = None

    def __enter__(self) -> "StreamingRenderer":
        spinner_text = "[dim]Generating...[/dim]"
        if self.interrupt_hint:
            spinner_text = f"{spinner_text} [dim]({self.interrupt_hint})[/dim]"
        try:
            spinner = Spinner(self.spinner_name, text=spinner_text)
        except Exception:
            spinner = Spinner("dots", text=spinner_text)
        self._live = Live(
            spinner,
            console=self.console,
            refresh_per_second=15,
            transient=True,  # Spinner disappears when content starts
        )
        self._live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._live:
            self._live.stop()
            self._live = None

        # Only print static panels for content that was NOT already shown
        # via a non-transient Live context.
        if exc_type is not None:
            return

        # If thinking was shown but never committed (no content followed),
        # print the thinking panel statically.
        if self.full_thought and not self._thinking_committed:
            self.console.print(render_thinking_panel(self.full_thought, self.render_mode, theme_name=self.theme_name))

        if self.full_content:
            self._begin_content_stream()
            self._flush_streamed_content(final=True)

    def update(self, chunk: CompletionChunk) -> None:
        if not self._live:
            return
        if chunk.usage:
            self.last_usage = chunk.usage
        if chunk.delta_thought:
            self.full_thought += chunk.delta_thought
        if chunk.delta_content:
            self.full_content += chunk.delta_content
        self._refresh_live()

    def _refresh_live(self) -> None:
        assert self._live is not None

        has_content = bool(self.full_content)
        has_thought = bool(self.full_thought)

        # Transition: first content token arrived
        if has_content and not self._content_started:
            self._begin_content_stream()
            self._flush_streamed_content(final=False)
            return

        if has_content:
            self._flush_streamed_content(final=False)
        elif has_thought:
            self._live.update(render_thinking_panel(self.full_thought, self.render_mode, active=True, theme_name=self.theme_name))

    def _begin_content_stream(self) -> None:
        if self._content_started:
            return
        has_thought = bool(self.full_thought)
        if self._live:
            self._live.stop()
        if has_thought and not self._thinking_committed:
            self.console.print(render_thinking_panel(self.full_thought, self.render_mode, theme_name=self.theme_name))
            self._thinking_committed = True
        self._content_started = True

    def _flush_streamed_content(self, *, final: bool) -> None:
        pending = self.full_content[self._flushed_chars:]
        if not pending:
            return
        boundary = self._find_flush_boundary(pending, final=final)
        if boundary <= 0:
            return
        chunk = pending[:boundary]
        self.console.print(
            render_assistant_stream_panel(
                chunk,
                self.model_name,
                render_mode=self.render_mode,
                usage=self.last_usage if final else None,
                theme_name=self.theme_name,
                initial=not self._stream_header_printed,
            )
        )
        self._stream_header_printed = True
        self._flushed_chars += boundary

    def _find_flush_boundary(self, pending: str, *, final: bool) -> int:
        if not pending:
            return 0
        if final:
            return len(pending)

        last_blank_boundary = -1
        last_line_boundary = -1
        offset = 0
        in_fence = False

        for line in pending.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = not in_fence
            offset += len(line)
            if in_fence:
                continue
            if line.endswith("\n"):
                last_line_boundary = offset
                if stripped.strip() == "":
                    last_blank_boundary = offset

        if last_blank_boundary > 0:
            return last_blank_boundary

        if len(pending) >= self.flush_min_chars and last_line_boundary > 0:
            if len(pending) - last_line_boundary >= self.tail_chars:
                return last_line_boundary

        if len(pending) >= self.flush_min_chars + self.tail_chars:
            return len(pending) - self.tail_chars

        return 0


# ---------------------------------------------------------------------------
# Session list display
# ---------------------------------------------------------------------------

def render_sessions_table(
    console: Console,
    sessions: List[SessionMetadata],
    title: str = "Recent Sessions",
) -> None:
    if not sessions:
        console.print("[dim]No previous sessions found.[/dim]")
        return

    table = Table(
        title=f"[bold cyan]{title}[/bold cyan]",
        show_header=True,
        header_style="bold",
        border_style="dim",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Session", style="cyan", width=10)
    table.add_column("Type", width=8)
    table.add_column("Summary", min_width=42)
    table.add_column("Themes", min_width=24)
    table.add_column("Msgs", justify="right", width=5)
    table.add_column("Updated", width=12)

    for i, s in enumerate(sessions, 1):
        summary_source = s.summary or (s.title if s.title != s.id else "")
        summary = (summary_source[:86] + "...") if len(summary_source) > 86 else (summary_source or "[dim]-[/dim]")
        themes_source = ", ".join(s.themes[:4]) if s.themes else "-"
        themes = (themes_source[:40] + "...") if len(themes_source) > 40 else themes_source
        updated = s.updated_at[:10] if s.updated_at else "?"
        table.add_row(str(i), s.id[:8], s.session_type, summary, themes, str(s.message_count), updated)

    console.print(table)


# ---------------------------------------------------------------------------
# Tool result display (in styled panel)
# ---------------------------------------------------------------------------

def render_tool_result(
    console: Console,
    spec: ToolDisplaySpec,
    result: ToolResult,
    content,
) -> None:
    """Display a tool execution result in a full, homogeneous panel."""
    if result.success:
        border = "green"
        label = "Tool Result"
        subtitle = "[green]Success[/green]"
    else:
        border = "red"
        label = "Tool Result"
        subtitle = "[red]Error[/red]"

    console.print(Panel(
        content,
        title=f"[bold {border}]{label}[/bold {border}] [dim]{spec.display_name}[/dim]",
        subtitle=f"{subtitle} [dim]{spec.category}[/dim]",
        border_style=border,
        expand=True,
        padding=(0, 1),
    ))


def render_notice(
    console: Console,
    message: str,
    level: str = "info",
    title: str | None = None,
) -> None:
    border, default_title = _STATUS_STYLES.get(level, _STATUS_STYLES["info"])
    console.print(Panel(
        render_content(message, "rich"),
        title=f"[bold {border}]{title or default_title}[/bold {border}]",
        border_style=border,
        expand=False,
        padding=(0, 1),
    ))


def render_local_runner_output(
    console: Console,
    *,
    title: str,
    content: str,
    cwd: str,
    border_style: str = "cyan",
) -> None:
    body = Text(content) if content.strip() else Text("(no output)", style="dim")
    console.print(Panel(
        body,
        title=f"[bold {border_style}]{title}[/bold {border_style}]",
        subtitle=f"[dim]{cwd}[/dim]" if cwd else None,
        border_style=border_style,
        expand=True,
        padding=(0, 1),
    ))


def render_key_value_panel(
    console: Console,
    title: str,
    rows: Sequence[Tuple[str, str]],
    border_style: str = "cyan",
) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("key", style="dim", no_wrap=True)
    table.add_column("value")
    for key, value in rows:
        table.add_row(key, value)
    console.print(Panel(table, title=f"[bold {border_style}]{title}[/bold {border_style}]", border_style=border_style))


def render_tool_proposal(
    console: Console,
    spec: ToolDisplaySpec,
    sections: Sequence[Tuple[str, object]],
    rationale: str = "",
    show_rationale: bool = True,
    requires_approval: bool = True,
) -> None:
    """Display a proposed tool call with rationale and exact JSON arguments."""
    from rich.console import Group

    parts = []

    if show_rationale and rationale:
        parts.append(Panel(
            render_content(rationale, "rich"),
            title="[bold cyan]Justification[/bold cyan]",
            border_style="cyan",
            expand=True,
            padding=(0, 1),
        ))

    for section_title, content in sections:
        if isinstance(content, list) and all(isinstance(row, tuple) and len(row) == 2 for row in content):
            kv = Table(show_header=False, box=None, padding=(0, 1))
            kv.add_column("field", style="dim", no_wrap=True)
            kv.add_column("value")
            for key, value in content:
                kv.add_row(str(key), str(value))
            content = kv
        parts.append(Panel(
            content,
            title=f"[bold white]{section_title}[/bold white]",
            border_style="white",
            expand=True,
            padding=(0, 1),
        ))

    danger_color, danger_label = _DANGER_STYLES.get(spec.danger_level, _DANGER_STYLES["low"])
    validation = (
        "[yellow]Validation required[/yellow] [dim]Press Enter/Y to approve, N to cancel, Ctrl+C to interrupt[/dim]"
        if requires_approval else
        "[dim]Auto-approved by policy[/dim]"
    )
    header = Group(
        Text.from_markup(
            f"[bold]{spec.display_name}[/bold]  [dim]{spec.category}[/dim]  "
            f"[{danger_color}]{danger_label}[/{danger_color}]"
        ),
        Rule(style=danger_color),
    )
    console.print(Panel(
        Group(header, *parts),
        title=f"[bold {spec.accent_color}]Tool Proposal[/bold {spec.accent_color}]",
        subtitle=validation,
        border_style=spec.accent_color,
        expand=True,
        padding=(0, 1),
    ))
