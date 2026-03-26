# src/k_ai/ui/render.py
"""
Rich-based rendering for k-ai: streaming responses, session tables,
tool proposals, and tool result display.
"""
from typing import List, Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from ..models import CompletionChunk, SessionMetadata, ToolResult, TokenUsage
from .markdown import render_content


# ---------------------------------------------------------------------------
# StreamingRenderer — handles the live streaming display
# ---------------------------------------------------------------------------

class StreamingRenderer:
    """
    Context-manager that renders a streaming LLM response in the terminal.

    Display states:
      1. Spinner       — waiting for the first token (transient)
      2. Thinking panel — <think>...</think> content (transient, then committed)
      3. Response panel — visible content (NON-transient = stays on screen)

    The response panel is always non-transient so it persists after Live stops.
    This avoids the double-print bug where __exit__ would re-print.
    """

    def __init__(self, console: Console, model_name: str, render_mode: str = "rich"):
        self.console = console
        self.model_name = model_name
        self.render_mode = render_mode

        self.full_content: str = ""
        self.full_thought: str = ""
        self.last_usage: Optional[TokenUsage] = None

        self._thinking_committed: bool = False
        self._content_started: bool = False
        self._live: Optional[Live] = None

    def __enter__(self) -> "StreamingRenderer":
        self._live = Live(
            Spinner("dots", text="[dim]Generating...[/dim]"),
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
            self.console.print(Panel(
                render_content(self.full_thought, self.render_mode),
                title="[bold yellow]Thinking[/bold yellow]",
                border_style="yellow",
            ))

        # Content panel: only print if we never switched to non-transient Live.
        # Once _content_started is True, the Live was non-transient and content
        # is already on screen — no need to re-print.
        if self.full_content and not self._content_started:
            self.console.print(self._content_panel())

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
            # Commit thinking if present
            if has_thought and not self._thinking_committed:
                self._live.stop()
                self.console.print(Panel(
                    render_content(self.full_thought, self.render_mode),
                    title="[bold yellow]Thinking[/bold yellow]",
                    border_style="yellow",
                ))
                self._thinking_committed = True
            else:
                # No thinking — stop the transient spinner
                self._live.stop()

            # Start a NEW non-transient Live for content (stays on screen)
            self._content_started = True
            self._live = Live(
                self._content_panel(),
                console=self.console,
                refresh_per_second=15,
                transient=False,  # Content persists after stop
            )
            self._live.start()
            return

        # Update existing live
        if has_content:
            self._live.update(self._content_panel())
        elif has_thought:
            self._live.update(Panel(
                render_content(self.full_thought, self.render_mode),
                title="[bold yellow]Thinking...[/bold yellow]",
                border_style="yellow",
                subtitle="[dim]processing[/dim]",
            ))

    def _content_panel(self) -> Panel:
        return Panel(
            render_content(self.full_content, self.render_mode),
            title=f"[bold green]{self.model_name}[/bold green]",
            border_style="green",
        )


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
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Title", min_width=20)
    table.add_column("Msgs", justify="right", width=5)
    table.add_column("Updated", width=12)

    for i, s in enumerate(sessions, 1):
        title_text = s.title if s.title != s.id else "[dim](untitled)[/dim]"
        updated = s.updated_at[:10] if s.updated_at else "?"
        table.add_row(str(i), s.id[:8], title_text, str(s.message_count), updated)

    console.print(table)


# ---------------------------------------------------------------------------
# Tool result display (in styled panel)
# ---------------------------------------------------------------------------

def render_tool_result(
    console: Console,
    name: str,
    result: ToolResult,
    max_display_length: int = 500,
) -> None:
    """Display a tool execution result in a styled panel with smart formatting."""
    from rich.syntax import Syntax

    if result.success:
        border = "green"
        label = "Agent"
    else:
        border = "red"
        label = "Agent Error"

    msg = result.message
    if len(msg) > max_display_length:
        msg = msg[:max_display_length] + "\n...(truncated)"

    # Smart content: detect if output looks like code/traceback
    if not result.success and ("Traceback" in msg or "Error" in msg):
        content = Syntax(msg, "pytb", theme="monokai", word_wrap=True)
    elif name in ("python_exec", "shell_exec") and result.success and msg.strip():
        content = Syntax(msg, "text", theme="monokai", word_wrap=True)
    else:
        content = Text(msg) if msg else Text("[dim](no output)[/dim]")

    console.print(Panel(
        content,
        title=f"[bold {border}]{label}[/bold {border}] [dim]{name}[/dim]",
        border_style=border,
        expand=False,
        padding=(0, 1),
    ))
