"""Textual-powered chat application for k-ai."""

from __future__ import annotations

import asyncio
import re
from contextlib import AbstractContextManager
from typing import Sequence

from rich.console import Console
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.events import Paste
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, RichLog, Static, TabbedContent, TabPane, TextArea

from ..models import CompletionChunk, SessionMetadata, TokenUsage, ToolResult
from ..tools.base import ToolDisplaySpec
from .presenter import AssistantStream, SessionUI
from .render import (
    build_notice_renderable,
    build_tool_proposal_renderable,
    build_tool_result_renderable,
    render_assistant_panel,
    render_assistant_stream_panel,
    render_user_panel,
)


TEXTUAL_CHAT_CSS = """
Screen {
  background: #0f1115;
  color: #e7edf3;
}

#app-shell {
  layout: vertical;
}

#body {
  layout: horizontal;
  height: 1fr;
}

#sidebar-title {
  height: 3;
  content-align: center middle;
  text-style: bold;
  color: #9fd6aa;
}

#center-column {
  width: 1fr;
  min-width: 40;
  padding: 0 1 0 1;
}

#boot-sessions {
  display: none;
  height: 12;
  margin-bottom: 1;
  border: round #2e8b57;
  background: #141821;
  padding: 0 1;
}

#boot-sessions.-visible {
  display: block;
}

#boot-sessions-header {
  height: 3;
  layout: horizontal;
}

#boot-sessions-table {
  height: 1fr;
}

#streaming-slot {
  display: none;
  margin-bottom: 1;
  border: round #2e8b57;
  background: #141a16;
}

#streaming-slot.-active {
  display: block;
}

#transcript {
  height: 1fr;
  border: round #313a46;
  background: #11151c;
}

#composer-shell {
  height: auto;
  margin-top: 1;
  border: round #2d4f6b;
  background: #111821;
  padding: 1;
}

#composer-title {
  height: 1;
  color: #87bfff;
  text-style: bold;
}

#composer-actions {
  height: 3;
  align: right middle;
}

#composer {
  height: 9;
  border: round #355b7a;
  background: #0d1218;
}

#composer-hints {
  margin-top: 1;
  height: auto;
  color: #8a96a3;
}

#inspector {
  width: 32;
  min-width: 26;
  border: round #5a6572;
  background: #141821;
  padding: 0 1;
}

#runtime-pane, #activity-pane {
  padding: 1 0;
}

#runtime-view, #activity-log {
  height: 1fr;
}

#activity-log {
  border: round #3f4752;
  background: #0f1318;
}

#approval-dialog {
  width: 78%;
  height: 78%;
  border: round #5f9ea0;
  background: #0f141b;
  padding: 1 2;
}

#approval-body {
  height: 1fr;
  border: round #3c566f;
  margin-bottom: 1;
  background: #111821;
}

#approval-actions {
  dock: bottom;
  height: 3;
  align: right middle;
}

Screen.-compact #inspector {
  display: none;
}

Screen.-compact #composer-hints {
  display: none;
}
"""

_ANSI_RE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\].*?(?:\x07|\x1b\\)|P.*?\x1b\\)", re.DOTALL)
_MOUSE_GARBAGE_RE = re.compile(r"\[<\d+(?:;\d+){2}[mM]")


def sanitize_composer_text(text: str) -> str:
    """Strip terminal control garbage that should never land in the composer."""
    cleaned = _ANSI_RE.sub("", text)
    cleaned = _MOUSE_GARBAGE_RE.sub("", cleaned)
    return cleaned


def build_runtime_text(snapshot: dict) -> Text:
    """Compact one-column runtime view for the Textual sidebar."""
    lines = [
        f"Provider  {snapshot.get('provider', '?')} / {snapshot.get('model', '?')}",
        f"Temp      {snapshot.get('temperature')}   Max {snapshot.get('max_tokens')}",
        "",
        f"Session   {str(snapshot.get('session_id') or '-')[:8]}   Type {snapshot.get('session_type') or '-'}",
        f"CWD       {snapshot.get('cwd') or '-'}",
        f"Context   {int(snapshot.get('estimated_context_tokens', 0) or 0):,} / {int(snapshot.get('context_window', 0) or 0):,} tok",
        f"Remain    {int(snapshot.get('remaining_context_tokens', 0) or 0):,} tok",
        f"History   {int(snapshot.get('history_messages', 0) or 0)} msg(s)",
        f"Tokens    {int(snapshot.get('prompt_tokens', 0) or 0):,} in / {int(snapshot.get('completion_tokens', 0) or 0):,} out",
        f"Skills    {snapshot.get('skills_summary') or '-'}",
        f"Hooks     {snapshot.get('hooks_summary') or '-'}",
        f"MCP       {snapshot.get('mcp_summary') or '-'}",
    ]
    summary = str(snapshot.get("session_summary") or "").strip()
    if summary:
        lines.extend(["", "Summary", summary])
    return Text("\n".join(lines))


class ToolApprovalScreen(ModalScreen[bool]):
    """Approval modal for tool execution."""

    BINDINGS = [
        Binding("escape", "dismiss(False)", "Cancel"),
        Binding("y", "approve", "Approve"),
        Binding("n", "dismiss(False)", "Reject"),
    ]

    def __init__(self, renderable, *, title: str):
        super().__init__()
        self._renderable = renderable
        self._title = title

    def compose(self) -> ComposeResult:
        with Container(id="approval-dialog"):
            yield Static(self._title, id="composer-title")
            yield Static(self._renderable, id="approval-body")
            with Horizontal(id="approval-actions"):
                yield Button("Reject", variant="default", id="reject")
                yield Button("Approve", variant="success", id="approve")

    def action_approve(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#approve")
    def _approve(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#reject")
    def _reject(self) -> None:
        self.dismiss(False)


class TextualConsoleIO:
    """Plain-text sink used by Rich Console fallbacks inside the Textual app."""

    def __init__(self, app: "TextualChatApp"):
        self.app = app
        self._buffer = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.app.append_activity(Text(line))
        return len(data)

    def flush(self) -> None:
        if self._buffer.strip():
            self.app.append_activity(Text(self._buffer.rstrip("\n")))
        self._buffer = ""

    def isatty(self) -> bool:
        return False


class TextualAssistantStream(AssistantStream):
    """Streaming sink that updates the dedicated Textual streaming slot."""

    def __init__(
        self,
        app: "TextualChatApp",
        *,
        model_name: str,
        render_mode: str,
        theme_name: str,
    ):
        self.app = app
        self.model_name = model_name
        self.render_mode = render_mode
        self.theme_name = theme_name
        self.full_content = ""
        self.full_thought = ""
        self.last_usage: TokenUsage | None = None

    def update(self, chunk: CompletionChunk) -> None:
        if chunk.usage:
            self.last_usage = chunk.usage
        if chunk.delta_thought:
            self.full_thought += chunk.delta_thought
        if chunk.delta_content:
            self.full_content += chunk.delta_content
        self.app.update_streaming_view(
            render_assistant_stream_panel(
                self.full_content or self.full_thought or "…",
                self.model_name,
                render_mode=self.render_mode,
                usage=self.last_usage,
                theme_name=self.theme_name,
                initial=True,
            )
        )


class _TextualStreamContext(AbstractContextManager[TextualAssistantStream]):
    def __init__(self, stream: TextualAssistantStream):
        self.stream = stream

    def __enter__(self) -> TextualAssistantStream:
        self.stream.app.begin_stream()
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None and self.stream.full_content:
            self.stream.app.commit_stream(
                render_assistant_panel(
                    self.stream.full_content,
                    self.stream.model_name,
                    render_mode=self.stream.render_mode,
                    usage=self.stream.last_usage,
                    theme_name=self.stream.theme_name,
                )
            )
        else:
            self.stream.app.clear_stream()


class TextualSessionUI(SessionUI):
    """Session UI bridge backed by a Textual app."""

    def __init__(self, app: "TextualChatApp"):
        self.app = app
        self.console_io = TextualConsoleIO(app)
        console = Console(file=self.console_io, force_terminal=False, color_system=None, width=120)
        super().__init__(console)

    def stream_assistant(
        self,
        *,
        model_name: str,
        render_mode: str,
        spinner_name: str,
        theme_name: str,
        flush_min_chars: int,
        tail_chars: int,
        interrupt_hint: str,
    ):
        stream = TextualAssistantStream(
            self.app,
            model_name=model_name,
            render_mode=render_mode,
            theme_name=theme_name,
        )
        return _TextualStreamContext(stream)

    def show_user(self, content: str, *, theme_name: str) -> None:
        self.app.append_transcript(render_user_panel(content, theme_name=theme_name))

    def show_assistant(self, content: str, *, model_name: str, render_mode: str = "rich", usage: TokenUsage | None = None, theme_name: str = "default") -> None:
        self.app.append_transcript(
            render_assistant_panel(
                content,
                model_name,
                render_mode=render_mode,
                usage=usage,
                theme_name=theme_name,
            )
        )

    def show_notice(self, message: str, *, level: str = "info", title: str | None = None) -> None:
        renderable = build_notice_renderable(message, level=level, title=title)
        self.app.append_transcript(renderable)
        if level in {"warning", "error"}:
            self.app.append_activity(renderable)

    def show_runtime(self, snapshot: dict, *, title: str = "Runtime Transparency", mode: str = "compact", theme_name: str = "default") -> None:
        self.app.update_runtime(snapshot)

    def show_sessions(self, sessions: Sequence[SessionMetadata], *, title: str = "Recent Sessions") -> None:
        self.app.update_sessions(sessions, title=title)

    def show_runner_output(self, *, title: str, content: str, cwd: str, border_style: str = "cyan") -> None:
        renderable = build_notice_renderable(content or "(no output)", level="info", title=f"{title} · {cwd}")
        self.app.append_transcript(renderable)
        self.app.append_activity(renderable)

    def show_tool_result(self, spec: ToolDisplaySpec, result: ToolResult, content) -> None:
        renderable = build_tool_result_renderable(spec, result, content)
        self.app.append_transcript(renderable)
        self.app.append_activity(renderable)

    async def confirm_tool_execution(
        self,
        spec: ToolDisplaySpec,
        sections: Sequence[tuple[str, object]],
        *,
        rationale: str,
        show_rationale: bool,
        requires_approval: bool,
    ) -> bool:
        renderable = build_tool_proposal_renderable(
            spec,
            sections,
            rationale=rationale,
            show_rationale=show_rationale,
            requires_approval=requires_approval,
        )
        if not requires_approval:
            self.app.append_activity(renderable)
            return True
        return bool(
            await self.app.push_screen_wait(
                ToolApprovalScreen(renderable, title=f"{spec.display_name} · approval required")
            )
        )

    def show_loaded_messages(self, messages, *, model_name: str, render_mode: str = "rich", theme_name: str = "default") -> None:
        for message in messages:
            if message.role.value == "system":
                continue
            if message.role.value == "user":
                self.show_user(message.content, theme_name=theme_name)
                continue
            if message.role.value == "assistant":
                self.show_assistant(message.content, model_name=model_name, render_mode=render_mode, theme_name=theme_name)
                continue
            self.app.append_transcript(build_notice_renderable(message.content, level="info", title=f"Tool {message.name or 'tool'}"))

    def suspend(self):
        return self.app.suspend()


class TextualChatApp(App[None]):
    """Full-screen Textual UI for interactive k-ai chat."""

    CSS = TEXTUAL_CHAT_CSS
    BINDINGS = [
        Binding("ctrl+enter", "submit", "Send", priority=True),
        Binding("ctrl+s", "submit", "Send", priority=True),
        Binding("f2", "submit", "Send", priority=True),
        Binding("ctrl+j", "focus_composer", "Composer", priority=True),
        Binding("ctrl+g", "focus_transcript", "Transcript", priority=True),
        Binding("ctrl+b", "focus_boot_sessions", "Sessions", priority=True),
        Binding("ctrl+r", "focus_runtime", "Runtime"),
        Binding("ctrl+l", "focus_activity", "Activity"),
        Binding("escape", "dismiss_boot_sessions", "Hide Sessions", show=False, priority=True),
        Binding("f5", "refresh_runtime", "Refresh"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, session):
        super().__init__()
        self.session = session
        self.presenter = TextualSessionUI(self)
        self.session.attach_ui(self.presenter)
        self._boot_task: asyncio.Task | None = None
        self._submit_lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="app-shell"):
            with Horizontal(id="body"):
                with Vertical(id="center-column"):
                    with Vertical(id="boot-sessions"):
                        with Horizontal(id="boot-sessions-header"):
                            yield Static("Recent Sessions", id="sidebar-title")
                            yield Button("Hide", id="hide-sessions", variant="default")
                        yield DataTable(id="boot-sessions-table", zebra_stripes=True)
                    yield Static(id="streaming-slot")
                    yield RichLog(id="transcript", wrap=True, markup=True, auto_scroll=True, highlight=True)
                    with Vertical(id="composer-shell"):
                        with Horizontal(id="composer-actions"):
                            yield Static("Composer", id="composer-title")
                            yield Button("Send", id="send-button", variant="primary")
                        yield TextArea(
                            id="composer",
                            language="markdown",
                            theme="monokai",
                            soft_wrap=True,
                            placeholder="Message... Ctrl+Enter pour envoyer.",
                        )
                        yield Static(
                            "Ctrl+Enter envoyer · Enter newline · Ctrl+B sessions · Ctrl+R runtime · Ctrl+L activity · /focus shell pour le runner",
                            id="composer-hints",
                        )
                with Vertical(id="inspector"):
                    yield Static("Runtime & Activity", id="sidebar-title")
                    with TabbedContent(initial="runtime-pane"):
                        with TabPane("Runtime", id="runtime-pane"):
                            yield Static(id="runtime-view")
                        with TabPane("Activity", id="activity-pane"):
                            yield RichLog(id="activity-log", wrap=True, markup=True, auto_scroll=True, highlight=True)
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "Textual Chat"
        table = self.query_one("#boot-sessions-table", DataTable)
        table.add_columns("#", "Session", "Type", "Summary", "Msgs")
        self.query_one("#composer", TextArea).clear()
        self.call_after_refresh(self.action_focus_composer)
        self._update_responsive_mode()
        self.run_worker(self._bootstrap(), exclusive=True, group="session")

    async def _bootstrap(self) -> None:
        await self.session.bootstrap()
        self._sanitize_composer()
        self.action_refresh_runtime()

    def update_sessions(self, sessions: Sequence[SessionMetadata], *, title: str = "Recent Sessions") -> None:
        table = self.query_one("#boot-sessions-table", DataTable)
        table.clear(columns=False)
        for index, session in enumerate(sessions, start=1):
            table.add_row(
                str(index),
                session.id[:8],
                session.session_type,
                (session.summary or session.title or "-")[:42],
                str(session.message_count),
                key=session.id,
            )
        container = self.query_one("#boot-sessions", Vertical)
        if sessions:
            container.add_class("-visible")
        else:
            container.remove_class("-visible")
        self.append_activity(Text(f"{title}: {len(sessions)} session(s)"))

    def update_runtime(self, renderable) -> None:
        if isinstance(renderable, dict):
            self.query_one("#runtime-view", Static).update(build_runtime_text(renderable))
            return
        self.query_one("#runtime-view", Static).update(renderable)

    def append_transcript(self, renderable) -> None:
        self.query_one("#transcript", RichLog).write(renderable, scroll_end=True)

    def append_activity(self, renderable) -> None:
        self.query_one("#activity-log", RichLog).write(renderable, scroll_end=True)

    def begin_stream(self) -> None:
        slot = self.query_one("#streaming-slot", Static)
        slot.add_class("-active")
        slot.update(Text("Generating…"))

    def update_streaming_view(self, renderable) -> None:
        slot = self.query_one("#streaming-slot", Static)
        slot.add_class("-active")
        slot.update(renderable)

    def commit_stream(self, renderable) -> None:
        self.append_transcript(renderable)
        self.clear_stream()

    def clear_stream(self) -> None:
        slot = self.query_one("#streaming-slot", Static)
        slot.remove_class("-active")
        slot.update("")

    async def submit_current_buffer(self) -> None:
        composer = self.query_one("#composer", TextArea)
        self._sanitize_composer()
        text = composer.text
        if not text.strip():
            return
        composer.clear()
        self.action_dismiss_boot_sessions()
        async with self._submit_lock:
            await self.session.submit_document(text)
            self.action_refresh_runtime()

    async def action_submit(self) -> None:
        await self.submit_current_buffer()

    def action_focus_composer(self) -> None:
        self.query_one("#composer", TextArea).focus()

    def action_focus_transcript(self) -> None:
        self.query_one("#transcript", RichLog).focus()

    def action_focus_boot_sessions(self) -> None:
        container = self.query_one("#boot-sessions", Vertical)
        if "-visible" in container.classes:
            self.query_one("#boot-sessions-table", DataTable).focus()
            return
        self.action_focus_composer()

    def action_dismiss_boot_sessions(self) -> None:
        self.query_one("#boot-sessions", Vertical).remove_class("-visible")

    def action_focus_runtime(self) -> None:
        self.query_one("#runtime-view", Static).focus()

    def action_focus_activity(self) -> None:
        self.query_one("#activity-log", RichLog).focus()

    def action_refresh_runtime(self) -> None:
        self.presenter.show_runtime(
            self.session.get_runtime_snapshot(),
            mode=self.session.cm.get_nested("cli", "runtime_stats_mode", default="compact"),
            theme_name=self.session.cm.get_nested("cli", "theme", default="default"),
        )

    def _sanitize_composer(self) -> None:
        composer = self.query_one("#composer", TextArea)
        cleaned = sanitize_composer_text(composer.text)
        if cleaned != composer.text:
            composer.load_text(cleaned)

    @on(TextArea.Changed, "#composer")
    def _handle_composer_changed(self, event: TextArea.Changed) -> None:
        self._sanitize_composer()

    @on(Paste)
    def _handle_paste(self, event: Paste) -> None:
        self.call_after_refresh(self._sanitize_composer)

    @on(Button.Pressed, "#send-button")
    async def _handle_send_button(self) -> None:
        await self.submit_current_buffer()

    @on(Button.Pressed, "#hide-sessions")
    def _handle_hide_sessions(self) -> None:
        self.action_dismiss_boot_sessions()

    @on(DataTable.RowSelected, "#boot-sessions-table")
    async def _load_selected_session(self, event: DataTable.RowSelected) -> None:
        row_key = event.row_key.value if event.row_key else None
        if row_key:
            self.action_dismiss_boot_sessions()
            await self.session.submit_document(f"/load {row_key}")
            self.action_refresh_runtime()

    def on_resize(self, event=None) -> None:
        self._update_responsive_mode()

    def _update_responsive_mode(self) -> None:
        if self.size.width < 120:
            self.add_class("-compact")
        else:
            self.remove_class("-compact")
