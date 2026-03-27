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
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual import events
from textual.events import Paste
from textual.message import Message
from textual.screen import ModalScreen
from textual.css.query import NoMatches
from textual.widgets import Button, DataTable, Footer, Header, RichLog, Static, TextArea

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
  height: 1fr;
  padding: 0 1;
}

#body {
  height: 1fr;
  min-height: 10;
}

#conversation-scroll {
  height: 1fr;
  border: round #313a46;
  background: #11151c;
}

#conversation-stack {
  layout: vertical;
  width: 100%;
  padding: 1;
}

.conversation-card {
  margin-bottom: 1;
}

#streaming-slot {
  display: none;
}

#streaming-slot.-active {
  display: block;
}

#composer-shell {
  height: auto;
  margin: 1 0 0 0;
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

#status-bar {
  height: auto;
  margin-top: 1;
  color: #97a7b4;
}

#approval-dialog {
  width: 88%;
  height: 84%;
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

#overlay-dialog {
  width: 92%;
  height: 88%;
  border: round #3f657f;
  background: #0f141b;
  padding: 1 2;
}

#overlay-header {
  height: 3;
  layout: horizontal;
}

#overlay-title {
  width: 1fr;
  content-align: left middle;
  color: #9fd6aa;
  text-style: bold;
}

#overlay-subtitle {
  height: auto;
  color: #7e8d99;
  margin-bottom: 1;
}

#overlay-body {
  height: 1fr;
  border: round #334657;
  background: #111821;
  padding: 0 1;
}

#overlay-log {
  height: 1fr;
}

#overlay-table {
  height: 1fr;
}

#sessions-inline {
  border: round #2e8b57;
  background: #141821;
  padding: 0 1 1 1;
}

#sessions-inline-header {
  height: 3;
}

#sessions-inline-table {
  height: 10;
}

#approval-inline {
  border: round #5f9ea0;
  background: #0f141b;
  padding: 1 1 0 1;
}

#approval-inline-body {
  height: auto;
}

#approval-inline-actions {
  height: 3;
  align: right middle;
}

Screen.-narrow #composer {
  min-height: 3;
}

Screen.-narrow #composer-hints {
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
    """Compact one-column runtime view for the Textual runtime overlay."""
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


def build_status_text(snapshot: dict) -> Text:
    provider = str(snapshot.get("provider") or "?")
    model = str(snapshot.get("model") or "?")
    temp = snapshot.get("temperature")
    max_tokens = snapshot.get("max_tokens")
    session_id = str(snapshot.get("session_id") or "-")[:8]
    session_type = str(snapshot.get("session_type") or "-")
    cwd = str(snapshot.get("cwd") or "-")
    auth = str(snapshot.get("auth_mode") or "-")
    context_used = int(snapshot.get("estimated_context_tokens", 0) or 0)
    context_total = int(snapshot.get("context_window", 0) or 0)
    context_percent = float(snapshot.get("context_percent", 0.0) or 0.0)
    remaining = int(snapshot.get("remaining_context_tokens", 0) or 0)
    compaction = int(snapshot.get("compaction_trigger_tokens", 0) or 0)
    compaction_pct = int(snapshot.get("compaction_trigger_percent", 0) or 0)
    prompt_tokens = int(snapshot.get("prompt_tokens", 0) or 0)
    completion_tokens = int(snapshot.get("completion_tokens", 0) or 0)
    total_tokens = int(snapshot.get("total_tokens", 0) or 0)
    history_messages = int(snapshot.get("history_messages", 0) or 0)
    token_source = str(snapshot.get("token_source") or "estimated")
    stream = "true" if snapshot.get("stream") else "false"
    skills = str(snapshot.get("skills_summary") or "-")
    hooks = str(snapshot.get("hooks_summary") or "-")
    mcp = str(snapshot.get("mcp_summary") or "-")
    lines = [
        Text.from_markup(
            f"[bold cyan]{provider}[/bold cyan]/[white]{model}[/white]  "
            f"[dim]temp[/dim] {temp}  [dim]max[/dim] {max_tokens}  "
            f"[dim]session[/dim] {session_id}  [dim]type[/dim] {session_type}  "
            f"[dim]auth[/dim] {auth}"
        ),
        Text.from_markup(
            f"[dim]cwd[/dim] {cwd}  "
            f"[dim]ctx[/dim] {context_used:,}/{context_total:,} ({context_percent:.1f}%)  "
            f"[dim]compact[/dim] {compaction:,} ({compaction_pct}%)  "
            f"[dim]remain[/dim] {remaining:,}"
        ),
        Text.from_markup(
            f"[dim]tok[/dim] {prompt_tokens:,} in / {completion_tokens:,} out / {total_tokens:,} total  "
            f"[dim]source[/dim] {token_source}  [dim]hist[/dim] {history_messages}  [dim]stream[/dim] {stream}"
        ),
        Text.from_markup(
            f"[dim]skills[/dim] {skills}  [dim]hooks[/dim] {hooks}  [dim]mcp[/dim] {mcp}"
        ),
    ]
    return Text("\n").join(lines)


def sanitize_runner_display(content: str) -> str:
    lines = [line.rstrip() for line in str(content or "").splitlines()]
    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if previous_blank:
                continue
            previous_blank = True
            cleaned.append("")
            continue
        previous_blank = False
        if stripped.startswith(("┌", "╭", "│", "╰", "└")):
            continue
        if "__KAI_" in stripped:
            continue
        if stripped.startswith("[kpihx@") and "ls ''\\004'" in stripped:
            stripped = stripped.split("ls ''\\004'", 1)[1].strip()
        if stripped.startswith(("[kpihx@", "%", "$")) and len(stripped) < 120:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def normalize_binding_name(binding: str) -> str:
    raw = str(binding or "").strip().lower()
    aliases = {
        "c-s": "ctrl+s",
        "c-j": "ctrl+j",
        "c-enter": "ctrl+enter",
        "ctrl-enter": "ctrl+enter",
        "escape-enter": "shift+enter",
        "s-enter": "shift+enter",
    }
    return aliases.get(raw, raw)


class ChatComposer(TextArea):
    class Submitted(Message):
        def __init__(self, sender: "ChatComposer"):
            super().__init__()
            self.sender = sender

    def __init__(self, *, submit_keys: Sequence[str], newline_keys: Sequence[str], **kwargs):
        super().__init__(**kwargs)
        self.submit_keys = {normalize_binding_name(item) for item in submit_keys}
        self.newline_keys = {normalize_binding_name(item) for item in newline_keys}

    async def _on_key(self, event: events.Key) -> None:
        key = normalize_binding_name(event.key)
        if key in self.submit_keys:
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self))
            return
        if key in self.newline_keys:
            event.stop()
            event.prevent_default()
            start, end = self.selection
            self._replace_via_keyboard("\n", start, end)
            return
        await super()._on_key(event)


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


class OverlayScreen(ModalScreen[None]):
    """Generic overlay used for runtime, activity, and session browser."""

    BINDINGS = [
        Binding("escape", "close_overlay", "Close", show=False, priority=True),
        Binding("q", "close_overlay", "Close", show=False),
    ]

    def __init__(self, *, title: str, subtitle: str = ""):
        super().__init__()
        self._title = title
        self._subtitle = subtitle

    def compose(self) -> ComposeResult:
        with Container(id="overlay-dialog"):
            with Horizontal(id="overlay-header"):
                yield Static(self._title, id="overlay-title")
                yield Button("Close", id="overlay-close", variant="default")
            if self._subtitle:
                yield Static(self._subtitle, id="overlay-subtitle")
            yield from self.compose_body()

    def compose_body(self) -> ComposeResult:
        yield Static("", id="overlay-body")

    def action_close_overlay(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#overlay-close")
    def _close_overlay(self) -> None:
        self.dismiss(None)


class RuntimeOverlayScreen(OverlayScreen):
    def __init__(self, renderable):
        super().__init__(title="Runtime Transparency", subtitle="Instant snapshot of the current session state.")
        self._renderable = renderable

    def compose_body(self) -> ComposeResult:
        yield Static(self._renderable, id="overlay-body")

    def update_renderable(self, renderable) -> None:
        self._renderable = renderable
        if self.is_mounted:
            self.query_one("#overlay-body", Static).update(renderable)


class ActivityOverlayScreen(OverlayScreen):
    def __init__(self, entries: Sequence[object]):
        super().__init__(title="Activity Log", subtitle="Operational trace for tools, runtime events, and notices.")
        self._entries = list(entries)

    def compose_body(self) -> ComposeResult:
        yield RichLog(id="overlay-log", wrap=True, markup=True, auto_scroll=True, highlight=True)

    def on_mount(self) -> None:
        log = self.query_one("#overlay-log", RichLog)
        for entry in self._entries:
            log.write(entry, scroll_end=True)

    def append_entry(self, renderable) -> None:
        self._entries.append(renderable)
        if self.is_mounted:
            self.query_one("#overlay-log", RichLog).write(renderable, scroll_end=True)


class SessionsOverlayScreen(OverlayScreen):
    BINDINGS = OverlayScreen.BINDINGS + [
        Binding("enter", "select_session", "Open", show=False, priority=True),
    ]

    def __init__(self, sessions: Sequence[SessionMetadata], *, title: str = "Recent Sessions"):
        super().__init__(title=title, subtitle="Open only when needed. The main chat stays centered on interaction.")
        self._sessions = list(sessions)

    def compose_body(self) -> ComposeResult:
        yield DataTable(id="overlay-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#overlay-table", DataTable)
        table.add_columns("#", "Session", "Type", "Summary", "Msgs")
        self._reload_table()
        self.call_after_refresh(table.focus)

    def _reload_table(self) -> None:
        table = self.query_one("#overlay-table", DataTable)
        table.clear(columns=False)
        for index, session in enumerate(self._sessions, start=1):
            table.add_row(
                str(index),
                session.id[:8],
                session.session_type,
                (session.summary or session.title or "-")[:56],
                str(session.message_count),
                key=session.id,
            )

    def update_sessions(self, sessions: Sequence[SessionMetadata]) -> None:
        self._sessions = list(sessions)
        if self.is_mounted:
            self._reload_table()

    def action_select_session(self) -> None:
        table = self.query_one("#overlay-table", DataTable)
        row_index = table.cursor_row
        if row_index is None or row_index < 0 or row_index >= len(self._sessions):
            return
        self.dismiss(self._sessions[row_index].id)

    @on(DataTable.RowSelected, "#overlay-table")
    def _select_row(self, event: DataTable.RowSelected) -> None:
        row_key = event.row_key.value if event.row_key else None
        if row_key:
            self.dismiss(row_key)


class ConversationRenderableCard(Static):
    def __init__(self, renderable, *, id: str | None = None):
        super().__init__(renderable, id=id, classes="conversation-card")


class InlineSessionsCard(Static):
    pass


class SessionsInlineWidget(Container):
    BINDINGS = [
        Binding("enter", "select_current_session", "Open", show=False, priority=True),
    ]

    def __init__(self, sessions: Sequence[SessionMetadata], *, title: str = "Recent Sessions"):
        super().__init__(id="sessions-inline", classes="conversation-card")
        self._sessions = list(sessions)
        self._title = title
        self.selected_session_id: str | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="sessions-inline-header"):
            yield Static(self._title, id="overlay-title")
            yield Button("Hide", id="sessions-inline-hide", variant="default")
        yield DataTable(id="sessions-inline-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#sessions-inline-table", DataTable)
        table.add_columns("#", "Session", "Type", "Summary", "Msgs")
        for index, session in enumerate(self._sessions, start=1):
            table.add_row(
                str(index),
                session.id[:8],
                session.session_type,
                (session.summary or session.title or "-")[:56],
                str(session.message_count),
                key=session.id,
            )
        self.call_after_refresh(table.focus)

    def update_sessions(self, sessions: Sequence[SessionMetadata]) -> None:
        self._sessions = list(sessions)
        if not self.is_mounted:
            return
        table = self.query_one("#sessions-inline-table", DataTable)
        table.clear(columns=False)
        for index, session in enumerate(self._sessions, start=1):
            table.add_row(
                str(index),
                session.id[:8],
                session.session_type,
                (session.summary or session.title or "-")[:56],
                str(session.message_count),
                key=session.id,
            )

    def action_select_current_session(self) -> None:
        table = self.query_one("#sessions-inline-table", DataTable)
        row_index = table.cursor_row
        if row_index is None or row_index < 0 or row_index >= len(self._sessions):
            return
        self.selected_session_id = self._sessions[row_index].id
        self.post_message(self.Selected(self, self.selected_session_id))

    @on(Button.Pressed, "#sessions-inline-hide")
    def _hide(self) -> None:
        self.remove()

    @on(DataTable.RowSelected, "#sessions-inline-table")
    def _select_row(self, event: DataTable.RowSelected) -> None:
        self.selected_session_id = event.row_key.value if event.row_key else None
        self.post_message(self.Selected(self, self.selected_session_id))

    class Selected(Message):
        def __init__(self, sender: "SessionsInlineWidget", session_id: str | None):
            super().__init__()
            self.sender = sender
            self.session_id = session_id


class InlineApprovalWidget(Container):
    BINDINGS = [
        Binding("y", "approve", "Approve", show=False, priority=True),
        Binding("n", "reject", "Reject", show=False, priority=True),
        Binding("escape", "reject", "Reject", show=False),
    ]

    class Decision(Message):
        def __init__(self, sender: "InlineApprovalWidget", approved: bool):
            super().__init__()
            self.sender = sender
            self.approved = approved

    def __init__(self, renderable, *, title: str):
        super().__init__(id="approval-inline", classes="conversation-card")
        self._renderable = renderable
        self._title = title

    def compose(self) -> ComposeResult:
        yield Static(self._title, id="overlay-title")
        yield Static(self._renderable, id="approval-inline-body")
        with Horizontal(id="approval-inline-actions"):
            yield Button("Reject", id="approval-inline-reject", variant="default")
            yield Button("Approve", id="approval-inline-approve", variant="success")

    def on_mount(self) -> None:
        self.call_after_refresh(lambda: self.query_one("#approval-inline-approve", Button).focus())

    def action_approve(self) -> None:
        self.post_message(self.Decision(self, True))

    def action_reject(self) -> None:
        self.post_message(self.Decision(self, False))

    @on(Button.Pressed, "#approval-inline-approve")
    def _approve(self) -> None:
        self.action_approve()

    @on(Button.Pressed, "#approval-inline-reject")
    def _reject(self) -> None:
        self.action_reject()


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
        flush_min_chars: int = 160,
    ):
        self.app = app
        self.model_name = model_name
        self.render_mode = render_mode
        self.theme_name = theme_name
        self.flush_min_chars = max(int(flush_min_chars or 0), 80)
        self.full_content = ""
        self.full_thought = ""
        self.last_usage: TokenUsage | None = None
        self._rendered_chars = 0

    def update(self, chunk: CompletionChunk) -> None:
        if chunk.usage:
            self.last_usage = chunk.usage
        if chunk.delta_thought:
            self.full_thought += chunk.delta_thought
        if chunk.delta_content:
            self.full_content += chunk.delta_content
        pending = len(self.full_content) - self._rendered_chars
        if not self.full_content:
            return
        if pending < self.flush_min_chars and not self.full_content.endswith(("\n", ".", "!", "?", ":", ";")):
            return
        self._rendered_chars = len(self.full_content)
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
            flush_min_chars=flush_min_chars,
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
        from .render import build_local_runner_output_renderable

        renderable = build_local_runner_output_renderable(
            title=title,
            content=sanitize_runner_display(content) or "(no output)",
            cwd=cwd,
            border_style=border_style,
        )
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
        return await self.app.show_inline_approval(
            renderable,
            title=f"{spec.display_name} · approval required",
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

    class AppendTranscript(Message):
        def __init__(self, renderable):
            super().__init__()
            self.renderable = renderable

    class AppendActivity(Message):
        def __init__(self, renderable):
            super().__init__()
            self.renderable = renderable

    class UpdateRuntime(Message):
        def __init__(self, renderable):
            super().__init__()
            self.renderable = renderable

    class UpdateSessions(Message):
        def __init__(self, sessions: Sequence[SessionMetadata], title: str):
            super().__init__()
            self.sessions = list(sessions)
            self.title = title

    class BeginStream(Message):
        pass

    class UpdateStream(Message):
        def __init__(self, renderable):
            super().__init__()
            self.renderable = renderable

    class CommitStream(Message):
        def __init__(self, renderable):
            super().__init__()
            self.renderable = renderable

    class ClearStream(Message):
        pass

    CSS = TEXTUAL_CHAT_CSS
    BINDINGS = [
        Binding("ctrl+enter", "submit", "Send", priority=True),
        Binding("ctrl+s", "submit", "", show=False, priority=True),
        Binding("f2", "submit", "", show=False, priority=True),
        Binding("ctrl+j", "focus_composer", "Composer", priority=True),
        Binding("ctrl+g", "focus_transcript", "Transcript", priority=True),
        Binding("ctrl+b", "show_sessions", "Sessions", priority=True),
        Binding("ctrl+r", "show_runtime", "Runtime"),
        Binding("ctrl+l", "show_activity", "Activity"),
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
        self._submit_keys = tuple(
            self.session.cm.get_nested("interaction", "input", "submit_bindings", default=["ctrl+enter", "ctrl+s", "f2"]) or ["ctrl+enter", "ctrl+s", "f2"]
        )
        self._newline_keys = tuple(
            self.session.cm.get_nested("interaction", "input", "newline_bindings", default=["enter"]) or ["enter"]
        )
        self._runtime_overlay: RuntimeOverlayScreen | None = None
        self._activity_overlay: ActivityOverlayScreen | None = None
        self._sessions_overlay: SessionsOverlayScreen | None = None
        self._activity_entries: list[object] = []
        self._recent_sessions: list[SessionMetadata] = []
        self._sessions_title = "Recent Sessions"
        self._boot_sessions_shown = False
        self._sessions_card: SessionsInlineWidget | None = None
        self._stream_widget: ConversationRenderableCard | None = None
        self._last_runtime_renderable = build_runtime_text(self.session.get_runtime_snapshot())
        self._pending_approval_widget: InlineApprovalWidget | None = None
        self._pending_approval_future: asyncio.Future[bool] | None = None
        self._last_status_text: Text = Text()
        self._submit_task: asyncio.Task[None] | None = None
        self._submitting: bool = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="app-shell"):
            with Vertical(id="body"):
                with VerticalScroll(id="conversation-scroll"):
                    with Vertical(id="conversation-stack"):
                        yield Static(id="streaming-slot")
            with Vertical(id="composer-shell"):
                with Horizontal(id="composer-actions"):
                    yield Static("Composer", id="composer-title")
                    yield Button("Send", id="send-button", variant="primary")
                yield ChatComposer(
                    id="composer",
                    submit_keys=self._submit_keys,
                    newline_keys=self._newline_keys,
                    language="markdown",
                    theme="monokai",
                    soft_wrap=True,
                    placeholder="Message...",
                )
                yield Static(
                    f"{' / '.join(self._submit_keys)} envoyer · {' / '.join(self._newline_keys)} newline · Ctrl+B sessions · Ctrl+R runtime · Ctrl+L activity · /focus shell pour le runner",
                    id="composer-hints",
                )
                yield Static(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "Textual Chat"
        self.query_one("#composer", TextArea).clear()
        self.call_after_refresh(self.action_focus_composer)
        self._update_responsive_mode()
        self.run_worker(self._bootstrap(), exclusive=True, group="session")

    async def _bootstrap(self) -> None:
        await self.session.bootstrap()
        self._sanitize_composer()
        self._update_composer_height()
        self.action_refresh_runtime()
        if self._recent_sessions and not self._boot_sessions_shown:
            self.call_after_refresh(self._show_sessions_inline)

    def update_sessions(self, sessions: Sequence[SessionMetadata], *, title: str = "Recent Sessions") -> None:
        self.post_message(self.UpdateSessions(sessions, title))

    def _update_sessions_impl(self, sessions: Sequence[SessionMetadata], *, title: str = "Recent Sessions") -> None:
        self._recent_sessions = list(sessions)
        self._sessions_title = title
        if self._sessions_card is not None:
            self._sessions_card.update_sessions(sessions)
        elif self.is_mounted and sessions and not self._boot_sessions_shown:
            self.call_after_refresh(self._show_sessions_inline)
        self._append_activity_impl(Text(f"{title}: {len(sessions)} session(s)"))

    def update_runtime(self, renderable) -> None:
        self.post_message(self.UpdateRuntime(renderable))

    def _update_runtime_impl(self, renderable) -> None:
        if isinstance(renderable, dict):
            renderable = build_runtime_text(renderable)
        if self._runtime_overlay is not None:
            self._runtime_overlay.update_renderable(renderable)
        self._last_runtime_renderable = renderable
        self._last_status_text = build_status_text(self.session.get_runtime_snapshot())
        self.query_one("#status-bar", Static).update(self._last_status_text)

    def append_transcript(self, renderable) -> None:
        self.post_message(self.AppendTranscript(renderable))

    def _append_transcript_impl(self, renderable) -> None:
        stack = self.query_one("#conversation-stack", Vertical)
        stack.mount(ConversationRenderableCard(renderable))
        self.call_after_refresh(self._scroll_conversation_end)

    def append_activity(self, renderable) -> None:
        self.post_message(self.AppendActivity(renderable))

    def _append_activity_impl(self, renderable) -> None:
        self._activity_entries.append(renderable)
        if self._activity_overlay is not None:
            self._activity_overlay.append_entry(renderable)

    def begin_stream(self) -> None:
        self.post_message(self.BeginStream())

    def _begin_stream_impl(self) -> None:
        slot = self.query_one("#streaming-slot", Static)
        slot.add_class("-active")
        if self._stream_widget is None:
            self._stream_widget = ConversationRenderableCard(Text("Generating…"), id="streaming-card")
            self.query_one("#conversation-stack", Vertical).mount(self._stream_widget)
        else:
            self._stream_widget.update(Text("Generating…"))
        self.call_after_refresh(self._scroll_conversation_end)

    def update_streaming_view(self, renderable) -> None:
        self.post_message(self.UpdateStream(renderable))

    def _update_streaming_view_impl(self, renderable) -> None:
        slot = self.query_one("#streaming-slot", Static)
        slot.add_class("-active")
        if self._stream_widget is None:
            self._stream_widget = ConversationRenderableCard(renderable, id="streaming-card")
            self.query_one("#conversation-stack", Vertical).mount(self._stream_widget)
        else:
            self._stream_widget.update(renderable)
        self.call_after_refresh(self._scroll_conversation_end)

    def commit_stream(self, renderable) -> None:
        self.post_message(self.CommitStream(renderable))

    def _commit_stream_impl(self, renderable) -> None:
        if self._stream_widget is None:
            self._append_transcript_impl(renderable)
        else:
            stream_widget = self._stream_widget
            if stream_widget.is_attached:
                stream_widget.remove()
            self._append_transcript_impl(renderable)
            self._stream_widget = None
        self._clear_stream_impl()
        self.call_after_refresh(self._scroll_conversation_end)

    def clear_stream(self) -> None:
        self.post_message(self.ClearStream())

    def _clear_stream_impl(self) -> None:
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
        self._update_composer_height()
        self._dismiss_sessions_inline()
        async with self._submit_lock:
            await self.session.submit_document(text)
            self.action_refresh_runtime()

    async def action_submit(self) -> None:
        self._queue_submit()

    def action_focus_composer(self) -> None:
        self.query_one("#composer", TextArea).focus()

    def action_focus_transcript(self) -> None:
        self.query_one("#conversation-scroll", VerticalScroll).focus()

    def action_show_sessions(self) -> None:
        self._show_sessions_inline()

    def action_show_runtime(self) -> None:
        self._show_runtime_overlay()

    def action_show_activity(self) -> None:
        self._show_activity_overlay()

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
        self._update_composer_height()

    @on(ChatComposer.Submitted)
    async def _handle_composer_submitted(self) -> None:
        self._queue_submit()

    @on(Paste)
    def _handle_paste(self, event: Paste) -> None:
        self.call_after_refresh(self._sanitize_composer)

    @on(Button.Pressed, "#send-button")
    async def _handle_send_button(self) -> None:
        self._queue_submit()

    @on(SessionsInlineWidget.Selected)
    def _handle_inline_session_selected(self, event: SessionsInlineWidget.Selected) -> None:
        if event.session_id:
            self._dismiss_sessions_inline()
            self.run_worker(self._load_session_from_inline(event.session_id), exclusive=True, group="load-session")

    @on(InlineApprovalWidget.Decision)
    def _handle_inline_approval(self, event: InlineApprovalWidget.Decision) -> None:
        if self._pending_approval_future is not None and not self._pending_approval_future.done():
            self._pending_approval_future.set_result(bool(event.approved))

    def on_resize(self, event=None) -> None:
        self._update_responsive_mode()
        self._update_composer_height()

    def _update_responsive_mode(self) -> None:
        if self.size.width < 110:
            self.add_class("-narrow")
        else:
            self.remove_class("-narrow")

    def _update_composer_height(self) -> None:
        composer = self.query_one("#composer", TextArea)
        line_count = max(1, len((composer.text or "").splitlines()) or 1)
        composer.styles.height = max(3, min(10, line_count + 2))

    def _scroll_conversation_end(self) -> None:
        self.query_one("#conversation-scroll", VerticalScroll).scroll_end(animate=False)

    @on(AppendTranscript)
    def _on_append_transcript(self, event: AppendTranscript) -> None:
        self._append_transcript_impl(event.renderable)
        event.stop()

    @on(AppendActivity)
    def _on_append_activity(self, event: AppendActivity) -> None:
        self._append_activity_impl(event.renderable)
        event.stop()

    @on(UpdateRuntime)
    def _on_update_runtime(self, event: UpdateRuntime) -> None:
        self._update_runtime_impl(event.renderable)
        event.stop()

    @on(UpdateSessions)
    def _on_update_sessions(self, event: UpdateSessions) -> None:
        self._update_sessions_impl(event.sessions, title=event.title)
        event.stop()

    @on(BeginStream)
    def _on_begin_stream(self, event: BeginStream) -> None:
        self._begin_stream_impl()
        event.stop()

    @on(UpdateStream)
    def _on_update_stream(self, event: UpdateStream) -> None:
        self._update_streaming_view_impl(event.renderable)
        event.stop()

    @on(CommitStream)
    def _on_commit_stream(self, event: CommitStream) -> None:
        self._commit_stream_impl(event.renderable)
        event.stop()

    @on(ClearStream)
    def _on_clear_stream(self, event: ClearStream) -> None:
        self._clear_stream_impl()
        event.stop()

    def _show_runtime_overlay(self) -> None:
        renderable = getattr(self, "_last_runtime_renderable", build_runtime_text(self.session.get_runtime_snapshot()))
        screen = RuntimeOverlayScreen(renderable)
        self._runtime_overlay = screen
        self.push_screen(screen, callback=lambda _: self._clear_overlay("runtime", screen))

    def _show_activity_overlay(self) -> None:
        screen = ActivityOverlayScreen(self._activity_entries)
        self._activity_overlay = screen
        self.push_screen(screen, callback=lambda _: self._clear_overlay("activity", screen))

    def _show_sessions_inline(self) -> None:
        if not self._recent_sessions:
            self.append_activity(Text("Recent Sessions: 0 session(s)"))
            return
        if self._sessions_card is None or not self._sessions_card.is_attached:
            self._sessions_card = SessionsInlineWidget(self._recent_sessions, title=self._sessions_title)
            self.query_one("#conversation-stack", Vertical).mount(self._sessions_card, before=self.query_one("#streaming-slot", Static))
        else:
            self._sessions_card.update_sessions(self._recent_sessions)
        self._boot_sessions_shown = True
        self.call_after_refresh(self._scroll_conversation_end)

    def _clear_overlay(self, overlay_kind: str, screen: ModalScreen) -> None:
        if overlay_kind == "runtime" and self._runtime_overlay is screen:
            self._runtime_overlay = None
        if overlay_kind == "activity" and self._activity_overlay is screen:
            self._activity_overlay = None

    async def _load_session_from_inline(self, session_id: str) -> None:
        await self.session.submit_document(f"/load {session_id}")
        self.action_refresh_runtime()

    def _dismiss_sessions_inline(self) -> None:
        if self._sessions_card is not None and self._sessions_card.is_attached:
            self._sessions_card.remove()
        self._sessions_card = None
        self.call_after_refresh(self.action_focus_composer)

    def on_unmount(self) -> None:
        if self._submit_task is not None and not self._submit_task.done():
            self._submit_task.cancel()
        self._submit_task = None

    async def show_inline_approval(self, renderable, *, title: str) -> bool:
        if self._pending_approval_future is not None and not self._pending_approval_future.done():
            self._pending_approval_future.set_result(False)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        widget = InlineApprovalWidget(renderable, title=title)
        self._pending_approval_widget = widget
        self._pending_approval_future = future
        self.query_one("#conversation-stack", Vertical).mount(widget)
        self.call_after_refresh(self._scroll_conversation_end)
        try:
            return await future
        finally:
            if widget.is_attached:
                widget.remove()
            if self._pending_approval_widget is widget:
                self._pending_approval_widget = None
            if self._pending_approval_future is future:
                self._pending_approval_future = None

    def _queue_submit(self) -> None:
        if self._submitting:
            return
        if self._submit_task is not None and not self._submit_task.done():
            return
        self._submit_task = asyncio.create_task(self._submit_in_background())
        self._submit_task.add_done_callback(self._handle_submit_task_done)

    async def _submit_in_background(self) -> None:
        self._set_submit_state(True)
        try:
            await self.submit_current_buffer()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.presenter.show_notice(
                f"Submit failed: {exc}",
                level="error",
                title="Submit Error",
            )
        finally:
            self._set_submit_state(False)

    def _handle_submit_task_done(self, task: asyncio.Task[None]) -> None:
        if self._submit_task is task:
            self._submit_task = None
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        self.presenter.show_notice(
            f"Background submit crashed: {exc}",
            level="error",
            title="Submit Error",
        )

    def _set_submit_state(self, active: bool) -> None:
        self._submitting = active
        try:
            composer = self.query_one("#composer", TextArea)
            button = self.query_one("#send-button", Button)
        except NoMatches:
            return
        composer.disabled = active
        button.disabled = active
        button.label = "Sending…" if active else "Send"
        if not active and self.is_mounted:
            self.call_after_refresh(self.action_focus_composer)
