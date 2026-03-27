"""UI presenter abstractions for classic console and Textual runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Any, Sequence

from rich.console import Console
from rich.prompt import Confirm

from ..models import CompletionChunk, Message, TokenUsage, ToolResult
from ..tools.base import ToolDisplaySpec
from .render import (
    StreamingRenderer,
    build_local_runner_output_renderable,
    build_notice_renderable,
    build_sessions_table_renderable,
    build_tool_proposal_renderable,
    build_tool_result_renderable,
    render_assistant_panel,
    render_runtime_panel,
    render_user_panel,
)


class AssistantStream(ABC):
    """Streaming sink abstraction used by ChatSession."""

    full_content: str
    full_thought: str
    last_usage: TokenUsage | None

    @abstractmethod
    def update(self, chunk: CompletionChunk) -> None:
        """Consume a streamed completion chunk."""


class SessionUI(ABC):
    """Minimal UI surface needed by ChatSession."""

    def __init__(self, console: Console):
        self.console = console

    @abstractmethod
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
        """Return a context manager yielding an AssistantStream."""

    def show_user(self, content: str, *, theme_name: str) -> None:
        self.console.print(render_user_panel(content, theme_name=theme_name))

    def show_assistant(self, content: str, *, model_name: str, render_mode: str = "rich", usage: TokenUsage | None = None, theme_name: str = "default") -> None:
        self.console.print(render_assistant_panel(content, model_name, render_mode=render_mode, usage=usage, theme_name=theme_name))

    def show_notice(self, message: str, *, level: str = "info", title: str | None = None) -> None:
        self.console.print(build_notice_renderable(message, level=level, title=title))

    def show_runtime(self, snapshot: dict[str, Any], *, title: str = "Runtime Transparency", mode: str = "compact", theme_name: str = "default") -> None:
        self.console.print(render_runtime_panel(snapshot, title=title, mode=mode, theme_name=theme_name))

    def show_sessions(self, sessions, *, title: str = "Recent Sessions") -> None:
        self.console.print(build_sessions_table_renderable(sessions, title=title))

    def show_runner_output(self, *, title: str, content: str, cwd: str, border_style: str = "cyan") -> None:
        self.console.print(
            build_local_runner_output_renderable(
                title=title,
                content=content,
                cwd=cwd,
                border_style=border_style,
            )
        )

    def show_tool_result(self, spec: ToolDisplaySpec, result: ToolResult, content) -> None:
        self.console.print(build_tool_result_renderable(spec, result, content))

    @abstractmethod
    async def confirm_tool_execution(
        self,
        spec: ToolDisplaySpec,
        sections: Sequence[tuple[str, object]],
        *,
        rationale: str,
        show_rationale: bool,
        requires_approval: bool,
    ) -> bool:
        """Display a tool proposal and optionally confirm it."""

    def show_loaded_messages(self, messages: Sequence[Message], *, model_name: str, render_mode: str = "rich", theme_name: str = "default") -> None:
        for message in messages:
            if message.role.value == "system":
                continue
            if message.role.value == "user":
                self.show_user(message.content, theme_name=theme_name)
                continue
            if message.role.value == "assistant":
                self.show_assistant(
                    message.content,
                    model_name=model_name,
                    render_mode=render_mode,
                    theme_name=theme_name,
                )
                continue
            self.show_notice(message.content[:300] + ("..." if len(message.content) > 300 else ""), level="info", title=f"Tool {message.name or 'tool'}")

    def suspend(self):
        """Suspend the active UI when raw terminal access is required."""
        return nullcontext()


class ClassicAssistantStream(AssistantStream):
    """Adapter around the existing Rich StreamingRenderer."""

    def __init__(self, renderer: StreamingRenderer):
        self._renderer = renderer

    @property
    def full_content(self) -> str:
        return self._renderer.full_content

    @property
    def full_thought(self) -> str:
        return self._renderer.full_thought

    @property
    def last_usage(self) -> TokenUsage | None:
        return self._renderer.last_usage

    def update(self, chunk: CompletionChunk) -> None:
        self._renderer.update(chunk)


class _ClassicStreamContext:
    def __init__(self, renderer: StreamingRenderer):
        self._renderer = renderer

    def __enter__(self) -> ClassicAssistantStream:
        self._renderer.__enter__()
        return ClassicAssistantStream(self._renderer)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._renderer.__exit__(exc_type, exc_val, exc_tb)


class ClassicSessionUI(SessionUI):
    """Existing Rich + prompt_toolkit console presentation."""

    def __init__(self, console: Console):
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
        renderer = StreamingRenderer(
            self.console,
            model_name,
            render_mode=render_mode,
            spinner_name=spinner_name,
            theme_name=theme_name,
            flush_min_chars=flush_min_chars,
            tail_chars=tail_chars,
            interrupt_hint=interrupt_hint,
        )
        return _ClassicStreamContext(renderer)

    async def confirm_tool_execution(
        self,
        spec: ToolDisplaySpec,
        sections: Sequence[tuple[str, object]],
        *,
        rationale: str,
        show_rationale: bool,
        requires_approval: bool,
    ) -> bool:
        self.console.print(
            build_tool_proposal_renderable(
                spec,
                sections,
                rationale=rationale,
                show_rationale=show_rationale,
                requires_approval=requires_approval,
            )
        )
        if not requires_approval:
            return True
        return Confirm.ask("Approve tool execution?", console=self.console, default=True)
