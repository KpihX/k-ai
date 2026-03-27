"""Tests for the Textual chat application."""

from __future__ import annotations

from dataclasses import dataclass, field
import asyncio

import pytest
from textual.containers import Vertical
from textual.widgets import DataTable, TextArea

from k_ai.models import CompletionChunk
from k_ai.ui.textual_chat import TextualChatApp, sanitize_composer_text


@dataclass
class _StubSession:
    cm: object
    submitted: list[str] = field(default_factory=list)
    attached_ui: object | None = None

    def attach_ui(self, ui) -> None:
        self.attached_ui = ui

    async def bootstrap(self) -> bool:
        return True

    async def submit_document(self, text: str) -> bool:
        self.submitted.append(text)
        return True

    def get_runtime_snapshot(self) -> dict:
        return {
            "provider": "test",
            "model": "demo",
            "temperature": 0.2,
            "max_tokens": 512,
            "session_id": "abc12345",
            "session_type": "classic",
            "cwd": "/tmp/demo",
            "estimated_context_tokens": 12,
            "context_window": 1024,
            "context_percent": 1.2,
            "compaction_trigger_tokens": 800,
            "compaction_trigger_percent": 80,
            "remaining_context_tokens": 1012,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "token_source": "estimated",
            "history_messages": len(self.submitted),
            "stream": True,
            "skills_summary": "",
            "skills_catalog_count": 0,
            "hooks_summary": "",
            "hooks_count": 0,
            "mcp_summary": "",
            "mcp_server_count": 0,
            "mcp_tool_count": 0,
            "mcp_resource_count": 0,
            "mcp_prompt_count": 0,
            "auth_mode": "api_key",
            "session_summary": "",
            "session_themes": [],
        }


@dataclass
class _StreamingStubSession(_StubSession):
    async def submit_document(self, text: str) -> bool:
        self.submitted.append(text)
        assert self.attached_ui is not None
        self.attached_ui.show_user(text, theme_name="default")
        with self.attached_ui.stream_assistant(
            model_name="demo-model",
            render_mode="rich",
            spinner_name="dots",
            theme_name="default",
            flush_min_chars=1,
            tail_chars=10,
            interrupt_hint="",
        ) as stream:
            await asyncio.sleep(0)
            stream.update(CompletionChunk(delta_content="réponse"))
        return True


@pytest.mark.asyncio
async def test_textual_chat_submits_composer_content(cm):
    session = _StubSession(cm)
    app = TextualChatApp(session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.text = "bonjour textual"
        await pilot.press("ctrl+s")

    assert session.submitted == ["bonjour textual"]


@pytest.mark.asyncio
async def test_textual_chat_enter_adds_newline_and_grows_composer(cm):
    session = _StubSession(cm)
    app = TextualChatApp(session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.text = "ligne 1"
        composer.focus()
        await pilot.pause()
        before = composer.styles.height
        await pilot.press("enter")
        await pilot.pause()
        assert "\n" in composer.text
        assert composer.styles.height != before
        assert session.submitted == []


@pytest.mark.asyncio
async def test_textual_chat_updates_sessions_and_stream(cm):
    session = _StubSession(cm)
    app = TextualChatApp(session)

    async with app.run_test() as pilot:
        app.update_sessions(
            [
                type(
                    "Meta",
                    (),
                    {
                        "id": "12345678abcdef",
                        "session_type": "classic",
                        "summary": "Résumé",
                        "title": "Titre",
                        "message_count": 4,
                    },
                )()
            ]
        )
        app.action_show_sessions()
        await pilot.pause()
        await pilot.pause()
        table = app.query_one("#sessions-inline-table", DataTable)
        assert table.row_count == 1

        with app.presenter.stream_assistant(
            model_name="demo-model",
            render_mode="rich",
            spinner_name="dots",
            theme_name="default",
            flush_min_chars=50,
            tail_chars=10,
            interrupt_hint="",
        ) as stream:
            stream.update(CompletionChunk(delta_content="réponse en cours"))

        await pilot.pause()
        assert "-active" not in app.query_one("#streaming-slot").classes


@pytest.mark.asyncio
async def test_textual_chat_opens_sessions_overlay_after_bootstrap(cm):
    session = _StubSession(cm)
    app = TextualChatApp(session)

    async with app.run_test() as pilot:
        app.update_sessions(
            [
                type(
                    "Meta",
                    (),
                    {
                        "id": "12345678abcdef",
                        "session_type": "classic",
                        "summary": "Résumé",
                        "title": "Titre",
                        "message_count": 4,
                    },
                )()
            ]
        )
        await pilot.pause()
        assert isinstance(app.query_one("#sessions-inline-table", DataTable), DataTable)


@pytest.mark.asyncio
async def test_textual_chat_submits_from_sessions_overlay_selection(cm):
    session = _StubSession(cm)
    app = TextualChatApp(session)

    async with app.run_test() as pilot:
        app.update_sessions(
            [
                type(
                    "Meta",
                    (),
                    {
                        "id": "12345678abcdef",
                        "session_type": "classic",
                        "summary": "Résumé",
                        "title": "Titre",
                        "message_count": 4,
                    },
                )()
            ]
        )
        app.action_show_sessions()
        await pilot.pause()
        table = app.query_one("#sessions-inline-table", DataTable)
        table.move_cursor(row=0, column=0)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert session.submitted == ["/load 12345678abcdef"]


@pytest.mark.asyncio
async def test_textual_chat_keeps_main_flow_after_submit(cm):
    session = _StubSession(cm)
    app = TextualChatApp(session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.text = "bonjour"
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert len(app.screen_stack) == 1
        assert session.submitted == ["bonjour"]


@pytest.mark.asyncio
async def test_textual_chat_submit_runs_in_background_and_commits_response(cm):
    session = _StreamingStubSession(cm)
    app = TextualChatApp(session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.text = "bonjour"
        await pilot.press("ctrl+s")
        await pilot.pause()
        await pilot.pause()
        stack = app.query_one("#conversation-stack", Vertical)
        assert len(stack.children) >= 2
        assert app._stream_widget is None
        assert session.submitted == ["bonjour"]


@pytest.mark.asyncio
async def test_textual_chat_status_bar_contains_runtime_details(cm):
    session = _StubSession(cm)
    app = TextualChatApp(session)

    async with app.run_test() as pilot:
        await pilot.pause()
        text = str(app._last_status_text)
        assert "test/demo" in text
        assert "ctx" in text
        assert "tok" in text
        assert "skills" in text


@pytest.mark.asyncio
async def test_textual_chat_inline_approval_resolves_yes(cm):
    session = _StubSession(cm)
    app = TextualChatApp(session)

    async with app.run_test() as pilot:
        task = asyncio.create_task(app.show_inline_approval("approval body", title="approval"))
        await pilot.pause()
        await pilot.press("y")
        await pilot.pause()
        assert await task is True


def test_sanitize_composer_text_strips_terminal_garbage():
    dirty = "\x1b[<35;71;27Mhello\x1b[200~"
    assert sanitize_composer_text(dirty) == "hello"
