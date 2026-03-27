"""Tests for the Textual chat application."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from textual.widgets import DataTable, TextArea

from k_ai.models import CompletionChunk
from k_ai.ui.textual_chat import TextualChatApp


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


@pytest.mark.asyncio
async def test_textual_chat_submits_composer_content(cm):
    session = _StubSession(cm)
    app = TextualChatApp(session)

    async with app.run_test() as pilot:
        composer = app.query_one("#composer", TextArea)
        composer.text = "bonjour textual"
        await pilot.press("ctrl+enter")

    assert session.submitted == ["bonjour textual"]


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
        table = app.query_one("#sessions-table", DataTable)
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
