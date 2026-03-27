# test/test_main.py
"""
Tests for CLI entry points around ask mode and cwd wiring.
"""
from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from k_ai.main import app


runner = CliRunner()


def test_root_prompt_invokes_ask(tmp_path):
    target = tmp_path / "repo"
    target.mkdir()
    with patch("k_ai.main.ChatSession") as session_cls:
        instance = session_cls.return_value
        instance.ask = AsyncMock(return_value="ok")
        result = runner.invoke(app, ["-C", str(target), "bonjour"])
    assert result.exit_code == 0
    assert "ok" in result.stdout
    assert session_cls.call_args.kwargs["workspace_root"] == str(target.resolve())


def test_chat_accepts_cwd_option(tmp_path):
    target = tmp_path / "repo"
    target.mkdir()
    with patch("k_ai.main.ChatSession") as session_cls:
        instance = session_cls.return_value
        instance.start = AsyncMock(return_value=None)
        result = runner.invoke(app, ["chat", "-C", str(target)])
    assert result.exit_code == 0
    assert session_cls.call_args.kwargs["workspace_root"] == str(target.resolve())


def test_chat_uses_textual_app_by_default(tmp_path):
    target = tmp_path / "repo"
    target.mkdir()
    with patch("k_ai.main.ChatSession") as session_cls, patch("k_ai.main.TextualChatApp") as app_cls:
        result = runner.invoke(app, ["chat", "-C", str(target)])
    assert result.exit_code == 0
    app_cls.assert_called_once_with(session_cls.return_value)
    app_cls.return_value.run.assert_called_once()


def test_chat_can_force_classic_ui(tmp_path):
    target = tmp_path / "repo"
    target.mkdir()
    with patch("k_ai.main.ChatSession") as session_cls, patch("k_ai.main.TextualChatApp") as app_cls:
        instance = session_cls.return_value
        instance.start = AsyncMock(return_value=None)
        result = runner.invoke(app, ["chat", "-C", str(target), "--classic-ui"])
    assert result.exit_code == 0
    instance.start.assert_awaited_once()
    app_cls.assert_not_called()
