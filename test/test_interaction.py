# test/test_interaction.py
"""
Tests for mixed input parsing, cwd normalization, persistent local runners, and ask mode.
"""
from pathlib import Path

import pytest

from k_ai.interaction import MixedInputParser, MixedInputParserError, normalize_workdir
from k_ai.session import ChatSession
from k_ai.models import CompletionChunk, Message, MessageRole


class TestNormalizeWorkdir:
    def test_expands_tilde_and_env(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        project = home / "project"
        project.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        resolved = normalize_workdir("$HOME/project")
        assert resolved == project.resolve()

    def test_rejects_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            normalize_workdir(tmp_path / "missing")


class TestMixedInputParser:
    def test_parses_mixed_blocks(self):
        parser = MixedInputParser()
        blocks = parser.parse("!pwd\n!ls\nexplique\n>import os\n>os.getcwd()")
        assert [block.kind.value for block in blocks] == ["shell", "llm", "python"]
        assert blocks[0].content == "pwd\nls"
        assert blocks[1].content == "explique"

    def test_parses_ephemeral_full_document(self):
        parser = MixedInputParser()
        blocks = parser.parse("/? explique\nce point")
        assert len(blocks) == 1
        assert blocks[0].kind.value == "ephemeral"
        assert blocks[0].content == "explique\nce point"

    def test_rejects_mid_document_ephemeral_prefix(self):
        parser = MixedInputParser()
        with pytest.raises(MixedInputParserError):
            parser.parse("bonjour\n/? ceci ne doit pas apparaître au milieu")


def _mock_llm(*parts: str):
    async def _chat_stream(messages, config=None, tools=None):
        for index, part in enumerate(parts):
            finish = "stop" if index == len(parts) - 1 else None
            yield CompletionChunk(delta_content=part, finish_reason=finish)

    return _chat_stream


class TestAskMode:
    @pytest.mark.asyncio
    async def test_ask_does_not_persist_history(self, cm, tmp_path):
        cm.set("provider", "ollama")
        cm.set("sessions.directory", str(tmp_path / "sessions"))
        cm.set("memory.internal_file", str(tmp_path / "MEMORY.json"))
        session = ChatSession(cm, workspace_root=str(tmp_path))
        session.llm.chat_stream = _mock_llm("réponse rapide")
        result = await session.ask("question")
        assert result == "réponse rapide"
        assert session.history == []

    @pytest.mark.asyncio
    async def test_ephemeral_uses_context_without_persisting(self, cm, tmp_path):
        cm.set("provider", "ollama")
        cm.set("sessions.directory", str(tmp_path / "sessions"))
        cm.set("memory.internal_file", str(tmp_path / "MEMORY.json"))
        session = ChatSession(cm, workspace_root=str(tmp_path))
        session.history.append(Message(role=MessageRole.USER, content="bonjour"))
        session.llm.chat_stream = _mock_llm("réponse contextuelle")
        result = await session._answer_ephemeral("petite question")
        assert result == "réponse contextuelle"
        assert len(session.history) == 1

    @pytest.mark.asyncio
    async def test_mixed_document_passes_local_outputs_into_llm_block(self, cm, tmp_path, monkeypatch):
        cm.set("provider", "ollama")
        cm.set("sessions.directory", str(tmp_path / "sessions"))
        cm.set("memory.internal_file", str(tmp_path / "MEMORY.json"))
        session = ChatSession(cm, workspace_root=str(tmp_path))
        captured = {}

        async def fake_shell(block):
            from k_ai.interaction.models import RunnerExecutionResult, RunnerKind
            return RunnerExecutionResult(
                runner=RunnerKind.SHELL,
                command=block.content,
                stdout="shell-output",
                success=True,
                cwd=Path(tmp_path),
            )

        async def fake_process(message, suppress_switch=False):
            captured["message"] = message

        monkeypatch.setattr(session, "_run_shell_block", fake_shell)
        monkeypatch.setattr(session, "_process_message", fake_process)
        await session._process_submitted_document("!pwd\nexplique ce résultat")
        assert "shell-output" in captured["message"]
        assert "explique ce résultat" in captured["message"]


class TestPersistentRunners:
    @pytest.mark.asyncio
    async def test_shell_runner_persists_cwd(self, cm, tmp_path):
        cm.set("provider", "ollama")
        cm.set("sessions.directory", str(tmp_path / "sessions"))
        cm.set("memory.internal_file", str(tmp_path / "MEMORY.json"))
        cm.set("interaction.runners.shell.command", "sh")
        session = ChatSession(cm, workspace_root=str(tmp_path))
        target = tmp_path / "repo"
        target.mkdir()
        first = await session._run_shell_block(type("Block", (), {"content": f"cd {target} && pwd"})())
        second = await session._run_shell_block(type("Block", (), {"content": "pwd"})())
        assert str(target) in first.stdout
        assert str(target) in second.stdout

    @pytest.mark.asyncio
    async def test_python_runner_persists_state(self, cm, tmp_path):
        cm.set("provider", "ollama")
        cm.set("sessions.directory", str(tmp_path / "sessions"))
        cm.set("memory.internal_file", str(tmp_path / "MEMORY.json"))
        session = ChatSession(cm, workspace_root=str(tmp_path))
        await session._run_python_block(type("Block", (), {"content": "x = 41"})())
        second = await session._run_python_block(type("Block", (), {"content": "print(x + 1)"})())
        assert "42" in second.stdout
