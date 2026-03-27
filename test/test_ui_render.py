# test/test_ui_render.py
"""Tests for UI streaming and PTY output normalization."""

from pathlib import Path
from rich.console import Console
from unittest.mock import patch

from k_ai.interaction.runners import PersistentPTYRunner, ShellRunner, TerminalOutputFilter
from k_ai.models import CompletionChunk
from k_ai.ui.render import StreamingRenderer


class _FakeLive:
    def __init__(self, renderable, console=None, refresh_per_second=15, transient=False):
        self.renderable = renderable
        self.console = console
        self.refresh_per_second = refresh_per_second
        self.transient = transient
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def update(self, renderable):
        self.renderable = renderable


class TestTerminalOutputFilter:
    def test_strips_terminal_control_sequences_across_chunk_boundaries(self):
        filt = TerminalOutputFilter()
        first = filt.feed("hello\x1b[?2004")
        second = filt.feed("h\r\nworld\x1b]0;title\x07")
        third = filt.flush()

        assert first == "hello"
        assert second == "\nworld"
        assert third == ""

    def test_normalizes_backspaces(self):
        filt = TerminalOutputFilter()
        assert filt.feed("abc\b\bZ") == "aZ"

    def test_sentinel_parser_ignores_echoed_printf_lines(self):
        runner = PersistentPTYRunner.__new__(PersistentPTYRunner)
        stdout = "\n".join([
            'printf "__TOKEN__ cwd=%s\\n" "$PWD"',
            "__TOKEN__ cwd=/tmp/demo",
            "real output",
        ])
        cleaned, cwd = runner._extract_sentinel(stdout, "__TOKEN__")
        assert cwd == Path("/tmp/demo")
        assert "printf" not in cleaned
        assert "real output" in cleaned

    def test_focus_runs_enter_and_exit_commands(self):
        runner = PersistentPTYRunner.__new__(PersistentPTYRunner)
        runner.console = Console(record=True, force_terminal=False)
        runner.focus_enter_commands = ["enter"]
        runner.focus_exit_commands = ["exit"]
        runner.ensure_alive = lambda: None
        calls = []
        runner._run_quiet_commands = lambda commands, token_prefix="__KAI_SENTINEL__": calls.append(list(commands))
        runner._interact_until_idle = lambda **kwargs: ("", False)
        runner.focus("notice")
        assert calls == [["enter"], ["exit"]]

    def test_shell_output_normalizer_strips_prompt_and_echo_artifacts(self):
        runner = ShellRunner.__new__(ShellRunner)
        cleaned = runner._normalize_block_output(
            'printf "hello\\n"',
            '\n__KAI_SENTINEL__abc cwd=/tmp/demo\n%\n\nprintf "hello\\n"\n\nhello\n%\ncwd=%s\\n" "$PWD"\n',
        )
        assert cleaned == "hello"


class TestStreamingRenderer:
    def test_prefers_paragraph_boundary_for_incremental_flush(self):
        console = Console(record=True, force_terminal=False, width=80)
        with patch("k_ai.ui.render.Live", _FakeLive):
            with StreamingRenderer(console, "demo-model", flush_min_chars=16, tail_chars=4) as renderer:
                renderer.update(CompletionChunk(delta_content="Premier bloc.\n\nDeux"))
                renderer.update(CompletionChunk(delta_content="ieme bloc final", finish_reason="stop"))

        output = console.export_text()
        assert output.count("Assistant") == 1
        assert "Premier bloc." in output
        assert "Deuxieme bloc final" in output

    def test_does_not_flush_inside_unclosed_fence(self):
        renderer = StreamingRenderer(Console(record=True, force_terminal=False), "demo", flush_min_chars=12, tail_chars=4)
        pending = "```python\nprint('x')\n"
        assert renderer._find_flush_boundary(pending, final=False) == 0
        assert renderer._find_flush_boundary(f"{pending}```\n\n", final=False) > 0
