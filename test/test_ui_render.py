# test/test_ui_render.py
"""Tests for UI streaming and PTY output normalization."""

import errno
from pathlib import Path
from rich.console import Console
from unittest.mock import ANY, MagicMock, patch

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
    def test_child_preexec_sets_controlling_tty_when_available(self):
        from k_ai.interaction.runners import _child_preexec

        with patch("k_ai.interaction.runners.os.setsid") as setsid:
            with patch("k_ai.interaction.runners._set_parent_death_signal") as deathsig:
                with patch("k_ai.interaction.runners.fcntl.ioctl") as ioctl:
                    _child_preexec(42)

        setsid.assert_called_once_with()
        ioctl.assert_called_once()
        deathsig.assert_called_once()

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

    def test_shell_output_normalizer_strips_prompt_prefixed_command_echo(self):
        runner = ShellRunner.__new__(ShellRunner)
        cleaned = runner._normalize_block_output("ls", "lls\nDesktop\nDocuments\n")
        assert cleaned == "Desktop\nDocuments"

    def test_shell_output_normalizer_strips_single_char_prefixed_echo(self):
        runner = ShellRunner.__new__(ShellRunner)
        cleaned = runner._normalize_block_output("ls -la", "lls -la\nfile.txt\n")
        assert cleaned == "file.txt"

    def test_drain_pending_output_ignores_eio_on_pty_close(self):
        runner = PersistentPTYRunner.__new__(PersistentPTYRunner)
        runner.master_fd = 10
        runner.read_chunk_size = 4096

        with patch("k_ai.interaction.runners.select.select", side_effect=[([10], [], []), ([], [], [])]):
            with patch("k_ai.interaction.runners.os.read", side_effect=OSError(errno.EIO, "Input/output error")):
                runner._drain_pending_output()

    def test_interactive_handoff_flushes_stdin_before_reading(self):
        runner = PersistentPTYRunner.__new__(PersistentPTYRunner)
        runner.console = MagicMock(file=MagicMock(isatty=lambda: False))
        runner.master_fd = 10
        runner.read_chunk_size = 4096
        runner._proc = MagicMock()
        runner._proc.poll.return_value = 0
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = True
        fake_stdin.fileno.return_value = 11

        with patch("k_ai.interaction.runners.sys.stdin", fake_stdin):
            with patch("k_ai.interaction.runners.termios.tcgetattr", return_value=[0, 0, 0, 0, 0, 0]):
                with patch("k_ai.interaction.runners.tty.setraw"):
                    with patch("k_ai.interaction.runners.termios.tcflush") as tcflush:
                        with patch("k_ai.interaction.runners.termios.tcsetattr"):
                            with patch("k_ai.interaction.runners.select.select", return_value=([10], [], [])):
                                with patch("k_ai.interaction.runners.os.read", side_effect=OSError(errno.EIO, "Input/output error")):
                                    runner._interact_until_idle(
                                        terminator=None,
                                        activation_token=None,
                                        capture_output=False,
                                        allow_escape=True,
                                        display_output=False,
                                        sanitize_capture=False,
                                        sanitize_display=False,
                                        route_stdin=True,
                                        input_notice=None,
                                        input_prompt_patterns=None,
                                        detach_sequences=(b"\x1b\x1b",),
                                    )

        assert tcflush.call_count == 2
        tcflush.assert_any_call(11, ANY)

    def test_interactive_handoff_uses_alternate_screen_when_tty(self):
        runner = PersistentPTYRunner.__new__(PersistentPTYRunner)
        terminal = MagicMock()
        terminal.isatty.return_value = True
        runner.console = MagicMock(file=terminal)
        runner.master_fd = 10
        runner.read_chunk_size = 4096
        runner._proc = MagicMock()
        runner._proc.poll.return_value = 0
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = False

        with patch("k_ai.interaction.runners.sys.stdin", fake_stdin):
            with patch("k_ai.interaction.runners.select.select", return_value=([10], [], [])):
                with patch("k_ai.interaction.runners.os.read", side_effect=OSError(errno.EIO, "Input/output error")):
                    runner._interact_until_idle(
                        terminator=None,
                        activation_token=None,
                        capture_output=False,
                        allow_escape=True,
                        display_output=True,
                        sanitize_capture=False,
                        sanitize_display=False,
                        route_stdin=True,
                        input_notice="hello",
                        input_prompt_patterns=None,
                        detach_sequences=(b"\x1b\x1b",),
                    )

        writes = "".join(call.args[0] for call in terminal.write.call_args_list)
        assert "\x1b[?1049h" in writes
        assert "\x1b[?1049l" in writes

    def test_drain_ready_stdin_consumes_only_immediately_available_bytes(self):
        with patch("k_ai.interaction.runners.select.select", side_effect=[([11], [], []), ([], [], [])]):
            with patch("k_ai.interaction.runners.os.read", return_value=b"\n"):
                PersistentPTYRunner._drain_ready_stdin(11)

    def test_strip_leading_newline_bytes(self):
        assert PersistentPTYRunner._strip_leading_newline_bytes(b"\r\nhello") == b"hello"
        assert PersistentPTYRunner._strip_leading_newline_bytes(b"hello") == b"hello"

    def test_interactive_handoff_waits_for_first_pty_output_before_forwarding_stdin(self):
        runner = PersistentPTYRunner.__new__(PersistentPTYRunner)
        runner.console = MagicMock(file=MagicMock(isatty=lambda: False))
        runner.master_fd = 10
        runner.read_chunk_size = 4096
        runner._proc = MagicMock()
        runner._proc.poll.return_value = 0
        fake_stdin = MagicMock()
        fake_stdin.isatty.return_value = True
        fake_stdin.fileno.return_value = 11

        def fake_read(fd, _size):
            if fd == 10:
                raise OSError(errno.EIO, "Input/output error")
            return b"secret\n"

        select_events = iter([([11], [], []), ([], [], []), ([10], [], [])])

        def fake_select(_fds, _w, _x, _timeout):
            return next(select_events, ([], [], []))

        with patch("k_ai.interaction.runners.sys.stdin", fake_stdin):
            with patch("k_ai.interaction.runners.termios.tcgetattr", return_value=[0, 0, 0, 0, 0, 0]):
                with patch("k_ai.interaction.runners.tty.setraw"):
                    with patch("k_ai.interaction.runners.termios.tcflush"):
                        with patch("k_ai.interaction.runners.termios.tcsetattr"):
                            with patch("k_ai.interaction.runners.select.select", side_effect=fake_select):
                                with patch("k_ai.interaction.runners.os.read", side_effect=fake_read):
                                    with patch("k_ai.interaction.runners.os.write") as os_write:
                                        runner._interact_until_idle(
                                            terminator=None,
                                            activation_token=None,
                                            capture_output=False,
                                            allow_escape=True,
                                            display_output=False,
                                            sanitize_capture=False,
                                            sanitize_display=False,
                                            route_stdin=True,
                                            input_notice=None,
                                            input_prompt_patterns=None,
                                            detach_sequences=(b"\x1b\x1b",),
                                        )

        os_write.assert_not_called()


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

    def test_hides_thinking_panel_when_disabled(self):
        console = Console(record=True, force_terminal=False, width=80)
        with patch("k_ai.ui.render.Live", _FakeLive):
            with StreamingRenderer(console, "demo-model", show_thinking=False, flush_min_chars=16, tail_chars=4) as renderer:
                renderer.update(CompletionChunk(delta_thought="private reasoning"))
                renderer.update(CompletionChunk(delta_content="Visible answer", finish_reason="stop"))

        output = console.export_text()
        assert "Reasoning" not in output
        assert "Visible answer" in output
