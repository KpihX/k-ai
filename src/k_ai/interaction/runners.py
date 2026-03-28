"""Persistent PTY-backed local runners for user shell and Python blocks."""

from __future__ import annotations

import atexit
import ctypes
import errno
import fcntl
import os
import pty
import re
import select
import shlex
import signal
import subprocess
import sys
import termios
import time
import threading
import tty
import uuid
import weakref
from contextlib import suppress
from pathlib import Path
from typing import Optional, Pattern

from rich.console import Console
from rich.panel import Panel

from .models import RunnerExecutionResult, RunnerKind


class PTYRunnerError(RuntimeError):
    """Raised when a PTY-backed user runner cannot be created or driven."""


_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ESC_RE = re.compile(r"\x1b[@-_]")
_DCS_RE = re.compile(r"\x1bP.*?(?:\x1b\\)", re.DOTALL)
_PR_SET_PDEATHSIG = 1
_LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform.startswith("linux") else None
_ACTIVE_RUNNERS: "weakref.WeakSet[PersistentPTYRunner]" = weakref.WeakSet()
_ACTIVE_PGIDS: set[int] = set()
_ACTIVE_LOCK = threading.RLock()


def _set_parent_death_signal(signum: int = signal.SIGTERM) -> None:
    if _LIBC is None:
        return
    result = _LIBC.prctl(_PR_SET_PDEATHSIG, signum, 0, 0, 0)
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    if os.getppid() == 1:
        os.kill(os.getpid(), signum)


def _child_preexec(slave_fd: int | None = None) -> None:
    os.setsid()
    if slave_fd is not None and hasattr(termios, "TIOCSCTTY"):
        with suppress(Exception):
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
    with suppress(Exception):
        _set_parent_death_signal(signal.SIGTERM)


def _register_process_group(pid: int) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PGIDS.add(pid)


def _unregister_process_group(pid: int) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PGIDS.discard(pid)


def _cleanup_registered_processes() -> None:
    with _ACTIVE_LOCK:
        pgids = list(_ACTIVE_PGIDS)
    if not pgids:
        return
    for pgid in pgids:
        with suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
    deadline = time.time() + 1.0
    while time.time() < deadline:
        remaining = []
        for pgid in pgids:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                continue
            remaining.append(pgid)
        if not remaining:
            break
        time.sleep(0.05)
        pgids = remaining
    for pgid in pgids:
        with suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
    with _ACTIVE_LOCK:
        _ACTIVE_PGIDS.difference_update(pgids)


def _cleanup_all_runners() -> None:
    for runner in list(_ACTIVE_RUNNERS):
        with suppress(Exception):
            runner.close()
    _cleanup_registered_processes()


atexit.register(_cleanup_all_runners)


class TerminalOutputFilter:
    """Normalize PTY output into stable text for block execution and capture."""

    def __init__(self) -> None:
        self._carry = ""

    def feed(self, text: str) -> str:
        combined = self._carry + text
        complete, self._carry = self._split_incomplete_escape(combined)
        return self._sanitize_complete_text(complete)

    def flush(self) -> str:
        if not self._carry:
            return ""
        leftover = self._sanitize_complete_text(self._carry)
        self._carry = ""
        return leftover

    @classmethod
    def _split_incomplete_escape(cls, text: str) -> tuple[str, str]:
        last_escape = text.rfind("\x1b")
        if last_escape < 0:
            return text, ""
        trailing = text[last_escape:]
        if cls._looks_like_complete_escape(trailing):
            return text, ""
        return text[:last_escape], trailing

    @staticmethod
    def _looks_like_complete_escape(fragment: str) -> bool:
        if not fragment.startswith("\x1b"):
            return True
        return bool(
            _OSC_RE.fullmatch(fragment)
            or _CSI_RE.fullmatch(fragment)
            or _ESC_RE.fullmatch(fragment)
            or _DCS_RE.fullmatch(fragment)
        )

    @staticmethod
    def _sanitize_complete_text(text: str) -> str:
        text = _OSC_RE.sub("", text)
        text = _DCS_RE.sub("", text)
        text = _CSI_RE.sub("", text)
        text = _ESC_RE.sub("", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
        text = TerminalOutputFilter._apply_backspaces(text)
        return text

    @staticmethod
    def _apply_backspaces(text: str) -> str:
        if "\b" not in text:
            return text
        rendered: list[str] = []
        for char in text:
            if char == "\b":
                if rendered:
                    rendered.pop()
                continue
            rendered.append(char)
        return "".join(rendered)


class PersistentPTYRunner:
    """Small PTY wrapper used for persistent user-side shell or Python execution."""

    kind: RunnerKind

    def __init__(
        self,
        *,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        console: Console,
        runner_name: str,
        escape_sequence: str = "\x1d",
        read_chunk_size: int = 4096,
        init_commands: list[str] | None = None,
        sanitize_output: bool = True,
        focus_enter_commands: list[str] | None = None,
        focus_exit_commands: list[str] | None = None,
    ) -> None:
        self.argv = list(argv)
        self.cwd = Path(cwd)
        self.env = dict(env)
        self.console = console
        self.runner_name = runner_name
        self.escape_sequence = escape_sequence.encode("utf-8")
        self.read_chunk_size = int(read_chunk_size)
        self.init_commands = [command for command in (init_commands or []) if str(command).strip()]
        self.sanitize_output = bool(sanitize_output)
        self.focus_enter_commands = [command for command in (focus_enter_commands or []) if str(command).strip()]
        self.focus_exit_commands = [command for command in (focus_exit_commands or []) if str(command).strip()]
        self.master_fd: Optional[int] = None
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._registered_pid: Optional[int] = None
        self._last_output: str = ""
        _ACTIVE_RUNNERS.add(self)
        self._start()

    def _start(self) -> None:
        master_fd, slave_fd = pty.openpty()
        attrs = termios.tcgetattr(slave_fd)
        attrs[3] = attrs[3] & ~termios.ECHO
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
        env = dict(self.env)
        env.setdefault("TERM", os.environ.get("TERM", "xterm-256color"))
        self._proc = subprocess.Popen(
            self.argv,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=str(self.cwd),
            env=env,
            close_fds=True,
            preexec_fn=lambda: _child_preexec(slave_fd),
        )
        os.close(slave_fd)
        os.set_blocking(master_fd, False)
        self.master_fd = master_fd
        self._registered_pid = self._proc.pid
        _register_process_group(self._registered_pid)
        self._run_initialization_commands()

    def ensure_alive(self) -> None:
        if self._proc is None or self.master_fd is None:
            raise PTYRunnerError(f"{self.runner_name} runner is not initialized.")
        if self._proc.poll() is not None:
            raise PTYRunnerError(f"{self.runner_name} runner exited with code {self._proc.returncode}.")

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        if self.master_fd is not None:
            with suppress(OSError):
                os.close(self.master_fd)
        if self._registered_pid is not None:
            _unregister_process_group(self._registered_pid)
        self.master_fd = None
        self._proc = None
        self._registered_pid = None

    def interrupt(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        with suppress(ProcessLookupError):
            os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)

    def focus(self, notice: str | None = None) -> None:
        self.ensure_alive()
        if notice:
            self.console.print(f"[bold cyan]{notice}[/bold cyan]")
        self._run_quiet_commands(self.focus_enter_commands)
        escape_sequence = getattr(self, "escape_sequence", b"\x1d")
        try:
            self._interact_until_idle(
                terminator=None,
                activation_token=None,
                capture_output=False,
                allow_escape=True,
                display_output=True,
                sanitize_capture=False,
                sanitize_display=False,
                route_stdin=True,
                detach_sequences=(escape_sequence,),
                input_notice=None,
                input_prompt_patterns=None,
            )
        finally:
            self._run_quiet_commands(self.focus_exit_commands)

    def refresh_cwd(self) -> Path:
        result = self.run_block(self._cwd_probe_command(), display_output=False)
        if result.cwd is not None:
            self.cwd = result.cwd
        return self.cwd

    def run_block(
        self,
        block: str,
        *,
        display_output: bool = True,
        route_stdin: bool = False,
        input_notice: str | None = None,
        input_prompt_patterns: list[str] | None = None,
        detach_sequences: tuple[bytes, ...] | None = None,
    ) -> RunnerExecutionResult:
        self.ensure_alive()
        token = f"__KAI_SENTINEL__{uuid.uuid4().hex}__"
        payload = self._wrap_block(block, token)
        stdout, interrupted, handoff_used, detached = self._execute_payload(
            payload,
            terminator=token,
            activation_token=None,
            capture_output=True,
            allow_escape=route_stdin,
            display_output=display_output,
            sanitize_capture=self.sanitize_output,
            sanitize_display=self.sanitize_output,
            route_stdin=route_stdin,
            input_notice=input_notice,
            input_prompt_patterns=input_prompt_patterns,
            detach_sequences=detach_sequences,
        )
        normalized_output, cwd = self._extract_sentinel(stdout, token)
        normalized_output = self._normalize_block_output(block, normalized_output)
        self.cwd = cwd or self.cwd
        return RunnerExecutionResult(
            runner=self.kind,
            command=block,
            stdout=normalized_output.strip(),
            success=not interrupted,
            returncode=0 if not interrupted else None,
            cwd=self.cwd,
            interrupted=interrupted,
            metadata={
                "interactive_handoff": handoff_used,
                "detached": detached,
            },
        )

    def _run_initialization_commands(self) -> None:
        self._run_quiet_commands(self.init_commands, token_prefix="__KAI_INIT__")

    def _run_quiet_commands(self, commands: list[str], *, token_prefix: str = "__KAI_SENTINEL__") -> None:
        for command in commands:
            token = f"{token_prefix}{uuid.uuid4().hex}__"
            payload = self._wrap_block(command, token)
            stdout, _, _, _ = self._execute_payload(
                payload,
                terminator=token,
                activation_token=None,
                capture_output=True,
                allow_escape=False,
                display_output=False,
                sanitize_capture=self.sanitize_output,
                sanitize_display=self.sanitize_output,
                route_stdin=False,
                input_notice=None,
                input_prompt_patterns=None,
                detach_sequences=None,
            )
            _, cwd = self._extract_sentinel(stdout, token)
            if cwd is not None:
                self.cwd = cwd

    def _execute_payload(
        self,
        payload: str,
        *,
        terminator: str | None,
        activation_token: str | None,
        capture_output: bool,
        allow_escape: bool,
        display_output: bool,
        sanitize_capture: bool,
        sanitize_display: bool,
        route_stdin: bool,
        input_notice: str | None,
        input_prompt_patterns: list[str] | None,
        detach_sequences: tuple[bytes, ...] | None,
        discard_initial_newline: bool = False,
    ) -> tuple[str, bool, bool, bool]:
        assert self.master_fd is not None
        self._drain_pending_output()
        os.write(self.master_fd, payload.encode("utf-8"))
        return self._interact_until_idle(
            terminator=terminator,
            activation_token=activation_token,
            capture_output=capture_output,
            allow_escape=allow_escape,
            display_output=display_output,
            sanitize_capture=sanitize_capture,
            sanitize_display=sanitize_display,
            route_stdin=route_stdin,
            input_notice=input_notice,
            input_prompt_patterns=input_prompt_patterns,
            detach_sequences=detach_sequences,
            discard_initial_newline=discard_initial_newline,
        )

    def _interact_until_idle(
        self,
        *,
        terminator: str | None,
        activation_token: str | None,
        capture_output: bool,
        allow_escape: bool,
        display_output: bool,
        sanitize_capture: bool,
        sanitize_display: bool,
        route_stdin: bool,
        input_notice: str | None,
        input_prompt_patterns: list[str] | None,
        detach_sequences: tuple[bytes, ...] | None,
        discard_initial_newline: bool = False,
    ) -> tuple[str, bool, bool, bool]:
        assert self.master_fd is not None
        output_parts: list[str] = []
        interrupted = False
        detached = False
        handoff_used = False
        stdin_fd: Optional[int] = None
        old_attrs = None
        alternate_screen = False
        terminal_output = getattr(getattr(self, "console", None), "file", None)
        capture_filter = TerminalOutputFilter() if sanitize_capture and capture_output else None
        display_filter = TerminalOutputFilter() if sanitize_display and display_output else None
        compiled_input_patterns = self._compile_patterns(input_prompt_patterns or [])
        handoff_active = route_stdin
        handoff_ready = not route_stdin
        notice_rendered = False
        pending_escape = False
        pending_initial_newline_discard = discard_initial_newline
        if (
            route_stdin
            and display_output
            and terminal_output is not None
            and hasattr(terminal_output, "isatty")
            and terminal_output.isatty()
        ):
            terminal_output.write("\x1b[?1049h\x1b[H\x1b[2J")
            terminal_output.flush()
            alternate_screen = True
        if handoff_active and input_notice:
            self.console.print(
                Panel(
                    input_notice,
                    title="[bold cyan]Interactive Input[/bold cyan]",
                    border_style="cyan",
                    expand=False,
                    padding=(0, 1),
                )
            )
            notice_rendered = True
            handoff_used = True

        if (route_stdin or compiled_input_patterns or allow_escape) and sys.stdin.isatty():
            stdin_fd = sys.stdin.fileno()
            with suppress(Exception):
                termios.tcflush(stdin_fd, termios.TCIFLUSH)
            old_attrs = termios.tcgetattr(stdin_fd)
            tty.setraw(stdin_fd)
            with suppress(Exception):
                termios.tcflush(stdin_fd, termios.TCIFLUSH)
            self._drain_ready_stdin(stdin_fd)

        try:
            while True:
                fds = [self.master_fd]
                if stdin_fd is not None:
                    fds.append(stdin_fd)
                ready, _, _ = select.select(fds, [], [], 0.05)
                if self.master_fd in ready:
                    try:
                        chunk = os.read(self.master_fd, self.read_chunk_size)
                    except BlockingIOError:
                        chunk = b""
                    except OSError as exc:
                        if exc.errno == errno.EIO:
                            chunk = b""
                        else:
                            raise
                    if chunk:
                        text = chunk.decode("utf-8", errors="replace")
                        display_text = display_filter.feed(text) if display_filter else text
                        capture_text = capture_filter.feed(text) if capture_filter else text
                        candidate_text = capture_text or display_text or text
                        if (
                            not handoff_active
                            and compiled_input_patterns
                            and self._matches_any_pattern(candidate_text, compiled_input_patterns)
                        ):
                            handoff_active = True
                        if handoff_active and not notice_rendered and input_notice:
                            self.console.print(
                                Panel(
                                    input_notice,
                                    title="[bold cyan]Interactive Input[/bold cyan]",
                                    border_style="cyan",
                                    expand=False,
                                    padding=(0, 1),
                                )
                            )
                            notice_rendered = True
                            handoff_used = True
                        if handoff_active and not handoff_ready and stdin_fd is not None:
                            self._drain_ready_stdin(stdin_fd)
                            handoff_ready = True
                        if display_output and display_text:
                            self.console.file.write(display_text)
                            self.console.file.flush()
                            self._last_output = (self._last_output + display_text)[-20000:]
                        if capture_output:
                            output_parts.append(capture_text)
                        if terminator and self._terminator_reached(
                            "".join(output_parts),
                            terminator=terminator,
                            activation_token=activation_token,
                        ):
                            break
                    elif self._proc is not None and self._proc.poll() is not None:
                        break
                if stdin_fd is not None and stdin_fd in ready and handoff_active:
                    if not handoff_ready:
                        self._drain_ready_stdin(stdin_fd)
                        continue
                    data = os.read(stdin_fd, 1024)
                    if pending_initial_newline_discard:
                        data = self._strip_leading_newline_bytes(data)
                        pending_initial_newline_discard = False
                        if not data:
                            continue
                    processed, detached, pending_escape = self._consume_detach_input(
                        data,
                        pending_escape=pending_escape,
                        allow_escape=allow_escape,
                        detach_sequences=detach_sequences,
                    )
                    if detached:
                        break
                    if data == b"\x03":
                        interrupted = True
                        self.interrupt()
                        continue
                    if processed:
                        os.write(self.master_fd, processed)
                if terminator is None and stdin_fd is None and not handoff_active:
                    if self._proc is not None and self._proc.poll() is not None:
                        break
                if terminator is None and not ready:
                    time.sleep(0.02)
        finally:
            if stdin_fd is not None and old_attrs is not None:
                with suppress(Exception):
                    termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_attrs)
            if alternate_screen and terminal_output is not None:
                with suppress(Exception):
                    terminal_output.write("\x1b[?1049l")
                    terminal_output.flush()
        if display_filter:
            leftover = display_filter.flush()
            if display_output and leftover:
                self.console.file.write(leftover)
                self.console.file.flush()
                self._last_output = (self._last_output + leftover)[-20000:]
        if capture_filter:
            leftover = capture_filter.flush()
            if capture_output and leftover:
                output_parts.append(leftover)
        return ("".join(output_parts), interrupted, handoff_used or route_stdin, detached)

    def _drain_pending_output(self) -> None:
        assert self.master_fd is not None
        while True:
            ready, _, _ = select.select([self.master_fd], [], [], 0)
            if self.master_fd not in ready:
                break
            try:
                chunk = os.read(self.master_fd, self.read_chunk_size)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break

    @staticmethod
    def _drain_ready_stdin(stdin_fd: int) -> None:
        while True:
            ready, _, _ = select.select([stdin_fd], [], [], 0)
            if stdin_fd not in ready:
                break
            try:
                chunk = os.read(stdin_fd, 1024)
            except BlockingIOError:
                break
            except OSError:
                break
            if not chunk:
                break

    @staticmethod
    def _strip_leading_newline_bytes(data: bytes) -> bytes:
        stripped = data
        while stripped.startswith((b"\r", b"\n")):
            stripped = stripped[1:]
        return stripped

    @staticmethod
    def _compile_patterns(patterns: list[str]) -> list[Pattern[str]]:
        compiled: list[Pattern[str]] = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                continue
        return compiled

    @staticmethod
    def _matches_any_pattern(text: str, patterns: list[Pattern[str]]) -> bool:
        return any(pattern.search(text) for pattern in patterns)

    def _consume_detach_input(
        self,
        data: bytes,
        *,
        pending_escape: bool,
        allow_escape: bool,
        detach_sequences: tuple[bytes, ...] | None,
    ) -> tuple[bytes, bool, bool]:
        processed = data
        detached = False
        sequences = list(detach_sequences or ())
        if allow_escape:
            sequences.append(self.escape_sequence)

        for sequence in sequences:
            if sequence == b"\x1b\x1b":
                continue
            if sequence and sequence in processed:
                processed = processed.replace(sequence, b"")
                detached = True

        if pending_escape:
            if processed.startswith(b"\x1b"):
                processed = processed[1:]
                detached = True
                pending_escape = False
            else:
                processed = b"\x1b" + processed
                pending_escape = False
        if processed.endswith(b"\x1b"):
            processed = processed[:-1]
            pending_escape = True
        if b"\x1b\x1b" in processed:
            processed = processed.replace(b"\x1b\x1b", b"")
            detached = True
            pending_escape = False
        return processed, detached, pending_escape

    @staticmethod
    def _terminator_reached(output: str, *, terminator: str, activation_token: str | None) -> bool:
        if activation_token:
            if activation_token not in output:
                return False
            output = output.split(activation_token, 1)[1]
        return terminator in output

    def _extract_sentinel(self, stdout: str, token: str) -> tuple[str, Path | None]:
        cwd: Path | None = None
        cleaned_lines: list[str] = []
        for line in stdout.splitlines():
            if token not in line:
                cleaned_lines.append(line)
                continue
            stripped = line.strip()
            if not stripped.startswith(token):
                continue
            suffix = stripped.split(token, 1)[1].strip()
            if suffix.startswith("cwd="):
                raw = suffix[len("cwd="):].strip()
                if raw:
                    cwd = Path(raw)
        return ("\n".join(cleaned_lines), cwd)

    def _wrap_block(self, block: str, token: str) -> str:
        raise NotImplementedError

    def _cwd_probe_command(self) -> str:
        raise NotImplementedError

    def _normalize_block_output(self, block: str, output: str) -> str:
        return output.strip()


class ShellRunner(PersistentPTYRunner):
    kind = RunnerKind.SHELL

    def __init__(
        self,
        *,
        shell_command: str,
        cwd: Path,
        console: Console,
        env: dict[str, str],
        escape_sequence: str = "\x1d",
        init_commands: list[str] | None = None,
        sanitize_output: bool = True,
        focus_enter_commands: list[str] | None = None,
        focus_exit_commands: list[str] | None = None,
    ) -> None:
        setup_commands = list(init_commands or [])
        argv = shlex.split(shell_command) if shell_command.strip() else [os.environ.get("SHELL", "/bin/sh"), "-l"]
        super().__init__(
            argv=argv,
            cwd=cwd,
            env=env,
            console=console,
            runner_name="shell",
            escape_sequence=escape_sequence,
            init_commands=setup_commands,
            sanitize_output=sanitize_output,
            focus_enter_commands=focus_enter_commands,
            focus_exit_commands=focus_exit_commands,
        )

    def run_block(
        self,
        block: str,
        *,
        display_output: bool = True,
        route_stdin: bool = False,
        input_notice: str | None = None,
        input_prompt_patterns: list[str] | None = None,
        detach_sequences: tuple[bytes, ...] | None = None,
        discard_initial_newline: bool = False,
    ) -> RunnerExecutionResult:
        if route_stdin:
            return self._run_interactive_shell_block(
                block,
                input_notice=input_notice,
                input_prompt_patterns=input_prompt_patterns,
                detach_sequences=detach_sequences,
                discard_initial_newline=discard_initial_newline,
            )
        return super().run_block(
            block,
            display_output=display_output,
            route_stdin=False,
            input_notice=None,
            input_prompt_patterns=input_prompt_patterns,
            detach_sequences=detach_sequences,
        )

    def _wrap_block(self, block: str, token: str) -> str:
        lines = [block.rstrip(), f'printf "{token} cwd=%s\\n" "$PWD"']
        return "\n".join(lines) + "\n"

    def _cwd_probe_command(self) -> str:
        return ":"

    def _interactive_shell_argv(self, command: str) -> list[str]:
        argv = list(self.argv)
        shell_path = argv[0] if argv else os.environ.get("SHELL", "/bin/sh")
        login_flags = [arg for arg in argv[1:] if arg in {"-l", "--login"}]
        return [shell_path, *login_flags, "-c", command]

    def _run_interactive_shell_block(
        self,
        block: str,
        *,
        input_notice: str | None,
        input_prompt_patterns: list[str] | None,
        detach_sequences: tuple[bytes, ...] | None,
        discard_initial_newline: bool = False,
    ) -> RunnerExecutionResult:
        wrapped = block.rstrip()
        master_fd, slave_fd = pty.openpty()
        attrs = termios.tcgetattr(slave_fd)
        attrs[3] = attrs[3] & ~termios.ECHO
        termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
        proc = subprocess.Popen(
            self._interactive_shell_argv(wrapped),
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=str(self.cwd),
            env=dict(self.env),
            close_fds=True,
            preexec_fn=lambda: _child_preexec(slave_fd),
        )
        os.close(slave_fd)
        os.set_blocking(master_fd, False)
        _register_process_group(proc.pid)
        original_fd = self.master_fd
        original_proc = self._proc
        self.master_fd = master_fd
        self._proc = proc
        try:
            stdout, interrupted, handoff_used, detached = self._interact_until_idle(
                terminator=None,
                activation_token=None,
                capture_output=True,
                allow_escape=True,
                display_output=True,
                sanitize_capture=self.sanitize_output,
                sanitize_display=False,
                route_stdin=True,
                input_notice=input_notice,
                input_prompt_patterns=input_prompt_patterns,
                detach_sequences=detach_sequences,
                discard_initial_newline=discard_initial_newline,
            )
            if detached and proc.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            if proc.poll() is None:
                with suppress(Exception):
                    proc.wait(timeout=1)
            cleaned = self._normalize_block_output(block, stdout)
            return RunnerExecutionResult(
                runner=self.kind,
                command=block,
                stdout=cleaned.strip(),
                success=not interrupted,
                returncode=proc.returncode if proc.poll() is not None else None,
                cwd=self.cwd,
                interrupted=interrupted,
                metadata={
                    "interactive_handoff": handoff_used,
                    "detached": detached,
                },
            )
        finally:
            self.master_fd = original_fd
            self._proc = original_proc
            with suppress(Exception):
                if proc.poll() is None:
                    proc.terminate()
            with suppress(Exception):
                proc.wait(timeout=1)
            _unregister_process_group(proc.pid)
            with suppress(OSError):
                os.close(master_fd)

    def _normalize_block_output(self, block: str, output: str) -> str:
        commands = [line.strip() for line in block.splitlines() if line.strip()]
        cleaned_lines: list[str] = []
        previous_blank = False
        for line in output.splitlines():
            line = line.replace("''$'\\004'", "").replace("$'\\004'", "").rstrip()
            stripped = line.strip()
            if stripped in commands:
                continue
            if any(
                len(stripped) <= len(command) + 2 and stripped.endswith(command)
                for command in commands
            ):
                continue
            if any(
                command
                and stripped == command[:1] + command
                for command in commands
            ):
                continue
            if stripped == "%":
                continue
            if stripped.startswith('printf "__KAI_'):
                continue
            if stripped.startswith("printf '__KAI_START__"):
                continue
            if stripped.startswith("__KAI_"):
                continue
            if 'cwd=%s\\n" "$PWD"' in stripped:
                continue
            if not stripped:
                if previous_blank:
                    continue
                previous_blank = True
                cleaned_lines.append("")
                continue
            previous_blank = False
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()


class PythonRunner(PersistentPTYRunner):
    kind = RunnerKind.PYTHON

    def __init__(
        self,
        *,
        python_executable: str,
        cwd: Path,
        console: Console,
        env: dict[str, str],
        escape_sequence: str = "\x1d",
        init_commands: list[str] | None = None,
        sanitize_output: bool = True,
        focus_enter_commands: list[str] | None = None,
        focus_exit_commands: list[str] | None = None,
    ) -> None:
        super().__init__(
            argv=[python_executable, "-u", "-i", "-q"],
            cwd=cwd,
            env=env,
            console=console,
            runner_name="python",
            escape_sequence=escape_sequence,
            init_commands=init_commands,
            sanitize_output=sanitize_output,
            focus_enter_commands=focus_enter_commands,
            focus_exit_commands=focus_exit_commands,
        )

    def _wrap_block(self, block: str, token: str) -> str:
        escaped = repr(block if block.endswith("\n") else block + "\n")
        payload = (
            "import os, traceback\n"
            "try:\n"
            f"    exec(compile({escaped}, '<k-ai-user>', 'exec'), globals(), globals())\n"
            "except BaseException:\n"
            "    traceback.print_exc()\n"
            f"print('{token} cwd=' + os.getcwd())\n"
        )
        return f"exec({payload!r}, globals(), globals())\n"

    def _cwd_probe_command(self) -> str:
        return "pass"
