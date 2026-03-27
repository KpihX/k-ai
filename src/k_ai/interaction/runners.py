"""Persistent PTY-backed local runners for user shell and Python blocks."""

from __future__ import annotations

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
import tty
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Optional

from rich.console import Console

from .models import RunnerExecutionResult, RunnerKind


class PTYRunnerError(RuntimeError):
    """Raised when a PTY-backed user runner cannot be created or driven."""


_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ESC_RE = re.compile(r"\x1b[@-_]")
_DCS_RE = re.compile(r"\x1bP.*?(?:\x1b\\)", re.DOTALL)


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
        self._last_output: str = ""
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
            preexec_fn=os.setsid,
        )
        os.close(slave_fd)
        os.set_blocking(master_fd, False)
        self.master_fd = master_fd
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
        self.master_fd = None
        self._proc = None

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
        try:
            self._interact_until_idle(
                terminator=None,
                capture_output=False,
                allow_escape=True,
                display_output=True,
                sanitize_output=False,
            )
        finally:
            self._run_quiet_commands(self.focus_exit_commands)

    def refresh_cwd(self) -> Path:
        result = self.run_block(self._cwd_probe_command(), display_output=False)
        if result.cwd is not None:
            self.cwd = result.cwd
        return self.cwd

    def run_block(self, block: str, *, display_output: bool = True) -> RunnerExecutionResult:
        self.ensure_alive()
        token = f"__KAI_SENTINEL__{uuid.uuid4().hex}__"
        payload = self._wrap_block(block, token)
        stdout, interrupted = self._execute_payload(
            payload,
            terminator=token,
            capture_output=True,
            allow_escape=False,
            display_output=display_output,
            sanitize_output=self.sanitize_output,
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
        )

    def _run_initialization_commands(self) -> None:
        self._run_quiet_commands(self.init_commands, token_prefix="__KAI_INIT__")

    def _run_quiet_commands(self, commands: list[str], *, token_prefix: str = "__KAI_SENTINEL__") -> None:
        for command in commands:
            token = f"{token_prefix}{uuid.uuid4().hex}__"
            payload = self._wrap_block(command, token)
            stdout, _ = self._execute_payload(
                payload,
                terminator=token,
                capture_output=True,
                allow_escape=False,
                display_output=False,
                sanitize_output=self.sanitize_output,
            )
            _, cwd = self._extract_sentinel(stdout, token)
            if cwd is not None:
                self.cwd = cwd

    def _execute_payload(
        self,
        payload: str,
        *,
        terminator: str | None,
        capture_output: bool,
        allow_escape: bool,
        display_output: bool,
        sanitize_output: bool,
    ) -> tuple[str, bool]:
        assert self.master_fd is not None
        os.write(self.master_fd, payload.encode("utf-8"))
        return self._interact_until_idle(
            terminator=terminator,
            capture_output=capture_output,
            allow_escape=allow_escape,
            display_output=display_output,
            sanitize_output=sanitize_output,
        )

    def _interact_until_idle(
        self,
        *,
        terminator: str | None,
        capture_output: bool,
        allow_escape: bool,
        display_output: bool,
        sanitize_output: bool,
    ) -> tuple[str, bool]:
        assert self.master_fd is not None
        output_parts: list[str] = []
        interrupted = False
        stdin_fd: Optional[int] = None
        old_attrs = None
        capture_filter = TerminalOutputFilter() if sanitize_output and capture_output else None
        display_filter = TerminalOutputFilter() if sanitize_output and display_output else None

        if sys.stdin.isatty():
            stdin_fd = sys.stdin.fileno()
            old_attrs = termios.tcgetattr(stdin_fd)
            tty.setraw(stdin_fd)

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
                    if chunk:
                        text = chunk.decode("utf-8", errors="replace")
                        display_text = display_filter.feed(text) if display_filter else text
                        capture_text = capture_filter.feed(text) if capture_filter else text
                        if display_output and display_text:
                            self.console.file.write(display_text)
                            self.console.file.flush()
                            self._last_output = (self._last_output + display_text)[-20000:]
                        if capture_output:
                            output_parts.append(capture_text)
                        if terminator and terminator in "".join(output_parts):
                            break
                    elif self._proc is not None and self._proc.poll() is not None:
                        break
                if stdin_fd is not None and stdin_fd in ready:
                    data = os.read(stdin_fd, 1024)
                    if allow_escape and self.escape_sequence in data:
                        break
                    if data == b"\x03":
                        interrupted = True
                        self.interrupt()
                        continue
                    if data:
                        os.write(self.master_fd, data)
                if terminator is None and stdin_fd is None:
                    if self._proc is not None and self._proc.poll() is not None:
                        break
                if terminator is None and not ready:
                    time.sleep(0.02)
        finally:
            if stdin_fd is not None and old_attrs is not None:
                with suppress(Exception):
                    termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_attrs)
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
        return ("".join(output_parts), interrupted)

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
        argv = shlex.split(shell_command) if shell_command.strip() else [os.environ.get("SHELL", "/bin/sh"), "-l"]
        super().__init__(
            argv=argv,
            cwd=cwd,
            env=env,
            console=console,
            runner_name="shell",
            escape_sequence=escape_sequence,
            init_commands=init_commands,
            sanitize_output=sanitize_output,
            focus_enter_commands=focus_enter_commands,
            focus_exit_commands=focus_exit_commands,
        )

    def _wrap_block(self, block: str, token: str) -> str:
        lines = [block.rstrip(), f'printf "{token} cwd=%s\\n" "$PWD"']
        return "\n".join(lines) + "\n"

    def _cwd_probe_command(self) -> str:
        return ":"

    def _normalize_block_output(self, block: str, output: str) -> str:
        commands = {line.strip() for line in block.splitlines() if line.strip()}
        cleaned_lines: list[str] = []
        previous_blank = False
        for line in output.splitlines():
            stripped = line.strip()
            if stripped in commands:
                continue
            if stripped == "%":
                continue
            if stripped.startswith('printf "__KAI_'):
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
            cleaned_lines.append(line.rstrip())
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
