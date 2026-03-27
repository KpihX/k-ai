"""Persistent PTY-backed local runners for user shell and Python blocks."""

from __future__ import annotations

import os
import pty
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
    ) -> None:
        self.argv = list(argv)
        self.cwd = Path(cwd)
        self.env = dict(env)
        self.console = console
        self.runner_name = runner_name
        self.escape_sequence = escape_sequence.encode("utf-8")
        self.read_chunk_size = int(read_chunk_size)
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
        self._interact_until_idle(terminator=None, capture_output=False, allow_escape=True)

    def refresh_cwd(self) -> Path:
        result = self.run_block(self._cwd_probe_command())
        if result.cwd is not None:
            self.cwd = result.cwd
        return self.cwd

    def run_block(self, block: str) -> RunnerExecutionResult:
        self.ensure_alive()
        token = f"__KAI_SENTINEL__{uuid.uuid4().hex}__"
        payload = self._wrap_block(block, token)
        os.write(self.master_fd, payload.encode("utf-8"))
        stdout, interrupted = self._interact_until_idle(terminator=token, capture_output=True, allow_escape=False)
        normalized_output, cwd = self._extract_sentinel(stdout, token)
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

    def _interact_until_idle(
        self,
        *,
        terminator: str | None,
        capture_output: bool,
        allow_escape: bool,
    ) -> tuple[str, bool]:
        assert self.master_fd is not None
        output_parts: list[str] = []
        interrupted = False
        stdin_fd: Optional[int] = None
        old_attrs = None

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
                        self.console.file.write(text)
                        self.console.file.flush()
                        self._last_output = (self._last_output + text)[-20000:]
                        if capture_output:
                            output_parts.append(text)
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
        return ("".join(output_parts), interrupted)

    def _extract_sentinel(self, stdout: str, token: str) -> tuple[str, Path | None]:
        cwd: Path | None = None
        cleaned_lines: list[str] = []
        for line in stdout.splitlines():
            if token not in line:
                cleaned_lines.append(line)
                continue
            suffix = line.split(token, 1)[1].strip()
            if suffix.startswith("cwd="):
                raw = suffix[len("cwd="):].strip()
                if raw:
                    cwd = Path(raw)
        return ("\n".join(cleaned_lines), cwd)

    def _wrap_block(self, block: str, token: str) -> str:
        raise NotImplementedError

    def _cwd_probe_command(self) -> str:
        raise NotImplementedError


class ShellRunner(PersistentPTYRunner):
    kind = RunnerKind.SHELL

    def __init__(self, *, shell_command: str, cwd: Path, console: Console, env: dict[str, str], escape_sequence: str = "\x1d") -> None:
        argv = shlex.split(shell_command) if shell_command.strip() else [os.environ.get("SHELL", "/bin/sh"), "-l"]
        super().__init__(
            argv=argv,
            cwd=cwd,
            env=env,
            console=console,
            runner_name="shell",
            escape_sequence=escape_sequence,
        )

    def _wrap_block(self, block: str, token: str) -> str:
        lines = [block.rstrip(), f'printf "{token} cwd=%s\\n" "$PWD"']
        return "\n".join(lines) + "\n"

    def _cwd_probe_command(self) -> str:
        return ":"


class PythonRunner(PersistentPTYRunner):
    kind = RunnerKind.PYTHON

    def __init__(self, *, python_executable: str, cwd: Path, console: Console, env: dict[str, str], escape_sequence: str = "\x1d") -> None:
        super().__init__(
            argv=[python_executable, "-u", "-i", "-q"],
            cwd=cwd,
            env=env,
            console=console,
            runner_name="python",
            escape_sequence=escape_sequence,
        )
        self.run_block("import sys; sys.ps1 = ''; sys.ps2 = ''")

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
