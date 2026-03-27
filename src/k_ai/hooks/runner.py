"""Command execution for hooks."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Mapping

from .models import HookCommand, HookExecution, HookMatcher


async def run_hook_command(
    *,
    matcher: HookMatcher,
    command: HookCommand,
    payload: Mapping[str, object],
    env: Mapping[str, str],
    cwd: Path,
    max_stdout_chars: int,
    max_stderr_chars: int,
) -> HookExecution:
    proc = await asyncio.create_subprocess_shell(
        command.command,
        cwd=str(cwd),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, **env},
    )
    stdin_data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(stdin_data), timeout=command.timeout_seconds)
        code = int(proc.returncode or 0)
    except asyncio.TimeoutError:
        proc.kill()
        stdout, stderr = await proc.communicate()
        code = 124
        stderr = (stderr or b"") + b"\nTimed out."
    return HookExecution(
        matcher=matcher,
        command=command,
        exit_code=code,
        stdout=(stdout.decode("utf-8", errors="replace")[:max_stdout_chars]).strip(),
        stderr=(stderr.decode("utf-8", errors="replace")[:max_stderr_chars]).strip(),
    )
