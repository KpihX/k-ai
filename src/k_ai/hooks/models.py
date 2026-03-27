"""Core models for the hooks runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class HookCommand:
    command: str
    timeout_seconds: int


@dataclass(frozen=True)
class HookMatcher:
    event: str
    matcher: str
    commands: Tuple[HookCommand, ...]
    source_file: Path
    scope: str
    precedence: int


@dataclass(frozen=True)
class HookIssue:
    path: Path
    message: str


@dataclass(frozen=True)
class HookCatalog:
    matchers: Tuple[HookMatcher, ...]
    issues: Tuple[HookIssue, ...] = ()


@dataclass(frozen=True)
class HookExecution:
    matcher: HookMatcher
    command: HookCommand
    exit_code: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class HookDispatchResult:
    blocked: bool = False
    message: str = ""
    context_lines: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    executions: Tuple[HookExecution, ...] = ()


@dataclass(frozen=True)
class HookRoot:
    path: Path
    scope: str
    precedence: int


HookPayload = Mapping[str, Any]
