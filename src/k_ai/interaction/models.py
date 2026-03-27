"""Core models for mixed interactive user input and persistent local runners."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class DocumentKind(str, Enum):
    LLM = "llm"
    SHELL = "shell"
    PYTHON = "python"
    EPHEMERAL = "ephemeral"


class RunnerKind(str, Enum):
    SHELL = "shell"
    PYTHON = "python"


@dataclass(frozen=True)
class DocumentBlock:
    kind: DocumentKind
    content: str
    start_line: int
    end_line: int


@dataclass
class RunnerExecutionResult:
    runner: RunnerKind
    command: str
    stdout: str
    success: bool
    returncode: Optional[int] = None
    cwd: Optional[Path] = None
    interrupted: bool = False
    metadata: dict[str, str] = field(default_factory=dict)

