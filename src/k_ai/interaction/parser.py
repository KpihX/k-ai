"""Parse multiline chat submissions into text, shell, python, or ephemeral blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .models import DocumentBlock, DocumentKind


class MixedInputParserError(ValueError):
    """Raised when a submitted document mixes incompatible directives."""


@dataclass(frozen=True)
class MixedInputSyntax:
    shell_prefix: str
    python_prefix: str
    ephemeral_prefix: str


class MixedInputParser:
    """Deterministic line-oriented parser for mixed chat submissions."""

    def __init__(self, *, shell_prefix: str = "!", python_prefix: str = ">", ephemeral_prefix: str = "/?"):
        self.syntax = MixedInputSyntax(
            shell_prefix=str(shell_prefix or "!"),
            python_prefix=str(python_prefix or ">"),
            ephemeral_prefix=str(ephemeral_prefix or "/?"),
        )

    def parse(self, text: str) -> list[DocumentBlock]:
        lines = text.splitlines()
        if not lines:
            return []
        first_nonempty = self._first_nonempty_line(lines)
        if first_nonempty is not None and first_nonempty[1].lstrip().startswith(self.syntax.ephemeral_prefix):
            return [self._parse_ephemeral(lines, first_nonempty[0])]
        return self._parse_blocks(lines)

    def _first_nonempty_line(self, lines: Iterable[str]) -> tuple[int, str] | None:
        for idx, line in enumerate(lines, start=1):
            if line.strip():
                return idx, line
        return None

    def _parse_ephemeral(self, lines: list[str], start_line: int) -> DocumentBlock:
        first = lines[start_line - 1]
        prefix = self.syntax.ephemeral_prefix
        stripped = first.lstrip()
        indent_len = len(first) - len(stripped)
        body = stripped[len(prefix):].lstrip()
        remainder = [body] if body else []
        remainder.extend(lines[start_line:])
        content = "\n".join(remainder).strip()
        if not content:
            raise MixedInputParserError("Ephemeral questions cannot be empty.")
        return DocumentBlock(
            kind=DocumentKind.EPHEMERAL,
            content=content,
            start_line=start_line,
            end_line=len(lines),
        )

    def _parse_blocks(self, lines: list[str]) -> List[DocumentBlock]:
        blocks: list[DocumentBlock] = []
        current_kind: DocumentKind | None = None
        current_lines: list[str] = []
        start_line = 1

        def flush(end_line: int) -> None:
            nonlocal current_kind, current_lines, start_line
            if current_kind is None:
                return
            content = "\n".join(current_lines).strip()
            if content:
                blocks.append(
                    DocumentBlock(
                        kind=current_kind,
                        content=content,
                        start_line=start_line,
                        end_line=end_line,
                    )
                )
            current_kind = None
            current_lines = []

        for idx, raw_line in enumerate(lines, start=1):
            kind, content = self._classify_line(raw_line)
            if current_kind is None:
                current_kind = kind
                current_lines = [content]
                start_line = idx
                continue
            if kind == current_kind:
                current_lines.append(content)
                continue
            flush(idx - 1)
            current_kind = kind
            current_lines = [content]
            start_line = idx
        flush(len(lines))
        return blocks

    def _classify_line(self, raw_line: str) -> tuple[DocumentKind, str]:
        stripped = raw_line.lstrip()
        if stripped.startswith(self.syntax.shell_prefix):
            return (DocumentKind.SHELL, stripped[len(self.syntax.shell_prefix):].lstrip())
        if stripped.startswith(self.syntax.python_prefix):
            return (DocumentKind.PYTHON, stripped[len(self.syntax.python_prefix):].lstrip())
        if stripped.startswith(self.syntax.ephemeral_prefix):
            raise MixedInputParserError("The ephemeral prefix must appear only at the beginning of the document.")
        return (DocumentKind.LLM, raw_line)
