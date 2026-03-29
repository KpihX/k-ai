# src/k_ai/memory.py
"""
Markdown-backed runtime context files for k-ai.

Two durable files matter:
  - AGENTS.md: agent instructions/context file loaded into the system prompt.
  - MEMORY.md: mutable k-ai-specific memory and operating notes.

The old JSON memory store is kept only as a compatibility layer at the API
level; the on-disk source of truth is now Markdown.
"""
from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import yaml

from .models import MemoryEntry

if TYPE_CHECKING:
    from .config import ConfigManager


_ENTRY_RE = re.compile(r"^\s*-\s*\[(\d+)\]\s+(.*?)\s*$", re.MULTILINE)
_PREFERRED_USER_RE = re.compile(
    r"(^|\n)-\s+(?:\[\d+\]\s+)?Preferred user name:\s*(.+?)\.?\s*(?=\n|$)",
    re.IGNORECASE,
)

_DEFAULT_MEMORY_TEMPLATE = """---
name: k-ai-memory
description: k-ai runtime memory, preferences, and operating notes.
license: private
metadata:
  author: KpihX
  version: "1.1.0"
  scope: runtime-memory
  target-agent: k-ai
allowed-tools: Read, Edit, Write
---

# k-ai Memory

> THE AGENT MUST PROACTIVELY UPDATE THIS MEMORY AS SOON AS IT FINDS SOMETHING KEY TO REMEMBER, WITHOUT ASKING FOR APPROVAL.

## Profile

## Notes
"""


def resolve_memory_path(cm: "ConfigManager", default: str = "~/.k-ai/MEMORY.md") -> Path:
    raw = (
        cm.get_nested("memory", "path", default=None)
        or cm.get_nested("memory", "internal_file", default=default)
        or default
    )
    return Path(str(raw)).expanduser()


def resolve_agents_path(cm: "ConfigManager", default: str = "~/.k-ai/AGENTS.md") -> Optional[Path]:
    raw = (
        cm.get_nested("memory", "agents_path", default=None)
        or cm.get_nested("memory", "external_file", default=default)
        or default
    )
    text = str(raw).strip()
    if not text:
        return None
    return Path(text).expanduser()


def load_context_file(path: Optional[str | Path]) -> str:
    """Load one markdown context file. Missing/unreadable files return an empty string."""
    if not path:
        return ""
    target = Path(path).expanduser()
    if not target.exists():
        return ""
    try:
        return target.read_text(encoding="utf-8")
    except Exception:
        return ""


load_external_memory = load_context_file


class MemoryStore:
    """Markdown-backed mutable runtime memory."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser()
        self.content: str = ""
        self.entries: List[MemoryEntry] = []
        self._next_id: int = 1

    def load(self) -> None:
        """Load or initialize MEMORY.md on disk."""
        if not self.path.exists():
            self.content = self.default_template()
            self._reindex()
            self.save()
            return
        try:
            self.content = self.path.read_text(encoding="utf-8")
            if not self.content.strip():
                self.content = self.default_template()
                self.save()
            self._validate_markdown(self.content)
            self._reindex()
        except Exception as exc:
            self._backup_and_reset(str(exc))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(self.content.rstrip() + "\n", encoding="utf-8")
        tmp.replace(self.path)
        self._reindex()

    def validate(self) -> tuple[bool, str]:
        if not self.path.exists():
            return True, "Memory file does not exist yet (template will be created on load)."
        try:
            text = self.path.read_text(encoding="utf-8")
            self._validate_markdown(text)
            entry_count = len(list(_ENTRY_RE.finditer(text)))
            return True, f"Valid Markdown memory. {entry_count} note entries."
        except Exception as exc:
            return False, f"Corrupt: {exc}"

    def list_entries(self) -> List[MemoryEntry]:
        return list(self.entries)

    def add(self, text: str) -> MemoryEntry:
        note = str(text or "").strip()
        if not note:
            raise ValueError("Memory entry text cannot be empty.")
        entry = MemoryEntry(id=self._next_id, text=note, created_at="")
        self._next_id += 1
        block = f"- [{entry.id}] {entry.text}"
        marker = "## Notes"
        if marker in self.content:
            head, tail = self.content.split(marker, 1)
            tail = tail.strip("\n")
            joined = f"{head}{marker}\n"
            if tail:
                joined += f"{tail}\n"
            joined += f"{block}\n"
            self.content = joined
        else:
            self.content = self.content.rstrip() + f"\n\n## Notes\n{block}\n"
        self.save()
        return entry

    def remove(self, entry_id: int) -> bool:
        removed = False
        lines: List[str] = []
        for line in self.content.splitlines():
            match = re.match(r"^\s*-\s*\[(\d+)\]\s+(.*?)\s*$", line)
            if match and int(match.group(1)) == int(entry_id):
                removed = True
                continue
            lines.append(line)
        if removed:
            self.content = "\n".join(lines).rstrip() + "\n"
            self.save()
        return removed

    def get_preferred_user_name(self) -> str:
        match = _PREFERRED_USER_RE.search(self.content)
        if not match:
            return ""
        return match.group(2).strip()

    def set_preferred_user_name(self, user_name: str) -> None:
        clean = str(user_name or "").strip().strip(".")
        if not clean:
            return
        replacement = f"- Preferred user name: {clean}."
        match = _PREFERRED_USER_RE.search(self.content)
        if match:
            self.content = self.content[:match.start()] + f"\n{replacement}" + self.content[match.end():]
        elif "## Profile" in self.content:
            self.content = self.content.replace("## Profile", f"## Profile\n\n{replacement}", 1)
        else:
            self.content = self.content.rstrip() + f"\n\n## Profile\n\n{replacement}\n"
        self.save()

    def has_meaningful_content(self) -> bool:
        return self.content.strip() != self.default_template().strip()

    @staticmethod
    def default_template() -> str:
        return _DEFAULT_MEMORY_TEMPLATE

    def _reindex(self) -> None:
        self.entries = [
            MemoryEntry(id=int(match.group(1)), text=match.group(2).strip(), created_at="")
            for match in _ENTRY_RE.finditer(self.content)
        ]
        self._next_id = max((entry.id for entry in self.entries), default=0) + 1

    @staticmethod
    def _validate_markdown(text: str) -> None:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError("missing YAML frontmatter opening delimiter")
        closing_index = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                closing_index = idx
                break
        if closing_index is None:
            raise ValueError("missing YAML frontmatter closing delimiter")
        parsed = yaml.safe_load("\n".join(lines[1:closing_index])) or {}
        if not isinstance(parsed, dict):
            raise ValueError("frontmatter must be a mapping")

    def _backup_and_reset(self, reason: str) -> None:
        backup = self.path.with_suffix(".bak")
        if self.path.exists():
            shutil.copy2(self.path, backup)
        self.content = self.default_template()
        self._reindex()
        self.save()
        warnings.warn(
            f"MEMORY.md was corrupt ({reason}). Backed up to {backup.name} and reset to the default template.",
            stacklevel=2,
        )
