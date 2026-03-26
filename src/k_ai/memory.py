# src/k_ai/memory.py
"""
Persistent memory store for k-ai.

Two levels:
  - External (read-only): a file loaded into the system prompt (e.g. KERNEL.md).
  - Internal (read-write): MEMORY.json with add/list/remove operations.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import MemoryEntry
from .exceptions import MemoryStoreError


_MEMORY_VERSION = 1


class MemoryStore:
    """
    Manages the internal MEMORY.json file.

    File format::

        {
            "version": 1,
            "entries": [
                {"id": 1, "text": "...", "created_at": "2026-03-26T14:30:00Z"},
                ...
            ]
        }
    """

    def __init__(self, path: Path):
        self.path = Path(path).expanduser()
        self.entries: List[MemoryEntry] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load entries from disk. Creates the file if missing."""
        if not self.path.exists():
            self.entries = []
            self._next_id = 1
            return

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._validate_raw(raw)
            self.entries = [
                MemoryEntry(
                    id=e["id"],
                    text=e["text"],
                    created_at=e.get("created_at", ""),
                )
                for e in raw.get("entries", [])
            ]
            self._next_id = max((e.id for e in self.entries), default=0) + 1
        except (json.JSONDecodeError, MemoryStoreError) as exc:
            self._backup_and_reset(str(exc))

    def save(self) -> None:
        """Persist current entries to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": _MEMORY_VERSION,
            "entries": [
                {"id": e.id, "text": e.text, "created_at": e.created_at}
                for e in self.entries
            ],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, text: str) -> MemoryEntry:
        """Add a new memory entry and persist."""
        entry = MemoryEntry(
            id=self._next_id,
            text=text.strip(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._next_id += 1
        self.entries.append(entry)
        self.save()
        return entry

    def list_entries(self) -> List[MemoryEntry]:
        """Return all entries (read-only copy)."""
        return list(self.entries)

    def remove(self, entry_id: int) -> bool:
        """Remove an entry by ID. Returns True if found and removed."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.id != entry_id]
        if len(self.entries) < before:
            self.save()
            return True
        return False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> tuple[bool, str]:
        """Check file integrity. Returns (ok, message)."""
        if not self.path.exists():
            return True, "Memory file does not exist yet (will be created on first add)."
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._validate_raw(raw)
            count = len(raw.get("entries", []))
            return True, f"Valid. {count} entries, version {raw.get('version')}."
        except (json.JSONDecodeError, MemoryStoreError) as exc:
            return False, f"Corrupt: {exc}"

    @staticmethod
    def _validate_raw(raw: dict) -> None:
        """Raise MemoryStoreError if the raw dict structure is invalid."""
        if not isinstance(raw, dict):
            raise MemoryStoreError(f"Expected dict, got {type(raw).__name__}")
        if raw.get("version") != _MEMORY_VERSION:
            raise MemoryStoreError(
                f"Unsupported version: {raw.get('version')} (expected {_MEMORY_VERSION})"
            )
        entries = raw.get("entries")
        if not isinstance(entries, list):
            raise MemoryStoreError("'entries' must be a list")
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise MemoryStoreError(f"Entry {i} is not a dict")
            if "id" not in entry or "text" not in entry:
                raise MemoryStoreError(f"Entry {i} missing 'id' or 'text'")

    def _backup_and_reset(self, reason: str) -> None:
        """Backup a corrupt file and reset to empty state."""
        backup = self.path.with_suffix(".bak")
        if self.path.exists():
            shutil.copy2(self.path, backup)
        self.entries = []
        self._next_id = 1
        self.save()
        import warnings
        warnings.warn(
            f"MEMORY.json was corrupt ({reason}). "
            f"Backed up to {backup.name} and reset to empty.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# External memory loader (read-only)
# ---------------------------------------------------------------------------

def load_external_memory(path: Optional[str]) -> str:
    """
    Load the external memory file content (e.g. KERNEL.md).
    Returns empty string if path is None or file doesn't exist.
    """
    if not path:
        return ""
    p = Path(path).expanduser()
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""
