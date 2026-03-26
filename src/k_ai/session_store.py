# src/k_ai/session_store.py
"""
Persistent session storage for k-ai.

Layout::

    <directory>/
    ├── index.json       # List of SessionMetadata dicts
    ├── <uuid>.jsonl     # Messages for session, one JSON per line
    └── ...
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import Message, SessionMetadata
from .exceptions import SessionStoreError


class SessionStore:
    """
    Manages session persistence on disk.

    Each session has:
      - An entry in index.json (metadata: id, title, summary, dates, etc.)
      - A .jsonl file with all messages (one JSON object per line)
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser()
        self._index: List[SessionMetadata] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Ensure directory exists and load the index."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._load_index()

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(
        self,
        provider: str = "",
        model: str = "",
    ) -> SessionMetadata:
        """Create a new session and return its metadata."""
        now = datetime.now(timezone.utc).isoformat()
        session_id = uuid.uuid4().hex[:12]
        meta = SessionMetadata(
            id=session_id,
            title=session_id,
            created_at=now,
            updated_at=now,
            provider=provider,
            model=model,
            themes=[],
        )
        self._index.append(meta)
        self._save_index()
        # Create empty message file
        self._messages_path(session_id).touch()
        return meta

    def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """Get session metadata by ID (or ID prefix)."""
        for meta in self._index:
            if meta.id == session_id or meta.id.startswith(session_id):
                return meta
        return None

    def list_sessions(self, limit: int = 10, order: str = "recent") -> List[SessionMetadata]:
        """Return sessions ordered by recency or age."""
        reverse = order != "oldest"
        sorted_sessions = sorted(
            self._index,
            key=lambda s: s.updated_at,
            reverse=reverse,
        )
        return sorted_sessions[:limit]

    def rename_session(self, session_id: str, title: str) -> bool:
        """Update a session's title. Returns True if found."""
        meta = self.get_session(session_id)
        if not meta:
            return False
        meta.title = title
        meta.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_index()
        return True

    def update_summary(self, session_id: str, summary: str, themes: Optional[List[str]] = None) -> bool:
        """Update a session's summary/themes. Returns True if found."""
        meta = self.get_session(session_id)
        if not meta:
            return False
        meta.summary = summary
        if themes is not None:
            meta.themes = list(themes)
        meta.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_index()
        return True

    def update_digest(self, session_id: str, summary: str, themes: Optional[List[str]] = None) -> bool:
        """Keep session title/summary/themes synchronized from a digest sentence."""
        meta = self.get_session(session_id)
        if not meta:
            return False
        digest = summary.strip()
        meta.title = digest or meta.title
        meta.summary = digest
        if themes is not None:
            meta.themes = list(themes)
        meta.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_index()
        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session (metadata + message file). Returns True if found."""
        meta = self.get_session(session_id)
        if not meta:
            return False
        self._index = [m for m in self._index if m.id != meta.id]
        self._save_index()
        msg_path = self._messages_path(meta.id)
        if msg_path.exists():
            msg_path.unlink()
        return True

    def update_meta(self, session_id: str, **kwargs) -> bool:
        """Update arbitrary fields on session metadata."""
        meta = self.get_session(session_id)
        if not meta:
            return False
        for key, value in kwargs.items():
            if hasattr(meta, key):
                setattr(meta, key, value)
        meta.updated_at = datetime.now(timezone.utc).isoformat()
        self._save_index()
        return True

    # ------------------------------------------------------------------
    # Message persistence
    # ------------------------------------------------------------------

    def save_message(self, session_id: str, message: Message) -> None:
        """Append a single message to the session's JSONL file."""
        path = self._messages_path(session_id)
        entry = {
            "role": message.role.value,
            "content": message.content,
        }
        if message.name:
            entry["name"] = message.name
        if message.tool_call_id:
            entry["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            entry["tool_calls"] = [tc.model_dump() for tc in message.tool_calls]

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Update message count
        meta = self.get_session(session_id)
        if meta:
            meta.message_count += 1
            meta.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_index()

    def load_messages(
        self,
        session_id: str,
        offset: int = 0,
        limit: Optional[int] = None,
        last_n: Optional[int] = None,
    ) -> List[Message]:
        """Load messages for a session, optionally sliced by offset/limit/last_n."""
        path = self._messages_path(session_id)
        if not path.exists():
            return []

        messages: List[Message] = []
        for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                messages.append(Message(**data))
            except Exception as exc:
                raise SessionStoreError(
                    f"Corrupt message at {path.name}:{line_num}: {exc}"
                ) from exc
        if last_n is not None and last_n > 0:
            return messages[-last_n:]
        if offset < 0:
            offset = 0
        if limit is None:
            return messages[offset:]
        return messages[offset:offset + max(limit, 0)]

    def rewrite_messages(self, session_id: str, messages: List[Message]) -> None:
        """Rewrite the full JSONL message log for a session."""
        path = self._messages_path(session_id)
        lines = []
        for message in messages:
            entry = {
                "role": message.role.value,
                "content": message.content,
            }
            if message.name:
                entry["name"] = message.name
            if message.tool_call_id:
                entry["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                entry["tool_calls"] = [tc.model_dump() for tc in message.tool_calls]
            lines.append(json.dumps(entry, ensure_ascii=False))

        payload = ("\n".join(lines) + "\n") if lines else ""
        path.write_text(payload, encoding="utf-8")

        meta = self.get_session(session_id)
        if meta:
            meta.message_count = len(messages)
            meta.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_index()

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.directory / "index.json"

    def _messages_path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.jsonl"

    def _load_index(self) -> None:
        """Load the session index from disk."""
        path = self._index_path()
        if not path.exists():
            self._index = []
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise SessionStoreError("index.json must be a JSON array")
            self._index = [SessionMetadata(**entry) for entry in raw]
        except (json.JSONDecodeError, SessionStoreError) as exc:
            raise SessionStoreError(f"Corrupt session index: {exc}") from exc

    def _save_index(self) -> None:
        """Persist the session index to disk atomically."""
        path = self._index_path()
        data = [meta.model_dump() for meta in self._index]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def session_count(self) -> int:
        return len(self._index)

    def total_size_bytes(self) -> int:
        """Total disk usage of all session files."""
        total = 0
        if self._index_path().exists():
            total += self._index_path().stat().st_size
        for meta in self._index:
            p = self._messages_path(meta.id)
            if p.exists():
                total += p.stat().st_size
        return total
