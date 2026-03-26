# test/test_session_store.py
"""
Tests for SessionStore: create, list, rename, delete, messages persistence.
"""
import json
import pytest
from pathlib import Path

from k_ai.session_store import SessionStore
from k_ai.models import Message, MessageRole
from k_ai.exceptions import SessionStoreError


@pytest.fixture
def store(tmp_path):
    s = SessionStore(tmp_path / "sessions")
    s.init()
    return s


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

class TestSessionCRUD:
    def test_create_session(self, store):
        meta = store.create_session(provider="ollama", model="phi4")
        assert meta.id != ""
        assert meta.provider == "ollama"
        assert meta.model == "phi4"
        assert meta.created_at != ""

    def test_create_multiple_sessions(self, store):
        m1 = store.create_session()
        m2 = store.create_session()
        assert m1.id != m2.id

    def test_get_session_by_full_id(self, store):
        meta = store.create_session()
        found = store.get_session(meta.id)
        assert found is not None
        assert found.id == meta.id

    def test_get_session_by_prefix(self, store):
        meta = store.create_session()
        found = store.get_session(meta.id[:4])
        assert found is not None
        assert found.id == meta.id

    def test_get_nonexistent_returns_none(self, store):
        assert store.get_session("nonexistent") is None

    def test_list_sessions_newest_first(self, store):
        m1 = store.create_session()
        m2 = store.create_session()
        sessions = store.list_sessions(limit=10)
        assert sessions[0].id == m2.id

    def test_list_sessions_limit(self, store):
        for _ in range(5):
            store.create_session()
        sessions = store.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_rename_session(self, store):
        meta = store.create_session()
        ok = store.rename_session(meta.id, "New Title")
        assert ok is True
        found = store.get_session(meta.id)
        assert found.title == "New Title"

    def test_rename_nonexistent(self, store):
        assert store.rename_session("nope", "title") is False

    def test_update_summary(self, store):
        meta = store.create_session()
        store.update_summary(meta.id, "This was about X.")
        found = store.get_session(meta.id)
        assert found.summary == "This was about X."

    def test_delete_session(self, store):
        meta = store.create_session()
        ok = store.delete_session(meta.id)
        assert ok is True
        assert store.get_session(meta.id) is None

    def test_delete_removes_jsonl_file(self, store):
        meta = store.create_session()
        msg_path = store._messages_path(meta.id)
        assert msg_path.exists()
        store.delete_session(meta.id)
        assert not msg_path.exists()

    def test_delete_nonexistent(self, store):
        assert store.delete_session("nope") is False

    def test_session_count(self, store):
        assert store.session_count() == 0
        store.create_session()
        store.create_session()
        assert store.session_count() == 2


# ---------------------------------------------------------------------------
# Message persistence
# ---------------------------------------------------------------------------

class TestMessagePersistence:
    def test_save_and_load_messages(self, store):
        meta = store.create_session()
        store.save_message(meta.id, Message(role=MessageRole.USER, content="hello"))
        store.save_message(meta.id, Message(role=MessageRole.ASSISTANT, content="hi"))

        messages = store.load_messages(meta.id)
        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "hello"
        assert messages[1].role == MessageRole.ASSISTANT

    def test_save_increments_message_count(self, store):
        meta = store.create_session()
        store.save_message(meta.id, Message(role=MessageRole.USER, content="a"))
        store.save_message(meta.id, Message(role=MessageRole.USER, content="b"))
        found = store.get_session(meta.id)
        assert found.message_count == 2

    def test_save_tool_message(self, store):
        meta = store.create_session()
        store.save_message(
            meta.id,
            Message(
                role=MessageRole.TOOL,
                content="result",
                tool_call_id="call_1",
                name="get_weather",
            ),
        )
        messages = store.load_messages(meta.id)
        assert messages[0].role == MessageRole.TOOL
        assert messages[0].tool_call_id == "call_1"
        assert messages[0].name == "get_weather"

    def test_load_empty_session(self, store):
        meta = store.create_session()
        messages = store.load_messages(meta.id)
        assert messages == []

    def test_load_nonexistent_session(self, store):
        messages = store.load_messages("no_such_id")
        assert messages == []


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

class TestIndexPersistence:
    def test_index_survives_reload(self, tmp_path):
        s1 = SessionStore(tmp_path / "sessions")
        s1.init()
        m = s1.create_session(provider="test", model="m1")
        s1.rename_session(m.id, "My Session")

        s2 = SessionStore(tmp_path / "sessions")
        s2.init()
        found = s2.get_session(m.id)
        assert found is not None
        assert found.title == "My Session"
        assert found.provider == "test"

    def test_corrupt_index_raises(self, tmp_path):
        d = tmp_path / "sessions"
        d.mkdir()
        (d / "index.json").write_text("NOT JSON")
        with pytest.raises(SessionStoreError):
            s = SessionStore(d)
            s.init()

    def test_total_size_bytes(self, store):
        meta = store.create_session()
        store.save_message(meta.id, Message(role=MessageRole.USER, content="hello world"))
        assert store.total_size_bytes() > 0
