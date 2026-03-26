# test/test_memory.py
"""
Tests for MemoryStore and load_external_memory.
"""
import json
import pytest
from pathlib import Path

from k_ai.memory import MemoryStore, load_external_memory
from k_ai.exceptions import MemoryStoreError


@pytest.fixture
def mem_path(tmp_path):
    return tmp_path / "MEMORY.json"


@pytest.fixture
def store(mem_path):
    s = MemoryStore(mem_path)
    s.load()
    return s


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

class TestMemoryStoreCRUD:
    def test_add_entry(self, store):
        entry = store.add("test fact")
        assert entry.id == 1
        assert entry.text == "test fact"
        assert entry.created_at != ""

    def test_add_increments_id(self, store):
        e1 = store.add("first")
        e2 = store.add("second")
        assert e2.id == e1.id + 1

    def test_list_entries(self, store):
        store.add("a")
        store.add("b")
        entries = store.list_entries()
        assert len(entries) == 2
        assert entries[0].text == "a"

    def test_remove_existing(self, store):
        store.add("keep")
        store.add("remove")
        assert store.remove(2) is True
        assert len(store.entries) == 1
        assert store.entries[0].text == "keep"

    def test_remove_nonexistent(self, store):
        store.add("only")
        assert store.remove(999) is False
        assert len(store.entries) == 1

    def test_add_strips_whitespace(self, store):
        entry = store.add("  padded  ")
        assert entry.text == "padded"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestMemoryStorePersistence:
    def test_save_creates_file(self, store, mem_path):
        store.add("persist this")
        assert mem_path.exists()

    def test_reload_preserves_entries(self, store, mem_path):
        store.add("fact 1")
        store.add("fact 2")

        store2 = MemoryStore(mem_path)
        store2.load()
        assert len(store2.entries) == 2
        assert store2.entries[0].text == "fact 1"

    def test_id_continuity_after_reload(self, store, mem_path):
        store.add("a")
        store.add("b")

        store2 = MemoryStore(mem_path)
        store2.load()
        e3 = store2.add("c")
        assert e3.id == 3

    def test_atomic_save(self, store, mem_path):
        """Save uses tmp + rename for atomicity."""
        store.add("test")
        # The .tmp file should not persist
        assert not mem_path.with_suffix(".tmp").exists()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestMemoryStoreValidation:
    def test_validate_empty(self, mem_path):
        store = MemoryStore(mem_path)
        ok, msg = store.validate()
        assert ok is True

    def test_validate_good_file(self, store, mem_path):
        store.add("good")
        ok, msg = store.validate()
        assert ok is True
        assert "1 entries" in msg

    def test_validate_corrupt_json(self, mem_path):
        mem_path.write_text("NOT JSON", encoding="utf-8")
        store = MemoryStore(mem_path)
        ok, msg = store.validate()
        assert ok is False
        assert "Corrupt" in msg

    def test_validate_wrong_version(self, mem_path):
        mem_path.write_text(json.dumps({"version": 99, "entries": []}), encoding="utf-8")
        store = MemoryStore(mem_path)
        ok, msg = store.validate()
        assert ok is False

    def test_validate_missing_entries(self, mem_path):
        mem_path.write_text(json.dumps({"version": 1}), encoding="utf-8")
        store = MemoryStore(mem_path)
        ok, msg = store.validate()
        assert ok is False

    def test_corrupt_file_backed_up_on_load(self, mem_path):
        mem_path.write_text("CORRUPT", encoding="utf-8")
        store = MemoryStore(mem_path)
        with pytest.warns(UserWarning, match="corrupt"):
            store.load()
        assert mem_path.with_suffix(".bak").exists()
        assert store.entries == []

    def test_load_nonexistent_file(self, tmp_path):
        store = MemoryStore(tmp_path / "nope.json")
        store.load()  # Should not raise
        assert store.entries == []


# ---------------------------------------------------------------------------
# External memory loader
# ---------------------------------------------------------------------------

class TestLoadExternalMemory:
    def test_load_existing_file(self, tmp_path):
        p = tmp_path / "KERNEL.md"
        p.write_text("# Context\nSome info", encoding="utf-8")
        content = load_external_memory(str(p))
        assert "Context" in content

    def test_load_nonexistent_returns_empty(self):
        assert load_external_memory("/nonexistent/file.md") == ""

    def test_load_none_returns_empty(self):
        assert load_external_memory(None) == ""

    def test_load_empty_string_returns_empty(self):
        assert load_external_memory("") == ""
