# test/test_memory.py
"""
Tests for MemoryStore and load_context_file.
"""
import pytest

from k_ai.memory import MemoryStore, load_context_file


@pytest.fixture
def mem_path(tmp_path):
    return tmp_path / "MEMORY.md"


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
        assert "- [1] test fact" in store.content

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
        assert "1 note entries" in msg

    def test_validate_corrupt_markdown(self, mem_path):
        mem_path.write_text("NOT MARKDOWN", encoding="utf-8")
        store = MemoryStore(mem_path)
        ok, msg = store.validate()
        assert ok is False
        assert "Corrupt" in msg

    def test_validate_missing_frontmatter_closing(self, mem_path):
        mem_path.write_text("---\nname: bad\n", encoding="utf-8")
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
        assert store.content.startswith("---")

    def test_load_nonexistent_file(self, tmp_path):
        target = tmp_path / "nope.md"
        store = MemoryStore(target)
        store.load()
        assert target.exists()
        assert store.entries == []


# ---------------------------------------------------------------------------
# External memory loader
# ---------------------------------------------------------------------------

class TestLoadContextFile:
    def test_load_existing_file(self, tmp_path):
        p = tmp_path / "AGENTS.md"
        p.write_text("# Context\nSome info", encoding="utf-8")
        content = load_context_file(str(p))
        assert "Context" in content

    def test_load_nonexistent_returns_empty(self):
        assert load_context_file("/nonexistent/file.md") == ""

    def test_load_none_returns_empty(self):
        assert load_context_file(None) == ""

    def test_load_empty_string_returns_empty(self):
        assert load_context_file("") == ""
