from pathlib import Path
from unittest.mock import MagicMock

from k_ai.config import ConfigManager
from k_ai.doctor import backup_runtime_state, run_doctor
from k_ai.memory import MemoryStore
from k_ai.models import Message, MessageRole
from k_ai.session_store import SessionStore


def _build_runtime(tmp_path: Path):
    cm = ConfigManager()
    runtime_dir = tmp_path / "runtime"
    sessions_dir = runtime_dir / "sessions"
    memory_path = runtime_dir / "MEMORY.json"
    config_path = runtime_dir / "config.yaml"

    cm.set("sessions.directory", str(sessions_dir))
    cm.set("memory.internal_file", str(memory_path))
    cm.set("config.persist_path", str(config_path))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(cm.dump_yaml(), encoding="utf-8")

    memory = MemoryStore(memory_path)
    memory.load()
    memory.add("keep me")

    store = SessionStore(sessions_dir)
    store.init()
    meta = store.create_session()
    store.save_message(meta.id, Message(role=MessageRole.USER, content="hello"))
    return cm, memory, store


def test_backup_runtime_state_copies_runtime_files(tmp_path):
    cm, memory, store = _build_runtime(tmp_path)
    backup_dir = backup_runtime_state(cm, memory, store)
    assert (backup_dir / "config.yaml").exists()
    assert (backup_dir / "MEMORY.json").exists()
    assert (backup_dir / "sessions").exists()


def test_run_doctor_can_reset_config_memory_and_sessions(tmp_path):
    cm, memory, store = _build_runtime(tmp_path)
    console = MagicMock()

    cm.set("tools.python.enabled", False)
    cm.save_active_yaml(str(Path(cm.get_nested("config", "persist_path"))))

    run_doctor(cm, memory, store, console, reset=["all"])

    assert cm.get_nested("tools", "python", "enabled") is True
    assert memory.entries == []
    assert store.session_count() == 0
