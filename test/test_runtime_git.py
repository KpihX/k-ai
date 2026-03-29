# test/test_runtime_git.py
import subprocess

from k_ai import ConfigManager
from k_ai.runtime_git import (
    RUNTIME_GITIGNORE_TEXT,
    commit_runtime_state,
    ensure_runtime_repo,
    ensure_runtime_repo_identity,
    runtime_store_root,
)


def _configure_runtime(cm: ConfigManager, runtime_root):
    cm.set("config.persist_path", str(runtime_root / "config.yaml"))
    cm.set("memory.path", str(runtime_root / "MEMORY.md"))
    cm.set("sessions.directory", str(runtime_root / "sessions"))


def test_ensure_runtime_repo_creates_managed_gitignore(tmp_path):
    runtime_root = tmp_path / ".k-ai"
    cm = ConfigManager()
    _configure_runtime(cm, runtime_root)

    result = ensure_runtime_repo(cm, overwrite_gitignore=True)

    assert result["ok"] is True
    assert runtime_store_root(cm) == runtime_root
    assert (runtime_root / ".gitignore").read_text(encoding="utf-8") == RUNTIME_GITIGNORE_TEXT


def test_commit_runtime_state_tracks_only_runtime_files(tmp_path):
    runtime_root = tmp_path / ".k-ai"
    cm = ConfigManager()
    _configure_runtime(cm, runtime_root)

    (runtime_root / "sessions").mkdir(parents=True, exist_ok=True)
    (runtime_root / "config.yaml").write_text("model: mistral\n", encoding="utf-8")
    (runtime_root / "MEMORY.md").write_text("# memory\n", encoding="utf-8")
    (runtime_root / "sessions" / "index.json").write_text("[]\n", encoding="utf-8")
    (runtime_root / "sessions" / "abc123.jsonl").write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")
    (runtime_root / "sandbox").mkdir(parents=True, exist_ok=True)
    (runtime_root / "sandbox" / "noise.txt").write_text("ignored\n", encoding="utf-8")

    ensure_runtime_repo(cm, overwrite_gitignore=True)
    subprocess.run(["git", "-C", str(runtime_root), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(runtime_root), "config", "user.name", "Test User"], check=True)

    result = commit_runtime_state(
        cm,
        summary="Conversation sur le runtime git",
        session_id="abc123",
        session_type="classic",
        themes=["git", "runtime"],
    )

    assert result["ok"] is True
    assert result["reason"] == "committed"
    tracked = subprocess.run(
        ["git", "-C", str(runtime_root), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert ".gitignore" in tracked
    assert "config.yaml" in tracked
    assert "MEMORY.md" in tracked
    assert "sessions/index.json" in tracked
    assert "sessions/abc123.jsonl" in tracked
    assert "sandbox/noise.txt" not in tracked


def test_ensure_runtime_repo_identity_sets_local_defaults(tmp_path):
    runtime_root = tmp_path / ".k-ai"
    cm = ConfigManager()
    _configure_runtime(cm, runtime_root)
    ensure_runtime_repo(cm, overwrite_gitignore=True)

    result = ensure_runtime_repo_identity(runtime_root)

    assert result["ok"] is True
    name = subprocess.run(
        ["git", "-C", str(runtime_root), "config", "--get", "user.name"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    email = subprocess.run(
        ["git", "-C", str(runtime_root), "config", "--get", "user.email"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert name == "k-ai runtime"
    assert email == "runtime@k-ai.local"


def test_commit_runtime_state_succeeds_without_preconfigured_git_identity(tmp_path):
    runtime_root = tmp_path / ".k-ai"
    cm = ConfigManager()
    _configure_runtime(cm, runtime_root)

    (runtime_root / "sessions").mkdir(parents=True, exist_ok=True)
    (runtime_root / "config.yaml").write_text("model: mistral\n", encoding="utf-8")
    (runtime_root / "MEMORY.md").write_text("# memory\n", encoding="utf-8")
    (runtime_root / "sessions" / "index.json").write_text("[]\n", encoding="utf-8")
    (runtime_root / "sessions" / "abc123.jsonl").write_text('{"role":"user","content":"hi"}\n', encoding="utf-8")

    result = commit_runtime_state(cm, summary="Premier commit runtime", session_id="abc123")

    assert result["ok"] is True
    assert result["reason"] == "committed"
