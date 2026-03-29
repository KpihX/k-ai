"""
Git management for the local k-ai runtime store (~/.k-ai by default).

This repo is intentionally narrow:
it tracks only the durable conversational state that matters to the user
(config, internal memory, session history/index) and ignores runtime-heavy
artifacts such as the Python sandbox.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .memory import resolve_memory_path

if TYPE_CHECKING:
    from .config import ConfigManager


RUNTIME_GITIGNORE_TEXT = """# Managed by k-ai
# Track only the durable runtime state that matters to the user.
*
!.gitignore
!MEMORY.md
!config.yaml
!sessions/
!sessions/index.json
!sessions/*.jsonl
"""

RUNTIME_GIT_USER_NAME = "k-ai runtime"
RUNTIME_GIT_USER_EMAIL = "runtime@k-ai.local"


def _expanded_path(value: str) -> Path:
    return Path(str(value)).expanduser()


def runtime_state_paths(cm: ConfigManager) -> Dict[str, Path]:
    config_path = _expanded_path(cm.get_nested("config", "persist_path", default="~/.k-ai/config.yaml"))
    memory_path = resolve_memory_path(cm)
    sessions_dir = _expanded_path(cm.get_nested("sessions", "directory", default="~/.k-ai/sessions"))
    return {
        "config": config_path,
        "memory": memory_path,
        "sessions": sessions_dir,
    }


def runtime_store_root(cm: ConfigManager) -> Optional[Path]:
    paths = runtime_state_paths(cm)
    parents = {paths["config"].parent, paths["memory"].parent, paths["sessions"].parent}
    if len(parents) != 1:
        return None
    return next(iter(parents))


def runtime_store_root_issues(cm: ConfigManager) -> List[str]:
    paths = runtime_state_paths(cm)
    parents = {name: path.parent for name, path in paths.items()}
    distinct = {str(path) for path in parents.values()}
    if len(distinct) <= 1:
        return []
    return [
        "config.persist_path, memory.path, and sessions.directory must share the same parent "
        "for runtime git tracking to work safely.",
        *(f"{name}: {path}" for name, path in parents.items()),
    ]


def write_runtime_gitignore(runtime_root: Path, overwrite: bool = True) -> Path:
    path = runtime_root / ".gitignore"
    if path.exists() and not overwrite:
        return path
    path.write_text(RUNTIME_GITIGNORE_TEXT, encoding="utf-8")
    return path


def _run_git(runtime_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(runtime_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def ensure_runtime_repo_identity(runtime_root: Path) -> Dict[str, Any]:
    name_result = _run_git(runtime_root, "config", "--get", "user.name")
    email_result = _run_git(runtime_root, "config", "--get", "user.email")
    changed = False

    if name_result.returncode != 0 or not name_result.stdout.strip():
        set_name = _run_git(runtime_root, "config", "user.name", RUNTIME_GIT_USER_NAME)
        if set_name.returncode != 0:
            return {
                "ok": False,
                "reason": "git_config_failed",
                "stderr": set_name.stderr.strip(),
                "runtime_root": runtime_root,
            }
        changed = True

    if email_result.returncode != 0 or not email_result.stdout.strip():
        set_email = _run_git(runtime_root, "config", "user.email", RUNTIME_GIT_USER_EMAIL)
        if set_email.returncode != 0:
            return {
                "ok": False,
                "reason": "git_config_failed",
                "stderr": set_email.stderr.strip(),
                "runtime_root": runtime_root,
            }
        changed = True

    return {
        "ok": True,
        "runtime_root": runtime_root,
        "changed": changed,
    }


def ensure_runtime_repo(
    cm: ConfigManager,
    *,
    overwrite_gitignore: bool = False,
    create_if_missing: bool = True,
) -> Dict[str, Any]:
    issues = runtime_store_root_issues(cm)
    if issues:
        return {"ok": False, "reason": "misaligned_paths", "issues": issues}

    runtime_root = runtime_store_root(cm)
    if runtime_root is None:
        return {"ok": False, "reason": "missing_root", "issues": issues}

    runtime_root.mkdir(parents=True, exist_ok=True)
    write_runtime_gitignore(runtime_root, overwrite=overwrite_gitignore or not (runtime_root / ".gitignore").exists())

    git_dir = runtime_root / ".git"
    initialized = git_dir.exists()
    if not initialized and create_if_missing:
        init_result = _run_git(runtime_root, "init", "-q")
        if init_result.returncode != 0:
            return {
                "ok": False,
                "reason": "git_init_failed",
                "stderr": init_result.stderr.strip(),
                "runtime_root": runtime_root,
            }
        initialized = True

    if initialized:
        identity_result = ensure_runtime_repo_identity(runtime_root)
        if not identity_result.get("ok"):
            return identity_result

    return {
        "ok": initialized,
        "runtime_root": runtime_root,
        "initialized": initialized,
        "gitignore": runtime_root / ".gitignore",
    }


def commit_runtime_state(
    cm: ConfigManager,
    *,
    summary: str = "",
    session_id: str = "",
    session_type: str = "",
    themes: Optional[List[str]] = None,
    create_if_missing: bool = True,
) -> Dict[str, Any]:
    if not bool(cm.get_nested("runtime_git", "enabled", default=True)):
        return {"ok": False, "reason": "disabled"}

    repo_state = ensure_runtime_repo(cm, create_if_missing=create_if_missing)
    if not repo_state.get("ok"):
        return repo_state

    runtime_root = repo_state["runtime_root"]
    add_result = _run_git(runtime_root, "add", ".")
    if add_result.returncode != 0:
        return {
            "ok": False,
            "reason": "git_add_failed",
            "stderr": add_result.stderr.strip(),
            "runtime_root": runtime_root,
        }

    dirty_result = _run_git(runtime_root, "diff", "--cached", "--quiet", "--exit-code")
    if dirty_result.returncode == 0:
        return {"ok": True, "reason": "clean", "runtime_root": runtime_root}
    if dirty_result.returncode not in {0, 1}:
        return {
            "ok": False,
            "reason": "git_diff_failed",
            "stderr": dirty_result.stderr.strip(),
            "runtime_root": runtime_root,
        }

    subject = _commit_subject(cm, summary=summary, session_id=session_id)
    body_lines = []
    if session_id:
        body_lines.append(f"Session: {session_id}")
    if session_type:
        body_lines.append(f"Type: {session_type}")
    clean_themes = [str(theme).strip() for theme in (themes or []) if str(theme).strip()]
    if clean_themes:
        body_lines.append("Themes: " + ", ".join(clean_themes[:8]))

    commit_args = ["commit", "-q", "-m", subject]
    if body_lines:
        commit_args.extend(["-m", "\n".join(body_lines)])
    commit_result = _run_git(runtime_root, *commit_args)
    if commit_result.returncode != 0:
        return {
            "ok": False,
            "reason": "git_commit_failed",
            "stderr": commit_result.stderr.strip(),
            "runtime_root": runtime_root,
            "subject": subject,
        }

    return {
        "ok": True,
        "reason": "committed",
        "runtime_root": runtime_root,
        "subject": subject,
    }


def _commit_subject(cm: ConfigManager, *, summary: str, session_id: str) -> str:
    prefix = str(cm.get_nested("runtime_git", "commit_prefix", default="chat:") or "chat:").strip()
    clean_summary = " ".join(str(summary or "").split()).strip()
    fallback = f"session {session_id}" if session_id else "runtime update"
    base = clean_summary or fallback
    max_len = int(cm.get_nested("runtime_git", "commit_subject_max_length", default=72))
    available = max(16, max_len - len(prefix) - 1)
    base = base[:available].rstrip(" .:-")
    if not base:
        base = fallback
    return f"{prefix} {base}".strip()
