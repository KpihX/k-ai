"""Working-directory helpers for chat and ask runtimes."""

from __future__ import annotations

import os
from pathlib import Path


def normalize_workdir(raw_path: str | os.PathLike[str] | None, *, base: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a user-provided working directory with env-var and tilde expansion."""
    if raw_path is None or str(raw_path).strip() == "":
        candidate = Path(base or Path.cwd())
    else:
        expanded = os.path.expandvars(os.path.expanduser(str(raw_path)))
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = Path(base or Path.cwd()) / candidate
    resolved = candidate.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Working directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Working directory is not a directory: {resolved}")
    return resolved
