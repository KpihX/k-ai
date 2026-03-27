"""Parsing and validation for SKILL.md documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from .models import SkillDocument, SkillSummary


_NAME_RE = re.compile(r"^(?!.*--)[a-z0-9](?:[a-z0-9-]{0,126}[a-z0-9])?$")


class SkillParseError(ValueError):
    """Raised when a skill file is invalid."""


def skill_file_signature(path: Path) -> Tuple[int, int]:
    """Return a stable cache key derived from file metadata."""
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def parse_skill_file(
    *,
    path: Path,
    root: Path,
    scope: str,
    precedence: int,
) -> SkillDocument:
    """Parse one SKILL.md file into a fully loaded document."""
    raw = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw, path)
    meta = _normalize_frontmatter(frontmatter, path)
    summary = SkillSummary(
        name=meta["name"],
        description=meta["description"],
        root=root,
        skill_dir=path.parent,
        skill_file=path,
        scope=scope,
        precedence=precedence,
        license=meta.get("license", ""),
        compatibility=meta.get("compatibility", ""),
        metadata=meta.get("metadata", {}),
        allowed_tools=tuple(meta.get("allowed_tools", ())),
    )
    return SkillDocument(summary=summary, body=body.strip())


def _split_frontmatter(text: str, path: Path) -> Tuple[Dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillParseError(f"{path}: missing YAML frontmatter opening delimiter.")

    closing_index = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing_index = idx
            break
    if closing_index is None:
        raise SkillParseError(f"{path}: missing YAML frontmatter closing delimiter.")

    frontmatter_text = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1:])
    try:
        parsed = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        raise SkillParseError(f"{path}: invalid YAML frontmatter ({exc}).") from exc
    if not isinstance(parsed, dict):
        raise SkillParseError(f"{path}: frontmatter must be a YAML mapping.")
    return parsed, body


def _normalize_frontmatter(meta: Dict[str, Any], path: Path) -> Dict[str, Any]:
    name = str(meta.get("name", "")).strip()
    if not name or not _NAME_RE.match(name):
        raise SkillParseError(
            f"{path}: invalid skill name '{name}'. Expected lowercase kebab-case."
        )

    description = str(meta.get("description", "")).strip()
    if not description or len(description) > 1024:
        raise SkillParseError(
            f"{path}: description is required and must be 1-1024 characters."
        )

    license_text = str(meta.get("license", "") or "").strip()
    compatibility = str(meta.get("compatibility", "") or "").strip()
    if compatibility and len(compatibility) > 500:
        raise SkillParseError(f"{path}: compatibility must be <= 500 characters.")

    normalized_meta: Dict[str, str] = {}
    raw_metadata = meta.get("metadata", {})
    if raw_metadata:
        if not isinstance(raw_metadata, dict):
            raise SkillParseError(f"{path}: metadata must be a mapping when provided.")
        normalized_meta = {
            str(key).strip(): str(value).strip()
            for key, value in raw_metadata.items()
            if str(key).strip()
        }

    allowed_tools = _normalize_allowed_tools(meta.get("allowed-tools", meta.get("allowed_tools", "")))

    return {
        "name": name,
        "description": description,
        "license": license_text,
        "compatibility": compatibility,
        "metadata": normalized_meta,
        "allowed_tools": allowed_tools,
    }


def _normalize_allowed_tools(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    if not text:
        return ()
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
    else:
        parts = [part.strip() for part in text.split()]
    return tuple(part for part in parts if part)
