"""Parsing for hook configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .models import HookCatalog, HookCommand, HookIssue, HookMatcher


class HookParseError(ValueError):
    """Raised when a hook configuration file is invalid."""


def parse_hook_file(
    *,
    path: Path,
    scope: str,
    precedence: int,
    default_timeout_seconds: int,
) -> HookCatalog:
    raw = path.read_text(encoding="utf-8")
    payload = _load_payload(path, raw)
    raw_hooks = payload.get("hooks", {})
    if not isinstance(raw_hooks, dict):
        raise HookParseError(f"{path}: top-level 'hooks' must be a mapping.")

    matchers: List[HookMatcher] = []
    issues: List[HookIssue] = []
    for event, entries in raw_hooks.items():
        if not isinstance(entries, list):
            raise HookParseError(f"{path}: hooks.{event} must be a list.")
        for index, entry in enumerate(entries):
            try:
                matchers.append(
                    _parse_matcher(
                        event=str(event).strip(),
                        entry=entry,
                        path=path,
                        scope=scope,
                        precedence=precedence,
                        default_timeout_seconds=default_timeout_seconds,
                    )
                )
            except HookParseError as exc:
                issues.append(HookIssue(path=path, message=f"{exc} (entry #{index + 1})"))
    return HookCatalog(matchers=tuple(matchers), issues=tuple(issues))


def _load_payload(path: Path, raw: str) -> Dict[str, Any]:
    if path.suffix.lower() == ".json":
        data = json.loads(raw or "{}")
    else:
        data = yaml.safe_load(raw) or {}
    if not isinstance(data, dict):
        raise HookParseError(f"{path}: hook config must be a mapping.")
    return data


def _parse_matcher(
    *,
    event: str,
    entry: Any,
    path: Path,
    scope: str,
    precedence: int,
    default_timeout_seconds: int,
) -> HookMatcher:
    if not isinstance(entry, dict):
        raise HookParseError(f"{path}: each hook entry for {event} must be a mapping.")
    matcher = str(entry.get("matcher", "*") or "*").strip() or "*"
    hooks = entry.get("hooks", [])
    if not isinstance(hooks, list) or not hooks:
        raise HookParseError(f"{path}: hooks.{event}.hooks must be a non-empty list.")
    commands: List[HookCommand] = []
    for hook_entry in hooks:
        if not isinstance(hook_entry, dict):
            raise HookParseError(f"{path}: each nested hook must be a mapping.")
        hook_type = str(hook_entry.get("type", "command") or "command").strip().lower()
        if hook_type != "command":
            raise HookParseError(f"{path}: unsupported hook type '{hook_type}'.")
        command = str(hook_entry.get("command", "") or "").strip()
        if not command:
            raise HookParseError(f"{path}: command hook must define a non-empty command.")
        timeout = int(hook_entry.get("timeout_seconds", default_timeout_seconds) or default_timeout_seconds)
        commands.append(HookCommand(command=command, timeout_seconds=max(1, timeout)))
    return HookMatcher(
        event=event,
        matcher=matcher,
        commands=tuple(commands),
        source_file=path,
        scope=scope,
        precedence=precedence,
    )
