"""Discovery and caching for hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .models import HookCatalog, HookIssue, HookMatcher, HookRoot
from .parser import HookParseError, parse_hook_file


class HookRegistry:
    def __init__(
        self,
        roots: Iterable[HookRoot],
        config_files: Iterable[str],
        default_timeout_seconds: int,
    ):
        self._roots = tuple(roots)
        self._config_files = tuple(str(name).strip() for name in config_files if str(name).strip())
        self._default_timeout_seconds = max(1, int(default_timeout_seconds))
        self._catalog = HookCatalog(matchers=(), issues=())

    def catalog(self, force_refresh: bool = False) -> HookCatalog:
        if force_refresh or (not self._catalog.matchers and not self._catalog.issues):
            self.refresh()
        return self._catalog

    def refresh(self) -> HookCatalog:
        matchers: List[HookMatcher] = []
        issues: List[HookIssue] = []
        for root in self._roots:
            if not root.path.exists() or not root.path.is_dir():
                continue
            for config_name in self._config_files:
                path = root.path / config_name
                if not path.exists() or not path.is_file():
                    continue
                try:
                    parsed = parse_hook_file(
                        path=path,
                        scope=root.scope,
                        precedence=root.precedence,
                        default_timeout_seconds=self._default_timeout_seconds,
                    )
                except HookParseError as exc:
                    issues.append(HookIssue(path=path, message=str(exc)))
                    continue
                matchers.extend(parsed.matchers)
                issues.extend(parsed.issues)
                break
        ordered = tuple(sorted(matchers, key=lambda item: (item.event, item.precedence, str(item.source_file), item.matcher)))
        self._catalog = HookCatalog(matchers=ordered, issues=tuple(issues))
        return self._catalog
