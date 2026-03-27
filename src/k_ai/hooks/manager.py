"""High-level hook orchestration."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ..config import ConfigManager
from .models import HookCatalog, HookDispatchResult, HookExecution, HookMatcher, HookRoot, HookCommand
from .registry import HookRegistry
from .runner import run_hook_command


class HookManager:
    def __init__(self, config: ConfigManager, workspace_root: Path | None = None):
        self._config = config
        self._workspace_root = (workspace_root or Path.cwd()).expanduser().resolve()
        self._registry = HookRegistry(
            roots=self._build_roots(),
            config_files=tuple(self._config.get_nested("hooks", "config_files", default=[]) or []),
            default_timeout_seconds=int(self._config.get_nested("hooks", "timeout_seconds", default=10) or 10),
        )

    def enabled(self) -> bool:
        return bool(self._config.get_nested("hooks", "enabled", default=True))

    def refresh(self) -> HookCatalog:
        return self._registry.refresh()

    def catalog(self, force_refresh: bool = False) -> HookCatalog:
        return self._registry.catalog(force_refresh=force_refresh)

    def runtime_summary(self) -> str:
        catalog = self.catalog()
        if not catalog.matchers:
            return self.config_text("hooks", "runtime", "discovery_none", default="No hooks discovered.")
        counts: Dict[str, int] = {}
        for matcher in catalog.matchers:
            counts[matcher.event] = counts.get(matcher.event, 0) + len(matcher.commands)
        return ", ".join(f"{event}:{count}" for event, count in sorted(counts.items()))

    async def dispatch(
        self,
        *,
        event: str,
        payload: Mapping[str, object],
        matcher_value: str = "*",
    ) -> HookDispatchResult:
        if not self.enabled():
            return HookDispatchResult()
        if not bool(self._config.get_nested("hooks", "events", event, "enabled", default=True)):
            return HookDispatchResult()

        selected = [item for item in self.catalog().matchers if item.event == event and self._matches(item.matcher, matcher_value)]
        if not selected:
            return HookDispatchResult()

        env = {
            "KAI_HOOK_EVENT": event,
            "KAI_PROJECT_DIR": str(self._workspace_root),
            "KAI_HOOK_MATCHER": matcher_value,
        }
        session_id = str(payload.get("session_id", "") or "").strip()
        if session_id:
            env["KAI_SESSION_ID"] = session_id

        executions: List[HookExecution] = []
        max_stdout = int(self._config.get_nested("hooks", "max_stdout_chars", default=4000) or 4000)
        max_stderr = int(self._config.get_nested("hooks", "max_stderr_chars", default=4000) or 4000)
        tasks = [
            self._safe_run_hook_command(
                matcher=matcher,
                command=command,
                payload=payload,
                env=env,
                max_stdout_chars=max_stdout,
                max_stderr_chars=max_stderr,
            )
            for matcher in selected
            for command in matcher.commands
        ]
        if bool(self._config.get_nested("hooks", "parallel", default=True)):
            executions.extend(await asyncio.gather(*tasks))
        else:
            for task in tasks:
                executions.append(await task)

        inject_stdout = bool(self._config.get_nested("hooks", "events", event, "inject_success_stdout", default=False))
        block_on_exit_two = bool(self._config.get_nested("hooks", "events", event, "block_on_exit_code_2", default=True))
        context_lines: List[str] = []
        warnings: List[str] = []
        blocked_message = ""

        for execution in executions:
            if execution.exit_code == 0:
                if inject_stdout and execution.stdout:
                    context_lines.append(execution.stdout[: int(self._config.get_nested("hooks", "max_context_chars", default=2000) or 2000)])
                continue
            if execution.exit_code == 2 and block_on_exit_two:
                blocked_message = execution.stderr or execution.stdout or f"Hook blocked event {event}."
                break
            warning_text = execution.stderr or execution.stdout or f"Hook exited with code {execution.exit_code}."
            warnings.append(warning_text)

        return HookDispatchResult(
            blocked=bool(blocked_message),
            message=blocked_message,
            context_lines=tuple(context_lines),
            warnings=tuple(warnings),
            executions=tuple(executions),
        )

    async def _safe_run_hook_command(
        self,
        *,
        matcher: HookMatcher,
        command: HookCommand,
        payload: Mapping[str, object],
        env: Mapping[str, str],
        max_stdout_chars: int,
        max_stderr_chars: int,
    ) -> HookExecution:
        try:
            return await run_hook_command(
                matcher=matcher,
                command=command,
                payload=payload,
                env=env,
                cwd=self._workspace_root,
                max_stdout_chars=max_stdout_chars,
                max_stderr_chars=max_stderr_chars,
            )
        except Exception as exc:
            return HookExecution(
                matcher=matcher,
                command=command,
                exit_code=1,
                stdout="",
                stderr=f"Hook execution failed: {exc}",
            )

    def config_text(self, *parts: str, default: str = "", **vars: str) -> str:
        raw = str(self._config.get_nested(*parts, default=default) or default)
        if not vars:
            return raw
        class _SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"
        return raw.format_map(_SafeDict({key: str(value) for key, value in vars.items()}))

    def _build_roots(self) -> tuple[HookRoot, ...]:
        configured = self._config.get_nested("hooks", "directories", default=[]) or []
        roots: List[HookRoot] = []
        seen: set[Path] = set()
        for precedence, raw in enumerate(configured):
            raw_text = str(raw or "").strip()
            if not raw_text:
                continue
            path = Path(raw_text).expanduser()
            if not path.is_absolute():
                path = (self._workspace_root / path).resolve()
                scope = "project"
            else:
                path = path.resolve()
                scope = "global" if str(path).startswith(str(Path("~/.agents").expanduser())) else "absolute"
            if path in seen:
                continue
            seen.add(path)
            roots.append(HookRoot(path=path, scope=scope, precedence=precedence))
        return tuple(roots)

    @staticmethod
    def _matches(pattern: str, value: str) -> bool:
        if pattern in {"", "*"}:
            return True
        try:
            return re.search(pattern, value or "") is not None
        except re.error:
            return pattern == value
