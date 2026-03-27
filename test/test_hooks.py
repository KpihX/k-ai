"""Tests for the hooks runtime."""

from pathlib import Path

import pytest

from k_ai import ConfigManager
from k_ai.hooks.manager import HookManager
from k_ai.hooks.parser import parse_hook_file


def write_hooks(root: Path, body: str, filename: str = "hooks.yaml") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    path.write_text(body, encoding="utf-8")
    return path


class TestHookParser:
    def test_parse_yaml_hook_file(self, tmp_path):
        path = write_hooks(
            tmp_path,
            """
hooks:
  PreToolUse:
    - matcher: "shell_exec"
      hooks:
        - type: command
          command: "python3 -c 'print(1)'"
          timeout_seconds: 5
""".strip() + "\n",
        )

        catalog = parse_hook_file(path=path, scope="project", precedence=0, default_timeout_seconds=10)

        assert len(catalog.matchers) == 1
        matcher = catalog.matchers[0]
        assert matcher.event == "PreToolUse"
        assert matcher.matcher == "shell_exec"
        assert matcher.commands[0].timeout_seconds == 5


class TestHookManager:
    @pytest.mark.asyncio
    async def test_user_prompt_submit_injects_context(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_hooks(
            tmp_path / ".k-ai" / "hooks",
            (
                '{\n'
                '  "hooks": {\n'
                '    "UserPromptSubmit": [\n'
                '      {\n'
                '        "matcher": "*",\n'
                '        "hooks": [\n'
                '          {\n'
                '            "type": "command",\n'
                '            "command": "python3 -c \\"import json,sys; data=json.load(sys.stdin); print(\'Hook says: \' + data[\'user_input\'])\\""\n'
                '          }\n'
                '        ]\n'
                '      }\n'
                '    ]\n'
                '  }\n'
                '}\n'
            ),
            filename="hooks.json",
        )
        cm = ConfigManager()
        manager = HookManager(cm, workspace_root=tmp_path)

        result = await manager.dispatch(event="UserPromptSubmit", payload={"user_input": "bonjour"}, matcher_value="*")

        assert result.blocked is False
        assert result.context_lines == ("Hook says: bonjour",)

    @pytest.mark.asyncio
    async def test_pre_tool_use_exit_code_two_blocks(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_hooks(
            tmp_path / ".k-ai" / "hooks",
            (
                '{\n'
                '  "hooks": {\n'
                '    "PreToolUse": [\n'
                '      {\n'
                '        "matcher": "clear_screen",\n'
                '        "hooks": [\n'
                '          {\n'
                '            "type": "command",\n'
                '            "command": "python3 -c \\"import sys; sys.stderr.write(\'blocked by hook\'); raise SystemExit(2)\\""\n'
                '          }\n'
                '        ]\n'
                '      }\n'
                '    ]\n'
                '  }\n'
                '}\n'
            ),
            filename="hooks.json",
        )
        cm = ConfigManager()
        manager = HookManager(cm, workspace_root=tmp_path)

        result = await manager.dispatch(
            event="PreToolUse",
            payload={"tool_name": "clear_screen", "tool_input": {}},
            matcher_value="clear_screen",
        )

        assert result.blocked is True
        assert "blocked by hook" in result.message
