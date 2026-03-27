# test/test_skills.py
"""Tests for the native skills runtime."""

from pathlib import Path

import pytest

from k_ai import ConfigManager
from k_ai.skills.manager import SkillManager
from k_ai.skills.parser import SkillParseError, parse_skill_file
from k_ai.skills.registry import SkillRegistry, SkillRoot
from k_ai.skills.selector import SkillSelector


def write_skill(root: Path, relative: str, *, name: str, description: str, body: str, extra: str = "") -> Path:
    skill_dir = root / relative
    skill_dir.mkdir(parents=True, exist_ok=True)
    path = skill_dir / "SKILL.md"
    text = (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "license: private\n"
        "metadata:\n"
        '  author: test\n'
        '  version: "1.0"\n'
        f"{extra}"
        "---\n\n"
        f"{body}\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


class TestSkillParser:
    def test_parse_valid_skill_file(self, tmp_path):
        path = write_skill(
            tmp_path,
            "python/review",
            name="python-review",
            description="Review Python code and identify bugs, regressions, and test gaps.",
            body="# Python Review\n\nInspect code carefully.",
            extra="allowed-tools: Bash(git:*) Read\n",
        )

        document = parse_skill_file(path=path, root=tmp_path, scope="project", precedence=0)

        assert document.summary.name == "python-review"
        assert document.summary.scope == "project"
        assert document.summary.allowed_tools == ("Bash(git:*)", "Read")
        assert "Inspect code carefully" in document.body

    def test_parse_accepts_comma_separated_allowed_tools(self, tmp_path):
        path = write_skill(
            tmp_path,
            "agents/codex",
            name="agent-runtime-codex",
            description="Codex runtime guidance with MCP and config details.",
            body="Use the runtime carefully.",
            extra="allowed-tools: Bash, Read, Write, Edit\n",
        )

        document = parse_skill_file(path=path, root=tmp_path, scope="global", precedence=2)

        assert document.summary.allowed_tools == ("Bash", "Read", "Write", "Edit")

    def test_parse_rejects_invalid_name(self, tmp_path):
        path = write_skill(
            tmp_path,
            "bad",
            name="BadSkill",
            description="Invalid name for spec validation.",
            body="Nope.",
        )

        with pytest.raises(SkillParseError):
            parse_skill_file(path=path, root=tmp_path, scope="project", precedence=0)


class TestSkillRegistry:
    def test_registry_prefers_higher_precedence_duplicate(self, tmp_path):
        project_root = tmp_path / ".k-ai" / "skills"
        global_root = tmp_path / "global"
        write_skill(
            project_root,
            "review",
            name="code-review",
            description="Project-local review flow.",
            body="Prefer project local rules.",
        )
        write_skill(
            global_root,
            "review",
            name="code-review",
            description="Global review flow.",
            body="Global rules.",
        )
        registry = SkillRegistry(
            [
                SkillRoot(path=project_root, scope="project", precedence=0),
                SkillRoot(path=global_root, scope="global", precedence=1),
            ]
        )

        catalog = registry.refresh()

        assert len(catalog.skills) == 1
        assert catalog.skills[0].description == "Project-local review flow."
        assert any("Duplicate skill name 'code-review'" in issue.message for issue in catalog.issues)

    def test_registry_reports_invalid_skill_without_crashing(self, tmp_path):
        root = tmp_path / "skills"
        write_skill(
            root,
            "good",
            name="good-skill",
            description="A valid skill.",
            body="Works.",
        )
        bad_dir = root / "bad"
        bad_dir.mkdir(parents=True, exist_ok=True)
        (bad_dir / "SKILL.md").write_text("---\nname: Bad\n---\n", encoding="utf-8")

        registry = SkillRegistry([SkillRoot(path=root, scope="project", precedence=0)])
        catalog = registry.refresh()

        assert [skill.name for skill in catalog.skills] == ["good-skill"]
        assert len(catalog.issues) == 1


class TestSkillSelector:
    def test_selector_matches_explicit_name(self, tmp_path):
        root = tmp_path / "skills"
        write_skill(
            root,
            "workspace",
            name="kpihx-workspace",
            description="Workspace and engineering standards.",
            body="Body.",
        )
        registry = SkillRegistry([SkillRoot(path=root, scope="project", precedence=0)])
        catalog = registry.refresh()
        selector = SkillSelector(max_active=3)

        selected = selector.select(user_input="use $kpihx-workspace now", available=catalog.skills)

        assert [item.summary.name for item in selected] == ["kpihx-workspace"]
        assert selected[0].explicit is True

    def test_selector_only_auto_activates_on_explicit_mentions_or_continuation(self, tmp_path):
        root = tmp_path / "skills"
        write_skill(
            root,
            "review",
            name="python-review",
            description="Review Python code, identify bugs, regressions, and missing tests.",
            body="Body.",
        )
        registry = SkillRegistry([SkillRoot(path=root, scope="project", precedence=0)])
        catalog = registry.refresh()
        selector = SkillSelector(max_active=3)

        selected = selector.select(
            user_input="can you review this python code and find regressions",
            available=catalog.skills,
        )

        assert selected == ()

    def test_selector_reuses_previous_skills_on_low_signal_turn(self):
        selector = SkillSelector(max_active=3)
        available = ()

        assert selector.is_low_signal_message("continue")
        assert selector.is_low_signal_message("et après")
        assert selector.is_low_signal_message("ok")
        assert selector.is_low_signal_message("peux-tu maintenant détailler le workflow complet") is False
        assert selector.select(user_input="continue", available=available, previous_names=("alpha",)) == ()


class TestSkillManager:
    def test_manager_uses_project_and_global_roots(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_skill(
            tmp_path / ".k-ai" / "skills",
            "local",
            name="local-skill",
            description="Local project skill.",
            body="Local body.",
        )
        write_skill(
            Path("~/.agents/skills").expanduser(),
            "global",
            name="global-skill",
            description="Global user skill.",
            body="Global body.",
        )
        cm = ConfigManager()
        manager = SkillManager(cm, workspace_root=tmp_path)

        names = [skill.name for skill in manager.catalog(force_refresh=True)]

        assert "local-skill" in names
        assert "global-skill" in names

    def test_manager_renders_active_skill_section(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_skill(
            tmp_path / ".k-ai" / "skills",
            "review",
            name="python-review",
            description="Review Python code for bugs and regressions.",
            body="# Review\n\nLook for bugs first.",
        )
        cm = ConfigManager()
        manager = SkillManager(cm, workspace_root=tmp_path)

        result = manager.resolve_for_turn(
            user_input="use $python-review to review this python code for bugs",
            force_refresh=True,
        )
        rendered = manager.render_active_section(result.activated)

        assert "python-review" in rendered
        assert "Look for bugs first." in rendered
