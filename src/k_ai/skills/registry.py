"""Discovery and caching for skills."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .models import SkillCatalog, SkillDocument, SkillIssue, SkillSummary
from .parser import SkillParseError, parse_skill_file, skill_file_signature


@dataclass(frozen=True)
class SkillRoot:
    """One configured discovery root."""

    path: Path
    scope: str
    precedence: int


class SkillRegistry:
    """Index discovered skills and lazily cache parsed documents."""

    def __init__(self, roots: Iterable[SkillRoot], max_discovered: int = 500):
        self._roots = tuple(roots)
        self._max_discovered = max(1, int(max_discovered))
        self._summary_cache: Dict[Path, Tuple[Tuple[int, int], SkillSummary]] = {}
        self._document_cache: Dict[Path, Tuple[Tuple[int, int], SkillDocument]] = {}
        self._catalog = SkillCatalog(skills=(), issues=())

    def catalog(self, force_refresh: bool = False) -> SkillCatalog:
        if force_refresh or not self._catalog.skills and not self._catalog.issues:
            self.refresh()
        return self._catalog

    def refresh(self) -> SkillCatalog:
        skills: Dict[str, SkillSummary] = {}
        issues: List[SkillIssue] = []
        discovered = 0

        for root in self._roots:
            if not root.path.exists() or not root.path.is_dir():
                continue
            for skill_file in self._iter_skill_files(root.path):
                discovered += 1
                if discovered > self._max_discovered:
                    issues.append(
                        SkillIssue(
                            path=root.path,
                            message=f"Discovery stopped after {self._max_discovered} skill files.",
                        )
                    )
                    break
                try:
                    summary = self._load_summary(skill_file, root)
                except SkillParseError as exc:
                    issues.append(SkillIssue(path=skill_file, message=str(exc)))
                    continue

                existing = skills.get(summary.name)
                if existing is None:
                    skills[summary.name] = summary
                    continue
                if summary.precedence < existing.precedence:
                    skills[summary.name] = summary
                    issues.append(
                        SkillIssue(
                            path=skill_file,
                            message=(
                                f"Duplicate skill name '{summary.name}' overrides {existing.skill_file} "
                                f"because its root has higher precedence."
                            ),
                        )
                    )
                else:
                    issues.append(
                        SkillIssue(
                            path=skill_file,
                            message=(
                                f"Duplicate skill name '{summary.name}' ignored in favor of {existing.skill_file}."
                            ),
                        )
                    )
            if discovered > self._max_discovered:
                break

        ordered = tuple(sorted(skills.values(), key=lambda item: (item.precedence, item.name)))
        self._catalog = SkillCatalog(skills=ordered, issues=tuple(issues))
        return self._catalog

    def list_skills(self) -> Tuple[SkillSummary, ...]:
        return self.catalog().skills

    def issues(self) -> Tuple[SkillIssue, ...]:
        return self.catalog().issues

    def get_summary(self, name: str) -> SkillSummary | None:
        normalized = str(name or "").strip().lower()
        if not normalized:
            return None
        for skill in self.catalog().skills:
            if skill.name == normalized:
                return skill
        return None

    def load_document(self, name: str) -> SkillDocument | None:
        summary = self.get_summary(name)
        if summary is None:
            return None
        signature = skill_file_signature(summary.skill_file)
        cached = self._document_cache.get(summary.skill_file)
        if cached and cached[0] == signature:
            return cached[1]
        document = parse_skill_file(
            path=summary.skill_file,
            root=summary.root,
            scope=summary.scope,
            precedence=summary.precedence,
        )
        self._document_cache[summary.skill_file] = (signature, document)
        self._summary_cache[summary.skill_file] = (signature, document.summary)
        return document

    def _load_summary(self, skill_file: Path, root: SkillRoot) -> SkillSummary:
        signature = skill_file_signature(skill_file)
        cached = self._summary_cache.get(skill_file)
        if cached and cached[0] == signature:
            return cached[1]
        document = parse_skill_file(
            path=skill_file,
            root=root.path,
            scope=root.scope,
            precedence=root.precedence,
        )
        self._summary_cache[skill_file] = (signature, document.summary)
        self._document_cache[skill_file] = (signature, document)
        return document.summary

    @staticmethod
    def _iter_skill_files(root: Path) -> Iterable[Path]:
        ignored_dirs = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv"}
        for path in root.rglob("SKILL.md"):
            if any(part in ignored_dirs for part in path.parts):
                continue
            yield path
