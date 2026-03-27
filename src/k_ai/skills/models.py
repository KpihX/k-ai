"""Core models for the k-ai skills runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SkillSummary:
    """Metadata-only view of one discovered skill."""

    name: str
    description: str
    root: Path
    skill_dir: Path
    skill_file: Path
    scope: str
    precedence: int
    license: str = ""
    compatibility: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    allowed_tools: Tuple[str, ...] = ()

    @property
    def slug_tokens(self) -> Tuple[str, ...]:
        return tuple(token for token in self.name.replace("-", " ").split() if token)


@dataclass(frozen=True)
class SkillDocument:
    """Fully loaded skill with body instructions."""

    summary: SkillSummary
    body: str


@dataclass(frozen=True)
class ActivatedSkill:
    """A skill selected for the current turn."""

    document: SkillDocument
    reason: str
    activation_source: str = "explicit"
    explicit: bool = False
    reused_from_context: bool = False


@dataclass(frozen=True)
class SkillIssue:
    """Non-fatal discovery or parsing issue."""

    path: Path
    message: str


@dataclass(frozen=True)
class SkillCatalog:
    """Snapshot of discovered skills plus non-fatal issues."""

    skills: Tuple[SkillSummary, ...]
    issues: Tuple[SkillIssue, ...]


@dataclass(frozen=True)
class SkillActivationResult:
    """Resolved skills for one turn."""

    activated: Tuple[ActivatedSkill, ...]
    available: Tuple[SkillSummary, ...]
    reused_context: bool = False
    issues: Tuple[SkillIssue, ...] = ()
    retained_names: Tuple[str, ...] = ()
