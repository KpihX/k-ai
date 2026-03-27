"""Deterministic skill activation with explicit and continuation routing."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import SkillSummary


@dataclass(frozen=True)
class SkillSelection:
    """One selected skill plus routing metadata."""

    summary: SkillSummary
    reason: str
    explicit: bool = False
    reused_from_context: bool = False


class SkillSelector:
    """Local skill selector aligned with explicit, progressive-disclosure routing."""

    def __init__(
        self,
        max_active: int = 3,
        explicit_prefixes: Sequence[str] = ("$",),
        reasons: Dict[str, str] | None = None,
    ):
        self._max_active = max(1, int(max_active))
        self._explicit_prefixes = tuple(prefix for prefix in explicit_prefixes if prefix)
        self._reasons = {
            "explicit_mention": "Explicit user mention of skill '{skill_name}'.",
            "reused_continuation": "Reused from previous active skills for a low-signal continuation turn.",
        }
        if reasons:
            for key, value in reasons.items():
                if key in self._reasons and str(value).strip():
                    self._reasons[key] = str(value)

    def is_low_signal_message(self, user_input: str) -> bool:
        tokens = self._tokenize(user_input)
        if not tokens:
            return True
        if len(tokens) <= 2:
            return True
        compact = " ".join(tokens)
        return len(compact) <= 16

    def select(
        self,
        *,
        user_input: str,
        available: Iterable[SkillSummary],
        previous_names: Sequence[str] = (),
        auto_activate_on_continuation: bool = True,
    ) -> Tuple[SkillSelection, ...]:
        text = self._normalize(user_input)
        skills = tuple(available)
        if not text or not skills:
            return ()

        explicit = self._explicit_matches(text, skills)
        if explicit:
            return tuple(explicit[: self._max_active])

        if auto_activate_on_continuation and previous_names and self.is_low_signal_message(user_input):
            reused = []
            previous_set = {name for name in previous_names}
            for skill in skills:
                if skill.name in previous_set:
                    reused.append(
                        SkillSelection(
                            summary=skill,
                            reason=self._reasons["reused_continuation"],
                            reused_from_context=True,
                        )
                    )
            if reused:
                return tuple(reused[: self._max_active])

        return ()

    def _explicit_matches(self, normalized_text: str, skills: Sequence[SkillSummary]) -> List[SkillSelection]:
        matches: List[SkillSelection] = []
        for skill in skills:
            if self._is_explicit_skill_mention(normalized_text, skill):
                matches.append(
                    SkillSelection(
                        summary=skill,
                        reason=self._reasons["explicit_mention"].format(skill_name=skill.name),
                        explicit=True,
                    )
                )
        return matches

    def _is_explicit_skill_mention(self, normalized_text: str, skill: SkillSummary) -> bool:
        if re.search(rf"\b{re.escape(skill.name)}\b", normalized_text):
            return True
        spaced_name = self._normalize(skill.name.replace("-", " "))
        if spaced_name and re.search(rf"\b{re.escape(spaced_name)}\b", normalized_text):
            return True
        for prefix in self._explicit_prefixes:
            escaped = re.escape(self._normalize(prefix))
            if re.search(rf"{escaped}{re.escape(skill.name)}\b", normalized_text):
                return True
        return False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        normalized = SkillSelector._normalize(text)
        return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", normalized)

    @staticmethod
    def _normalize(text: str) -> str:
        raw = unicodedata.normalize("NFKD", str(text or ""))
        raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
        return raw.lower()
