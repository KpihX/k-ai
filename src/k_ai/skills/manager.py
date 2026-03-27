"""High-level skill runtime orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from ..config import ConfigManager
from .models import ActivatedSkill, SkillActivationResult, SkillIssue, SkillSummary
from .registry import SkillRegistry, SkillRoot
from .selector import SkillSelector


class SkillManager:
    """Resolve and render skills for k-ai turns."""

    def __init__(self, config: ConfigManager, workspace_root: Path | None = None):
        self._config = config
        self._workspace_root = (workspace_root or Path.cwd()).expanduser().resolve()
        self._registry = SkillRegistry(
            self._build_roots(),
            max_discovered=int(self._config.get_nested("skills", "max_discovered", default=500) or 500),
        )
        self._selector = SkillSelector(
            max_active=int(self._config.get_nested("skills", "max_active_per_turn", default=3) or 3),
            explicit_prefixes=tuple(self._config.get_nested("skills", "explicit_prefixes", default=["$"]) or ["$"]),
            reasons=dict(self._config.get_nested("skills", "selection", "reasons", default={}) or {}),
        )

    def enabled(self) -> bool:
        return bool(self._config.get_nested("skills", "enabled", default=True))

    def refresh(self) -> None:
        self._registry.refresh()

    def catalog(self, force_refresh: bool = False) -> Tuple[SkillSummary, ...]:
        return self._registry.catalog(force_refresh=force_refresh).skills

    def issues(self) -> Tuple[SkillIssue, ...]:
        return self._registry.catalog().issues

    def list_active_names(self, result: SkillActivationResult) -> Tuple[str, ...]:
        return tuple(item.document.summary.name for item in result.activated)

    def should_keep_previous(self, user_input: str) -> bool:
        return self._selector.is_low_signal_message(user_input)

    def resolve_for_turn(
        self,
        *,
        user_input: str,
        previous_names: Sequence[str] = (),
        force_refresh: bool = False,
    ) -> SkillActivationResult:
        if not self.enabled():
            return SkillActivationResult(activated=(), available=(), issues=())

        catalog = self._registry.catalog(force_refresh=force_refresh)
        available = catalog.skills
        auto_activate = bool(self._config.get_nested("skills", "auto_activate_from_user_input", default=True))
        if not auto_activate:
            return SkillActivationResult(
                activated=(),
                available=available,
                issues=catalog.issues,
                retained_names=tuple(previous_names),
            )

        selections = self._selector.select(
            user_input=user_input,
            available=available,
            previous_names=previous_names,
            auto_activate_on_continuation=bool(
                self._config.get_nested("skills", "auto_activate_on_continuation", default=True)
            ),
        )

        activated: List[ActivatedSkill] = []
        for selection in selections:
            document = self._registry.load_document(selection.summary.name)
            if document is None:
                continue
            activated.append(
                ActivatedSkill(
                    document=document,
                    reason=selection.reason,
                    activation_source="explicit" if selection.explicit else "continuation",
                    explicit=selection.explicit,
                    reused_from_context=selection.reused_from_context,
                )
            )

        retained = tuple(item.document.summary.name for item in activated) or tuple(previous_names)
        return SkillActivationResult(
            activated=tuple(activated),
            available=available,
            reused_context=any(item.reused_from_context for item in activated),
            issues=catalog.issues,
            retained_names=tuple(item.document.summary.name for item in activated),
        )

    def load_named_skill(self, name: str):
        return self._registry.load_document(name)

    def config_text(self, *parts: str, default: str = "", **vars: str) -> str:
        raw = str(self._config.get_nested(*parts, default=default) or default)
        if not vars:
            return raw
        class _SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"
        return raw.format_map(_SafeDict({key: str(value) for key, value in vars.items()}))

    def section_title(self, key: str, default: str) -> str:
        return self.config_text("skills", "runtime", "section_titles", key, default=default).strip()

    def activation_visibility_mode(self) -> str:
        mode = self.config_text(
            "skills",
            "runtime",
            "auto_activation",
            "visibility",
            default="announce",
        ).strip().lower()
        if mode not in {"silent", "announce", "tool"}:
            return "announce"
        return mode

    def format_activation_notice(self, activated: Sequence[ActivatedSkill]) -> str:
        labels = [self._activation_label(item) for item in activated if self._activation_label(item)]
        if not labels:
            return ""
        return self.config_text(
            "skills",
            "runtime",
            "auto_activation",
            "notice_message",
            default="Loaded for this turn: {skill_list}",
            skill_list=", ".join(labels),
        ).strip()

    def runtime_status_summary(
        self,
        *,
        active_names: Sequence[str] = (),
        retained_names: Sequence[str] = (),
    ) -> str:
        active = [str(name).strip() for name in active_names if str(name).strip()]
        retained = [str(name).strip() for name in retained_names if str(name).strip()]
        none = self.config_text(
            "skills",
            "runtime",
            "auto_activation",
            "runtime_none_message",
            default="none",
        ).strip()
        if active and retained:
            return self.config_text(
                "skills",
                "runtime",
                "auto_activation",
                "runtime_combined_template",
                default="active: {active_skills} | retained: {retained_skills}",
                active_skills=", ".join(active),
                retained_skills=", ".join(retained),
            ).strip()
        if active:
            return self.config_text(
                "skills",
                "runtime",
                "auto_activation",
                "runtime_active_template",
                default="{skill_list}",
                skill_list=", ".join(active),
            ).strip()
        if retained:
            return self.config_text(
                "skills",
                "runtime",
                "auto_activation",
                "runtime_retained_template",
                default="retained: {skill_list}",
                skill_list=", ".join(retained),
            ).strip()
        return none

    def render_catalog_section(self) -> str:
        if not self.enabled():
            return ""
        if not bool(self._config.get_nested("skills", "include_catalog_in_system_prompt", default=True)):
            return ""
        skills = self.catalog()
        if not skills:
            return ""
        limit = int(self._config.get_nested("skills", "catalog_limit", default=32) or 32)
        intro = str(self._config.get_nested("prompts", "skill_catalog_intro", default="The following skills are available in this runtime.") or "").strip()
        lines = [intro] if intro else []
        for skill in skills[:limit]:
            lines.append(
                self.config_text(
                    "skills",
                    "runtime",
                    "catalog",
                    "line_template",
                    default="- {skill_name}: {skill_description} [scope={skill_scope}]",
                    skill_name=skill.name,
                    skill_description=skill.description,
                    skill_scope=skill.scope,
                )
            )
        if len(skills) > limit:
            lines.append(
                self.config_text(
                    "skills",
                    "runtime",
                    "catalog",
                    "overflow_template",
                    default="- ... and {remaining_count} more skill(s)",
                    remaining_count=str(len(skills) - limit),
                )
            )
        usage_line = self.config_text(
            "skills",
            "runtime",
            "catalog",
            "usage_line",
            default="Use activate_skill with the exact skill name when you need to load one.",
        ).strip()
        if usage_line:
            lines.append(usage_line)
        return "\n".join(lines).strip()

    def render_active_section(self, activated: Sequence[ActivatedSkill]) -> str:
        if not activated:
            return ""
        intro = str(self._config.get_nested("prompts", "active_skills_intro", default="The following skills were activated for the current turn.") or "").strip()
        chunks: List[str] = [intro] if intro else []
        for item in activated:
            summary = item.document.summary
            chunks.append(
                self.config_text(
                    "skills",
                    "runtime",
                    "active",
                    "block_template",
                    default=(
                        "### Skill: {skill_name}\n"
                        "- source: {skill_source}\n"
                        "- scope: {skill_scope}\n"
                        "- reason: {skill_reason}\n\n"
                        "{skill_body}"
                    ),
                    skill_name=summary.name,
                    skill_source=str(summary.skill_file),
                    skill_scope=summary.scope,
                    skill_reason=item.reason,
                    skill_body=item.document.body,
                ).strip()
            )
        return "\n\n".join(chunk for chunk in chunks if chunk).strip()

    def _activation_label(self, item: ActivatedSkill) -> str:
        summary = item.document.summary
        template_key = "item_template"
        if item.activation_source == "semantic":
            template_key = "semantic_item_template"
        elif item.reused_from_context:
            template_key = "reused_item_template"
        elif item.explicit:
            template_key = "explicit_item_template"
        return self.config_text(
            "skills",
            "runtime",
            "auto_activation",
            template_key,
            default="{skill_name}",
            skill_name=summary.name,
        ).strip()

    def _build_roots(self) -> Tuple[SkillRoot, ...]:
        configured = self._config.get_nested("skills", "directories", default=[]) or []
        roots: List[SkillRoot] = []
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
            roots.append(SkillRoot(path=path, scope=scope, precedence=precedence))
        return tuple(roots)
