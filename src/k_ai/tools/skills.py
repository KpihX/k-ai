"""Internal tools for skill activation."""

from __future__ import annotations

from typing import Any, Dict

from ..config import ConfigManager
from ..models import ToolResult
from .base import InternalTool, ToolContext, ToolRegistry


class ActivateSkillTool(InternalTool):
    name = "activate_skill"
    category = "skills"
    danger_level = "low"
    accent_color = "magenta"
    requires_approval = False

    def __init__(self, config: ConfigManager):
        self._config = config
        self.display_name = self._config_text("display_name", "Activate Skill")
        self.description = self._config_text(
            "description",
            "Load one available SKILL.md into the current turn context before answering. Use this when the skill catalog indicates a relevant specialized workflow.",
        )
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": self._config_text("argument_description", "Exact skill name from the visible skill catalog."),
                },
            },
            "required": ["skill_name"],
        }

    def _config_text(self, key: str, default: str, **vars: str) -> str:
        raw = str(self._config.get_nested("skills", "runtime", "tool", key, default=default) or default)
        if not vars:
            return raw
        class _SafeDict(dict):
            def __missing__(self, name):
                return "{" + name + "}"
        return raw.format_map(_SafeDict({name: str(value) for name, value in vars.items()}))

    async def execute(self, arguments: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.activate_skill is None:
            return ToolResult(
                success=False,
                message=str(
                    ctx.config.get_nested(
                        "skills",
                        "runtime",
                        "activation",
                        "unavailable_message",
                        default="Skill runtime is unavailable.",
                    ) or "Skill runtime is unavailable."
                ),
            )
        result = ctx.activate_skill(str(arguments.get("skill_name", "") or ""))
        if not result.get("success"):
            fallback = str(
                ctx.config.get_nested(
                    "skills",
                    "runtime",
                    "activation",
                    "failed_message",
                    default="Skill activation failed.",
                ) or "Skill activation failed."
            )
            return ToolResult(success=False, message=str(result.get("message", fallback)))
        fallback = str(
            ctx.config.get_nested(
                "skills",
                "runtime",
                "activation",
                "activated_message",
                default="Skill activated.",
            ) or "Skill activated."
        )
        return ToolResult(success=True, message=str(result.get("message", fallback)), data=result)

    def proposal_rationale(self, arguments: Dict[str, Any]) -> str:
        skill_name = str(arguments.get("skill_name", "") or "").strip() or "(unspecified)"
        return self._config_text(
            "proposal_rationale",
            "Load the skill '{skill_name}' into the current turn context before answering.",
            skill_name=skill_name,
        )


def register_skill_tools(registry: ToolRegistry, ctx: ToolContext) -> None:
    registry.register(ActivateSkillTool(ctx.config))
