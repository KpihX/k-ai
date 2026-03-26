"""Shared UI theme helpers used across renderers and syntax blocks."""

from typing import Dict

_UI_THEMES: Dict[str, Dict[str, str]] = {
    "default": {
        "syntax": "monokai",
        "assistant_border": "green",
        "user_border": "bright_black",
        "user_style": "on #1f2329",
        "runtime_border": "cyan",
        "thinking_border": "yellow",
    }
}


def resolve_ui_theme(theme_name: str | None) -> dict:
    return _UI_THEMES.get(str(theme_name or "default").strip().lower(), _UI_THEMES["default"])


def resolve_syntax_theme(theme_name: str | None) -> str:
    return str(resolve_ui_theme(theme_name).get("syntax", "monokai"))
