# src/k_ai/ui/__init__.py
"""UI components for k-ai: rendering, panels, math, tool results."""
from .render import (
    StreamingRenderer,
    render_assistant_panel,
    render_key_value_panel,
    render_local_runner_output,
    render_notice,
    render_runtime_panel,
    render_sessions_table,
    render_tool_proposal,
    render_tool_result,
    render_user_panel,
)
from .input import (
    ephemeral_confirm,
    ephemeral_input,
    erase_terminal_lines,
    estimate_document_prompt_lines,
)
from .markdown import MathAwareMarkdown, render_content
from .math import latex_to_unicode, MATH_REGEX

__all__ = [
    "StreamingRenderer",
    "render_assistant_panel",
    "render_key_value_panel",
    "render_local_runner_output",
    "render_notice",
    "render_runtime_panel",
    "render_sessions_table",
    "render_tool_proposal",
    "render_tool_result",
    "render_user_panel",
    "ephemeral_confirm",
    "ephemeral_input",
    "erase_terminal_lines",
    "estimate_document_prompt_lines",
    "MathAwareMarkdown",
    "render_content",
    "latex_to_unicode",
]
