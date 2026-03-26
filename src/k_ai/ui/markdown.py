# src/k_ai/ui/markdown.py
"""
Math-aware Markdown rendering for k-ai.

Provides a Rich renderable that handles Markdown with embedded LaTeX:
  - Inline math ($...$) rendered inline without breaking text flow
  - Block math ($$...$$) rendered as distinct elements
  - Configurable render modes: raw, markdown, rich (with math)
"""
import re
from typing import Dict

from rich.console import Console, ConsoleOptions, RenderResult as RichRenderResult
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from rich.segment import Segment

from .math import MATH_REGEX, latex_to_unicode, is_block_math


class MathAwareMarkdown:
    """
    Rich renderable: Markdown with inline/block math converted to Unicode.

    Strategy for inline math:
      1. Replace $...$ with unique placeholders (code spans)
      2. Render surrounding text via Rich Markdown
      3. Intercept generated segments and substitute placeholders
         with pre-rendered Unicode math
    """

    def __init__(self, markup: str, style: str = "none"):
        self.markup = markup
        self.style = style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RichRenderResult:
        segments = MATH_REGEX.split(self.markup)
        text_buffer = []
        inline_placeholders: Dict[str, Text] = {}
        placeholder_idx = 0

        def flush_buffer():
            if not text_buffer:
                return
            raw_text = "".join(text_buffer)
            rich_md = RichMarkdown(raw_text, style=self.style)
            rendered = console.render(rich_md, options)

            for seg in rendered:
                if not seg.text or "__M" not in seg.text or not inline_placeholders:
                    yield seg
                    continue

                pattern = re.compile("|".join(re.escape(k) for k in inline_placeholders.keys()))
                parts = pattern.split(seg.text)
                matches = pattern.findall(seg.text)

                for i, part in enumerate(parts):
                    if part:
                        yield Segment(part, seg.style, seg.control)
                    if i < len(matches):
                        math_text = inline_placeholders[matches[i]]
                        for s in console.render(math_text, options):
                            if s.text == '\n' and not s.control:
                                continue
                            yield s

            text_buffer.clear()
            inline_placeholders.clear()
            nonlocal placeholder_idx
            placeholder_idx = 0

        for segment in segments:
            if not segment:
                continue

            if MATH_REGEX.match(segment):
                if is_block_math(segment):
                    yield from flush_buffer()
                    unicode_text = latex_to_unicode(segment)
                    yield Text(f"\n  {unicode_text}\n", style="bold cyan")
                else:
                    key = f"__M{placeholder_idx}__"
                    placeholder_idx += 1
                    unicode_text = latex_to_unicode(segment)
                    inline_placeholders[key] = Text(unicode_text, style="bold cyan")
                    text_buffer.append(f"`{key}`")
            else:
                text_buffer.append(segment)

        yield from flush_buffer()


def render_content(text: str, mode: str = "rich") -> object:
    """
    Render text content according to the display mode.

    Args:
        text: The content to render.
        mode: One of "raw", "markdown", "rich".

    Returns:
        A Rich renderable (or plain string for raw mode).
    """
    if mode == "raw":
        return Text(text)
    elif mode == "markdown":
        return RichMarkdown(text)
    elif mode == "rich":
        if "$" in text or "\\(" in text or "\\[" in text:
            return MathAwareMarkdown(text)
        return RichMarkdown(text)
    return RichMarkdown(text)
