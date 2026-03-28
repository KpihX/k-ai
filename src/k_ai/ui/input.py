"""Ephemeral terminal prompts that are erased after user input."""

from __future__ import annotations

from math import ceil

from rich.console import Console


def _console_width(console: Console) -> int:
    try:
        width = int(getattr(console.size, "width", 80) or 80)
    except Exception:
        width = 80
    return max(width, 20)


def _wrapped_rows(text: str, width: int) -> int:
    if not text:
        return 1
    return max(ceil(len(text) / max(width, 1)), 1)


def estimate_document_prompt_lines(
    text: str,
    *,
    prompt_label: str,
    continuation_label: str,
    console: Console,
) -> int:
    width = _console_width(console)
    lines = text.split("\n")
    total = 0
    for index, line in enumerate(lines):
        label = prompt_label if index == 0 else continuation_label
        total += _wrapped_rows(f"{label}{line}", width)
    return max(total, 1)


def estimate_inline_prompt_lines(prompt: str, response: str, console: Console) -> int:
    return _wrapped_rows(f"{prompt}{response}", _console_width(console))


def erase_terminal_lines(console: Console, count: int) -> None:
    if count <= 0:
        return
    output = getattr(console, "file", None)
    if output is None or not hasattr(output, "isatty") or not output.isatty():
        return
    output.write("\r")
    for index in range(count):
        output.write("\x1b[1A")
        output.write("\x1b[2K")
        if index < count - 1:
            output.write("\r")
    output.write("\r")
    output.flush()


def ephemeral_input(console: Console, prompt: str) -> str:
    raw = str(console.input(prompt))
    erase_terminal_lines(console, estimate_inline_prompt_lines(prompt, raw, console))
    return raw


def ephemeral_confirm(console: Console, prompt: str, *, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = ephemeral_input(console, f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    if raw in {"y", "yes"}:
        return True
    if raw in {"n", "no"}:
        return False
    return default
