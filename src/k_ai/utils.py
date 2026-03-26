# src/k_ai/utils.py
"""
Utility functions for k-ai.
"""
from typing import Optional, Tuple


class ThinkingParser:
    """
    Stateful streaming parser for <think>...</think> tags.
    Separates chain-of-thought content from the regular response.
    Handles tags that arrive split across multiple stream chunks.
    """

    def __init__(self):
        self._in_thinking = False
        self._buffer = ""  # Holds bytes that may be part of an incomplete tag

    def parse(self, raw: str) -> Tuple[Optional[str], str]:
        """
        Process a single streaming chunk.

        Returns:
            (thought_delta, content_delta) where thought_delta is None when
            there is no thinking content in this chunk.
        """
        if not raw:
            return None, ""

        self._buffer += raw
        thought_parts: list[str] = []
        content_parts: list[str] = []

        while self._buffer:
            if self._in_thinking:
                end_idx = self._buffer.find("</think>")
                if end_idx != -1:
                    thought_parts.append(self._buffer[:end_idx])
                    self._buffer = self._buffer[end_idx + len("</think>"):]
                    self._in_thinking = False
                else:
                    partial = self._partial_suffix(self._buffer, "</think>")
                    if partial > 0:
                        thought_parts.append(self._buffer[:-partial])
                        self._buffer = self._buffer[-partial:]
                        break
                    else:
                        thought_parts.append(self._buffer)
                        self._buffer = ""
            else:
                start_idx = self._buffer.find("<think>")
                if start_idx != -1:
                    if start_idx > 0:
                        content_parts.append(self._buffer[:start_idx])
                    self._buffer = self._buffer[start_idx + len("<think>"):]
                    self._in_thinking = True
                else:
                    partial = self._partial_suffix(self._buffer, "<think>")
                    if partial > 0:
                        content_parts.append(self._buffer[:-partial])
                        self._buffer = self._buffer[-partial:]
                        break
                    else:
                        content_parts.append(self._buffer)
                        self._buffer = ""

        thought = "".join(thought_parts) or None
        content = "".join(content_parts)
        return thought, content

    @staticmethod
    def _partial_suffix(text: str, tag: str) -> int:
        """
        Returns the length of the longest suffix of `text` that is a prefix of `tag`.
        Returns 0 if no such partial match exists.
        """
        for length in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:length]):
                return length
        return 0
