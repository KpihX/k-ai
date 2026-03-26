# test/test_utils.py
"""
Tests for ThinkingParser — the stateful <think>...</think> streaming parser.
"""
import pytest
from k_ai.utils import ThinkingParser


def parse_all(chunks: list[str]) -> tuple[str, str]:
    """Helper: feed all chunks through a fresh parser, return (thought, content)."""
    p = ThinkingParser()
    thoughts = []
    contents = []
    for chunk in chunks:
        t, c = p.parse(chunk)
        if t:
            thoughts.append(t)
        contents.append(c)
    return "".join(thoughts), "".join(contents)


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------

class TestThinkingParserBasic:
    def test_empty_input(self):
        p = ThinkingParser()
        t, c = p.parse("")
        assert t is None
        assert c == ""

    def test_plain_content_no_tags(self):
        p = ThinkingParser()
        t, c = p.parse("Hello world")
        assert t is None
        assert c == "Hello world"

    def test_full_think_block_single_chunk(self):
        p = ThinkingParser()
        t, c = p.parse("<think>I am thinking</think>")
        assert t == "I am thinking"
        assert c == ""

    def test_content_before_think(self):
        p = ThinkingParser()
        t, c = p.parse("Intro <think>thought</think>")
        assert t == "thought"
        assert c == "Intro "

    def test_content_after_think(self):
        p = ThinkingParser()
        t, c = p.parse("<think>thought</think> Answer here.")
        assert t == "thought"
        assert c == " Answer here."

    def test_content_before_and_after_think(self):
        p = ThinkingParser()
        t, c = p.parse("Before <think>thinking</think> after.")
        assert t == "thinking"
        assert c == "Before  after."

    def test_no_think_tag_returns_none_for_thought(self):
        p = ThinkingParser()
        t, c = p.parse("Just text")
        assert t is None

    def test_empty_think_block(self):
        p = ThinkingParser()
        t, c = p.parse("<think></think>")
        # empty string → thought is None (falsy filtered)
        assert c == ""


# ---------------------------------------------------------------------------
# Streaming: tags split across chunks
# ---------------------------------------------------------------------------

class TestThinkingParserSplitTags:
    def test_open_tag_split_across_two_chunks(self):
        """<thi | nk> split: no content leaked before tag is complete."""
        thought, content = parse_all(["<thi", "nk>thinking</think>"])
        assert thought == "thinking"
        assert content == ""

    def test_open_tag_split_one_char(self):
        thought, content = parse_all(["<", "think>inside</think>"])
        assert thought == "inside"
        assert content == ""

    def test_close_tag_split_across_two_chunks(self):
        thought, content = parse_all(["<think>thinking</thi", "nk>"])
        assert thought == "thinking"
        assert content == ""

    def test_close_tag_split_one_char(self):
        thought, content = parse_all(["<think>abc</", "think>"])
        assert thought == "abc"
        assert content == ""

    def test_full_tag_delivered_one_char_at_a_time(self):
        full = "<think>X</think>"
        chunks = list(full)
        thought, content = parse_all(chunks)
        assert thought == "X"
        assert content == ""

    def test_content_and_think_split(self):
        thought, content = parse_all(["Hello <thi", "nk>deep</think> world"])
        assert thought == "deep"
        assert content == "Hello  world"

    def test_think_content_split_mid_word(self):
        thought, content = parse_all(["<think>one ", "two</think>text"])
        assert thought == "one two"
        assert content == "text"


# ---------------------------------------------------------------------------
# Multi-chunk accumulation
# ---------------------------------------------------------------------------

class TestThinkingParserMultiChunk:
    def test_plain_content_across_chunks(self):
        thought, content = parse_all(["Hello ", "world", "!"])
        assert thought == ""
        assert content == "Hello world!"

    def test_think_content_across_three_chunks(self):
        thought, content = parse_all(["<think>", "deep thought", "</think>"])
        assert thought == "deep thought"
        assert content == ""

    def test_interleaved_content_and_thought(self):
        """Pre-text, then think block, then post-text across chunks."""
        thought, content = parse_all(["Preamble", "<think>reasoning</think>", " conclusion"])
        assert thought == "reasoning"
        assert content == "Preamble conclusion"

    def test_multiple_sequential_think_blocks(self):
        """Two separate <think> blocks in sequence."""
        thought, content = parse_all([
            "<think>first</think>",
            "<think>second</think>",
            "final",
        ])
        assert thought == "firstsecond"
        assert content == "final"


# ---------------------------------------------------------------------------
# _partial_suffix (static helper)
# ---------------------------------------------------------------------------

class TestPartialSuffix:
    def test_no_partial(self):
        assert ThinkingParser._partial_suffix("hello", "<think>") == 0

    def test_single_char_match(self):
        # "hello<" ends with the first char of "<think>"
        assert ThinkingParser._partial_suffix("hello<", "<think>") == 1

    def test_longer_partial(self):
        # "abc<thi" ends with "<thi" which is first 4 chars of "<think>"
        assert ThinkingParser._partial_suffix("abc<thi", "<think>") == 4

    def test_full_tag_minus_one(self):
        tag = "<think>"
        text = "x" + tag[:-1]  # all but last char
        assert ThinkingParser._partial_suffix(text, tag) == len(tag) - 1

    def test_empty_text(self):
        assert ThinkingParser._partial_suffix("", "<think>") == 0

    def test_close_tag_partial(self):
        assert ThinkingParser._partial_suffix("</thi", "</think>") == 5


# ---------------------------------------------------------------------------
# State isolation: each parser instance is independent
# ---------------------------------------------------------------------------

class TestThinkingParserStateIsolation:
    def test_two_parsers_independent(self):
        p1 = ThinkingParser()
        p2 = ThinkingParser()
        p1.parse("<think>only in p1")
        t2, c2 = p2.parse("not in p1")
        assert t2 is None
        assert c2 == "not in p1"

    def test_state_persists_within_same_parser(self):
        """
        The parser emits thought content eagerly unless the buffer ends with a
        partial suffix of the closing tag.  "part1 " has no such suffix, so it
        is emitted on the first chunk; "part2" is emitted on the second.
        """
        p = ThinkingParser()
        t1, c1 = p.parse("<think>part1 ")
        t2, c2 = p.parse("part2</think>done")
        # First chunk: "part1 " is inside think and has no </think> partial → emitted
        assert t1 == "part1 "
        assert c1 == ""
        # Second chunk: "part2" before closing tag, "done" is content
        assert t2 == "part2"
        assert c2 == "done"
