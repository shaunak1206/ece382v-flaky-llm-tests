"""Unit tests for iterable_to_regexp, PythonFormatter and the w2re package API."""
import re

import pytest

import w2re
from w2re import (
    BaseFormatter,
    PythonFormatter,
    PythonWordMatchFormatter,
    iterable_to_regexp,
)
from w2re import utils as w2re_utils
from w2re.prefix_tree.primitives import PrefixTreeNode

# Words used to check which strings a generated pattern accepts (full match).
PROBES = [
    "", ".", "a", "b", "c", "d", "x", "y", "z", "xy",
    "ca", "cat", "cats", "car", "cab", "cart", "ct", "cta", "cot",
    "do", "dog", "dogs", "dot",
    "a.b", "axb", "ab", "a+b", "a*b",
    "aa", "aaa", "aaaa", "aaaaa",
    "abc", "abcab", "abcabc", "abcabcabc",
    "caf", "caff", "cafe", "café",
]


def _accepts(pattern, word):
    return re.fullmatch(pattern, word) is not None


def _accepted(pattern, words=PROBES):
    return {word for word in words if _accepts(pattern, word)}


def test_empty_iterable_matches_only_empty_string():
    assert iterable_to_regexp([]) == r"\A\Z"
    assert iterable_to_regexp(["", ""]) == r"\A\Z"
    assert iterable_to_regexp([], PythonWordMatchFormatter) == r"\A\Z"
    assert _accepted(r"\A\Z") == {""}


def test_single_word_is_literal_regexp():
    pattern = iterable_to_regexp(["cat"])
    assert pattern == "cat"
    assert _accepted(pattern) == {"cat"}


def test_single_letter_and_metacharacter_words_are_escaped():
    assert iterable_to_regexp(["a"]) == "a"
    assert iterable_to_regexp(["."]) == r"\."
    assert _accepted(iterable_to_regexp(["."])) == {"."}


def test_dotted_word_escapes_metacharacters():
    pattern = iterable_to_regexp(["a.b"])
    assert pattern == r"a\.b"
    assert _accepted(pattern) == {"a.b"}


def test_shared_prefix_words_branch_correctly():
    pattern = iterable_to_regexp(["cat", "car"])
    assert _accepted(pattern) == {"cat", "car"}


def test_prefix_word_makes_tail_optional_in_either_order():
    for words in (["ca", "cat"], ["cat", "ca"]):
        assert _accepted(iterable_to_regexp(words)) == {"ca", "cat"}


def test_word_order_does_not_change_language():
    words = ["cat", "car", "dog", "do", "ca"]
    orderings = [
        words,
        list(reversed(words)),
        sorted(words),
        words[3:] + words[:3],
    ]
    expected = {"ca", "cat", "car", "do", "dog"}
    for ordering in orderings:
        assert _accepted(iterable_to_regexp(ordering)) == expected


def test_duplicate_words_do_not_change_pattern():
    assert iterable_to_regexp(["cat", "cat"]) == iterable_to_regexp(["cat"]) == "cat"


def test_generator_and_list_inputs_match():
    words = ["cat", "dog", "bird"]
    assert iterable_to_regexp(word for word in words) == iterable_to_regexp(words)
    assert iterable_to_regexp(tuple(words)) == iterable_to_regexp(words)


def test_single_letters_collapse_into_character_class():
    pattern = iterable_to_regexp(["a", "b", "c"])
    assert pattern == "[a-c]"
    assert _accepted(pattern) == {"a", "b", "c"}


def test_single_letters_mixed_with_words_use_alternation():
    pattern = iterable_to_regexp(["x", "y", "cat"])
    assert _accepted(pattern) == {"x", "y", "cat"}


def test_repeated_block_is_compressed_without_changing_language():
    pattern = iterable_to_regexp(["abcabc"])
    assert _accepted(pattern) == {"abcabc"}


def test_repeated_letters_language_is_preserved():
    pattern = iterable_to_regexp(["aa", "aaaa"])
    assert _accepted(pattern) == {"aa", "aaaa"}


def test_multi_character_branches_with_metacharacters_are_escaped():
    pattern = iterable_to_regexp(["a+b", "a*b"])
    assert _accepted(pattern) == {"a+b", "a*b"}


def test_unicode_words_are_supported():
    pattern = iterable_to_regexp(["café", "cafe"])
    assert _accepted(pattern) == {"café", "cafe"}


def test_word_match_formatter_exact_pattern():
    pattern = iterable_to_regexp(["cat", "dog"], PythonWordMatchFormatter)
    assert pattern == r"(?:\W+|\A)((?:cat|dog))(?=\W+|\Z)"


def test_word_match_formatter_only_matches_whole_words():
    pattern = iterable_to_regexp(["cat"], PythonWordMatchFormatter)
    assert re.search(pattern, "the cat sat") is not None
    assert re.search(pattern, "cat") is not None
    assert re.search(pattern, "concatenate") is None
    assert re.search(pattern, "cats") is None


def test_default_formatter_is_python_formatter_and_matches_substrings():
    words = ["cat", "dog"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words, PythonFormatter)
    assert re.search(iterable_to_regexp(["cat"]), "concatenate") is not None
    assert PythonFormatter.wrap_regexp(PrefixTreeNode()) == r"\A\Z"


def test_formatter_codes_and_descriptions():
    assert PythonFormatter.code() == "py"
    assert PythonFormatter.description() == "Python regular expression"
    assert PythonWordMatchFormatter.code() == "pyw"
    assert PythonWordMatchFormatter.description() == "Python word matching regular expression"
    assert issubclass(PythonWordMatchFormatter, PythonFormatter)


def test_base_formatter_wrap_is_not_implemented():
    with pytest.raises(NotImplementedError):
        iterable_to_regexp(["a"], formatter=BaseFormatter)


def test_package_metadata_and_exports():
    assert re.fullmatch(r"\d+\.\d+\.\d+", w2re.__version__)
    assert w2re.APPLICATION_NAME == "w2re"
    assert w2re.CHANGELOG_URL.startswith("https://")
    assert w2re.CHANGELOG_URL.endswith("/CHANGELOG.md")
    assert w2re.iterable_to_regexp is w2re_utils.iterable_to_regexp
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.PythonWordMatchFormatter is PythonWordMatchFormatter
    assert w2re.BaseFormatter is BaseFormatter


@pytest.mark.xfail(
    strict=False,
    reason="single-character metacharacter is emitted unescaped when a longer word "
           "shares the alternation, so '.' also matches any character",
)
def test_metacharacter_single_letter_beside_longer_word_is_escaped():
    pattern = iterable_to_regexp([".", "ab"])
    assert _accepts(pattern, ".")
    assert not _accepts(pattern, "x")


@pytest.mark.xfail(
    strict=False,
    reason="a caret that is the first character of a character class negates the class",
)
def test_caret_letter_in_character_class_is_literal():
    pattern = iterable_to_regexp(["^", "a"])
    assert _accepts(pattern, "^")
    assert _accepts(pattern, "a")
