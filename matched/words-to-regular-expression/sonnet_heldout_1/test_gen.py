import io
import os
import re

import pytest

import w2re
from w2re import (
    BaseFormatter,
    PythonFormatter,
    PythonWordMatchFormatter,
    iterable_to_regexp,
    stream_to_regexp,
)
from w2re import utils


def _assert_matches_exactly(pattern, words, non_words=()):
    compiled = re.compile(pattern)
    for word in words:
        assert compiled.fullmatch(word), f"{word!r} should match {pattern!r}"
    for word in non_words:
        assert not compiled.fullmatch(word), f"{word!r} should not match {pattern!r}"


def test_package_exports_public_names():
    assert w2re.iterable_to_regexp is utils.iterable_to_regexp
    assert w2re.stream_to_regexp is utils.stream_to_regexp
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.PythonWordMatchFormatter is PythonWordMatchFormatter
    assert w2re.BaseFormatter is BaseFormatter


def test_package_metadata():
    assert re.fullmatch(r"\d+\.\d+\.\d+", w2re.__version__)
    assert w2re.APPLICATION_NAME == "w2re"
    assert w2re.CHANGELOG_URL.startswith("https://")
    assert w2re.CHANGELOG_URL.endswith("CHANGELOG.md")


def test_formatters_inherit_from_base_formatter():
    assert issubclass(PythonFormatter, BaseFormatter)
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_iterable_to_regexp_returns_string():
    assert isinstance(iterable_to_regexp(["abc", "abd"]), str)


def test_single_word_matches_only_itself():
    pattern = iterable_to_regexp(["hello"])
    _assert_matches_exactly(pattern, ["hello"], ["hell", "helloo", "world", ""])


def test_multiple_words_with_common_prefix():
    words = ["abc", "abd", "abe", "xyz"]
    pattern = iterable_to_regexp(words)
    _assert_matches_exactly(pattern, words, ["ab", "abf", "xy", "xyzz", "abcd"])


def test_words_that_are_prefixes_of_other_words():
    words = ["ab", "abc", "abcd"]
    pattern = iterable_to_regexp(words)
    _assert_matches_exactly(pattern, words, ["a", "abcde", "abd", ""])


def test_words_with_common_suffix():
    words = ["walking", "talking", "balking"]
    pattern = iterable_to_regexp(words)
    _assert_matches_exactly(pattern, words, ["alking", "walkin", "kalking"])


def test_duplicate_words_are_handled():
    words = ["cat", "cat", "car"]
    pattern = iterable_to_regexp(words)
    _assert_matches_exactly(pattern, ["cat", "car"], ["ca", "cart", "catcat"])


def test_accepts_generator_input():
    pattern = iterable_to_regexp(w for w in ["one", "two", "three"])
    _assert_matches_exactly(pattern, ["one", "two", "three"], ["on", "four"])


@pytest.mark.parametrize("container", [list, tuple, set, frozenset])
def test_accepts_various_iterable_types(container):
    words = ["red", "green", "blue"]
    pattern = iterable_to_regexp(container(words))
    _assert_matches_exactly(pattern, words, ["gre", "yellow"])


def test_input_order_does_not_change_matching():
    words = ["alpha", "alpine", "beta", "bet"]
    p1 = iterable_to_regexp(words)
    p2 = iterable_to_regexp(list(reversed(words)))
    candidates = words + ["alp", "be", "betaa", "alpin", "gamma"]
    c1, c2 = re.compile(p1), re.compile(p2)
    for candidate in candidates:
        assert bool(c1.fullmatch(candidate)) == bool(c2.fullmatch(candidate))


def test_special_regex_characters_are_escaped():
    words = ["a.c", "x+y", "(z)"]
    pattern = iterable_to_regexp(words)
    _assert_matches_exactly(pattern, words, ["abc", "xxy", "z"])


def test_unicode_words():
    words = ["café", "cafe", "naïve"]
    pattern = iterable_to_regexp(words)
    _assert_matches_exactly(pattern, words, ["caf", "naive"])


def test_default_formatter_is_python_formatter():
    words = ["foo", "foobar", "baz"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words, PythonFormatter)


def test_word_match_formatter_matches_words():
    words = ["cat", "car", "dog"]
    pattern = iterable_to_regexp(words, PythonWordMatchFormatter)
    compiled = re.compile(pattern)
    for word in words:
        assert compiled.fullmatch(word)
    assert not compiled.fullmatch("bird")


def test_stream_to_regexp_matches_iterable_to_regexp():
    words = ["apple", "apply", "banana"]
    stream = io.StringIO(os.linesep.join(words))
    from_stream = stream_to_regexp(stream)
    _assert_matches_exactly(from_stream, words, ["appl", "bananas"])
    assert from_stream == iterable_to_regexp(words)


def test_stream_to_regexp_ignores_empty_lines():
    words = ["one", "two"]
    text = os.linesep + os.linesep.join(words) + os.linesep + os.linesep
    pattern = stream_to_regexp(io.StringIO(text))
    _assert_matches_exactly(pattern, words, ["", "on", "three"])
