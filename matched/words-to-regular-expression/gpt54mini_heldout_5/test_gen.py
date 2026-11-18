import re
from io import StringIO

import pytest

import w2re
from w2re import (
    APPLICATION_NAME,
    CHANGELOG_URL,
    BaseFormatter,
    PrefixTree,
    PythonFormatter,
    PythonWordMatchFormatter,
    iterable_to_regexp,
    stream_to_regexp,
)


def _assert_matches_all(pattern, words):
    for word in words:
        assert re.fullmatch(pattern, word), f"Pattern {pattern!r} did not match {word!r}"


def _assert_does_not_match(pattern, words):
    for word in words:
        assert not re.fullmatch(pattern, word), f"Pattern {pattern!r} unexpectedly matched {word!r}"


def test_package_version_constant():
    assert w2re.__version__ == "3.1.0"


def test_package_metadata_constants():
    assert APPLICATION_NAME == "w2re"
    assert CHANGELOG_URL == (
        "https://github.com/radeklat/words-to-regular-expression/"
        "blob/develop/CHANGELOG.md"
    )


def test_re_exports_are_available_from_package_root():
    assert iterable_to_regexp is w2re.iterable_to_regexp
    assert stream_to_regexp is w2re.stream_to_regexp
    assert PrefixTree is w2re.PrefixTree
    assert PythonFormatter is w2re.PythonFormatter
    assert BaseFormatter is w2re.BaseFormatter
    assert PythonWordMatchFormatter is w2re.PythonWordMatchFormatter


def test_python_formatter_is_a_base_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)


def test_python_word_match_formatter_is_a_base_formatter_subclass():
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_iterable_to_regexp_returns_a_string():
    pattern = iterable_to_regexp(["cat", "dog"])
    assert isinstance(pattern, str)
    assert pattern


def test_iterable_to_regexp_matches_all_words_in_iterable():
    words = ["cat", "car", "dog"]
    pattern = iterable_to_regexp(words)
    _assert_matches_all(pattern, words)


def test_iterable_to_regexp_does_not_match_unrelated_words():
    words = ["cat", "car", "dog"]
    pattern = iterable_to_regexp(words)
    _assert_does_not_match(pattern, ["cab", "do", "cats", "mouse"])


def test_iterable_to_regexp_handles_shared_prefixes():
    words = ["prefix", "preform", "prevent"]
    pattern = iterable_to_regexp(words)
    _assert_matches_all(pattern, words)
    _assert_does_not_match(pattern, ["pre", "pref", "prefixes"])


def test_iterable_to_regexp_handles_nested_prefixes():
    words = ["a", "ab", "abc"]
    pattern = iterable_to_regexp(words)
    _assert_matches_all(pattern, words)
    _assert_does_not_match(pattern, ["", "abcd", "b"])


def test_iterable_to_regexp_accepts_any_iterable_not_just_lists():
    words = (word for word in ["alpha", "beta", "gamma"])
    pattern = iterable_to_regexp(words)
    _assert_matches_all(pattern, ["alpha", "beta", "gamma"])
    _assert_does_not_match(pattern, ["alp", "bet", "gammad"])


def test_iterable_to_regexp_allows_explicit_formatter_argument():
    words = ["one", "two", "three"]
    default_pattern = iterable_to_regexp(words)
    explicit_pattern = iterable_to_regexp(words, formatter=PythonFormatter)
    assert explicit_pattern == default_pattern


def test_stream_to_regexp_reads_from_text_stream():
    stream = StringIO("red\nblue\ngreen")
    pattern = stream_to_regexp(stream)
    _assert_matches_all(pattern, ["red", "blue", "green"])
    _assert_does_not_match(pattern, ["re", "blu", "yellow"])


def test_stream_to_regexp_ignores_empty_lines():
    stream = StringIO("apple\n\nbanana\n\ncherry\n")
    pattern = stream_to_regexp(stream)
    _assert_matches_all(pattern, ["apple", "banana", "cherry"])
    _assert_does_not_match(pattern, ["", "apples", "banan"])


def test_iterable_to_regexp_and_stream_to_regexp_agree_on_same_input():
    words = ["sun", "moon", "stars"]
    iterable_pattern = iterable_to_regexp(words)
    stream_pattern = stream_to_regexp(StringIO("\n".join(words)))
    assert iterable_pattern == stream_pattern
