import io
import os
import re

import pytest

import w2re
from w2re import (
    BaseFormatter,
    PrefixTree,
    PythonFormatter,
    PythonWordMatchFormatter,
    iterable_to_regexp,
    stream_to_regexp,
)
from w2re import utils


def _assert_matches_exactly(regexp, accepted, rejected):
    compiled = re.compile(regexp)
    for word in accepted:
        assert compiled.fullmatch(word), f"{word!r} should match {regexp!r}"
    for word in rejected:
        assert not compiled.fullmatch(word), f"{word!r} should not match {regexp!r}"


def test_iterable_to_regexp_returns_string():
    assert isinstance(iterable_to_regexp(["cat", "dog"]), str)


def test_iterable_to_regexp_single_word():
    regexp = iterable_to_regexp(["hello"])
    _assert_matches_exactly(regexp, ["hello"], ["hell", "hello!", "world", ""])


def test_iterable_to_regexp_multiple_distinct_words():
    words = ["apple", "banana", "cherry"]
    regexp = iterable_to_regexp(words)
    _assert_matches_exactly(regexp, words, ["app", "bananas", "cherr", "grape"])


def test_iterable_to_regexp_words_sharing_prefix():
    words = ["car", "cart", "carton", "cat"]
    regexp = iterable_to_regexp(words)
    _assert_matches_exactly(regexp, words, ["ca", "cartons", "c", "cars"])


def test_iterable_to_regexp_is_order_independent_semantically():
    words = ["alpha", "alphabet", "beta", "gamma"]
    r1 = iterable_to_regexp(words)
    r2 = iterable_to_regexp(list(reversed(words)))
    rejected = ["alph", "alphabets", "bet", "delta"]
    _assert_matches_exactly(r1, words, rejected)
    _assert_matches_exactly(r2, words, rejected)


def test_iterable_to_regexp_handles_duplicates():
    regexp = iterable_to_regexp(["dup", "dup", "dupe"])
    _assert_matches_exactly(regexp, ["dup", "dupe"], ["du", "dupee"])


def test_iterable_to_regexp_accepts_generator():
    regexp = iterable_to_regexp(w for w in ["one", "two", "three"])
    _assert_matches_exactly(regexp, ["one", "two", "three"], ["on", "four"])


def test_iterable_to_regexp_accepts_tuple_and_set():
    _assert_matches_exactly(iterable_to_regexp(("ab", "ac")), ["ab", "ac"], ["a", "ad"])
    _assert_matches_exactly(iterable_to_regexp({"xy", "xz"}), ["xy", "xz"], ["x", "xw"])


def test_iterable_to_regexp_single_character_words():
    regexp = iterable_to_regexp(["a", "b", "c"])
    _assert_matches_exactly(regexp, ["a", "b", "c"], ["d", "ab", ""])


def test_iterable_to_regexp_escapes_special_characters():
    words = ["a.b", "c+d", "(e)"]
    regexp = iterable_to_regexp(words)
    _assert_matches_exactly(regexp, words, ["axb", "cd", "ccd", "e"])


def test_iterable_to_regexp_unicode_words():
    words = ["café", "naïve", "日本語"]
    regexp = iterable_to_regexp(words)
    _assert_matches_exactly(regexp, words, ["cafe", "日本"])


def test_iterable_to_regexp_default_formatter_is_python_formatter():
    words = ["foo", "foobar", "baz"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words, PythonFormatter)


def test_iterable_to_regexp_is_deterministic():
    words = ["one", "only", "once", "two"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words)


def test_iterable_to_regexp_with_word_match_formatter():
    words = ["cat", "cow"]
    regexp = iterable_to_regexp(words, PythonWordMatchFormatter)
    assert isinstance(regexp, str)
    compiled = re.compile(regexp)
    for word in words:
        assert compiled.fullmatch(word)
    assert not compiled.fullmatch("cot")


def test_python_formatter_is_base_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_python_formatter_used_via_utils_module_matches_package_export():
    assert utils.PythonFormatter is PythonFormatter
    assert utils.iterable_to_regexp is iterable_to_regexp


def test_python_formatter_output_is_valid_compilable_regex():
    regexp = iterable_to_regexp(["abc", "abd", "xyz", "x"], PythonFormatter)
    pattern = re.compile(regexp)
    assert pattern.fullmatch("abc")
    assert pattern.fullmatch("x")
    assert not pattern.fullmatch("ab")


def test_stream_to_regexp_matches_iterable_semantics():
    words = ["red", "green", "blue"]
    stream = io.StringIO(os.linesep.join(words) + os.linesep)
    regexp = stream_to_regexp(stream)
    _assert_matches_exactly(regexp, words, ["re", "gree", "yellow"])


def test_package_metadata_and_exports():
    assert w2re.APPLICATION_NAME == "w2re"
    assert re.fullmatch(r"\d+\.\d+\.\d+", w2re.__version__)
    assert w2re.CHANGELOG_URL.startswith("https://")
    assert w2re.CHANGELOG_URL.endswith("CHANGELOG.md")
    assert w2re.PrefixTree is PrefixTree
    assert callable(w2re.stream_to_regexp)
