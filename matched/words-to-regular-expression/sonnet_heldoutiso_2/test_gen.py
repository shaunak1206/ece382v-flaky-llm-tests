import io
import os
import re
from typing import Iterable, List

import pytest

import w2re
from w2re import (
    BaseFormatter,
    PythonFormatter,
    PythonWordMatchFormatter,
    iterable_to_regexp,
    stream_to_regexp,
)
from w2re.utils import iterable_to_regexp as utils_iterable_to_regexp


def _matches(regexp: str, text: str) -> bool:
    return re.fullmatch(regexp, text) is not None


def _assert_matches_exactly(regexp: str, words: Iterable[str], non_words: Iterable[str] = ()):
    for word in words:
        assert _matches(regexp, word), f"{word!r} should match {regexp!r}"
    for word in non_words:
        assert not _matches(regexp, word), f"{word!r} should not match {regexp!r}"


def test_package_exports_same_function_as_utils():
    assert w2re.iterable_to_regexp is utils_iterable_to_regexp


def test_package_metadata_constants():
    assert w2re.__version__ == "3.1.0"
    assert w2re.APPLICATION_NAME == "w2re"
    assert w2re.CHANGELOG_URL.startswith("https://github.com/")
    assert w2re.CHANGELOG_URL.endswith("CHANGELOG.md")


def test_formatters_are_base_formatter_subclasses():
    assert issubclass(PythonFormatter, BaseFormatter)
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_returns_compilable_string():
    regexp = iterable_to_regexp(["abc", "abd", "xyz"])
    assert isinstance(regexp, str)
    re.compile(regexp)


def test_single_word_matches_only_itself():
    regexp = iterable_to_regexp(["hello"])
    _assert_matches_exactly(regexp, ["hello"], ["hell", "hello!", "ello", "", "Hello"])


def test_multiple_words_without_common_prefix():
    regexp = iterable_to_regexp(["apple", "banana", "cherry"])
    _assert_matches_exactly(
        regexp,
        ["apple", "banana", "cherry"],
        ["app", "bananas", "cherr", "applebanana", ""],
    )


def test_words_with_common_prefix():
    regexp = iterable_to_regexp(["car", "cart", "cat", "dog"])
    _assert_matches_exactly(
        regexp,
        ["car", "cart", "cat", "dog"],
        ["ca", "c", "carts", "do", "cats", "cog"],
    )


def test_word_that_is_prefix_of_another_word():
    regexp = iterable_to_regexp(["test", "testing", "tester"])
    _assert_matches_exactly(
        regexp,
        ["test", "testing", "tester"],
        ["tes", "testin", "testers", "testingg"],
    )


def test_duplicates_do_not_change_matched_language():
    with_duplicates = iterable_to_regexp(["a", "ab", "a", "ab", "abc"])
    _assert_matches_exactly(with_duplicates, ["a", "ab", "abc"], ["", "b", "abcd", "abb"])


def test_input_order_does_not_change_matched_language():
    words = ["alpha", "alphabet", "beta", "bet", "gamma"]
    probes = words + ["alph", "be", "gam", "alphabets", "", "betaa"]
    forward = iterable_to_regexp(words)
    backward = iterable_to_regexp(list(reversed(words)))
    for probe in probes:
        assert _matches(forward, probe) == _matches(backward, probe)
    _assert_matches_exactly(forward, words)


def test_accepts_generator_and_tuple_inputs():
    words: List[str] = ["one", "two", "three"]
    from_generator = iterable_to_regexp(w for w in words)
    from_tuple = iterable_to_regexp(tuple(words))
    for regexp in (from_generator, from_tuple):
        _assert_matches_exactly(regexp, words, ["on", "tw", "threee"])


def test_special_regex_characters_are_escaped():
    regexp = iterable_to_regexp(["a.b", "c+d", "(x)"])
    _assert_matches_exactly(regexp, ["a.b", "c+d", "(x)"], ["axb", "ccd", "cd", "x"])


def test_unicode_words():
    words = ["příliš", "žluťoučký", "kůň"]
    regexp = iterable_to_regexp(words)
    _assert_matches_exactly(regexp, words, ["pril", "kun"])


def test_default_formatter_equals_explicit_python_formatter():
    words = ["red", "green", "greed", "blue"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words, PythonFormatter)


def test_word_match_formatter_matches_words():
    words = ["cat", "car", "dog"]
    regexp = iterable_to_regexp(words, PythonWordMatchFormatter)
    assert isinstance(regexp, str)
    compiled = re.compile(regexp)
    for word in words:
        assert compiled.fullmatch(word) is not None


def test_stream_to_regexp_matches_same_language_as_iterable():
    words = ["foo", "foobar", "baz"]
    stream = io.StringIO(os.linesep.join(words) + os.linesep)
    from_stream = stream_to_regexp(stream)
    from_iterable = iterable_to_regexp(words)
    probes = words + ["fo", "foob", "ba", "bazz", ""]
    for probe in probes:
        assert _matches(from_stream, probe) == _matches(from_iterable, probe)
