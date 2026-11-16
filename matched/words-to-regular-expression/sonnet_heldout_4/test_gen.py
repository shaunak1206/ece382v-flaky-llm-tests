import inspect
import io
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


def _assert_matches_exactly(pattern, words, non_words=()):
    compiled = re.compile(pattern)
    for word in words:
        assert compiled.fullmatch(word), f"{word!r} should match {pattern!r}"
    for word in non_words:
        assert not compiled.fullmatch(word), f"{word!r} should not match {pattern!r}"


def test_package_metadata():
    assert w2re.__version__ == "3.1.0"
    assert w2re.APPLICATION_NAME == "w2re"
    assert w2re.CHANGELOG_URL.startswith("https://github.com/")
    assert w2re.CHANGELOG_URL.endswith("CHANGELOG.md")


def test_package_exports_expected_names():
    assert issubclass(PythonFormatter, BaseFormatter)
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)
    assert w2re.iterable_to_regexp is utils.iterable_to_regexp
    assert w2re.stream_to_regexp is utils.stream_to_regexp
    assert PrefixTree is not None


def test_utils_uses_package_python_formatter():
    assert utils.PythonFormatter is w2re.PythonFormatter


def test_default_formatter_is_python_formatter():
    signature = inspect.signature(iterable_to_regexp)
    assert signature.parameters["formatter"].default is PythonFormatter


def test_returns_string():
    assert isinstance(iterable_to_regexp(["abc", "abd"]), str)


def test_single_word():
    pattern = iterable_to_regexp(["hello"])
    _assert_matches_exactly(pattern, ["hello"], ["hell", "helloo", "world", ""])


def test_words_with_common_prefix():
    pattern = iterable_to_regexp(["abc", "abd", "abe"])
    _assert_matches_exactly(pattern, ["abc", "abd", "abe"], ["ab", "abf", "abcd"])


def test_words_with_different_first_letters():
    pattern = iterable_to_regexp(["cat", "dog", "bird"])
    _assert_matches_exactly(pattern, ["cat", "dog", "bird"], ["ca", "do", "birds", "cow"])


def test_word_that_is_prefix_of_another():
    pattern = iterable_to_regexp(["car", "cart", "carpet"])
    _assert_matches_exactly(pattern, ["car", "cart", "carpet"], ["ca", "carp", "carts"])


def test_common_suffix_words():
    pattern = iterable_to_regexp(["walking", "talking", "working"])
    _assert_matches_exactly(pattern, ["walking", "talking", "working"], ["king", "walkin"])


def test_duplicates_are_harmless():
    pattern = iterable_to_regexp(["abc", "abc", "abd", "abd"])
    _assert_matches_exactly(pattern, ["abc", "abd"], ["ab", "abcd"])


def test_input_order_does_not_change_matching():
    words = ["apple", "apply", "ape", "banana", "band"]
    forward = iterable_to_regexp(words)
    backward = iterable_to_regexp(list(reversed(words)))
    non_words = ["app", "ban", "bananas", "applee"]
    _assert_matches_exactly(forward, words, non_words)
    _assert_matches_exactly(backward, words, non_words)


def test_accepts_generator_and_set():
    words = ["one", "two", "three"]
    from_generator = iterable_to_regexp(w for w in words)
    from_set = iterable_to_regexp(set(words))
    for pattern in (from_generator, from_set):
        _assert_matches_exactly(pattern, words, ["on", "tw", "threee"])


def test_unicode_words():
    words = ["café", "cafè", "naïve"]
    pattern = iterable_to_regexp(words)
    _assert_matches_exactly(pattern, words, ["cafe", "naive"])


def test_explicit_python_formatter_equals_default():
    words = ["abc", "abd", "xyz"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words, PythonFormatter)


def test_same_input_gives_same_output():
    words = ["alpha", "alpine", "beta", "betray"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words)


def test_word_match_formatter_finds_words_in_text():
    pattern = iterable_to_regexp(["cat", "dog"], PythonWordMatchFormatter)
    compiled = re.compile(pattern)
    assert compiled.search("the cat sat")
    assert compiled.search("a dog barked")
    assert not compiled.search("concatenate")
    assert not compiled.search("hotdogs")


def test_stream_to_regexp_matches_words():
    stream = io.StringIO("cat\ncar\ndog\n")
    pattern = stream_to_regexp(stream)
    _assert_matches_exactly(pattern, ["cat", "car", "dog"], ["ca", "cow", ""])


def test_stream_to_regexp_ignores_empty_lines():
    stream = io.StringIO("\ncat\n\n\ndog\n\n")
    pattern = stream_to_regexp(stream)
    _assert_matches_exactly(pattern, ["cat", "dog"], ["", "ca"])
