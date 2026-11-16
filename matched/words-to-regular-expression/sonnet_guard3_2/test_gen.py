import io
import re

import pytest

import w2re
from w2re import (
    APPLICATION_NAME,
    BaseFormatter,
    PythonFormatter,
    PythonWordMatchFormatter,
    iterable_to_regexp,
    stream_to_regexp,
)
from w2re import utils as w2re_utils


def _compile(words, **kwargs):
    return re.compile(iterable_to_regexp(words, **kwargs))


def test_returns_string():
    assert isinstance(iterable_to_regexp(["abc", "abd"]), str)


def test_single_word_matches_only_itself():
    pattern = _compile(["hello"])
    assert pattern.fullmatch("hello")
    assert not pattern.fullmatch("hell")
    assert not pattern.fullmatch("hello!")
    assert not pattern.fullmatch("")


def test_words_with_common_prefix_all_match():
    words = ["car", "cart", "cat", "dog"]
    pattern = _compile(words)
    for word in words:
        assert pattern.fullmatch(word), word


def test_non_listed_words_do_not_match():
    pattern = _compile(["car", "cart", "cat", "dog"])
    for word in ["ca", "c", "carts", "cats", "do", "dogs", "bat", "", "cow"]:
        assert not pattern.fullmatch(word), word


def test_duplicates_in_input_are_harmless():
    pattern = _compile(["apple", "apple", "apply", "apple"])
    assert pattern.fullmatch("apple")
    assert pattern.fullmatch("apply")
    assert not pattern.fullmatch("appl")


def test_accepts_generator_input():
    words = ["red", "green", "blue"]
    pattern = _compile(w for w in words)
    for word in words:
        assert pattern.fullmatch(word)
    assert not pattern.fullmatch("yellow")


def test_accepts_tuple_input():
    pattern = _compile(("one", "two", "three"))
    assert pattern.fullmatch("one")
    assert pattern.fullmatch("three")
    assert not pattern.fullmatch("four")


def test_input_order_does_not_change_matched_words():
    words = ["alpha", "alphabet", "beta", "gamma", "gam"]
    candidates = words + ["alph", "bet", "gamm", "delta", "alphabets", ""]
    forward = _compile(words)
    backward = _compile(list(reversed(words)))
    for candidate in candidates:
        assert bool(forward.fullmatch(candidate)) == bool(backward.fullmatch(candidate))
        assert bool(forward.fullmatch(candidate)) == (candidate in words)


def test_special_regex_characters_are_treated_literally():
    pattern = _compile(["a.b", "c+d"])
    assert pattern.fullmatch("a.b")
    assert pattern.fullmatch("c+d")
    assert not pattern.fullmatch("axb")
    assert not pattern.fullmatch("ccd")
    assert not pattern.fullmatch("cd")


def test_digits_and_unicode_words():
    words = ["123", "12", "café", "cafe"]
    pattern = _compile(words)
    for word in words:
        assert pattern.fullmatch(word), word
    assert not pattern.fullmatch("1")
    assert not pattern.fullmatch("cafés")


def test_explicit_python_formatter_equals_default():
    words = ["tree", "trees", "trie"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words, formatter=PythonFormatter)


def test_same_input_gives_same_output():
    words = ["north", "northern", "south", "east", "west"]
    assert iterable_to_regexp(list(words)) == iterable_to_regexp(list(words))


def test_word_match_formatter_matches_words():
    words = ["cat", "cats", "dog"]
    pattern = _compile(words, formatter=PythonWordMatchFormatter)
    for word in words:
        assert pattern.fullmatch(word), word
    assert not pattern.fullmatch("bird")


def test_formatters_are_base_formatter_subclasses():
    assert issubclass(PythonFormatter, BaseFormatter)
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_package_exports_and_metadata():
    assert w2re.iterable_to_regexp is w2re_utils.iterable_to_regexp
    assert w2re.stream_to_regexp is w2re_utils.stream_to_regexp
    assert w2re.PythonFormatter is PythonFormatter
    assert APPLICATION_NAME == "w2re"
    assert isinstance(w2re.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", w2re.__version__)
    assert w2re.CHANGELOG_URL.startswith("https://github.com/")


def test_stream_to_regexp_matches_lines():
    import os

    stream = io.StringIO(os.linesep.join(["foo", "foobar", "baz"]))
    pattern = re.compile(stream_to_regexp(stream))
    assert pattern.fullmatch("foo")
    assert pattern.fullmatch("foobar")
    assert pattern.fullmatch("baz")
    assert not pattern.fullmatch("fooba")
