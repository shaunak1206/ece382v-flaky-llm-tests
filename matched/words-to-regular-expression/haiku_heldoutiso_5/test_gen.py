import re

import pytest

import w2re
from w2re import (
    APPLICATION_NAME,
    CHANGELOG_URL,
    BaseFormatter,
    PrefixTree,
    PythonFormatter,
    PythonWordMatchFormatter,
    __version__,
    iterable_to_regexp,
    stream_to_regexp,
)
from w2re import utils
from w2re.prefix_tree.tree import PrefixTree as TreePrefixTree


def _assert_matches_exactly(pattern, words, non_words):
    compiled = re.compile(pattern)
    for word in words:
        assert compiled.fullmatch(word) is not None, f"{word!r} should match {pattern!r}"
    for non_word in non_words:
        assert compiled.fullmatch(non_word) is None, f"{non_word!r} should not match {pattern!r}"


# --- iterable_to_regexp ---

def test_iterable_to_regexp_returns_str():
    assert isinstance(iterable_to_regexp(["cat", "car"]), str)


@pytest.mark.parametrize(
    "words",
    [
        ["cat", "car", "cart"],
        ["apple", "apply", "app"],
        ["dog", "dogs", "dogma"],
    ],
)
def test_iterable_to_regexp_matches_all_input_words(words):
    pattern = iterable_to_regexp(words)
    for word in words:
        assert re.fullmatch(pattern, word) is not None


@pytest.mark.parametrize(
    "words, non_words",
    [
        (["abc", "abd"], ["ab", "abcd", "abe", "", "xyz"]),
        (["color", "colour"], ["colo", "colourful", "cold"]),
    ],
)
def test_iterable_to_regexp_rejects_non_words(words, non_words):
    _assert_matches_exactly(iterable_to_regexp(words), words, non_words)


def test_iterable_to_regexp_output_compiles():
    pattern = iterable_to_regexp(["one", "two", "three"])
    re.compile(pattern)


def test_iterable_to_regexp_default_formatter_is_python_formatter():
    words = ["hello", "help", "world"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words, PythonFormatter)


def test_iterable_to_regexp_accepts_generator():
    words = ["red", "green", "blue"]
    pattern = iterable_to_regexp(word for word in words)
    _assert_matches_exactly(pattern, words, ["re", "gree", "bluee", "yellow"])


def test_iterable_to_regexp_accepts_tuple_and_set():
    words = ["north", "south", "east", "west"]
    from_tuple = iterable_to_regexp(tuple(words))
    from_set = iterable_to_regexp(set(words))
    _assert_matches_exactly(from_tuple, words, ["norths", "up"])
    _assert_matches_exactly(from_set, words, ["norths", "up"])


def test_iterable_to_regexp_duplicates_do_not_change_result():
    assert iterable_to_regexp(["cat", "cat", "dog"]) == iterable_to_regexp(["cat", "dog"])


def test_iterable_to_regexp_single_word():
    pattern = iterable_to_regexp(["solo"])
    _assert_matches_exactly(pattern, ["solo"], ["sol", "solos", "so"])


def test_iterable_to_regexp_escapes_regex_metacharacters():
    words = ["a.b", "a+b", "(x)"]
    _assert_matches_exactly(iterable_to_regexp(words), words, ["axb", "aab", "x", "ab"])


def test_iterable_to_regexp_handles_unicode_words():
    words = ["žluť", "kůň"]
    _assert_matches_exactly(iterable_to_regexp(words), words, ["zluť", "kun"])


def test_iterable_to_regexp_is_deterministic():
    words = ["alpha", "alphabet", "beta"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words)


# --- PythonFormatter ---

def test_python_formatter_is_base_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)


def test_python_formatter_is_the_one_used_in_utils():
    assert utils.PythonFormatter is PythonFormatter


def test_python_word_match_formatter_is_base_formatter_subclass():
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_python_formatter_output_is_usable_with_re():
    words = ["tree", "trees", "trie"]
    pattern = iterable_to_regexp(words, PythonFormatter)
    _assert_matches_exactly(pattern, words, ["tre", "treess", "tr"])


# --- w2re/__init__.py ---

def test_package_version():
    assert __version__ == "3.1.0"
    assert w2re.__version__ == "3.1.0"


def test_application_name():
    assert APPLICATION_NAME == "w2re"


def test_changelog_url_format():
    assert CHANGELOG_URL.startswith("https://")
    assert CHANGELOG_URL.endswith("CHANGELOG.md")


def test_package_exports_match_submodule_objects():
    assert iterable_to_regexp is utils.iterable_to_regexp
    assert stream_to_regexp is utils.stream_to_regexp
    assert PrefixTree is TreePrefixTree
    assert PythonFormatter is w2re.formatters.PythonFormatter
    assert BaseFormatter is w2re.formatters.BaseFormatter
    assert PythonWordMatchFormatter is w2re.formatters.PythonWordMatchFormatter
