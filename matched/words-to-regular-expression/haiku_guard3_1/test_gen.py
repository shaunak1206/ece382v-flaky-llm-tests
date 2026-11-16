import io
import os
import re

import pytest

import w2re
from w2re import (
    BaseFormatter,
    PythonFormatter,
    PythonWordMatchFormatter,
    PrefixTree,
    __version__,
    APPLICATION_NAME,
    CHANGELOG_URL,
    iterable_to_regexp,
    stream_to_regexp,
)
from w2re import utils


def _assert_matches_exactly(pattern, words, non_words=()):
    """Every word fully matches the pattern, and no non-word does."""
    compiled = re.compile(pattern)
    for word in words:
        assert compiled.fullmatch(word) is not None, "{!r} should match".format(word)
    for non_word in non_words:
        assert compiled.fullmatch(non_word) is None, "{!r} should not match".format(non_word)


WORDS = ["hello", "help", "helium", "world"]


def test_version_constant():
    assert __version__ == "3.1.0"


def test_application_name_constant():
    assert APPLICATION_NAME == "w2re"


def test_changelog_url_is_github_changelog():
    assert CHANGELOG_URL.startswith("https://github.com/")
    assert CHANGELOG_URL.endswith("CHANGELOG.md")


def test_package_exports_iterable_to_regexp_from_utils():
    assert w2re.iterable_to_regexp is utils.iterable_to_regexp


def test_package_exports_stream_to_regexp_from_utils():
    assert w2re.stream_to_regexp is utils.stream_to_regexp


def test_package_exports_formatter_classes():
    for cls in (BaseFormatter, PythonFormatter, PythonWordMatchFormatter):
        assert isinstance(cls, type)


def test_python_formatter_is_base_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)


def test_python_formatter_is_default_formatter_for_iterable_to_regexp():
    assert iterable_to_regexp(WORDS) == iterable_to_regexp(WORDS, PythonFormatter)


def test_iterable_to_regexp_returns_str():
    result = iterable_to_regexp(WORDS)
    assert isinstance(result, str)
    assert result != ""


def test_iterable_to_regexp_result_compiles_and_matches_all_words():
    pattern = iterable_to_regexp(WORDS)
    re.compile(pattern)
    _assert_matches_exactly(pattern, WORDS, non_words=["hel", "wor", "helloo", "x"])


def test_iterable_to_regexp_single_word():
    pattern = iterable_to_regexp(["python"])
    _assert_matches_exactly(pattern, ["python"], non_words=["pytho", "pythons", ""])


def test_iterable_to_regexp_shared_prefixes_do_not_leak_to_prefix_matches():
    pattern = iterable_to_regexp(["hell", "hello"])
    _assert_matches_exactly(pattern, ["hell", "hello"], non_words=["hel", "helli"])


def test_iterable_to_regexp_escapes_special_characters():
    words = ["a.b", "c+d"]
    pattern = iterable_to_regexp(words)
    _assert_matches_exactly(pattern, words, non_words=["axb", "cd", "ccd"])


def test_iterable_to_regexp_accepts_generator():
    generator = (word for word in WORDS)
    pattern = iterable_to_regexp(generator)
    _assert_matches_exactly(pattern, WORDS)


def test_iterable_to_regexp_accepts_frozenset_and_duplicates():
    pattern = iterable_to_regexp(["help", "help", "hello"])
    _assert_matches_exactly(pattern, ["help", "hello"], non_words=["hel"])
    pattern_from_set = iterable_to_regexp(frozenset(["help", "hello"]))
    _assert_matches_exactly(pattern_from_set, ["help", "hello"], non_words=["hel"])


def test_iterable_to_regexp_explicit_python_formatter_keyword():
    pattern = iterable_to_regexp(WORDS, formatter=PythonFormatter)
    _assert_matches_exactly(pattern, WORDS)


def test_prefix_tree_to_regexp_agrees_with_iterable_to_regexp():
    pattern = PrefixTree(WORDS).to_regexp(PythonFormatter)
    _assert_matches_exactly(pattern, WORDS, non_words=["hel"])


def test_python_word_match_formatter_matches_each_word():
    pattern = iterable_to_regexp(WORDS, PythonWordMatchFormatter)
    re.compile(pattern)
    for word in WORDS:
        assert re.search(pattern, word) is not None, "{!r} should match".format(word)


def test_stream_to_regexp_matches_lines_of_stream():
    text = os.linesep.join(WORDS) + os.linesep
    pattern = stream_to_regexp(io.StringIO(text))
    _assert_matches_exactly(pattern, WORDS, non_words=["hel", ""])


def test_stream_to_regexp_ignores_empty_lines():
    text = os.linesep.join(["hello", "", "help", ""])
    pattern = stream_to_regexp(io.StringIO(text))
    _assert_matches_exactly(pattern, ["hello", "help"], non_words=[""])


def test_stream_to_regexp_uses_os_linesep_at_call_time(monkeypatch):
    monkeypatch.setattr(os, "linesep", "|")
    pattern = stream_to_regexp(io.StringIO("hello|help|"))
    _assert_matches_exactly(pattern, ["hello", "help"], non_words=["hello|help"])
