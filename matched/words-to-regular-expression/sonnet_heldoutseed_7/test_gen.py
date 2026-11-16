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


def _full(pattern, text):
    return re.fullmatch(pattern, text) is not None


def test_package_exposes_public_names():
    for name in (
        "BaseFormatter",
        "PythonFormatter",
        "PythonWordMatchFormatter",
        "PrefixTree",
        "iterable_to_regexp",
        "stream_to_regexp",
    ):
        assert hasattr(w2re, name)


def test_package_reexports_same_function_objects():
    assert w2re.iterable_to_regexp is utils.iterable_to_regexp
    assert w2re.stream_to_regexp is utils.stream_to_regexp


def test_version_looks_like_semver():
    assert isinstance(w2re.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", w2re.__version__)


def test_application_name_and_changelog_url():
    assert w2re.APPLICATION_NAME == "w2re"
    assert w2re.CHANGELOG_URL.startswith("https://")
    assert "CHANGELOG" in w2re.CHANGELOG_URL


def test_python_formatter_is_base_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_returns_string_that_compiles():
    result = iterable_to_regexp(["apple", "apricot", "banana"])
    assert isinstance(result, str)
    re.compile(result)


def test_all_words_match_and_unrelated_do_not():
    words = ["apple", "apricot", "banana"]
    pattern = iterable_to_regexp(words)
    for word in words:
        assert _full(pattern, word)
    for other in ("cherry", "appl", "bananas", "xapple"):
        assert not _full(pattern, other)


def test_single_word():
    pattern = iterable_to_regexp(["hello"])
    assert _full(pattern, "hello")
    assert not _full(pattern, "hell")
    assert not _full(pattern, "helloo")


def test_words_sharing_prefix_only_listed_variants_match():
    pattern = iterable_to_regexp(["abc", "abd"])
    assert _full(pattern, "abc")
    assert _full(pattern, "abd")
    assert not _full(pattern, "abe")
    assert not _full(pattern, "ab")


def test_word_that_is_prefix_of_another():
    pattern = iterable_to_regexp(["car", "cart", "carpet"])
    for word in ("car", "cart", "carpet"):
        assert _full(pattern, word)
    for other in ("ca", "carp", "carte"):
        assert not _full(pattern, other)


def test_special_regex_characters_are_escaped():
    pattern = iterable_to_regexp(["a.b", "a+b", "(x)"])
    for word in ("a.b", "a+b", "(x)"):
        assert _full(pattern, word)
    assert not _full(pattern, "axb")
    assert not _full(pattern, "aab")
    assert not _full(pattern, "x")


def test_duplicates_and_order_do_not_change_matching():
    words = ["dog", "dot", "door"]
    shuffled = ["door", "dog", "dot", "dog", "door"]
    p1 = iterable_to_regexp(words)
    p2 = iterable_to_regexp(shuffled)
    for word in words + ["do", "dock", "doors"]:
        assert _full(p1, word) == _full(p2, word)


def test_accepts_generator_and_set():
    words = {"red", "green", "blue"}
    pattern_gen = iterable_to_regexp(w for w in sorted(words))
    pattern_set = iterable_to_regexp(words)
    for pattern in (pattern_gen, pattern_set):
        for word in words:
            assert _full(pattern, word)
        assert not _full(pattern, "yellow")


def test_unicode_words():
    pattern = iterable_to_regexp(["café", "cafe", "naïve"])
    for word in ("café", "cafe", "naïve"):
        assert _full(pattern, word)
    assert not _full(pattern, "naive")


def test_default_formatter_is_python_formatter():
    words = ["alpha", "alps", "beta"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words, PythonFormatter)


def test_word_match_formatter_matches_words():
    words = ["one", "only", "two"]
    pattern = iterable_to_regexp(words, PythonWordMatchFormatter)
    compiled = re.compile(pattern)
    for word in words:
        assert compiled.search("say " + word + " now")
    assert not compiled.search("zzz")


def test_stream_to_regexp_matches_iterable_version():
    words = ["cat", "cow", "bat"]
    stream = io.StringIO(os.linesep.join(words))
    from_stream = stream_to_regexp(stream)
    from_iter = iterable_to_regexp(words)
    for word in words + ["ca", "bow"]:
        assert _full(from_stream, word) == _full(from_iter, word)
