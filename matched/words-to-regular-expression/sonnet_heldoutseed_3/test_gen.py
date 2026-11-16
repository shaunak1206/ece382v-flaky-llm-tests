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


def _assert_matches_exactly(regexp, words, non_words=()):
    compiled = re.compile(regexp)
    for word in words:
        assert compiled.fullmatch(word), f"{word!r} should match {regexp!r}"
    for word in non_words:
        assert not compiled.fullmatch(word), f"{word!r} should not match {regexp!r}"


def test_package_metadata_constants():
    assert isinstance(w2re.__version__, str)
    assert w2re.__version__.count(".") == 2
    assert w2re.APPLICATION_NAME == "w2re"
    assert w2re.CHANGELOG_URL.startswith("https://")
    assert w2re.CHANGELOG_URL.endswith("CHANGELOG.md")


def test_package_reexports_public_api():
    assert w2re.iterable_to_regexp is utils.iterable_to_regexp
    assert w2re.stream_to_regexp is utils.stream_to_regexp
    assert w2re.PrefixTree is PrefixTree


def test_formatters_inherit_from_base_formatter():
    assert issubclass(PythonFormatter, BaseFormatter)
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_iterable_to_regexp_returns_string():
    assert isinstance(iterable_to_regexp(["cat", "car"]), str)


def test_iterable_to_regexp_single_word():
    regexp = iterable_to_regexp(["hello"])
    _assert_matches_exactly(regexp, ["hello"], ["hell", "hello!", "", "world"])


def test_iterable_to_regexp_multiple_words_with_common_prefix():
    words = ["cat", "car", "cart", "cap"]
    regexp = iterable_to_regexp(words)
    _assert_matches_exactly(regexp, words, ["ca", "cab", "carts", "dog", "c"])


def test_iterable_to_regexp_words_that_are_prefixes_of_each_other():
    words = ["car", "cart", "cartoon"]
    regexp = iterable_to_regexp(words)
    _assert_matches_exactly(regexp, words, ["ca", "carto", "cartoons", "cars"])


def test_iterable_to_regexp_unrelated_words():
    words = ["apple", "banana", "cherry"]
    regexp = iterable_to_regexp(words)
    _assert_matches_exactly(regexp, words, ["apples", "ban", "cherr", "grape"])


def test_iterable_to_regexp_order_independent():
    words = ["alpha", "alphabet", "beta", "betamax", "gamma"]
    forward = iterable_to_regexp(words)
    backward = iterable_to_regexp(list(reversed(words)))
    non_words = ["alph", "alphabets", "bet", "gam", "delta"]
    _assert_matches_exactly(forward, words, non_words)
    _assert_matches_exactly(backward, words, non_words)


def test_iterable_to_regexp_accepts_generator():
    words = ["red", "green", "blue"]
    regexp = iterable_to_regexp(w for w in words)
    _assert_matches_exactly(regexp, words, ["gre", "yellow"])


def test_iterable_to_regexp_handles_duplicates():
    regexp = iterable_to_regexp(["dup", "dup", "dupe"])
    _assert_matches_exactly(regexp, ["dup", "dupe"], ["du", "dupee"])


def test_iterable_to_regexp_escapes_special_characters():
    words = ["a.b", "c++", "d*e", "(f)", "g|h"]
    regexp = iterable_to_regexp(words)
    _assert_matches_exactly(regexp, words, ["axb", "c", "ddde", "f", "g", "h"])


def test_iterable_to_regexp_unicode_words():
    words = ["čaj", "čas", "kává"]
    regexp = iterable_to_regexp(words)
    _assert_matches_exactly(regexp, words, ["ča", "kava"])


def test_iterable_to_regexp_compresses_common_prefixes():
    words = ["internationalization", "internationalisation"]
    regexp = iterable_to_regexp(words)
    naive = "|".join(re.escape(w) for w in words)
    assert len(regexp) < len(naive)


def test_default_formatter_is_python_formatter():
    words = ["one", "only", "once", "two"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words, PythonFormatter)


def test_python_formatter_output_matches_prefix_tree_to_regexp():
    words = ["tree", "trie", "try"]
    assert iterable_to_regexp(words, PythonFormatter) == PrefixTree(words).to_regexp(
        PythonFormatter
    )


def test_word_match_formatter_finds_whole_words_in_text():
    words = ["cat", "car"]
    regexp = iterable_to_regexp(words, PythonWordMatchFormatter)
    compiled = re.compile(regexp)
    assert compiled.search("I drive a car today")
    assert compiled.search("the cat sat")
    assert not compiled.search("concatenate")


def test_stream_to_regexp_matches_words_and_ignores_empty_lines():
    words = ["foo", "foobar", "baz"]
    stream = io.StringIO(os.linesep.join(words) + os.linesep + os.linesep)
    regexp = stream_to_regexp(stream)
    _assert_matches_exactly(regexp, words, ["fo", "foob", "ba", ""])
