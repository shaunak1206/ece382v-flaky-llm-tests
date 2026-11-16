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
from w2re import utils as w2re_utils


WORDS = ["car", "card", "care", "cat", "dog", "do"]
NON_WORDS = ["ca", "cards", "cars", "d", "dogs", "c", "", "cart", "xcar"]


def _fullmatches(regexp, text):
    return re.fullmatch(regexp, text) is not None


def test_package_metadata():
    assert w2re.__version__ == "3.1.0"
    assert w2re.APPLICATION_NAME == "w2re"
    assert w2re.CHANGELOG_URL.startswith("https://github.com/")
    assert w2re.CHANGELOG_URL.endswith("CHANGELOG.md")


def test_package_reexports_utils_and_formatters():
    assert w2re.iterable_to_regexp is w2re_utils.iterable_to_regexp
    assert w2re.stream_to_regexp is w2re_utils.stream_to_regexp
    assert w2re_utils.PythonFormatter is PythonFormatter


def test_formatters_are_base_formatter_subclasses():
    assert issubclass(PythonFormatter, BaseFormatter)
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_iterable_to_regexp_returns_compilable_string():
    regexp = iterable_to_regexp(WORDS)
    assert isinstance(regexp, str)
    re.compile(regexp)


def test_iterable_to_regexp_matches_all_words():
    regexp = iterable_to_regexp(WORDS)
    for word in WORDS:
        assert _fullmatches(regexp, word), word


def test_iterable_to_regexp_rejects_non_words():
    regexp = iterable_to_regexp(WORDS)
    for text in NON_WORDS:
        assert not _fullmatches(regexp, text), text


def test_iterable_to_regexp_single_word():
    regexp = iterable_to_regexp(["hello"])
    assert _fullmatches(regexp, "hello")
    assert not _fullmatches(regexp, "hell")
    assert not _fullmatches(regexp, "hello!")


def test_iterable_to_regexp_accepts_generator():
    regexp = iterable_to_regexp(word for word in WORDS)
    for word in WORDS:
        assert _fullmatches(regexp, word)
    assert not _fullmatches(regexp, "ca")


def test_iterable_to_regexp_accepts_tuple():
    regexp = iterable_to_regexp(tuple(WORDS))
    for word in WORDS:
        assert _fullmatches(regexp, word)


def test_iterable_to_regexp_input_order_does_not_change_matching():
    forward = iterable_to_regexp(WORDS)
    backward = iterable_to_regexp(list(reversed(WORDS)))
    for text in WORDS + NON_WORDS:
        assert _fullmatches(forward, text) == _fullmatches(backward, text), text


def test_iterable_to_regexp_duplicates_do_not_change_matching():
    regexp = iterable_to_regexp(WORDS + WORDS)
    for word in WORDS:
        assert _fullmatches(regexp, word)
    for text in NON_WORDS:
        assert not _fullmatches(regexp, text)


def test_iterable_to_regexp_is_deterministic():
    assert iterable_to_regexp(WORDS) == iterable_to_regexp(list(WORDS))


def test_default_formatter_is_python_formatter():
    assert iterable_to_regexp(WORDS) == iterable_to_regexp(WORDS, PythonFormatter)


def test_word_match_formatter_matches_words():
    regexp = iterable_to_regexp(WORDS, PythonWordMatchFormatter)
    assert isinstance(regexp, str)
    compiled = re.compile(regexp)
    for word in WORDS:
        assert compiled.fullmatch(word) is not None, word


def test_stream_to_regexp_matches_words_and_skips_empty_lines():
    text = os.linesep.join(["car", "", "card", "dog", ""])
    regexp = stream_to_regexp(io.StringIO(text))
    assert _fullmatches(regexp, "car")
    assert _fullmatches(regexp, "card")
    assert _fullmatches(regexp, "dog")
    assert not _fullmatches(regexp, "ca")
    assert not _fullmatches(regexp, "")
