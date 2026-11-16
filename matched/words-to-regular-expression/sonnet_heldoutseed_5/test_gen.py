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
    iterable_to_regexp,
    stream_to_regexp,
)
from w2re import utils


def _full(regexp, text):
    return re.fullmatch(regexp, text) is not None


def test_package_exports():
    assert w2re.iterable_to_regexp is utils.iterable_to_regexp
    assert w2re.stream_to_regexp is utils.stream_to_regexp
    assert w2re.PrefixTree is PrefixTree


def test_package_metadata():
    assert w2re.APPLICATION_NAME == 'w2re'
    assert re.fullmatch(r'\d+\.\d+\.\d+', w2re.__version__)
    assert w2re.CHANGELOG_URL.startswith('https://')


def test_formatters_inheritance():
    assert issubclass(PythonFormatter, BaseFormatter)
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_utils_uses_python_formatter_as_default():
    assert utils.PythonFormatter is PythonFormatter


def test_returns_string_and_valid_regexp():
    result = iterable_to_regexp(['cat', 'dog'])
    assert isinstance(result, str)
    re.compile(result)


def test_single_word():
    regexp = iterable_to_regexp(['hello'])
    assert _full(regexp, 'hello')
    assert not _full(regexp, 'hell')
    assert not _full(regexp, 'hello!')


def test_multiple_words_match():
    words = ['cat', 'dog', 'bird']
    regexp = iterable_to_regexp(words)
    for word in words:
        assert _full(regexp, word)


def test_non_listed_words_do_not_match():
    regexp = iterable_to_regexp(['cat', 'dog'])
    for text in ['ca', 'cats', 'do', 'bird', 'cd', '']:
        assert not _full(regexp, text)


def test_words_sharing_prefixes():
    words = ['car', 'cart', 'cartoon', 'care', 'cat']
    regexp = iterable_to_regexp(words)
    for word in words:
        assert _full(regexp, word)
    for text in ['ca', 'carto', 'cartoons', 'c', 'cartoo']:
        assert not _full(regexp, text)


def test_word_that_is_prefix_of_another():
    regexp = iterable_to_regexp(['a', 'ab', 'abc'])
    for word in ['a', 'ab', 'abc']:
        assert _full(regexp, word)
    assert not _full(regexp, 'abcd')
    assert not _full(regexp, 'b')


def test_accepts_generator_and_tuple():
    words = ['one', 'two', 'three']
    for iterable in (tuple(words), (w for w in words)):
        regexp = iterable_to_regexp(iterable)
        for word in words:
            assert _full(regexp, word)


def test_duplicates_and_order_do_not_matter():
    words = ['alpha', 'beta', 'alpha', 'gamma', 'beta']
    regexp = iterable_to_regexp(words)
    reversed_regexp = iterable_to_regexp(list(reversed(words)))
    for word in set(words):
        assert _full(regexp, word)
        assert _full(reversed_regexp, word)


def test_special_characters_are_escaped():
    regexp = iterable_to_regexp(['a.b', 'c+d', '(e)'])
    for word in ['a.b', 'c+d', '(e)']:
        assert _full(regexp, word)
    assert not _full(regexp, 'axb')
    assert not _full(regexp, 'ccd')


def test_unicode_words():
    words = ['žluťoučký', 'kůň', 'häuser']
    regexp = iterable_to_regexp(words)
    for word in words:
        assert _full(regexp, word)


def test_explicit_python_formatter_matches_default():
    words = ['red', 'green', 'blue']
    assert iterable_to_regexp(words, PythonFormatter) == iterable_to_regexp(words)


def test_word_match_formatter_matches_words():
    words = ['red', 'green', 'blue']
    regexp = iterable_to_regexp(words, PythonWordMatchFormatter)
    re.compile(regexp)
    for word in words:
        assert re.search(regexp, word) is not None


def test_stream_to_regexp_matches_iterable_version():
    words = ['apple', 'apricot', 'banana']
    stream = io.StringIO(os.linesep.join(words) + os.linesep)
    regexp = stream_to_regexp(stream)
    for word in words:
        assert _full(regexp, word)
    assert not _full(regexp, 'app')
