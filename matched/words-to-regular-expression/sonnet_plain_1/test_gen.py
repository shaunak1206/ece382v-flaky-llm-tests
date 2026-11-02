import re

import pytest

import w2re
from w2re import (
    BaseFormatter,
    PrefixTree,
    PythonFormatter,
    PythonWordMatchFormatter,
    iterable_to_regexp,
)
from w2re.prefix_tree.primitives import PrefixTreeNode


def _matches(pattern, text):
    return re.fullmatch(pattern, text) is not None


def test_package_exports_and_metadata():
    assert w2re.APPLICATION_NAME == 'w2re'
    assert w2re.__version__ == '3.1.0'
    assert w2re.CHANGELOG_URL.startswith('https://github.com/')
    assert w2re.iterable_to_regexp is iterable_to_regexp
    assert callable(w2re.stream_to_regexp)
    assert w2re.PrefixTree is PrefixTree


def test_python_formatter_metadata():
    assert PythonFormatter.code() == 'py'
    assert PythonFormatter.description() == 'Python regular expression'
    assert issubclass(PythonFormatter, BaseFormatter)


def test_base_formatter_is_abstract():
    with pytest.raises(NotImplementedError):
        BaseFormatter.wrap_regexp(PrefixTreeNode())
    with pytest.raises(AssertionError):
        BaseFormatter.code()
    with pytest.raises(AssertionError):
        BaseFormatter.description()


def test_empty_iterable_matches_only_empty_string():
    pattern = iterable_to_regexp([])
    assert pattern == r"\A\Z"
    assert _matches(pattern, '')
    assert not _matches(pattern, 'a')


def test_iterable_with_only_empty_string_matches_empty_string():
    pattern = iterable_to_regexp([''])
    assert pattern == r"\A\Z"
    assert _matches(pattern, '')
    assert not _matches(pattern, 'x')


def test_python_formatter_wrap_regexp_empty_tree():
    assert PythonFormatter.wrap_regexp(PrefixTreeNode()) == r"\A\Z"


def test_python_formatter_wrap_regexp_non_empty_tree():
    tree = PrefixTree(['cat'])
    assert tree.to_regexp(PythonFormatter) == 'cat'


def test_single_word_is_returned_literally():
    assert iterable_to_regexp(['cat']) == 'cat'


def test_single_character_word():
    assert iterable_to_regexp(['a']) == 'a'


def test_common_prefix_is_factored_out():
    pattern = iterable_to_regexp(['cat', 'car'])
    assert pattern == 'ca[rt]'
    assert _matches(pattern, 'cat')
    assert _matches(pattern, 'car')
    assert not _matches(pattern, 'ca')
    assert not _matches(pattern, 'cab')


def test_word_that_is_prefix_of_another_becomes_optional_suffix():
    assert iterable_to_regexp(['a', 'ab']) == 'ab?'
    assert iterable_to_regexp(['cat', 'cats']) == 'cats?'
    # insertion order must not matter
    assert iterable_to_regexp(['cats', 'cat']) == 'cats?'


def test_consecutive_letters_are_collapsed_into_range():
    assert iterable_to_regexp(['a', 'b', 'c']) == '[a-c]'
    assert iterable_to_regexp(['a1', 'a2', 'a3']) == 'a[1-3]'


def test_alternation_of_unrelated_words():
    pattern = iterable_to_regexp(['cat', 'dog'])
    assert pattern == '(?:cat|dog)'
    assert _matches(pattern, 'cat')
    assert _matches(pattern, 'dog')
    assert not _matches(pattern, 'catdog')
    assert not _matches(pattern, '')


def test_duplicates_do_not_change_result():
    assert iterable_to_regexp(['cat', 'cat', 'cat']) == iterable_to_regexp(['cat'])


def test_accepts_generator_and_tuple_input():
    from_generator = iterable_to_regexp(w for w in ('cat', 'car'))
    from_tuple = iterable_to_regexp(('cat', 'car'))
    assert from_generator == from_tuple == 'ca[rt]'


def test_special_regex_characters_are_escaped():
    words = ['a.b', 'a+b', 'a*b', '(x)', '[y]', 'a|b', '$1', '^z', 'q?', 'back\\slash']
    pattern = iterable_to_regexp(words)
    for word in words:
        assert _matches(pattern, word), word
    for non_word in ['axb', 'aab', 'x', 'y', 'ab', 'z', 'q', 'back', 'backslash']:
        assert not _matches(pattern, non_word), non_word


def test_single_special_characters_match_literally():
    pattern = iterable_to_regexp(['.'])
    assert _matches(pattern, '.')
    assert not _matches(pattern, 'x')

    pattern = iterable_to_regexp(['.', 'x', ']', '\\', '-'])
    for char in ['.', 'x', ']', '\\', '-']:
        assert _matches(pattern, char), char
    assert not _matches(pattern, 'y')
    assert not _matches(pattern, '[')


def test_repeated_substrings_are_compressed_but_still_match_exactly():
    pattern = iterable_to_regexp(['aaaa'])
    assert len(pattern) > len('aaaa') or '{' in pattern
    assert '{' in pattern
    assert _matches(pattern, 'aaaa')
    assert not _matches(pattern, 'aaa')
    assert not _matches(pattern, 'aaaaa')

    pattern = iterable_to_regexp(['banana'])
    assert pattern == 'b(?:an){2}a'
    assert _matches(pattern, 'banana')
    assert not _matches(pattern, 'bana')


def test_round_trip_for_mixed_word_list():
    words = [
        'apple', 'apply', 'ape', 'apt', 'april', 'banana', 'band', 'bandana',
        'can', 'candy', 'cane', 'x', 'xy', 'xyz', 'hello world', 'hello', 'a1b2',
        'Zebra', 'zebra', '123', '12', '1',
    ]
    pattern = iterable_to_regexp(words)
    for word in words:
        assert _matches(pattern, word), word
    for non_word in ['', 'app', 'ban', 'ca', 'xz', 'hello w', 'ZEBRA', '1234', 'aple']:
        assert not _matches(pattern, non_word), non_word


def test_unicode_words_round_trip():
    words = ['café', 'cafe', 'über', '你好', '你']
    pattern = iterable_to_regexp(words)
    for word in words:
        assert _matches(pattern, word), word
    assert not _matches(pattern, 'caf')
    assert not _matches(pattern, '好')


def test_explicit_python_formatter_equals_default():
    words = ['one', 'two', 'three', 'tree']
    assert iterable_to_regexp(words, PythonFormatter) == iterable_to_regexp(words)


def test_word_match_formatter_wraps_pattern_with_boundaries():
    pattern = iterable_to_regexp(['cat', 'car'], PythonWordMatchFormatter)
    assert pattern == r"(?:\W+|\A)(ca[rt])(?=\W+|\Z)"
    found = [m.group(1) for m in re.finditer(pattern, 'the cat and car, not cart')]
    assert found == ['cat', 'car']
    assert iterable_to_regexp([], PythonWordMatchFormatter) == r"\A\Z"
