import io
import re
import pytest

from w2re.utils import iterable_to_regexp
from w2re import PythonFormatter
from w2re import iterable_to_regexp as w2re_iterable_to_regexp
from w2re import __version__, APPLICATION_NAME, CHANGELOG_URL


def test_iterable_to_regexp_basic_words():
    words = ['cat', 'car', 'dog']
    regexp = iterable_to_regexp(words)
    # regexp should match all words exactly
    for word in words:
        assert re.fullmatch(regexp, word)
    # and not match unrelated word
    assert not re.fullmatch(regexp, 'cow')


def test_iterable_to_regexp_empty_iterable():
    regexp = iterable_to_regexp([])
    # Should produce something that matches nothing
    pattern = re.compile(regexp)
    assert not pattern.pattern == ''  # should not be empty string regex
    assert not pattern.fullmatch('anything')


def test_iterable_to_regexp_single_word():
    word = ['hello']
    regexp = iterable_to_regexp(word)
    assert re.fullmatch(regexp, 'hello')
    assert not re.fullmatch(regexp, 'hell')


def test_iterable_to_regexp_duplicates():
    words = ['test', 'test', 'testing']
    regexp = iterable_to_regexp(words)
    assert re.fullmatch(regexp, 'test')
    assert re.fullmatch(regexp, 'testing')
    assert not re.fullmatch(regexp, 'tester')


def test_iterable_to_regexp_with_common_prefix():
    words = ['apple', 'applet', 'application']
    regexp = iterable_to_regexp(words)
    for word in words:
        assert re.fullmatch(regexp, word)
    assert not re.fullmatch(regexp, 'app')
    assert not re.fullmatch(regexp, 'apples')


def test_iterable_to_regexp_with_empty_string_in_list():
    words = ['', 'a', 'ab']
    regexp = iterable_to_regexp(words)
    assert re.fullmatch(regexp, '')
    assert re.fullmatch(regexp, 'a')
    assert re.fullmatch(regexp, 'ab')
    assert not re.fullmatch(regexp, 'abc')


def test_iterable_to_regexp_non_ascii_words():
    words = ['naïve', 'café', 'résumé']
    regexp = iterable_to_regexp(words)
    for word in words:
        assert re.fullmatch(regexp, word)
    assert not re.fullmatch(regexp, 'naive')


def test_python_formatter_default_behavior():
    # Confirm that the returned regex is a string and can be compiled
    words = ['foo', 'bar']
    regexp = iterable_to_regexp(words, formatter=PythonFormatter)
    assert isinstance(regexp, str)
    compiled = re.compile(regexp)
    for word in words:
        assert compiled.fullmatch(word)
    assert not compiled.fullmatch('baz')


def test_version_and_constants():
    # Test that imported module constants exist and are correct type
    assert isinstance(__version__, str)
    assert isinstance(APPLICATION_NAME, str)
    assert isinstance(CHANGELOG_URL, str)
    assert __version__.count('.') == 2  # semantic versioning format


# Because PythonFormatter is a class, test that it has expected interface:
def test_python_formatter_interface():
    formatter = PythonFormatter()
    assert hasattr(formatter, "format")
    # format() is expected to be a method; can't do full test without source here


# Additional test using a more complex input to iterable_to_regexp
def test_iterable_to_regexp_large_input():
    words = [f'word{i}' for i in range(100)]
    regexp = iterable_to_regexp(words)
    compiled = re.compile(regexp)
    for i in range(100):
        assert compiled.fullmatch(f'word{i}')
    assert not compiled.fullmatch('word100')


# Test behavior with an iterable that is not a list (e.g., generator)
def test_iterable_to_regexp_generator_input():
    words = (f"gen{i}" for i in range(5))
    regexp = iterable_to_regexp(words)
    compiled = re.compile(regexp)
    for i in range(5):
        assert compiled.fullmatch(f"gen{i}")
    assert not compiled.fullmatch('gen5')


# Misc: test that iterable_to_regexp is imported and callable from w2re package
def test_iterable_to_regexp_imported_from_w2re():
    regexp = w2re_iterable_to_regexp(['a', 'b'])
    compiled = re.compile(regexp)
    assert compiled.fullmatch('a')
    assert compiled.fullmatch('b')


@pytest.mark.parametrize("words,expected_matches,expected_not_matches", [
    (['a', 'ab', 'abc'], ['a', 'ab', 'abc'], ['']),
    (['a', '', 'c'], ['a', '', 'c'], ['b']),
    (['123', '124', '456'], ['123', '124', '456'], ['12', '457']),
])
def test_iterable_to_regexp_parametrized(words, expected_matches, expected_not_matches):
    regexp = iterable_to_regexp(words)
    compiled = re.compile(regexp)
    for word in expected_matches:
        assert compiled.fullmatch(word)
    for word in expected_not_matches:
        assert not compiled.fullmatch(word)
