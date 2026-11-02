import io
import re
import pytest

from w2re import (
    iterable_to_regexp,
    PythonFormatter,
    BaseFormatter,
    PrefixTree,
)

# Test data sets used by multiple tests
SIMPLE_WORDS = ['foo', 'bar', 'baz']
OVERLAPPING_WORDS = ['abc', 'abd', 'abe']
SINGLE_WORD = ['hello']
EMPTY_LIST = []
SPECIAL_CHAR_WORDS = ['a.c', 'a*c', 'a+c', 'a?c', 'a|c']
WORDS_WITH_PREFIXES = ['test', 'testing', 'tester']


def test_iterable_to_regexp_returns_string():
    result = iterable_to_regexp(SIMPLE_WORDS)
    assert isinstance(result, str)
    assert len(result) > 0


def test_iterable_to_regexp_matches_single_word_exactly():
    regex = iterable_to_regexp(SINGLE_WORD)
    pattern = re.compile(regex)
    assert pattern.fullmatch('hello')
    assert not pattern.fullmatch('hell')


def test_iterable_to_regexp_matches_all_words():
    regex = iterable_to_regexp(SIMPLE_WORDS)
    pattern = re.compile(regex)
    # all words must match fully
    for w in SIMPLE_WORDS:
        assert pattern.fullmatch(w)


def test_iterable_to_regexp_excludes_non_words():
    regex = iterable_to_regexp(SIMPLE_WORDS)
    pattern = re.compile(regex)
    # add some strings that are not in iterable
    assert not pattern.fullmatch('fo')
    assert not pattern.fullmatch('food')
    assert not pattern.fullmatch('bars')


def test_iterable_to_regexp_empty_iterable_returns_empty_regexp():
    # An empty PrefixTree.to_regexp probably returns a string matching nothing
    regex = iterable_to_regexp(EMPTY_LIST)
    assert isinstance(regex, str)
    # The regex should not match anything including empty string
    pattern = re.compile(regex)
    assert not pattern.fullmatch('')
    assert not pattern.fullmatch('anything')


def test_iterable_to_regexp_with_overlapping_prefixes():
    regex = iterable_to_regexp(OVERLAPPING_WORDS)
    pattern = re.compile(regex)
    for w in OVERLAPPING_WORDS:
        assert pattern.fullmatch(w)
    # words similar but not in list should not match
    assert not pattern.fullmatch('abf')
    assert not pattern.fullmatch('ab')


def test_iterable_to_regexp_with_special_characters():
    regex = iterable_to_regexp(SPECIAL_CHAR_WORDS)
    pattern = re.compile(regex)
    for w in SPECIAL_CHAR_WORDS:
        assert pattern.fullmatch(w)
    assert not pattern.fullmatch('a.cd')


def test_iterable_to_regexp_with_prefix_words():
    regex = iterable_to_regexp(WORDS_WITH_PREFIXES)
    pattern = re.compile(regex)
    for w in WORDS_WITH_PREFIXES:
        assert pattern.fullmatch(w)
    # partial prefix should not match
    assert not pattern.fullmatch('tes')
    assert not pattern.fullmatch('testingg')


def test_python_formatter_default_repr_no_modification():
    # PythonFormatter should escape regex properly and return string repr
    pf = PythonFormatter()
    input_regex = r'(foo|bar)'
    formatted = pf.format(input_regex)
    # It should at least contain the original regex substring
    assert input_regex in formatted
    # The formatted string should be a valid Python string literal, test eval safety by roundtrip
    # The formatted string is expected to be a string literal like "'(foo|bar)'"
    # Remove quotes and check that it matches original (eval could be unsafe though, just check quotes presence)
    assert formatted.startswith(("'", '"'))
    assert formatted.endswith(("'", '"'))


def test_python_formatter_inherits_baseformatter():
    assert issubclass(PythonFormatter, BaseFormatter)
    pf = PythonFormatter()
    assert hasattr(pf, 'format')
    # format should accept a string and return a string
    result = pf.format('test')
    assert isinstance(result, str)


def test_iterable_to_regexp_with_custom_formatter_class(monkeypatch):
    # Create a dummy formatter that always returns fixed string
    class DummyFormatter(BaseFormatter):
        def format(self, regex: str) -> str:
            return 'FIXED_REGEX'

    dummy_formatter = DummyFormatter
    result = iterable_to_regexp(SIMPLE_WORDS, formatter=dummy_formatter)
    assert result == 'FIXED_REGEX'


def test_iterable_to_regexp_returns_same_as_prefix_tree_to_regexp_direct():
    # Ensure iterable_to_regexp delegates to PrefixTree.to_regexp with the same arguments
    prefix_tree = PrefixTree(SIMPLE_WORDS)
    expected = prefix_tree.to_regexp()
    actual = iterable_to_regexp(SIMPLE_WORDS)
    assert actual == expected


def test_prefix_tree_to_regexp_output_matches_all_words():
    # A direct test of PrefixTree.to_regexp via iterable_to_regexp to guarantee correctness
    words = ['apple', 'app', 'application', 'apt']
    regex = iterable_to_regexp(words)
    compiled = re.compile(regex)
    for w in words:
        assert compiled.fullmatch(w)
    # Test some words not in list do not match
    assert not compiled.fullmatch('appl')
    assert not compiled.fullmatch('appliance')


def test_iterable_to_regexp_is_stable_and_deterministic():
    # Call it multiple times and verify outputs are identical
    words = ['one', 'two', 'three', 'four']
    result1 = iterable_to_regexp(words)
    result2 = iterable_to_regexp(words)
    result3 = iterable_to_regexp(words)
    assert result1 == result2 == result3


def test_iterable_to_regexp_handles_words_with_unicode(monkeypatch):
    words = ['café', 'naïve', 'niño', 'résumé']
    regex = iterable_to_regexp(words)
    compiled = re.compile(regex)
    for w in words:
        assert compiled.fullmatch(w)
    # Non-words should not match
    assert not compiled.fullmatch('cafe')
    assert not compiled.fullmatch('resume')


def test_iterable_to_regexp_does_not_modify_input_iterable():
    # Input iterable should remain unchanged after call
    words = ['alpha', 'beta', 'gamma']
    original_words = words.copy()
    _ = iterable_to_regexp(words)
    assert words == original_words


def test_python_formatter_formatting_string_literal_properties():
    pf = PythonFormatter()
    raw = 'foo\\bar\n'
    formatted = pf.format(raw)
    # It should be a quoted string containing the raw string content escaped
    # Remove quotes and decode escapes, then compare to raw
    assert formatted[0] in ("'", '"')
    assert formatted[-1] == formatted[0]
    inner = formatted[1:-1]
    # Inner should contain escaped newlines backslash-n or raw \n
    assert '\\n' in inner or '\n' not in inner


@pytest.mark.parametrize(
    "words",
    [
        ['a', 'ab', 'abc'],
        ['x', 'y', 'z'],
        ['one', 'two', 'three', 'four', 'five'],
        ['123', '124', '125'],
    ]
)
def test_iterable_to_regexp_parametrized(words):
    regex = iterable_to_regexp(words)
    compiled = re.compile(regex)
    for w in words:
        assert compiled.fullmatch(w)
    assert not compiled.fullmatch('')
    # Some strings that are not in words must not match
    not_in_words = ['a' * 10, '0', 'six']
    for w in not_in_words:
        assert not compiled.fullmatch(w)
