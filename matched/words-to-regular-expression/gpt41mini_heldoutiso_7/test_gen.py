import io
import re
import pytest

from w2re import iterable_to_regexp, PythonFormatter
from w2re.formatters import BaseFormatter
from w2re.prefix_tree.tree import PrefixTree


def test_iterable_to_regexp_single_word():
    pattern = iterable_to_regexp(['abc'])
    # The regexp for one word "abc" should match exactly "abc"
    assert re.fullmatch(pattern, 'abc')
    assert not re.fullmatch(pattern, 'abcd')
    assert not re.fullmatch(pattern, 'ab')


def test_iterable_to_regexp_multiple_words_non_overlapping():
    words = ['cat', 'dog', 'bird']
    pattern = iterable_to_regexp(words)
    for w in words:
        assert re.fullmatch(pattern, w)
    assert not re.fullmatch(pattern, 'fish')
    assert not re.fullmatch(pattern, 'ca')
    assert not re.fullmatch(pattern, 'dogs')


def test_iterable_to_regexp_multiple_words_with_shared_prefix():
    words = ['play', 'player', 'playing']
    pattern = iterable_to_regexp(words)
    for w in words:
        assert re.fullmatch(pattern, w)
    # Partial matches fail
    assert not re.fullmatch(pattern, 'playe')
    assert not re.fullmatch(pattern, 'plays')


def test_iterable_to_regexp_empty_iterable():
    pattern = iterable_to_regexp([])
    # The regexp should not match anything except possibly empty input,
    # but let's check it is a string and does not error on compilation
    assert isinstance(pattern, str)
    # Matching empty string should fail if no '' in iterable
    assert re.fullmatch(pattern, '') is None


def test_iterable_to_regexp_repeated_words():
    words = ['repeat', 'repeat', 'repeat']
    pattern = iterable_to_regexp(words)
    assert re.fullmatch(pattern, 'repeat')
    assert not re.fullmatch(pattern, 'repeats')


def test_iterable_to_regexp_with_empty_string_in_iterable():
    words = ['', 'a', 'abc']
    pattern = iterable_to_regexp(words)
    # It should match empty string as well
    assert re.fullmatch(pattern, '')
    assert re.fullmatch(pattern, 'a')
    assert re.fullmatch(pattern, 'abc')
    assert not re.fullmatch(pattern, 'ab')


def test_iterable_to_regexp_with_non_ascii_words():
    words = ['café', 'naïve', 'coöperate']
    pattern = iterable_to_regexp(words)
    for w in words:
        assert re.fullmatch(pattern, w)
    assert not re.fullmatch(pattern, 'cafe')  # accents matter


def test_pythonformatter_format_method_basic_usage():
    fmt = PythonFormatter()
    # The format method expects a string, returns string
    sample_regexp = 'abc|def'
    formatted = fmt.format(sample_regexp)
    assert isinstance(formatted, str)
    # It should contain the original regexp somewhere
    assert sample_regexp in formatted


def test_pythonformatter_format_output_contains_re_module_import():
    fmt = PythonFormatter()
    regex = 'abc'
    result = fmt.format(regex)
    assert 'import re' in result


def test_pythonformatter_compile_and_match():
    fmt = PythonFormatter()
    regexp = iterable_to_regexp(['foo', 'bar'])
    formatted_code = fmt.format(regexp)
    # The formatted code should be a Python snippet that imports re and stores compiled regex
    # Try to exec it safely, capture 'patt' variable
    namespace = {}
    exec(formatted_code, namespace)
    assert 'patt' in namespace
    patt = namespace['patt']
    assert hasattr(patt, 'match')
    assert patt.match('foo')
    assert patt.match('bar')
    assert not patt.match('baz')


def test_iterable_to_regexp_returns_string_using_default_formatter():
    # Should produce a string even for huge input
    big_input = ['word' + str(i) for i in range(100)]
    result = iterable_to_regexp(big_input)
    assert isinstance(result, str)
    # Basic sanity - test partial matching of some input words
    patt = re.compile(result)
    assert patt.match('word0')
    assert patt.match('word50')
    assert not patt.match('word100')


def test_iterable_to_regexp_accepts_other_formatter(monkeypatch):
    # We create a dummy formatter that uppercases the pattern string
    class DummyFormatter(BaseFormatter):
        def format(self, regexp: str) -> str:
            return regexp.upper()

    words = ['abc', 'def']
    result = iterable_to_regexp(words, formatter=DummyFormatter)
    assert isinstance(result, str)
    # The result should be uppercase version of some regexp pattern covering given words
    # Check letters uppercase and that 'ABC' and 'DEF' appear in result (indirect check)
    assert 'ABC' in result or 'DEF' in result


def test_iterable_to_regexp_with_prefix_tree_directly(monkeypatch):
    # monkeypatch PrefixTree.to_regexp to verify it is called correctly
    called = {}

    orig_to_regexp = PrefixTree.to_regexp

    def fake_to_regexp(self, formatter):
        called['formatter'] = formatter
        return 'FAKE_REGEX'

    monkeypatch.setattr(PrefixTree, 'to_regexp', fake_to_regexp)

    dummy_iterable = ['x', 'y']
    result = iterable_to_regexp(dummy_iterable, formatter=PythonFormatter)
    assert result == 'FAKE_REGEX'
    assert called['formatter'] == PythonFormatter

    monkeypatch.setattr(PrefixTree, 'to_regexp', orig_to_regexp)


def test_iterable_to_regexp_with_generator_input():
    # input as generator instead of list
    gen = (str(i) for i in range(3))
    pattern = iterable_to_regexp(gen)
    patt = re.compile(pattern)
    for i in range(3):
        assert patt.match(str(i))
    assert not patt.match('3')


@pytest.mark.parametrize('words, should_match, should_not_match', [
    (['a', 'ab', 'abc'], ['a', 'ab', 'abc'], ['b', 'ac', '']),
    ([''], [''], ['a']),
    (['prefix', 'prefixes'], ['prefix', 'prefixes'], ['pref']),
])
def test_iterable_to_regexp_parametrized(words, should_match, should_not_match):
    pattern = iterable_to_regexp(words)
    patt = re.compile(pattern)
    for w in should_match:
        assert patt.match(w)
    for w in should_not_match:
        assert not patt.match(w)
