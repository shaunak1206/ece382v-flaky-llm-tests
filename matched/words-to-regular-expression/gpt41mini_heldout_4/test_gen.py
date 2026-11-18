import io
import re
import pytest

from w2re import iterable_to_regexp, PythonFormatter
from w2re.formatters import BaseFormatter
from w2re.prefix_tree.tree import PrefixTree


def test_iterable_to_regexp_returns_string():
    words = ['cat', 'car', 'dog']
    result = iterable_to_regexp(words)
    assert isinstance(result, str)
    assert 'cat' in result or 'car' in result or 'dog' in result


def test_iterable_to_regexp_empty_iterable():
    result = iterable_to_regexp([])
    # Empty prefix tree should produce a regexp that matches nothing or empty string
    # The exact output may depend on implementation - check it's string and no error
    assert isinstance(result, str)


def test_iterable_to_regexp_single_word():
    word = 'hello'
    result = iterable_to_regexp([word])
    pattern = re.compile(result)
    assert pattern.match(word)
    assert not pattern.match('hell')


def test_iterable_to_regexp_substrings():
    words = ['he', 'hell', 'hello']
    result = iterable_to_regexp(words)
    pattern = re.compile(result)
    for w in words:
        assert pattern.match(w)
    # Should not match unrelated word
    assert not pattern.match('hey')


def test_iterable_to_regexp_with_pythonformatter(monkeypatch):
    # Custom PythonFormatter that uppercases output for testing
    class DummyFormatter(PythonFormatter):
        def format(self, regexp: str) -> str:
            return regexp.upper()

    words = ['abc', 'abd']
    result = iterable_to_regexp(words, formatter=DummyFormatter)
    assert result == result.upper()


def test_iterable_to_regexp_non_strings():
    words = [b'abc', 'abd', 123]
    # PrefixTree expects iterable of strings, so non-str should raise or be filtered
    # The code does not explicitly filter, so it may raise TypeError when processing
    with pytest.raises(TypeError):
        iterable_to_regexp(words)


def test_pythonformatter_format_basic():
    pf = PythonFormatter()
    regexp = 'abc|def'
    formatted = pf.format(regexp)
    # PythonFormatter adds r'' quoting
    assert formatted.startswith('r\'')
    assert formatted.endswith('\'')
    assert regexp in formatted


def test_pythonformatter_format_regex_special_chars():
    pf = PythonFormatter()
    regexp = 'a.c*'
    formatted = pf.format(regexp)
    assert 'a.c*' in formatted


def test_prefix_tree_to_regexp_called_by_iterable_to_regexp(monkeypatch):
    called = {}

    orig_to_regexp = PrefixTree.to_regexp

    def fake_to_regexp(self, formatter):
        called['yes'] = True
        # We just return a fixed string for testing
        return 'FAKE_REGEX'

    monkeypatch.setattr(PrefixTree, 'to_regexp', fake_to_regexp)

    result = iterable_to_regexp(['x'])
    assert called.get('yes')
    assert result == 'FAKE_REGEX'

    # Restore monkeypatch not necessary as fixture takes care of that


def test_iterable_to_regexp_handles_duplicates():
    words = ['foo', 'foo', 'bar']
    result = iterable_to_regexp(words)
    pattern = re.compile(result)
    assert pattern.match('foo')
    assert pattern.match('bar')
    assert not pattern.match('baz')


def test_iterable_to_regexp_with_multiline_words():
    words = ['a\nb\nc'.split('\n')]
    # The input is a list of lists, so flatten for test
    words_flat = sum(words, [])
    result = iterable_to_regexp(words_flat)
    pattern = re.compile(result)
    for w in words_flat:
        assert pattern.match(w)


def test_iterable_to_regexp_accepts_iterators():
    words = iter(['x', 'y', 'z'])
    result = iterable_to_regexp(words)
    assert isinstance(result, str)
    pattern = re.compile(result)
    for ch in ['x', 'y', 'z']:
        assert pattern.match(ch)


def test_iterable_to_regexp_handles_large_input():
    words = [f'word{i}' for i in range(1000)]
    result = iterable_to_regexp(words)
    pattern = re.compile(result)
    for i in (0, 10, 999):
        assert pattern.match(f'word{i}')
    assert not pattern.match('word1001')


def test_pythonformatter_format_with_empty_string():
    pf = PythonFormatter()
    formatted = pf.format('')
    # Should still return raw string, essentially "r''"
    assert formatted == "r''"


def test_pythonformatter_class_is_subclass_of_baseformatter():
    assert issubclass(PythonFormatter, BaseFormatter)
