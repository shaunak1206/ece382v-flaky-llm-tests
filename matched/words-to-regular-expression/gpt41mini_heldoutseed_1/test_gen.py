import io
import re
import pytest

from w2re.utils import iterable_to_regexp
from w2re import PythonFormatter, BaseFormatter, iterable_to_regexp, stream_to_regexp

# We import PrefixTree only to cross-check regex results in some tests
from w2re.prefix_tree.tree import PrefixTree


def test_iterable_to_regexp_basic():
    words = ['cat', 'car', 'dog']
    regex = iterable_to_regexp(words)
    # regex should match exactly these words
    for w in words:
        assert re.fullmatch(regex, w)
    assert not re.fullmatch(regex, 'cow')
    assert not re.fullmatch(regex, 'cats')


def test_iterable_to_regexp_empty():
    regex = iterable_to_regexp([])
    # An empty iterable means a regex that matches nothing, usually ''
    # But PrefixTree might generate something minimal, check that no match for anything nonempty
    assert not re.fullmatch(regex, 'anything')
    assert not re.fullmatch(regex, 'a')


def test_iterable_to_regexp_single_word():
    word = 'hello'
    regex = iterable_to_regexp([word])
    assert re.fullmatch(regex, word)
    assert not re.fullmatch(regex, 'hell')


def test_iterable_to_regexp_with_empty_string():
    # Iterable includes empty string
    regex = iterable_to_regexp(['', 'a'])
    # regex should match '' and 'a'
    assert re.fullmatch(regex, '')
    assert re.fullmatch(regex, 'a')
    assert not re.fullmatch(regex, 'aa')


def test_iterable_to_regexp_with_duplicates():
    words = ['foo', 'foo', 'bar']
    regex = iterable_to_regexp(words)
    assert re.fullmatch(regex, 'foo')
    assert re.fullmatch(regex, 'bar')
    assert not re.fullmatch(regex, 'baz')


def test_iterable_to_regexp_uses_formatter(monkeypatch):
    class DummyFormatter(BaseFormatter):
        def format(self, regexp: str) -> str:
            return f"DUMMY({regexp})"

    words = ['a']
    regex = iterable_to_regexp(words, formatter=DummyFormatter)
    # The regex should be the DummyFormatter result
    assert regex.startswith('DUMMY(')
    assert 'a' in regex


def test_iterable_to_regexp_large_input():
    words = [f"word{i}" for i in range(1000)]
    regex = iterable_to_regexp(words)
    # Sample some matches
    assert re.fullmatch(regex, "word0")
    assert re.fullmatch(regex, "word999")
    assert not re.fullmatch(regex, "word1000")


def test_iterable_to_regexp_preserves_all_matches():
    words = ['abc', 'abcd', 'abce']
    regex = iterable_to_regexp(words)
    for word in words:
        assert re.fullmatch(regex, word)


def test_stream_to_regexp_matches_iterable_to_regexp_equally():
    # The utils.py function stream_to_regexp is similar to iterable_to_regexp,
    # but working with stream input.
    test_words = ['alpha', 'beta', 'gamma']
    stream_content = '\n'.join(test_words)
    stream = io.StringIO(stream_content)
    regex_from_stream = stream_to_regexp(stream)
    regex_from_iter = iterable_to_regexp(test_words)
    # Both regexes should match the same words
    for w in test_words:
        assert re.fullmatch(regex_from_stream, w)
        assert re.fullmatch(regex_from_iter, w)
    # Should not match words not in list
    assert not re.fullmatch(regex_from_stream, 'delta')
    assert not re.fullmatch(regex_from_iter, 'delta')


def test_stream_to_regexp_empty_stream():
    stream = io.StringIO('')
    regex = stream_to_regexp(stream)
    # Should not match any string except perhaps empty string
    assert not re.fullmatch(regex, 'a')


def test_pythonformatter_format_output_contains_expected_parts():
    # PythonFormatter formats to a Python regex string (raw)
    formatter = PythonFormatter()
    pattern = 'abc|def'
    formatted = formatter.format(pattern)
    # It should return a string with 'r"' or 'r\' and the pattern
    assert isinstance(formatted, str)
    # Typically, PythonFormatter uses raw string with quotes
    assert formatted.startswith('r"') or formatted.startswith("r'")


def test_pythonformatter_format_returns_string():
    formatter = PythonFormatter()
    s = 'testpattern'
    out = formatter.format(s)
    assert isinstance(out, str)
    assert 'testpattern' in out


def test_imported_constants_and_versions():
    import w2re
    assert hasattr(w2re, '__version__')
    assert w2re.__version__ == '3.1.0'
    assert hasattr(w2re, 'APPLICATION_NAME')
    assert w2re.APPLICATION_NAME == 'w2re'
    assert hasattr(w2re, 'CHANGELOG_URL')
    assert w2re.CHANGELOG_URL.startswith("https://github.com/")


def test_imports_exist_and_are_correct_types():
    import w2re
    assert hasattr(w2re, 'iterable_to_regexp')
    assert callable(w2re.iterable_to_regexp)
    assert hasattr(w2re, 'stream_to_regexp')
    assert callable(w2re.stream_to_regexp)
    assert hasattr(w2re, 'PythonFormatter')
    assert callable(w2re.PythonFormatter)
    # BaseFormatter is a class
    assert isinstance(w2re.BaseFormatter, type)


def test_iterable_to_regexp_preserves_order_indifference():
    # The generated regex should match all, order of input does not affect matches
    words1 = ['foo', 'bar', 'baz']
    words2 = ['baz', 'bar', 'foo']
    regex1 = iterable_to_regexp(words1)
    regex2 = iterable_to_regexp(words2)
    for w in words1:
        assert re.fullmatch(regex1, w)
        assert re.fullmatch(regex2, w)


def test_iterable_to_regexp_handles_unicode_words():
    words = ['café', 'naïve', 'résumé']
    regex = iterable_to_regexp(words)
    for w in words:
        assert re.fullmatch(regex, w)
    assert not re.fullmatch(regex, 'cafe')


def test_stream_to_regexp_strip_empty_lines():
    stream = io.StringIO('\n\ncat\n\ndog\n\n')
    regex = stream_to_regexp(stream)
    assert re.fullmatch(regex, 'cat')
    assert re.fullmatch(regex, 'dog')
    assert not re.fullmatch(regex, '')


def test_stream_to_regexp_uses_specified_formatter():
    class UpperFormatter(BaseFormatter):
        def format(self, regexp: str) -> str:
            return regexp.upper()

    words = ['a', 'b']
    stream = io.StringIO('\n'.join(words))
    regex = stream_to_regexp(stream, formatter=UpperFormatter)
    # It should be uppercase
    assert regex.isupper()
    # To verify at least 'A' in regex for 'a'
    assert 'A' in regex

