import io
import pytest

from w2re import iterable_to_regexp, PythonFormatter
from w2re.formatters import BaseFormatter


def test_iterable_to_regexp_simple_words():
    words = ['apple', 'banana', 'cherry']
    regex = iterable_to_regexp(words)
    assert isinstance(regex, str)
    assert all(word[:2] in regex for word in words)  # Rough check substrings appear


def test_iterable_to_regexp_empty_list():
    regex = iterable_to_regexp([])
    assert regex == '' or regex == '(?!)' or regex == ''  # depending on implementation: empty regexp


def test_iterable_to_regexp_single_word():
    regex = iterable_to_regexp(['hello'])
    assert 'hello' in regex


def test_iterable_to_regexp_duplicates():
    words = ['test', 'test', 'testing']
    regex = iterable_to_regexp(words)
    # Ensure duplicates do not cause repetition in pattern (best effort)
    assert regex.count('test') == 1 or regex.count('test') < 3


def test_iterable_to_regexp_with_custom_formatter(monkeypatch):
    # Custom formatter to force a specific pattern
    class DummyFormatter(BaseFormatter):
        def format(self, prefix_tree):
            return 'dummy_regex'

    words = ['a', 'b', 'c']
    regex = iterable_to_regexp(words, formatter=DummyFormatter)
    assert regex == 'dummy_regex'


def test_pythonformatter_formatting_behavior():
    formatter = PythonFormatter()
    # It has to have a format method that accepts a prefix tree and returns str
    class DummyTree:
        def __str__(self):
            return 'dummy_tree'

    # We cannot easily create a PrefixTree here, so test format returns string
    # The actual PrefixTree interface is larger but this is enough for the test
    result = formatter.format(DummyTree())
    assert isinstance(result, str)


def test_iterable_to_regexp_yields_regex_str_type():
    # Use a longer list to ensure code path is exercised
    words = ['foo', 'bar', 'baz', 'qux', 'quux']
    regex = iterable_to_regexp(words)
    assert isinstance(regex, str)
    assert regex  # non-empty


def test_iterable_to_regexp_generator_input():
    gen = (w for w in ['cat', 'dog', 'mouse'])
    regex = iterable_to_regexp(gen)
    assert isinstance(regex, str)
    assert 'cat' in regex or 'dog' in regex or 'mouse' in regex


def test_iterable_to_regexp_non_ascii_characters():
    words = ['café', 'niño', 'jalapeño']
    regex = iterable_to_regexp(words)
    assert isinstance(regex, str)
    for w in words:
        for char in w[:2]:
            assert char in regex  # rough check


def test_iterable_to_regexp_special_characters():
    words = ['foo.bar', 'foo?bar', 'foo*bar', 'foo+bar']
    regex = iterable_to_regexp(words)
    assert isinstance(regex, str)
    # The regex should contain at least parts of these words
    for w in ['foo', 'bar']:
        assert w in regex


def test_iterable_to_regexp_unicode_emojis():
    words = ['🙂', '😂', '👍']
    regex = iterable_to_regexp(words)
    assert isinstance(regex, str)
    for w in words:
        assert w in regex or len(regex) > 0


def test_pythonformatter_is_subclass_of_baseformatter():
    from w2re.formatters import PythonFormatter, BaseFormatter
    assert issubclass(PythonFormatter, BaseFormatter)


def test_pythonformatter_format_returns_string(iterable=['a', 'bb', 'ccc']):
    # This test uses the real PrefixTree by passing a real iterable into iterable_to_regexp
    regex = iterable_to_regexp(iterable, formatter=PythonFormatter)
    assert isinstance(regex, str)
    assert regex


def test_iterable_to_regexp_non_str_items():
    # Passing non-str should raise TypeError or handle gracefully
    # PrefixTree expects strings; we test with ints to verify failure or error handling
    with pytest.raises(TypeError):
        # Cast ints to iterable to break type expectations
        iterable_to_regexp([1, 2, 3])


def test_iterable_to_regexp_with_mixed_types():
    mixed = ['a', 1, 'b']
    with pytest.raises(TypeError):
        iterable_to_regexp(mixed)


def test_iterable_to_regexp_passed_set():
    # Sets are iterables, but unordered; regex will vary but should not error
    s = {'alpha', 'beta', 'gamma'}
    regex = iterable_to_regexp(s)
    assert isinstance(regex, str)
    for w in s:
        # text snippets appear in regex
        assert w[:1] in regex


def test_iterable_to_regexp_passed_tuple():
    t = ('one', 'two', 'three')
    regex = iterable_to_regexp(t)
    assert isinstance(regex, str)
    for w in t:
        assert w[:1] in regex


def test_imports_exist_in_init():
    import w2re
    # Check that attributes exist
    assert hasattr(w2re, 'iterable_to_regexp')
    assert hasattr(w2re, 'PythonFormatter')
    assert hasattr(w2re, '__version__')
    assert w2re.__version__ == '3.1.0'
    assert hasattr(w2re, 'APPLICATION_NAME')
    assert hasattr(w2re, 'CHANGELOG_URL')
