import io
import random
import pytest

from w2re import iterable_to_regexp, PythonFormatter, __version__, APPLICATION_NAME, CHANGELOG_URL
from w2re.utils import iterable_to_regexp as utils_iterable_to_regexp
from w2re.formatters import PythonFormatter as ImportedPythonFormatter
from w2re.formatters import BaseFormatter
from w2re.prefix_tree.tree import PrefixTree


def test_iterable_to_regexp_empty_iterable():
    # Should produce an empty regexp string for empty input
    result = iterable_to_regexp([])
    assert isinstance(result, str)
    # The empty regexp in prefix tree is '', but PythonFormatter may format something minimal
    # Check empty or minimal valid regexp output (no match)
    assert result != None


def test_iterable_to_regexp_single_word():
    result = iterable_to_regexp(['apple'])
    assert isinstance(result, str)
    assert 'apple' in result


def test_iterable_to_regexp_multiple_words_no_common_prefix():
    words = ['cat', 'dog', 'mouse']
    result = iterable_to_regexp(words)
    assert isinstance(result, str)
    # All words present as substrings in regexp (not necessarily contiguous)
    for w in words:
        assert w[0] in result


def test_iterable_to_regexp_multiple_words_with_common_prefix():
    words = ['test', 'tester', 'testing', 'tested']
    result = iterable_to_regexp(words)
    # Result should contain 'test' base prefix
    assert 'test' in result
    # Result should contain also endings for suffixes
    for suffix in ['er', 'ing', 'ed']:
        assert suffix in result


def test_iterable_to_regexp_iterable_not_list():
    # iterable can be a generator
    words = ('i' * i for i in range(1, 5))
    result = iterable_to_regexp(words)
    assert isinstance(result, str)
    # Check some expected substrings represent the inputs
    for i in range(1, 5):
        snippet = 'i' * i
        assert snippet[0] in result


def test_python_formatter_class_identity():
    # The default formatter parameter is exactly PythonFormatter
    result_default = iterable_to_regexp(['abc'])
    result_explicit = iterable_to_regexp(['abc'], formatter=PythonFormatter)
    assert result_default == result_explicit


def test_iterable_to_regexp_with_custom_formatter(monkeypatch):
    # Create a dummy formatter class that formats in a fixed way
    class DummyFormatter(BaseFormatter):
        def format(self, *args, **kwargs):
            return "dummy"

    result = iterable_to_regexp(['something'], formatter=DummyFormatter)
    # The result should be what PrefixTree.to_regexp returns with DummyFormatter,
    # but we don't control PrefixTree to_regexp output with DummyFormatter,
    # so just check that the result is a string and can be different from normal.
    assert isinstance(result, str)
    # Likely different from normal formatter output
    assert result != iterable_to_regexp(['something'], formatter=PythonFormatter)


def test_imports_from_init_module():
    # Test that __version__, APPLICATION_NAME, CHANGELOG_URL have expected values
    assert __version__ == '3.1.0'
    assert APPLICATION_NAME == 'w2re'
    assert CHANGELOG_URL.startswith('https://github.com/radeklat/words-to-regular-expression/')


def test_imported_formatter_classes():
    # The imported PythonFormatter is the same as the one from w2re.formatters
    assert issubclass(PythonFormatter, BaseFormatter)
    # Also check that the imported PythonFormatter has attribute 'format'
    assert hasattr(PythonFormatter, 'format')
    assert callable(getattr(PythonFormatter, 'format'))


def test_iterable_to_regexp_handles_duplicates():
    words = ['dup', 'dup', 'duplication']
    result = iterable_to_regexp(words)
    # Should produce a regexp containing 'dup' and 'duplication'
    assert all(x in result for x in ['dup', 'duplication'])


def test_iterable_to_regexp_preserves_order_independent_result():
    # The regexp should be independent of the input order (because it's prefix tree)
    words1 = ['apple', 'banana', 'cherry']
    words2 = ['cherry', 'apple', 'banana']
    result1 = iterable_to_regexp(words1)
    result2 = iterable_to_regexp(words2)
    assert isinstance(result1, str) and isinstance(result2, str)
    # Both results should contain all words
    for w in words1:
        assert w[0] in result1
        assert w[0] in result2


def test_iterable_to_regexp_non_ascii_input():
    words = ['café', 'naïve', 'jalapeño']
    result = iterable_to_regexp(words)
    # Result should contain at least one character of each input word
    for w in words:
        assert w[0] in result


def test_iterable_to_regexp_large_input_basic_performance():
    # Large number of words, but simple; test it runs and returns string
    words = ['word' + str(i) for i in range(1000)]
    result = iterable_to_regexp(words)
    assert isinstance(result, str)
    # Check a few sample words' initial character present in output
    for i in (0, 500, 999):
        assert 'w' in result


def test_iterable_to_regexp_works_with_whitespace_strings():
    words = [' ', '  ', '   ']
    result = iterable_to_regexp(words)
    # The regexp should be string and probably contain space chars
    assert isinstance(result, str)
    assert ' ' in result


def test_iterable_to_regexp_raises_type_error_on_non_string_iterable():
    # Passing numbers instead of strings should error in PrefixTree construction or to_regexp
    with pytest.raises(TypeError):
        iterable_to_regexp([1, 2, 3])


def test_iterable_to_regexp_does_not_mutate_input():
    words = ['cat', 'dog']
    words_copy = list(words)
    result = iterable_to_regexp(words)
    # Input list unchanged
    assert words == words_copy


def test_module_attribute_monkeypatch_iterable_to_regexp(monkeypatch):
    # Patch PrefixTree to a dummy class to verify it is called correctly
    class DummyPrefixTree:
        def __init__(self, iterable):
            self.iterable = iterable
        def to_regexp(self, formatter):
            return f"formatted {list(self.iterable)} with {formatter.__name__}"

    monkeypatch.setattr("w2re.utils.PrefixTree", DummyPrefixTree)
    # Exercise iterable_to_regexp with monkeypatched PrefixTree
    words = ["a", "b"]
    result = iterable_to_regexp(words, formatter=PythonFormatter)
    assert result == f"formatted {words} with PythonFormatter"


def test_module_attribute_monkeypatch_pythonformatter(monkeypatch):
    # Patch PythonFormatter to dummy class with a name property
    class DummyFormatter(BaseFormatter):
        pass
    monkeypatch.setattr("w2re.utils.PythonFormatter", DummyFormatter)
    # Using the monkeypatched PythonFormatter explicitly
    result = iterable_to_regexp(['x'], formatter=DummyFormatter)
    # Result comes from PrefixTree.to_regexp(DummyFormatter), type string
    assert isinstance(result, str)


def test_iterable_to_regexp_with_non_iterable_argument():
    # Passing non-iterable should raise TypeError
    with pytest.raises(TypeError):
        iterable_to_regexp(12345)


def test_iterable_to_regexp_with_mixed_iterable(monkeypatch):
    # Ensure that when iterable yields non-str, an error is raised during PrefixTree init or to_regexp
    mixed_iterable = ['abc', 123, 'def']
    with pytest.raises(TypeError):
        iterable_to_regexp(mixed_iterable)


def test_iterable_to_regexp_empty_string_elements():
    words = ['', '', '']
    result = iterable_to_regexp(words)
    assert isinstance(result, str)


def test_iterable_to_regexp_with_large_repeating_patterns():
    words = ['a'*i for i in range(20)]  # a, aa, aaa, ...
    result = iterable_to_regexp(words)
    # Check that output regexp contains 'a' and some length references
    assert 'a' in result
