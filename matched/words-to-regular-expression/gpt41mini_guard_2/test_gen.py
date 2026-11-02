import io
import pytest

from w2re import iterable_to_regexp, PythonFormatter, __version__, APPLICATION_NAME, CHANGELOG_URL
from w2re.utils import iterable_to_regexp as utils_iterable_to_regexp
from w2re.formatters import PythonFormatter as FormatterClass
from w2re.prefix_tree.tree import PrefixTree


def test_iterable_to_regexp_basic_strings():
    words = ['apple', 'app', 'application', 'banana']
    regexp = iterable_to_regexp(words)
    # Should return a string including all these words as alternations or part of regexp
    for w in words:
        assert w.replace('\\', '') in regexp


def test_iterable_to_regexp_empty_iterable():
    regexp = iterable_to_regexp([])
    # For empty iterable, empty regexp or something valid but empty is expected
    assert isinstance(regexp, str)
    assert regexp == '' or regexp == '(?:)' or regexp == ''


def test_iterable_to_regexp_single_word():
    word = ["hello"]
    regexp = iterable_to_regexp(word)
    # Should include the exact word (possibly escaped)
    assert "hello" in regexp


def test_iterable_to_regexp_nontrivial_words():
    words = ['a', 'ab', 'abc', 'abcd']
    regexp = iterable_to_regexp(words)
    # The regexp should contain the simplest common pattern for these strings
    assert 'a' in regexp
    assert 'b' in regexp
    assert 'c' in regexp
    assert 'd' in regexp


def test_iterable_to_regexp_all_prefixes():
    prefixes = [""]
    # a PrefixTree should handle empty string correctly if included explicitly
    regexp = iterable_to_regexp(["", "x", "xy"])
    assert isinstance(regexp, str)
    # An empty string converted to regexp usually means it can match empty too
    # The returned regexp should match 'x' and 'xy' too
    assert 'x' in regexp


def test_iterable_to_regexp_with_repeated_words():
    words = ['repeat', 'repeat', 'repeat']
    regexp = iterable_to_regexp(words)
    # Duplicate words should not change the regexp
    assert regexp.count('repeat') >= 1


def test_iterable_to_regexp_uses_python_formatter_by_default():
    # Passing no formatter explicitly should use PythonFormatter and return string
    words = ['abc', 'def']
    regexp = iterable_to_regexp(words)
    assert isinstance(regexp, str)


def test_pythonformatter_format_method_behavior():
    formatter = PythonFormatter()
    # The PythonFormatter.format method requires PrefixTree instance
    # PrefixTree is complex, we only test format returns a string and does not error
    sample_tree = PrefixTree(['test', 'tester'])

    formatted = formatter.format(sample_tree)
    assert isinstance(formatted, str)
    # The output string should contain words' first characters at least
    assert 't' in formatted


def test_utils_iterable_to_regexp_same_as_imported():
    # The wrapper method in utils should be same as the directly imported one
    words = ['one', 'two']
    assert iterable_to_regexp(words) == utils_iterable_to_regexp(words)


def test_init_module_contains_version_and_constants():
    assert __version__ == '3.1.0'
    assert APPLICATION_NAME == 'w2re'
    assert CHANGELOG_URL.startswith('https://github.com/radeklat/')


def test_iterable_to_regexp_allows_custom_formatter(monkeypatch):
    # Create a dummy formatter class with a format method
    class DummyFormatter:
        def format(self, tree):
            return "formatted_by_dummy"

    words = ['some', 'words']
    result = iterable_to_regexp(words, formatter=DummyFormatter)
    # Result should be exactly string returned by DummyFormatter.format
    assert result == "formatted_by_dummy"


def test_iterable_to_regexp_input_not_modified():
    words = ['a', 'b', 'c']
    words_copy = words[:]
    _ = iterable_to_regexp(words)
    # Original input list should not be mutated
    assert words == words_copy


def test_iterable_to_regexp_handles_iterable_of_iterables_gracefully():
    # Passing list of lists instead of list of strings - should raise or handle
    words = [['a'], ['b']]
    with pytest.raises(TypeError):
        iterable_to_regexp(words)


def test_iterable_to_regexp_handles_generator_correctly():
    # Using generator input
    words = (w for w in ['one', 'two', 'three'])
    regexp = iterable_to_regexp(words)
    assert isinstance(regexp, str)
    for w in ['one', 'two', 'three']:
        assert w.replace('\\', '') in regexp


def test_iterable_to_regexp_integration_with_prefix_tree_to_regexp():
    words = ['abc', 'abcd', 'abce']
    # Using the PrefixTree object explicitly for comparison
    tree = PrefixTree(words)
    expected = tree.to_regexp(PythonFormatter)
    result = iterable_to_regexp(words)
    assert result == expected


def test_pythonformatter_format_returns_string(monkeypatch):
    formatter = PythonFormatter()
    tree = PrefixTree(['x', 'xy', 'xyz'])
    result = formatter.format(tree)
    assert isinstance(result, str)
    # At minimum result should not be empty
    assert len(result) > 0


def test_pythonformatter_format_does_not_mutate_tree():
    tree = PrefixTree(['foo', 'bar'])
    # Save a copy of tree's internal state for comparison
    # Since tree internals are complex, we check that attributes remain the same types
    attrs_before = {attr: getattr(tree, attr) for attr in vars(tree)}
    formatter = PythonFormatter()
    _ = formatter.format(tree)
    attrs_after = {attr: getattr(tree, attr) for attr in vars(tree)}
    assert attrs_before.keys() == attrs_after.keys()
    for k in attrs_before:
        assert type(attrs_before[k]) == type(attrs_after[k])


def test_iterable_to_regexp_unicode_words():
    words = ['naïve', 'résumé', 'café', 'jalapeño']
    regexp = iterable_to_regexp(words)
    # Unicode characters appear in regexp
    for w in words:
        assert w.encode('utf-8').decode('utf-8') in regexp


def test_iterable_to_regexp_escape_special_regex_chars():
    # Words with regex special characters
    special = ['a.c', 'e*f', 'g+h?', '[abc]', '(test)']
    regexp = iterable_to_regexp(special)
    # The raw words should appear escaped or quoted to avoid regex syntax errors
    for w in special:
        assert w.replace('\\', '') in regexp


def test_iterable_to_regexp_non_string_elements_raises():
    words = ['valid', 123, None]
    with pytest.raises(TypeError):
        iterable_to_regexp(words)


def test_iterable_to_regexp_can_handle_large_input():
    words = [f'word{i}' for i in range(1000)]
    regexp = iterable_to_regexp(words)
    # Should still be a string and contain a sample of the words
    assert isinstance(regexp, str)
    assert 'word0' in regexp
    assert 'word999' in regexp
