import io
import re
import pytest

from w2re.utils import iterable_to_regexp
from w2re import PythonFormatter, iterable_to_regexp as w2re_iterable_to_regexp
from w2re import __version__, APPLICATION_NAME, CHANGELOG_URL


def test_iterable_to_regexp_empty_list():
    result = iterable_to_regexp([])
    # Should produce a regexp matching nothing, typically something like (?:)
    # Depends on PrefixTree implementation, but should not error and return a string
    assert isinstance(result, str)
    # An empty list might produce a non-matching regexp, so at least not fail to compile
    re.compile(result)


def test_iterable_to_regexp_single_word():
    result = iterable_to_regexp(['hello'])
    assert isinstance(result, str)
    # The regexp should match "hello"
    pattern = re.compile(result)
    assert pattern.fullmatch("hello")
    assert not pattern.fullmatch("hell")


def test_iterable_to_regexp_multiple_words_with_common_prefix():
    words = ["cat", "car", "cart", "dog"]
    regexp = iterable_to_regexp(words)
    pattern = re.compile(regexp)
    for word in words:
        assert pattern.fullmatch(word)
    assert not pattern.fullmatch("ca")
    assert not pattern.fullmatch("cattle")
    assert not pattern.fullmatch("dogs")


def test_iterable_to_regexp_multiple_words_no_common_prefix():
    words = ["apple", "banana", "cherry"]
    regexp = iterable_to_regexp(words)
    pattern = re.compile(regexp)
    for word in words:
        assert pattern.fullmatch(word)
    assert not pattern.fullmatch("app")
    assert not pattern.fullmatch("bananaa")
    assert not pattern.fullmatch("berry")


def test_iterable_to_regexp_words_with_special_chars():
    words = ["foo.bar", "foo?bar", "foo*bar"]
    regexp = iterable_to_regexp(words)
    pattern = re.compile(regexp)
    for word in words:
        assert pattern.fullmatch(word)


def test_iterable_to_regexp_non_string_inputs_raise_typeerror():
    with pytest.raises(TypeError):
        # Passing integers should raise because PrefixTree expects strings
        iterable_to_regexp([1, 2, 3])

    with pytest.raises(TypeError):
        iterable_to_regexp([None])


def test_pythonformatter_formatting_output_is_string():
    # Use PythonFormatter explicitly with iterable_to_regexp
    words = ["foo", "foobar", "bar"]
    result = iterable_to_regexp(words, formatter=PythonFormatter)
    assert isinstance(result, str)
    # The output may contain python code (regex patterns assigned to variables)
    assert "re.compile" in result or "pattern =" in result or result


def test_iterable_to_regexp_with_custom_formatter_dummy():
    # Define a dummy formatter that returns a fixed string
    from w2re.formatters import BaseFormatter

    class DummyFormatter(BaseFormatter):
        def format(self, prefix_tree):
            return "dummy_regex"

    words = ["any", "words", "here"]
    regexp = iterable_to_regexp(words, formatter=DummyFormatter)
    assert regexp == "dummy_regex"


def test_init_module_exports_version_and_constants():
    assert isinstance(__version__, str)
    assert __version__ == '3.1.0'
    assert APPLICATION_NAME == 'w2re'
    assert CHANGELOG_URL.startswith('https://github.com/radeklat/words-to-regular-expression')


def test_iterable_to_regexp_unicode_words():
    words = ["café", "naïve", "résumé"]
    regexp = iterable_to_regexp(words)
    pattern = re.compile(regexp)
    for w in words:
        assert pattern.fullmatch(w)


def test_iterable_to_regexp_long_words():
    words = ["a" * 100, "a" * 101, "a" * 99 + "b"]
    regexp = iterable_to_regexp(words)
    pattern = re.compile(regexp)
    for w in words:
        assert pattern.fullmatch(w)
    assert not pattern.fullmatch("a" * 50)


def test_iterable_to_regexp_identical_words():
    words = ["repeat", "repeat", "repeat"]
    regexp = iterable_to_regexp(words)
    pattern = re.compile(regexp)
    assert pattern.fullmatch("repeat")
    assert not pattern.fullmatch("repea")


def test_iterable_to_regexp_words_with_overlap():
    words = ["a", "ab", "abc", "abcd"]
    regexp = iterable_to_regexp(words)
    pattern = re.compile(regexp)
    for w in words:
        assert pattern.fullmatch(w)


def test_iterable_to_regexp_words_with_empty_string_included():
    words = ["", "nonempty"]
    regexp = iterable_to_regexp(words)
    pattern = re.compile(regexp)
    assert pattern.fullmatch("")
    assert pattern.fullmatch("nonempty")
    assert not pattern.fullmatch("nonempt")


def test_iterable_to_regexp_large_input_performance():
    # Just test no crash / error for a large number of words
    words = [f"word{i}" for i in range(1000)]
    regexp = iterable_to_regexp(words)
    pattern = re.compile(regexp)
    # Test some samples
    assert pattern.fullmatch("word0")
    assert pattern.fullmatch("word999")
    assert not pattern.fullmatch("word1000")


def test_iterable_to_regexp_input_is_generator():
    words = (f"w{i}" for i in range(10))
    regexp = iterable_to_regexp(words)
    pattern = re.compile(regexp)
    assert pattern.fullmatch("w0")
    assert pattern.fullmatch("w9")
    assert not pattern.fullmatch("w10")
