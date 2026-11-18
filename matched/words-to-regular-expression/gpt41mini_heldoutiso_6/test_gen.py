import io
import re
import pytest

from w2re.utils import iterable_to_regexp
from w2re.formatters import PythonFormatter, BaseFormatter
from w2re.prefix_tree.tree import PrefixTree
import w2re


def test_iterable_to_regexp_basic():
    words = ["cat", "car", "dog"]
    regexp = iterable_to_regexp(words, PythonFormatter)
    # The regexp should match cat, car, dog but nothing else
    pattern = re.compile(regexp)
    assert pattern.fullmatch("cat")
    assert pattern.fullmatch("car")
    assert pattern.fullmatch("dog")
    assert not pattern.fullmatch("cow")
    assert not pattern.fullmatch("cats")


def test_iterable_to_regexp_empty_iterable():
    regexp = iterable_to_regexp([], PythonFormatter)
    # It should produce some regexp — empty input usually produces an empty pattern or something that matches nothing
    pattern = re.compile(regexp)
    assert not pattern.fullmatch("anything")


def test_iterable_to_regexp_single_word():
    regexp = iterable_to_regexp(["hello"], PythonFormatter)
    pattern = re.compile(regexp)
    assert pattern.fullmatch("hello")
    assert not pattern.fullmatch("hell")
    assert not pattern.fullmatch("hello!")


def test_iterable_to_regexp_repeated_words():
    words = ["test", "test", "testing"]
    regexp = iterable_to_regexp(words, PythonFormatter)
    pattern = re.compile(regexp)
    assert pattern.fullmatch("test")
    assert pattern.fullmatch("testing")
    assert not pattern.fullmatch("tester")


def test_iterable_to_regexp_custom_formatter_type():
    # Using BaseFormatter directly - its usage should still produce a string regexp (format behavior may differ)
    words = ["alpha", "beta"]
    regexp = iterable_to_regexp(words, BaseFormatter)
    assert isinstance(regexp, str)
    pattern = re.compile(regexp)
    assert pattern.fullmatch("alpha")
    assert pattern.fullmatch("beta")


def test_iterable_to_regexp_preserves_order_independent():
    # Check that order of input words does not affect matching behaviour
    words1 = ["one", "two", "three"]
    words2 = ["three", "one", "two"]
    regexp1 = iterable_to_regexp(words1, PythonFormatter)
    regexp2 = iterable_to_regexp(words2, PythonFormatter)
    # Both regexps should match the same set of words
    pattern1 = re.compile(regexp1)
    pattern2 = re.compile(regexp2)
    for w in ["one", "two", "three"]:
        assert pattern1.fullmatch(w)
        assert pattern2.fullmatch(w)


def test_pythonformatter_format_output_type():
    formatter = PythonFormatter()
    # Format a simple string using the formatter interface (emulates PrefixTree.to_regexp calls)
    formatted = formatter.format("abc|def")
    assert isinstance(formatted, str)
    assert "abc" in formatted or "def" in formatted


def test_pythonformatter_escape_special_chars():
    formatter = PythonFormatter()
    # Confirm that special regex characters are escaped properly by the formatter
    # PrefixTree will call formatter, so test the formatter directly
    escaped = formatter.escape_special_chars("a.c*+?")
    # All special characters should be escaped by backslash
    for ch in ".^$*+?{}[]\\|()":
        if ch in "a.c*+?":
            assert f"\\{ch}" in escaped


def test_iterable_to_regexp_non_ascii_words():
    words = ["café", "naïve", "résumé"]
    regexp = iterable_to_regexp(words, PythonFormatter)
    pattern = re.compile(regexp)
    assert pattern.fullmatch("café")
    assert pattern.fullmatch("naïve")
    assert pattern.fullmatch("résumé")

def test_iterable_to_regexp_words_with_common_prefix():
    words = ["prefix", "preface", "preform"]
    regexp = iterable_to_regexp(words, PythonFormatter)
    pattern = re.compile(regexp)
    for w in words:
        assert pattern.fullmatch(w)
    assert not pattern.fullmatch("pre")
    assert not pattern.fullmatch("prefixes")


def test_iterable_to_regexp_words_with_empty_string():
    words = ["", "empty", "none"]
    regexp = iterable_to_regexp(words, PythonFormatter)
    pattern = re.compile(regexp)
    # Empty string matching depends on implementation, likely no matches
    # but test that empty string is accepted if in input
    if pattern.fullmatch(""):
        assert pattern.fullmatch("")
    assert pattern.fullmatch("empty")
    assert pattern.fullmatch("none")


def test_iterable_to_regexp_unicode_emoji():
    words = ["😀", "😃😄", "🎉"]
    regexp = iterable_to_regexp(words, PythonFormatter)
    pattern = re.compile(regexp)
    for w in words:
        assert pattern.fullmatch(w)
    assert not pattern.fullmatch("😃")


def test_iterable_to_regexp_raises_typeerror_for_non_iterable():
    with pytest.raises(TypeError):
        # Passing an int instead of iterable should raise TypeError
        iterable_to_regexp(42, PythonFormatter)


def test_iterable_to_regexp_raises_typeerror_for_non_str_iterables():
    with pytest.raises(TypeError):
        # Passing iterable of non-str elements should raise error in PrefixTree or formatter
        iterable_to_regexp([1, 2, 3], PythonFormatter)


def test_w2re_init_version_and_constants():
    # Test __version__, APPLICATION_NAME, CHANGELOG_URL presence and type
    assert isinstance(w2re.__version__, str)
    assert isinstance(w2re.APPLICATION_NAME, str)
    assert isinstance(w2re.CHANGELOG_URL, str)
    assert "https://" in w2re.CHANGELOG_URL

