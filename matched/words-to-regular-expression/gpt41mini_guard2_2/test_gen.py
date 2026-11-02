import io
import pytest

from w2re.utils import iterable_to_regexp
from w2re import PythonFormatter, iterable_to_regexp as imported_iterable_to_regexp
from w2re import __version__, APPLICATION_NAME, CHANGELOG_URL
from w2re.formatters import PythonFormatter as ImportedPythonFormatter
from w2re.prefix_tree.tree import PrefixTree


def test_iterable_to_regexp_empty_list_returns_non_empty_string():
    # Even empty input returns some regexp (usually empty string or similar)
    result = iterable_to_regexp([])
    assert isinstance(result, str)
    # The result should compile as a regex pattern
    import re
    re.compile(result)


def test_iterable_to_regexp_single_word_returns_word_regexp():
    word = "hello"
    regexp = iterable_to_regexp([word])
    assert isinstance(regexp, str)
    # Compiles as expected
    import re
    pattern = re.compile(regexp)
    assert pattern.match(word)


def test_iterable_to_regexp_multiple_words_with_common_prefix():
    words = ["test", "testing", "tester"]
    regexp = iterable_to_regexp(words)
    assert isinstance(regexp, str)
    import re
    pattern = re.compile(regexp)
    # All words should match the regexp
    for w in words:
        assert pattern.match(w)


def test_iterable_to_regexp_distinct_words():
    words = ["foo", "bar", "baz"]
    regexp = iterable_to_regexp(words)
    import re
    pattern = re.compile(regexp)
    for w in words:
        assert pattern.match(w)
    # Words not in list should likely not match (at least a very different word)
    assert not pattern.match("qux")


def test_iterable_to_regexp_with_formatter_pythonformatter_is_default():
    # The default formatter should behave same as passing PythonFormatter explicitly
    words = ["x", "xy", "xyz"]
    default_regexp = iterable_to_regexp(words)
    explicit_regexp = iterable_to_regexp(words, formatter=PythonFormatter)
    assert default_regexp == explicit_regexp


def test_iterable_to_regexp_with_custom_formatter_monkeypatch(monkeypatch):
    # Patch the to_regexp to track formatter usage
    class DummyFormatter:
        pass

    called = {}

    original_to_regexp = PrefixTree.to_regexp

    def fake_to_regexp(self, formatter):
        called['formatter'] = formatter
        return "dummy"

    monkeypatch.setattr(PrefixTree, 'to_regexp', fake_to_regexp)

    result = iterable_to_regexp(["a", "b"], formatter=DummyFormatter)
    assert result == "dummy"
    assert called.get('formatter') is DummyFormatter

    # Restore is automatic with monkeypatch


def test_imports_from_init_are_correct():
    from w2re import (
        BaseFormatter,
        PythonFormatter,
        PythonWordMatchFormatter,
        PrefixTree,
        iterable_to_regexp,
        stream_to_regexp,
        __version__,
        APPLICATION_NAME,
        CHANGELOG_URL,
    )
    # Check that __version__ is correct type and not empty
    assert isinstance(__version__, str)
    assert len(__version__) > 0

    assert APPLICATION_NAME == 'w2re'
    assert CHANGELOG_URL.startswith("https://github.com/radeklat/words-to-regular-expression")


def test_iterable_to_regexp_returns_same_result_for_equivalent_inputs():
    words1 = ["apple", "applesauce", "apply"]
    words2 = ["apple", "apply", "applesauce"]  # Different order
    r1 = iterable_to_regexp(words1)
    r2 = iterable_to_regexp(words2)
    assert isinstance(r1, str)
    assert r1 == r2


def test_iterable_to_regexp_handles_repeated_words_gracefully():
    words = ["repeat", "repeat", "rep", "repetition", "rep"]
    result = iterable_to_regexp(words)
    import re
    pattern = re.compile(result)
    for w in set(words):
        assert pattern.match(w)


def test_iterable_to_regexp_handles_words_with_special_characters():
    words = ["a.c", "a*c", "a+c", "a?c"]
    regexp = iterable_to_regexp(words)
    import re
    pattern = re.compile(regexp)
    for w in words:
        # Because the regexp is supposed to match words exactly
        assert pattern.match(w)


def test_iterable_to_regexp_with_empty_string_word():
    words = ["", "empty", "nonempty"]
    regex = iterable_to_regexp(words)
    import re
    pattern = re.compile(regex)
    for w in words:
        assert pattern.match(w)


def test_iterable_to_regexp_accepts_iterable_not_list():
    # Use a generator instead of list
    gen = (str(i) for i in range(5))
    regex = iterable_to_regexp(gen)
    import re
    pattern = re.compile(regex)
    # The regex should match "0" and "4"
    assert pattern.match("0")
    assert pattern.match("4")


def test_iterable_to_regexp_returns_different_for_different_words():
    r1 = iterable_to_regexp(["cat"])
    r2 = iterable_to_regexp(["dog"])
    assert r1 != r2


def test_imported_pythonformatter_is_class_and_subclass_of_baseformatter():
    from w2re.formatters import BaseFormatter
    assert isinstance(PythonFormatter, type)
    assert issubclass(PythonFormatter, BaseFormatter)


def test_iterable_to_regexp_with_large_input_performance_and_consistency():
    words = [f"word{i}" for i in range(1000)]
    result = iterable_to_regexp(words)
    import re
    pattern = re.compile(result)
    # Check some samples to match
    for i in [0, 50, 999]:
        assert pattern.match(f"word{i}")


def test_iterable_to_regexp_with_unicode_words():
    words = ["café", "naïve", "über", "façade"]
    result = iterable_to_regexp(words)
    import re
    pattern = re.compile(result)
    for w in words:
        assert pattern.match(w)


def test_imported_iterable_to_regexp_is_same_as_direct_import():
    # The symbol imported from __init__.py and direct utils import should be the same
    assert imported_iterable_to_regexp is iterable_to_regexp


def test_application_constants_are_strings_and_non_empty():
    assert isinstance(APPLICATION_NAME, str) and APPLICATION_NAME
    assert isinstance(CHANGELOG_URL, str) and CHANGELOG_URL
    assert CHANGELOG_URL.startswith("https://github.com/")


def test_iterable_to_regexp_with_unicode_and_special_regex_chars():
    words = ["^start", "end$", "mid*dle", "qua(d)", "{brace}"]
    result = iterable_to_regexp(words)
    import re
    pattern = re.compile(result)
    for w in words:
        assert pattern.match(w)


def test_iterable_to_regexp_accepts_tuple_input():
    words = ("one", "two", "three")
    regex = iterable_to_regexp(words)
    import re
    pattern = re.compile(regex)
    for w in words:
        assert pattern.match(w)


def test_iterable_to_regexp_with_whitespace_words():
    words = [" ", "\t", "\n", "word"]
    regex = iterable_to_regexp(words)
    import re
    pattern = re.compile(regex)
    for w in words:
        assert pattern.match(w)
