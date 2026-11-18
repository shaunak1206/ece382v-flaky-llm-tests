import io
import os
import re

import pytest

import w2re
from w2re import (
    APPLICATION_NAME,
    CHANGELOG_URL,
    BaseFormatter,
    PrefixTree,
    PythonFormatter,
    PythonWordMatchFormatter,
    __version__,
    iterable_to_regexp,
    stream_to_regexp,
)


def test_public_metadata_constants():
    assert __version__ == "3.1.0"
    assert APPLICATION_NAME == "w2re"
    assert CHANGELOG_URL.endswith("/blob/develop/CHANGELOG.md")
    assert CHANGELOG_URL.startswith("https://github.com/radeklat/words-to-regular-expression/")


def test_init_reexports_are_the_same_objects():
    assert w2re.iterable_to_regexp is iterable_to_regexp
    assert w2re.stream_to_regexp is stream_to_regexp
    assert w2re.PrefixTree is PrefixTree
    assert w2re.BaseFormatter is BaseFormatter
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.PythonWordMatchFormatter is PythonWordMatchFormatter


def test_python_formatter_is_a_base_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)


def test_python_formatter_is_reexported_from_package_root():
    assert w2re.PythonFormatter is PythonFormatter


def test_iterable_to_regexp_uses_prefix_tree_and_default_python_formatter(monkeypatch):
    calls = []

    class DummyTree:
        def __init__(self, iterable):
            self.items = list(iterable)

        def to_regexp(self, formatter):
            calls.append((self.items, formatter))
            return "dummy-regexp"

    monkeypatch.setattr("w2re.utils.PrefixTree", DummyTree)

    result = iterable_to_regexp(["alpha", "beta"])

    assert result == "dummy-regexp"
    assert calls == [(["alpha", "beta"], PythonFormatter)]


def test_iterable_to_regexp_accepts_custom_formatter(monkeypatch):
    calls = []

    class CustomFormatter:
        pass

    class DummyTree:
        def __init__(self, iterable):
            self.items = list(iterable)

        def to_regexp(self, formatter):
            calls.append((self.items, formatter))
            return "custom-regexp"

    monkeypatch.setattr("w2re.utils.PrefixTree", DummyTree)

    result = iterable_to_regexp(["alpha"], formatter=CustomFormatter)

    assert result == "custom-regexp"
    assert calls == [(["alpha"], CustomFormatter)]


def test_stream_to_regexp_uses_lines_from_stream_and_default_formatter(monkeypatch):
    calls = []

    class DummyTree:
        def __init__(self, iterable):
            self.items = list(iterable)

        def to_regexp(self, formatter):
            calls.append((self.items, formatter))
            return "stream-regexp"

    monkeypatch.setattr("w2re.utils.PrefixTree", DummyTree)

    content = os.linesep.join(["one", "", "two", ""])
    result = stream_to_regexp(io.StringIO(content))

    assert result == "stream-regexp"
    assert calls == [(["one", "two"], PythonFormatter)]


def test_iterable_to_regexp_matches_all_given_words():
    regex = iterable_to_regexp(["cat", "car", "dog"])
    pattern = re.compile(regex)

    assert pattern.fullmatch("cat")
    assert pattern.fullmatch("car")
    assert pattern.fullmatch("dog")
    assert not pattern.fullmatch("cow")


def test_iterable_to_regexp_works_with_generator_iterables():
    words = (word for word in ["alpha", "beta", "gamma"])
    regex = iterable_to_regexp(words)
    pattern = re.compile(regex)

    assert pattern.fullmatch("alpha")
    assert pattern.fullmatch("beta")
    assert pattern.fullmatch("gamma")
    assert not pattern.fullmatch("delta")


def test_iterable_to_regexp_can_handle_words_with_shared_prefixes():
    regex = iterable_to_regexp(["car", "cart", "cat"])
    pattern = re.compile(regex)

    assert pattern.fullmatch("car")
    assert pattern.fullmatch("cart")
    assert pattern.fullmatch("cat")
    assert not pattern.fullmatch("can")


def test_iterable_to_regexp_single_word_is_matchable():
    regex = iterable_to_regexp(["hello"])
    pattern = re.compile(regex)

    assert pattern.fullmatch("hello")
    assert not pattern.fullmatch("hell")
    assert not pattern.fullmatch("hello!")


def test_iterable_to_regexp_output_is_a_valid_python_regex():
    regex = iterable_to_regexp(["a", "ab", "abc"])
    re.compile(regex)  # should not raise


def test_iterable_to_regexp_matches_whitespace_containing_words():
    regex = iterable_to_regexp(["new york", "san francisco"])
    pattern = re.compile(regex)

    assert pattern.fullmatch("new york")
    assert pattern.fullmatch("san francisco")
    assert not pattern.fullmatch("newyork")
