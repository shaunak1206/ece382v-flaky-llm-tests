import io
import os
import re

import pytest

import w2re
from w2re import (
    BaseFormatter,
    PrefixTree,
    PythonFormatter,
    PythonWordMatchFormatter,
    iterable_to_regexp,
    stream_to_regexp,
)
import w2re.utils as utils


def assert_matches(pattern, positives=(), negatives=()):
    compiled = re.compile(pattern)
    for word in positives:
        assert compiled.fullmatch(word), f"{pattern!r} should match {word!r}"
    for word in negatives:
        assert not compiled.fullmatch(word), f"{pattern!r} should not match {word!r}"


def test_package_reexports_public_api():
    assert w2re.iterable_to_regexp is iterable_to_regexp
    assert w2re.stream_to_regexp is stream_to_regexp
    assert w2re.PrefixTree is PrefixTree
    assert w2re.BaseFormatter is BaseFormatter
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.PythonWordMatchFormatter is PythonWordMatchFormatter


def test_package_metadata_constants():
    assert w2re.__version__ == "3.1.0"
    assert w2re.APPLICATION_NAME == "w2re"
    assert w2re.CHANGELOG_URL.endswith("/blob/develop/CHANGELOG.md")
    assert "github.com/radeklat/words-to-regular-expression" in w2re.CHANGELOG_URL


def test_iterable_to_regexp_uses_default_python_formatter(monkeypatch):
    captured = {}

    class DummyPrefixTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "dummy-pattern"

    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    result = utils.iterable_to_regexp(["alpha", "beta"])

    assert result == "dummy-pattern"
    assert captured["iterable"] == ["alpha", "beta"]
    assert captured["formatter"] is PythonFormatter


def test_iterable_to_regexp_passes_custom_formatter(monkeypatch):
    captured = {}

    class DummyFormatter:
        pass

    class DummyPrefixTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "dummy-pattern"

    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    result = utils.iterable_to_regexp(["alpha"], formatter=DummyFormatter)

    assert result == "dummy-pattern"
    assert captured["iterable"] == ["alpha"]
    assert captured["formatter"] is DummyFormatter


def test_iterable_to_regexp_accepts_generator_input():
    words = (word for word in ["cat", "car", "dog"])
    pattern = iterable_to_regexp(words)

    assert_matches(pattern, positives=["cat", "car", "dog"], negatives=["cab", "do"])


def test_iterable_to_regexp_produces_regex_for_single_word():
    pattern = iterable_to_regexp(["hello"])
    assert_matches(pattern, positives=["hello"], negatives=["hell", "hello!"])


def test_iterable_to_regexp_handles_shared_prefixes():
    pattern = iterable_to_regexp(["cat", "car"])
    assert_matches(pattern, positives=["cat", "car"], negatives=["cab", "ca", "cart"])


def test_iterable_to_regexp_handles_prefix_words():
    pattern = iterable_to_regexp(["a", "ab"])
    assert_matches(pattern, positives=["a", "ab"], negatives=["", "abc", "b"])


def test_iterable_to_regexp_ignores_duplicate_words_semantically():
    pattern = iterable_to_regexp(["same", "same", "same"])
    assert_matches(pattern, positives=["same"], negatives=["sam", "samex"])


def test_stream_to_regexp_uses_default_python_formatter(monkeypatch):
    captured = {}

    class DummyPrefixTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "dummy-pattern"

    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    stream = io.StringIO(f"one{os.linesep}two{os.linesep}")
    result = utils.stream_to_regexp(stream)

    assert result == "dummy-pattern"
    assert captured["iterable"] == ["one", "two"]
    assert captured["formatter"] is PythonFormatter


def test_stream_to_regexp_filters_empty_lines():
    stream = io.StringIO(f"alpha{os.linesep}{os.linesep}beta{os.linesep}{os.linesep}")
    pattern = stream_to_regexp(stream)

    assert_matches(pattern, positives=["alpha", "beta"], negatives=["", "gamma"])


def test_stream_to_regexp_passes_custom_formatter(monkeypatch):
    captured = {}

    class DummyFormatter:
        pass

    class DummyPrefixTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "dummy-pattern"

    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    stream = io.StringIO(f"a{os.linesep}b{os.linesep}")
    result = utils.stream_to_regexp(stream, formatter=DummyFormatter)

    assert result == "dummy-pattern"
    assert captured["iterable"] == ["a", "b"]
    assert captured["formatter"] is DummyFormatter


def test_stream_to_regexp_with_single_line():
    stream = io.StringIO(f"lonely{os.linesep}")
    pattern = stream_to_regexp(stream)

    assert_matches(pattern, positives=["lonely"], negatives=["", "lonelyx", "xlonely"])


def test_prefix_tree_to_regexp_works_with_python_formatter():
    tree = PrefixTree(["alpha", "alphabet", "alpine"])
    pattern = tree.to_regexp(PythonFormatter)

    assert_matches(
        pattern,
        positives=["alpha", "alphabet", "alpine"],
        negatives=["alp", "alphan", "beta"],
    )
