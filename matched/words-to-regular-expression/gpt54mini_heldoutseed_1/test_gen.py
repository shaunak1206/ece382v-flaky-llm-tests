import io
import os
import re

import pytest

import w2re
from w2re import BaseFormatter, PrefixTree, PythonFormatter, PythonWordMatchFormatter
from w2re.utils import iterable_to_regexp, stream_to_regexp


def test_init_exports_expected_symbols():
    assert w2re.BaseFormatter is BaseFormatter
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.PythonWordMatchFormatter is PythonWordMatchFormatter
    assert w2re.PrefixTree is PrefixTree


def test_init_exports_expected_metadata():
    assert w2re.__version__ == "3.1.0"
    assert w2re.APPLICATION_NAME == "w2re"
    assert "github.com/radeklat/words-to-regular-expression/" in w2re.CHANGELOG_URL


def test_python_formatter_is_available_and_instantiable():
    formatter = PythonFormatter()
    assert isinstance(formatter, BaseFormatter)
    assert formatter.__class__.__name__ == "PythonFormatter"


def test_iterable_to_regexp_matches_all_words():
    pattern = iterable_to_regexp(["cat", "car"])
    assert re.fullmatch(pattern, "cat")
    assert re.fullmatch(pattern, "car")
    assert not re.fullmatch(pattern, "cab")


def test_iterable_to_regexp_handles_prefix_words():
    pattern = iterable_to_regexp(["a", "ab", "abc"])
    assert re.fullmatch(pattern, "a")
    assert re.fullmatch(pattern, "ab")
    assert re.fullmatch(pattern, "abc")
    assert not re.fullmatch(pattern, "abcd")


def test_iterable_to_regexp_escapes_regex_metacharacters():
    words = ["a.b", "a+b", "a?b", "a(b)"]
    pattern = iterable_to_regexp(words)
    for word in words:
        assert re.fullmatch(pattern, word)
    assert not re.fullmatch(pattern, "axb")


def test_iterable_to_regexp_accepts_generator_iterables():
    pattern = iterable_to_regexp(word for word in ["hello", "world"])
    assert re.fullmatch(pattern, "hello")
    assert re.fullmatch(pattern, "world")
    assert not re.fullmatch(pattern, "helloworld")


def test_iterable_to_regexp_produces_same_pattern_for_repeated_input():
    words = ["alpha", "beta", "gamma"]
    pattern1 = iterable_to_regexp(words)
    pattern2 = iterable_to_regexp(tuple(words))
    assert pattern1 == pattern2


def test_stream_to_regexp_matches_lines_from_stream():
    content = os.linesep.join(["dog", "cat", "bird"])
    pattern = stream_to_regexp(io.StringIO(content))
    for word in ["dog", "cat", "bird"]:
        assert re.fullmatch(pattern, word)
    assert not re.fullmatch(pattern, "fish")


def test_stream_to_regexp_ignores_empty_lines():
    content = os.linesep.join(["alpha", "", "beta", "", "gamma"])
    pattern = stream_to_regexp(io.StringIO(content))
    assert re.fullmatch(pattern, "alpha")
    assert re.fullmatch(pattern, "beta")
    assert re.fullmatch(pattern, "gamma")
    assert not re.fullmatch(pattern, "")


def test_stream_to_regexp_uses_os_linesep_splitting():
    content = "one\n" if os.linesep == "\n" else "one\r\n"
    content += f"two{os.linesep}three"
    pattern = stream_to_regexp(io.StringIO(content))
    assert re.fullmatch(pattern, "one")
    assert re.fullmatch(pattern, "two")
    assert re.fullmatch(pattern, "three")


def test_iterable_to_regexp_passes_iterable_into_prefix_tree(monkeypatch):
    captured = {}

    class DummyPrefixTree:
        def __init__(self, iterable):
            captured["items"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "dummy-pattern"

    monkeypatch.setattr("w2re.utils.PrefixTree", DummyPrefixTree)
    result = iterable_to_regexp(["x", "y", "z"], formatter=PythonFormatter)
    assert result == "dummy-pattern"
    assert captured["items"] == ["x", "y", "z"]
    assert captured["formatter"] is PythonFormatter


def test_stream_to_regexp_filters_empty_lines_before_prefix_tree(monkeypatch):
    captured = {}

    class DummyPrefixTree:
        def __init__(self, iterable):
            captured["items"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "stream-pattern"

    monkeypatch.setattr("w2re.utils.PrefixTree", DummyPrefixTree)
    stream = io.StringIO(f"a{os.linesep}{os.linesep}b{os.linesep}c{os.linesep}")
    result = stream_to_regexp(stream, formatter=PythonWordMatchFormatter)
    assert result == "stream-pattern"
    assert captured["items"] == ["a", "b", "c"]
    assert captured["formatter"] is PythonWordMatchFormatter
