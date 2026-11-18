import io
import re

import pytest

import w2re
import w2re.utils as utils
from w2re.formatters import BaseFormatter, PythonFormatter, PythonWordMatchFormatter


def test_package_exports_expected_names():
    assert w2re.BaseFormatter is BaseFormatter
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.PythonWordMatchFormatter is PythonWordMatchFormatter
    assert w2re.PrefixTree is utils.PrefixTree
    assert w2re.iterable_to_regexp is utils.iterable_to_regexp
    assert w2re.stream_to_regexp is utils.stream_to_regexp


def test_package_metadata_values():
    assert w2re.__version__ == "3.1.0"
    assert w2re.APPLICATION_NAME == "w2re"
    assert w2re.CHANGELOG_URL.endswith("blob/develop/CHANGELOG.md")
    assert "github.com/radeklat/words-to-regular-expression/" in w2re.CHANGELOG_URL


def test_python_formatter_is_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)


def test_python_word_match_formatter_is_formatter_subclass():
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_iterable_to_regexp_uses_default_python_formatter(monkeypatch):
    captured = {}

    class FakePrefixTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "fake-regexp"

    monkeypatch.setattr(utils, "PrefixTree", FakePrefixTree)

    result = utils.iterable_to_regexp(["alpha", "beta"])

    assert result == "fake-regexp"
    assert captured["iterable"] == ["alpha", "beta"]
    assert captured["formatter"] is PythonFormatter


def test_iterable_to_regexp_passes_custom_formatter(monkeypatch):
    captured = {}

    class CustomFormatter:
        pass

    class FakePrefixTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "custom-regexp"

    monkeypatch.setattr(utils, "PrefixTree", FakePrefixTree)

    result = utils.iterable_to_regexp(["x"], formatter=CustomFormatter)

    assert result == "custom-regexp"
    assert captured["iterable"] == ["x"]
    assert captured["formatter"] is CustomFormatter


def test_iterable_to_regexp_consumes_iterable_once(monkeypatch):
    consumed = []

    def generator():
        for item in ["one", "two", "three"]:
            consumed.append(item)
            yield item

    class FakePrefixTree:
        def __init__(self, iterable):
            self.items = list(iterable)

        def to_regexp(self, formatter):
            return ",".join(self.items)

    monkeypatch.setattr(utils, "PrefixTree", FakePrefixTree)

    result = utils.iterable_to_regexp(generator())

    assert result == "one,two,three"
    assert consumed == ["one", "two", "three"]


def test_stream_to_regexp_uses_default_python_formatter(monkeypatch):
    captured = {}

    class FakePrefixTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "stream-regexp"

    monkeypatch.setattr(utils, "PrefixTree", FakePrefixTree)

    result = utils.stream_to_regexp(io.StringIO("alpha\nbeta\n"))

    assert result == "stream-regexp"
    assert captured["iterable"] == ["alpha", "beta"]
    assert captured["formatter"] is PythonFormatter


def test_stream_to_regexp_filters_empty_lines(monkeypatch):
    captured = {}

    class FakePrefixTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            return "ignored"

    monkeypatch.setattr(utils, "PrefixTree", FakePrefixTree)

    utils.stream_to_regexp(io.StringIO("\nalpha\n\nbeta\n"))

    assert captured["iterable"] == ["alpha", "beta"]


def test_stream_to_regexp_passes_custom_formatter(monkeypatch):
    captured = {}

    class CustomFormatter:
        pass

    class FakePrefixTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "stream-custom"

    monkeypatch.setattr(utils, "PrefixTree", FakePrefixTree)

    result = utils.stream_to_regexp(io.StringIO("x\ny"), formatter=CustomFormatter)

    assert result == "stream-custom"
    assert captured["iterable"] == ["x", "y"]
    assert captured["formatter"] is CustomFormatter


def test_iterable_to_regexp_produces_regex_matching_all_words():
    result = utils.iterable_to_regexp(["cat", "car", "dog"])

    assert re.fullmatch(result, "cat")
    assert re.fullmatch(result, "car")
    assert re.fullmatch(result, "dog")
    assert not re.fullmatch(result, "cow")


def test_iterable_to_regexp_for_single_word_matches_only_that_word():
    result = utils.iterable_to_regexp(["hello"])

    assert re.fullmatch(result, "hello")
    assert not re.fullmatch(result, "hell")
    assert not re.fullmatch(result, "hello!")


def test_iterable_to_regexp_for_duplicate_words_still_matches_the_word():
    result = utils.iterable_to_regexp(["repeat", "repeat", "repeat"])

    assert re.fullmatch(result, "repeat")
    assert not re.fullmatch(result, "repe")
