import io
import os

import pytest

import w2re
import w2re.utils as utils


class DummyPrefixTree:
    instances = []

    def __init__(self, iterable):
        self.items = list(iterable)
        self.to_regexp_calls = []
        DummyPrefixTree.instances.append(self)

    def to_regexp(self, formatter):
        self.to_regexp_calls.append(formatter)
        return f"regexp:{formatter.__name__}:{'|'.join(self.items)}"


class DummyFormatter:
    pass


class AnotherDummyFormatter:
    pass


def test_iterable_to_regexp_uses_prefix_tree_and_default_formatter(monkeypatch):
    DummyPrefixTree.instances.clear()
    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    result = utils.iterable_to_regexp(["cat", "dog"])

    assert result == "regexp:PythonFormatter:cat|dog"
    assert len(DummyPrefixTree.instances) == 1
    assert DummyPrefixTree.instances[0].items == ["cat", "dog"]
    assert DummyPrefixTree.instances[0].to_regexp_calls == [w2re.PythonFormatter]


def test_iterable_to_regexp_passes_custom_formatter(monkeypatch):
    DummyPrefixTree.instances.clear()
    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    result = utils.iterable_to_regexp(["a", "b"], formatter=DummyFormatter)

    assert result == "regexp:DummyFormatter:a|b"
    assert DummyPrefixTree.instances[0].to_regexp_calls == [DummyFormatter]


def test_iterable_to_regexp_preserves_generator_input_order(monkeypatch):
    DummyPrefixTree.instances.clear()
    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    def gen():
        yield "x"
        yield "y"
        yield "z"

    result = utils.iterable_to_regexp(gen())

    assert result == "regexp:PythonFormatter:x|y|z"
    assert DummyPrefixTree.instances[0].items == ["x", "y", "z"]


def test_iterable_to_regexp_accepts_empty_iterable(monkeypatch):
    DummyPrefixTree.instances.clear()
    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    result = utils.iterable_to_regexp([])

    assert result == "regexp:PythonFormatter:"
    assert DummyPrefixTree.instances[0].items == []


def test_iterable_to_regexp_does_not_filter_whitespace_strings(monkeypatch):
    DummyPrefixTree.instances.clear()
    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    result = utils.iterable_to_regexp([" ", "\t", "word"])

    assert result == "regexp:PythonFormatter: |\t|word"
    assert DummyPrefixTree.instances[0].items == [" ", "\t", "word"]


def test_iterable_to_regexp_allows_duplicate_items(monkeypatch):
    DummyPrefixTree.instances.clear()
    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    result = utils.iterable_to_regexp(["same", "same", "other"])

    assert result == "regexp:PythonFormatter:same|same|other"
    assert DummyPrefixTree.instances[0].items == ["same", "same", "other"]


def test_stream_to_regexp_uses_os_linesep_and_ignores_empty_lines(monkeypatch):
    DummyPrefixTree.instances.clear()
    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    stream = io.StringIO(f"alpha{os.linesep}{os.linesep}beta{os.linesep}gamma{os.linesep}")

    result = utils.stream_to_regexp(stream)

    assert result == "regexp:PythonFormatter:alpha|beta|gamma"
    assert DummyPrefixTree.instances[0].items == ["alpha", "beta", "gamma"]


def test_stream_to_regexp_keeps_lines_that_contain_only_spaces(monkeypatch):
    DummyPrefixTree.instances.clear()
    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    stream = io.StringIO(f"  {os.linesep}x{os.linesep}")

    result = utils.stream_to_regexp(stream)

    assert result == "regexp:PythonFormatter:  |x"
    assert DummyPrefixTree.instances[0].items == ["  ", "x"]


def test_stream_to_regexp_passes_custom_formatter(monkeypatch):
    DummyPrefixTree.instances.clear()
    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    stream = io.StringIO(f"a{os.linesep}b")
    result = utils.stream_to_regexp(stream, formatter=AnotherDummyFormatter)

    assert result == "regexp:AnotherDummyFormatter:a|b"
    assert DummyPrefixTree.instances[0].to_regexp_calls == [AnotherDummyFormatter]


def test_stream_to_regexp_does_not_strip_internal_whitespace(monkeypatch):
    DummyPrefixTree.instances.clear()
    monkeypatch.setattr(utils, "PrefixTree", DummyPrefixTree)

    stream = io.StringIO(f"foo bar{os.linesep}baz\tqux")
    result = utils.stream_to_regexp(stream)

    assert result == "regexp:PythonFormatter:foo bar|baz\tqux"
    assert DummyPrefixTree.instances[0].items == ["foo bar", "baz\tqux"]


def test_pythonformatter_is_reexported_from_w2re():
    assert w2re.PythonFormatter is utils.PythonFormatter
    assert w2re.BaseFormatter is utils.BaseFormatter
    assert w2re.PythonWordMatchFormatter is utils.PythonWordMatchFormatter


def test_prefix_tree_is_reexported_from_w2re():
    assert w2re.PrefixTree is utils.PrefixTree


def test_iterable_to_regexp_is_reexported_from_w2re():
    assert w2re.iterable_to_regexp is utils.iterable_to_regexp


def test_stream_to_regexp_is_reexported_from_w2re():
    assert w2re.stream_to_regexp is utils.stream_to_regexp


def test_w2re_metadata_constants():
    assert w2re.__version__ == "3.1.0"
    assert w2re.APPLICATION_NAME == "w2re"
    assert "github.com/radeklat/words-to-regular-expression/" in w2re.CHANGELOG_URL
    assert w2re.CHANGELOG_URL.endswith("CHANGELOG.md")


def test_pythonformatter_class_name_is_stable():
    assert w2re.PythonFormatter.__name__ == "PythonFormatter"
    assert utils.PythonFormatter.__name__ == "PythonFormatter"
