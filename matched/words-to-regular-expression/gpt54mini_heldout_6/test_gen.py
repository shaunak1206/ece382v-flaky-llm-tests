import io
import os

import pytest

import w2re
import w2re.utils as utils


def test_package_exports_version_and_constants():
    assert w2re.__version__ == "3.1.0"
    assert w2re.APPLICATION_NAME == "w2re"
    assert w2re.CHANGELOG_URL.endswith("/blob/develop/CHANGELOG.md")


def test_package_reexports_formatter_classes():
    from w2re.formatters import BaseFormatter, PythonFormatter, PythonWordMatchFormatter

    assert w2re.BaseFormatter is BaseFormatter
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.PythonWordMatchFormatter is PythonWordMatchFormatter


def test_package_reexports_utils_functions():
    assert w2re.iterable_to_regexp is utils.iterable_to_regexp
    assert w2re.stream_to_regexp is utils.stream_to_regexp


def test_iterable_to_regexp_uses_prefix_tree_and_default_formatter(monkeypatch):
    captured = {}

    class FakeTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "fake-regexp"

    monkeypatch.setattr(utils, "PrefixTree", FakeTree)

    result = utils.iterable_to_regexp(["alpha", "beta"])

    assert result == "fake-regexp"
    assert captured["iterable"] == ["alpha", "beta"]
    assert captured["formatter"] is w2re.PythonFormatter


def test_iterable_to_regexp_accepts_custom_formatter(monkeypatch):
    captured = {}

    class FakeFormatter:
        pass

    class FakeTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "custom"

    monkeypatch.setattr(utils, "PrefixTree", FakeTree)

    result = utils.iterable_to_regexp(["one"], formatter=FakeFormatter)

    assert result == "custom"
    assert captured["iterable"] == ["one"]
    assert captured["formatter"] is FakeFormatter


def test_iterable_to_regexp_passes_generator_without_consuming_extra(monkeypatch):
    seen = []

    class FakeTree:
        def __init__(self, iterable):
            seen.extend(iterable)

        def to_regexp(self, formatter):
            return "ok"

    monkeypatch.setattr(utils, "PrefixTree", FakeTree)

    gen = (word for word in ["a", "b", "c"])
    assert utils.iterable_to_regexp(gen) == "ok"
    assert seen == ["a", "b", "c"]


def test_stream_to_regexp_filters_empty_lines_and_uses_os_linesep(monkeypatch):
    captured = {}

    class FakeTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "stream-regexp"

    monkeypatch.setattr(utils, "PrefixTree", FakeTree)

    stream = io.StringIO(f"alpha{os.linesep}{os.linesep}beta{os.linesep}")
    result = utils.stream_to_regexp(stream)

    assert result == "stream-regexp"
    assert captured["iterable"] == ["alpha", "beta"]
    assert captured["formatter"] is w2re.PythonFormatter


def test_stream_to_regexp_accepts_custom_formatter(monkeypatch):
    captured = {}

    class FakeFormatter:
        pass

    class FakeTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            captured["formatter"] = formatter
            return "custom-stream"

    monkeypatch.setattr(utils, "PrefixTree", FakeTree)

    stream = io.StringIO("x\ny")
    result = utils.stream_to_regexp(stream, formatter=FakeFormatter)

    assert result == "custom-stream"
    assert captured["iterable"] == ["x\ny"]
    assert captured["formatter"] is FakeFormatter


def test_stream_to_regexp_returns_empty_when_input_only_blank_lines(monkeypatch):
    captured = {}

    class FakeTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            return "empty"

    monkeypatch.setattr(utils, "PrefixTree", FakeTree)

    stream = io.StringIO(f"{os.linesep}{os.linesep}")
    result = utils.stream_to_regexp(stream)

    assert result == "empty"
    assert captured["iterable"] == []


def test_iterable_to_regexp_can_handle_empty_iterable(monkeypatch):
    captured = {}

    class FakeTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            return "empty"

    monkeypatch.setattr(utils, "PrefixTree", FakeTree)

    result = utils.iterable_to_regexp([])

    assert result == "empty"
    assert captured["iterable"] == []


def test_stream_to_regexp_does_not_strip_whitespace_only_lines(monkeypatch):
    captured = {}

    class FakeTree:
        def __init__(self, iterable):
            captured["iterable"] = list(iterable)

        def to_regexp(self, formatter):
            return "ok"

    monkeypatch.setattr(utils, "PrefixTree", FakeTree)

    stream = io.StringIO(f"   {os.linesep}word{os.linesep}")
    utils.stream_to_regexp(stream)

    assert captured["iterable"] == ["   ", "word"]


def test_iterable_to_regexp_uses_prefix_tree_to_regexp_return_value(monkeypatch):
    class FakeTree:
        def __init__(self, iterable):
            self.items = list(iterable)

        def to_regexp(self, formatter):
            return f"{len(self.items)}-items"

    monkeypatch.setattr(utils, "PrefixTree", FakeTree)

    assert utils.iterable_to_regexp(["a", "b", "c"]) == "3-items"


def test_imported_functions_work_from_package_namespace():
    assert w2re.iterable_to_regexp(["hello"]) == utils.iterable_to_regexp(["hello"])
    assert w2re.stream_to_regexp(io.StringIO("hello")) == utils.stream_to_regexp(io.StringIO("hello"))
