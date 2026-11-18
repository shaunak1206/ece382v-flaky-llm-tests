import io
import os

import w2re
import w2re.utils as utils


class DummyTree:
    instances = []

    def __init__(self, iterable):
        self.items = list(iterable)
        self.to_regexp_calls = []
        DummyTree.instances.append(self)

    def to_regexp(self, formatter):
        self.to_regexp_calls.append(formatter)
        return f"regexp:{formatter.__name__}:{'|'.join(self.items)}"


def test_iterable_to_regexp_uses_default_python_formatter(monkeypatch):
    monkeypatch.setattr(utils, "PrefixTree", DummyTree)
    DummyTree.instances.clear()

    result = utils.iterable_to_regexp(["a", "b"])

    assert result == "regexp:PythonFormatter:a|b"
    assert len(DummyTree.instances) == 1
    assert DummyTree.instances[0].items == ["a", "b"]
    assert DummyTree.instances[0].to_regexp_calls == [w2re.PythonFormatter]


def test_iterable_to_regexp_accepts_custom_formatter(monkeypatch):
    monkeypatch.setattr(utils, "PrefixTree", DummyTree)
    DummyTree.instances.clear()

    class CustomFormatter:
        pass

    result = utils.iterable_to_regexp(["x"], formatter=CustomFormatter)

    assert result == "regexp:CustomFormatter:x"
    assert DummyTree.instances[0].to_regexp_calls == [CustomFormatter]


def test_iterable_to_regexp_passes_generator_through(monkeypatch):
    monkeypatch.setattr(utils, "PrefixTree", DummyTree)
    DummyTree.instances.clear()

    generator = (str(i) for i in range(3))
    result = utils.iterable_to_regexp(generator)

    assert result == "regexp:PythonFormatter:0|1|2"
    assert DummyTree.instances[0].items == ["0", "1", "2"]


def test_iterable_to_regexp_does_not_drop_duplicate_items(monkeypatch):
    monkeypatch.setattr(utils, "PrefixTree", DummyTree)
    DummyTree.instances.clear()

    result = utils.iterable_to_regexp(["same", "same", "other"])

    assert result == "regexp:PythonFormatter:same|same|other"
    assert DummyTree.instances[0].items == ["same", "same", "other"]


def test_stream_to_regexp_uses_default_python_formatter(monkeypatch):
    monkeypatch.setattr(utils, "PrefixTree", DummyTree)
    DummyTree.instances.clear()

    content = os.linesep.join(["alpha", "beta"])
    result = utils.stream_to_regexp(io.StringIO(content))

    assert result == "regexp:PythonFormatter:alpha|beta"
    assert len(DummyTree.instances) == 1
    assert DummyTree.instances[0].items == ["alpha", "beta"]
    assert DummyTree.instances[0].to_regexp_calls == [w2re.PythonFormatter]


def test_stream_to_regexp_filters_empty_lines(monkeypatch):
    monkeypatch.setattr(utils, "PrefixTree", DummyTree)
    DummyTree.instances.clear()

    content = os.linesep.join(["first", "", "second", "", "third", ""])
    result = utils.stream_to_regexp(io.StringIO(content))

    assert result == "regexp:PythonFormatter:first|second|third"
    assert DummyTree.instances[0].items == ["first", "second", "third"]


def test_stream_to_regexp_accepts_custom_formatter(monkeypatch):
    monkeypatch.setattr(utils, "PrefixTree", DummyTree)
    DummyTree.instances.clear()

    class CustomFormatter:
        pass

    result = utils.stream_to_regexp(io.StringIO("only"), formatter=CustomFormatter)

    assert result == "regexp:CustomFormatter:only"
    assert DummyTree.instances[0].to_regexp_calls == [CustomFormatter]


def test_stream_to_regexp_uses_os_linesep_for_splitting(monkeypatch):
    monkeypatch.setattr(utils, "PrefixTree", DummyTree)
    DummyTree.instances.clear()

    content = f"a{os.linesep}b{os.linesep}c"
    utils.stream_to_regexp(io.StringIO(content))

    assert DummyTree.instances[0].items == ["a", "b", "c"]


def test_w2re_version_is_exposed():
    assert w2re.__version__ == "3.1.0"


def test_w2re_application_name_is_exposed():
    assert w2re.APPLICATION_NAME == "w2re"


def test_w2re_changelog_url_is_exposed_and_points_to_github():
    assert w2re.CHANGELOG_URL.startswith("https://github.com/radeklat/words-to-regular-expression/")
    assert w2re.CHANGELOG_URL.endswith("CHANGELOG.md")


def test_w2re_exports_utils_functions():
    assert w2re.iterable_to_regexp is utils.iterable_to_regexp
    assert w2re.stream_to_regexp is utils.stream_to_regexp


def test_w2re_exports_formatter_classes_and_prefix_tree():
    from w2re.formatters import BaseFormatter, PythonFormatter, PythonWordMatchFormatter
    from w2re.prefix_tree.tree import PrefixTree

    assert w2re.BaseFormatter is BaseFormatter
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.PythonWordMatchFormatter is PythonWordMatchFormatter
    assert w2re.PrefixTree is PrefixTree
