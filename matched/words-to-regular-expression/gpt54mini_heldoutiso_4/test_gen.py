import re
from io import StringIO

import w2re
from w2re import (
    APPLICATION_NAME,
    CHANGELOG_URL,
    BaseFormatter,
    PrefixTree,
    PythonFormatter,
    PythonWordMatchFormatter,
    iterable_to_regexp,
    stream_to_regexp,
)


def _compiled(pattern):
    return re.compile(pattern)


def _matches_all(pattern, words):
    rx = _compiled(pattern)
    return all(rx.fullmatch(word) for word in words)


def _matches_none(pattern, words):
    rx = _compiled(pattern)
    return all(rx.fullmatch(word) is None for word in words)


def test_package_version_and_metadata_constants():
    assert w2re.__version__ == "3.1.0"
    assert APPLICATION_NAME == "w2re"
    assert CHANGELOG_URL.startswith("https://github.com/radeklat/words-to-regular-expression/")


def test_init_reexports_expected_classes_and_functions():
    assert BaseFormatter is not None
    assert PythonFormatter is not None
    assert PythonWordMatchFormatter is not None
    assert PrefixTree is not None
    assert iterable_to_regexp is not None
    assert stream_to_regexp is not None


def test_pythonformatter_is_reexported_from_package():
    from w2re.formatters import PythonFormatter as DirectPythonFormatter

    assert PythonFormatter is DirectPythonFormatter


def test_pythonwordmatchformatter_is_reexported_from_package():
    from w2re.formatters import PythonWordMatchFormatter as DirectPythonWordMatchFormatter

    assert PythonWordMatchFormatter is DirectPythonWordMatchFormatter


def test_iterable_to_regexp_returns_string_and_compiles_for_single_word():
    pattern = iterable_to_regexp(["hello"])
    assert isinstance(pattern, str)
    assert _matches_all(pattern, ["hello"])
    assert _matches_none(pattern, ["hell", "hello!"])


def test_iterable_to_regexp_matches_multiple_distinct_words():
    words = ["cat", "dog", "bird"]
    pattern = iterable_to_regexp(words)
    assert _matches_all(pattern, words)
    assert _matches_none(pattern, ["ca", "do", "bir", "cats"])


def test_iterable_to_regexp_handles_shared_prefixes():
    words = ["car", "cart", "carbon"]
    pattern = iterable_to_regexp(words)
    assert _matches_all(pattern, words)
    assert _matches_none(pattern, ["ca", "care", "carb", "cars"])


def test_iterable_to_regexp_order_independent_for_same_word_set():
    words1 = ["alpha", "beta", "gamma", "delta"]
    words2 = list(reversed(words1))
    pattern1 = iterable_to_regexp(words1)
    pattern2 = iterable_to_regexp(words2)

    assert _matches_all(pattern1, words1)
    assert _matches_all(pattern2, words1)
    assert _matches_none(pattern1, ["alp", "bet", "gamm", "delt"])
    assert _matches_none(pattern2, ["alp", "bet", "gamm", "delt"])


def test_iterable_to_regexp_accepts_tuples():
    words = ("sun", "moon", "stars")
    pattern = iterable_to_regexp(words)
    assert _matches_all(pattern, words)
    assert _matches_none(pattern, ["su", "moons", "star"])


def test_iterable_to_regexp_accepts_generators():
    words = ["one", "two", "three"]

    def generator():
        for word in words:
            yield word

    pattern = iterable_to_regexp(generator())
    assert _matches_all(pattern, words)
    assert _matches_none(pattern, ["on", "tw", "thre"])


def test_stream_to_regexp_uses_default_formatter_and_matches_words():
    stream = StringIO("red\ngreen\nblue")
    pattern = stream_to_regexp(stream)
    assert isinstance(pattern, str)
    assert _matches_all(pattern, ["red", "green", "blue"])
    assert _matches_none(pattern, ["re", "gren", "blu"])


def test_pythonformatter_default_output_from_prefix_tree_compiles_and_matches():
    tree = PrefixTree(["a", "ab", "abc"])
    pattern = tree.to_regexp(PythonFormatter)
    assert isinstance(pattern, str)
    assert _matches_all(pattern, ["a", "ab", "abc"])
    assert _matches_none(pattern, ["", "b", "abcd"])


def test_custom_formatter_class_can_be_passed_to_iterable_to_regexp():
    class CustomFormatter(PythonFormatter):
        pass

    pattern = iterable_to_regexp(["x", "xy"], formatter=CustomFormatter)
    assert isinstance(pattern, str)
    assert _matches_all(pattern, ["x", "xy"])
    assert _matches_none(pattern, ["", "xyz"])
