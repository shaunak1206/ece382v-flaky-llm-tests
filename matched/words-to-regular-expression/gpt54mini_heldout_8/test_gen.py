import io
import re

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
import w2re
import w2re.utils


def _assert_regex_matches(pattern, word):
    assert re.fullmatch(pattern, word), f"{pattern!r} should match {word!r}"


def _assert_regex_does_not_match(pattern, word):
    assert not re.fullmatch(pattern, word), f"{pattern!r} should not match {word!r}"


def test_package_exports_expected_metadata():
    assert __version__ == "3.1.0"
    assert APPLICATION_NAME == "w2re"
    assert CHANGELOG_URL.startswith("https://github.com/radeklat/words-to-regular-expression/")


def test_package_reexports_formatter_and_tree_symbols():
    assert w2re.BaseFormatter is BaseFormatter
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.PythonWordMatchFormatter is PythonWordMatchFormatter
    assert w2re.PrefixTree is PrefixTree


def test_package_reexports_helper_functions():
    assert w2re.iterable_to_regexp is iterable_to_regexp
    assert w2re.stream_to_regexp is stream_to_regexp


def test_python_formatter_is_exposed_from_utils_and_package():
    assert w2re.utils.PythonFormatter is PythonFormatter
    assert issubclass(PythonFormatter, BaseFormatter)


def test_iterable_to_regexp_returns_string():
    pattern = iterable_to_regexp(["alpha", "beta"])
    assert isinstance(pattern, str)
    assert pattern


def test_iterable_to_regexp_matches_single_word_exactly():
    pattern = iterable_to_regexp(["hello"])
    _assert_regex_matches(pattern, "hello")
    _assert_regex_does_not_match(pattern, "hell")
    _assert_regex_does_not_match(pattern, "hello!")


def test_iterable_to_regexp_matches_multiple_words():
    pattern = iterable_to_regexp(["cat", "dog", "bird"])
    for word in ["cat", "dog", "bird"]:
        _assert_regex_matches(pattern, word)


def test_iterable_to_regexp_does_not_match_missing_word():
    pattern = iterable_to_regexp(["cat", "dog", "bird"])
    _assert_regex_does_not_match(pattern, "cow")
    _assert_regex_does_not_match(pattern, "cats")


def test_iterable_to_regexp_handles_prefix_words():
    pattern = iterable_to_regexp(["car", "cart", "care"])
    for word in ["car", "cart", "care"]:
        _assert_regex_matches(pattern, word)
    _assert_regex_does_not_match(pattern, "ca")
    _assert_regex_does_not_match(pattern, "carts")


def test_iterable_to_regexp_ignores_duplicates_semantically():
    unique_pattern = iterable_to_regexp(["same", "other"])
    duplicate_pattern = iterable_to_regexp(["same", "same", "other"])
    for word in ["same", "other"]:
        _assert_regex_matches(unique_pattern, word)
        _assert_regex_matches(duplicate_pattern, word)
    _assert_regex_does_not_match(duplicate_pattern, "different")


def test_iterable_to_regexp_accepts_custom_formatter_argument():
    default_pattern = iterable_to_regexp(["foo", "bar"])
    explicit_pattern = iterable_to_regexp(["foo", "bar"], PythonFormatter)
    assert default_pattern == explicit_pattern
    for word in ["foo", "bar"]:
        _assert_regex_matches(explicit_pattern, word)


def test_stream_to_regexp_ignores_blank_lines():
    stream = io.StringIO("alpha\n\nbeta\n")
    pattern = stream_to_regexp(stream)
    _assert_regex_matches(pattern, "alpha")
    _assert_regex_matches(pattern, "beta")
    _assert_regex_does_not_match(pattern, "")


def test_stream_to_regexp_uses_os_line_separators():
    linesep = __import__("os").linesep
    stream = io.StringIO(f"red{linesep}green{linesep}{linesep}blue")
    pattern = stream_to_regexp(stream)
    for word in ["red", "green", "blue"]:
        _assert_regex_matches(pattern, word)
    _assert_regex_does_not_match(pattern, "yellow")
