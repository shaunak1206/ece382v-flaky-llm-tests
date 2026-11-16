import inspect
import re
from unittest import mock

import w2re
import w2re.utils as w2re_utils
from w2re import (
    BaseFormatter,
    PythonFormatter,
    PythonWordMatchFormatter,
    PrefixTree,
    iterable_to_regexp,
    stream_to_regexp,
)
from w2re.formatters import PythonFormatter as FormattersPythonFormatter
from w2re.formatters import PythonWordMatchFormatter as FormattersPythonWordMatchFormatter


def _all_fullmatch(pattern, words):
    return all(re.fullmatch(pattern, word) is not None for word in words)


def test_iterable_to_regexp_returns_str():
    result = iterable_to_regexp(["hello", "help"])
    assert isinstance(result, str)


def test_iterable_to_regexp_matches_all_given_words():
    words = ["hello", "help", "hero", "world"]
    pattern = iterable_to_regexp(words)
    assert _all_fullmatch(pattern, words)


def test_iterable_to_regexp_rejects_strings_not_in_input():
    pattern = iterable_to_regexp(["hello", "help"])
    for candidate in ["hel", "helloo", "he", "", "hellp"]:
        assert re.fullmatch(pattern, candidate) is None, candidate


def test_iterable_to_regexp_accepts_generator_input():
    words = ["cat", "car", "dog"]
    pattern = iterable_to_regexp(word for word in words)
    assert _all_fullmatch(pattern, words)
    assert re.fullmatch(pattern, "cot") is None


def test_iterable_to_regexp_accepts_tuple_input():
    words = ("alpha", "beta")
    pattern = iterable_to_regexp(words)
    assert _all_fullmatch(pattern, words)


def test_iterable_to_regexp_duplicate_words_do_not_change_result():
    assert iterable_to_regexp(["cat", "cat", "car"]) == iterable_to_regexp(["cat", "car"])


def test_iterable_to_regexp_single_word_matches_only_itself():
    pattern = iterable_to_regexp(["python"])
    assert re.fullmatch(pattern, "python") is not None
    assert re.fullmatch(pattern, "pythons") is None
    assert re.fullmatch(pattern, "pytho") is None


def test_iterable_to_regexp_escapes_special_characters():
    pattern = iterable_to_regexp(["a.c"])
    assert re.fullmatch(pattern, "a.c") is not None
    assert re.fullmatch(pattern, "abc") is None


def test_iterable_to_regexp_supports_unicode_words():
    words = ["naïve", "café"]
    pattern = iterable_to_regexp(words)
    assert _all_fullmatch(pattern, words)


def test_iterable_to_regexp_result_compiles_as_python_regex():
    pattern = iterable_to_regexp(["one", "two", "three", "four"])
    compiled = re.compile(pattern)
    assert compiled.fullmatch("three") is not None


def test_iterable_to_regexp_default_formatter_is_python_formatter():
    signature = inspect.signature(iterable_to_regexp)
    assert signature.parameters["formatter"].default is PythonFormatter


def test_iterable_to_regexp_explicit_python_formatter_matches_default():
    words = ["flower", "flow", "flight"]
    assert iterable_to_regexp(words, PythonFormatter) == iterable_to_regexp(words)


def test_iterable_to_regexp_passes_iterable_to_prefix_tree():
    words = ["red", "green"]
    with mock.patch.object(w2re_utils, "PrefixTree") as tree_cls:
        tree_cls.return_value.to_regexp.return_value = "SENTINEL"
        result = iterable_to_regexp(words, PythonWordMatchFormatter)

    assert result == "SENTINEL"
    tree_cls.assert_called_once_with(words)


def test_iterable_to_regexp_passes_formatter_to_to_regexp():
    with mock.patch.object(w2re_utils, "PrefixTree") as tree_cls:
        tree_cls.return_value.to_regexp.return_value = "SENTINEL"
        iterable_to_regexp(["x"], PythonWordMatchFormatter)

    tree_cls.return_value.to_regexp.assert_called_once_with(PythonWordMatchFormatter)


def test_python_formatter_is_base_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_python_formatter_identity_across_modules():
    assert PythonFormatter is FormattersPythonFormatter
    assert PythonWordMatchFormatter is FormattersPythonWordMatchFormatter


def test_python_word_match_formatter_output_fullmatches_words():
    words = ["hello", "help"]
    pattern = iterable_to_regexp(words, PythonWordMatchFormatter)
    assert isinstance(pattern, str)
    assert re.compile(pattern) is not None
    assert _all_fullmatch(pattern, words)


def test_package_version():
    assert w2re.__version__ == "3.1.0"


def test_package_application_name():
    assert w2re.APPLICATION_NAME == "w2re"


def test_package_changelog_url():
    assert w2re.CHANGELOG_URL.startswith("https://")
    assert w2re.CHANGELOG_URL.endswith("CHANGELOG.md")


def test_package_exports_public_api():
    assert w2re.iterable_to_regexp is w2re_utils.iterable_to_regexp
    assert w2re.stream_to_regexp is w2re_utils.stream_to_regexp
    assert w2re.PrefixTree is PrefixTree
    assert stream_to_regexp is w2re_utils.stream_to_regexp
