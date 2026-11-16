import re

import pytest

import w2re
from w2re import (
    APPLICATION_NAME,
    CHANGELOG_URL,
    BaseFormatter,
    PythonFormatter,
    PythonWordMatchFormatter,
    PrefixTree,
    __version__,
    iterable_to_regexp,
)
from w2re import utils
from w2re.formatters import BaseFormatter as FormattersBaseFormatter
from w2re.formatters import PythonFormatter as FormattersPythonFormatter
from w2re.utils import iterable_to_regexp as utils_iterable_to_regexp


WORDS = ["cat", "car", "cart", "dog", "dogma"]


def _fullmatches(pattern: str, text: str) -> bool:
    return re.fullmatch(pattern, text) is not None


# --- iterable_to_regexp -----------------------------------------------------

def test_iterable_to_regexp_returns_str():
    assert isinstance(iterable_to_regexp(WORDS), str)


def test_iterable_to_regexp_result_compiles():
    re.compile(iterable_to_regexp(WORDS))


@pytest.mark.parametrize("word", WORDS)
def test_iterable_to_regexp_matches_every_input_word(word):
    pattern = iterable_to_regexp(WORDS)
    assert _fullmatches(pattern, word)


@pytest.mark.parametrize("text", ["ca", "do", "dogs", "carts", "bird", ""])
def test_iterable_to_regexp_rejects_strings_not_in_input(text):
    pattern = iterable_to_regexp(WORDS)
    assert not _fullmatches(pattern, text)


def test_iterable_to_regexp_accepts_generator_like_list():
    from_list = iterable_to_regexp(list(WORDS))
    from_generator = iterable_to_regexp(word for word in WORDS)
    assert from_generator == from_list


def test_iterable_to_regexp_accepts_tuple_and_set():
    expected = iterable_to_regexp(WORDS)
    assert iterable_to_regexp(tuple(WORDS)) == expected
    assert iterable_to_regexp(set(WORDS)) == expected


def test_iterable_to_regexp_empty_input_returns_compilable_str():
    result = iterable_to_regexp([])
    assert isinstance(result, str)
    re.compile(result)


def test_iterable_to_regexp_duplicates_do_not_change_matching():
    pattern = iterable_to_regexp(["cat", "cat", "dog", "cat"])
    assert _fullmatches(pattern, "cat")
    assert _fullmatches(pattern, "dog")
    assert not _fullmatches(pattern, "ca")


def test_iterable_to_regexp_escapes_regex_metacharacters():
    words = ["a.b", "c+d", "(e)", "f*"]
    pattern = iterable_to_regexp(words)
    for word in words:
        assert _fullmatches(pattern, word)
    assert not _fullmatches(pattern, "axb")
    assert not _fullmatches(pattern, "cccd")
    assert not _fullmatches(pattern, "e")


def test_iterable_to_regexp_supports_unicode_words():
    words = ["café", "naïve", "über"]
    pattern = iterable_to_regexp(words)
    for word in words:
        assert _fullmatches(pattern, word)
    assert not _fullmatches(pattern, "cafe")


def test_iterable_to_regexp_default_formatter_is_python_formatter():
    assert iterable_to_regexp(WORDS) == iterable_to_regexp(WORDS, PythonFormatter)


def test_iterable_to_regexp_with_word_match_formatter_still_matches_words():
    pattern = iterable_to_regexp(WORDS, PythonWordMatchFormatter)
    re.compile(pattern)
    for word in WORDS:
        assert _fullmatches(pattern, word)


# --- PythonFormatter (imported into w2re.utils) ------------------------------

def test_python_formatter_is_base_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)


def test_python_word_match_formatter_is_base_formatter_subclass():
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_utils_module_exposes_same_python_formatter_class():
    assert utils.PythonFormatter is FormattersPythonFormatter
    assert FormattersPythonFormatter is PythonFormatter


def test_package_exports_same_formatter_classes_as_formatters_module():
    assert BaseFormatter is FormattersBaseFormatter
    assert PythonFormatter is FormattersPythonFormatter


# --- w2re/__init__.py --------------------------------------------------------

def test_package_exports_iterable_to_regexp_from_utils():
    assert iterable_to_regexp is utils_iterable_to_regexp
    assert w2re.iterable_to_regexp is utils_iterable_to_regexp


def test_package_exports_prefix_tree_class():
    from w2re.prefix_tree.tree import PrefixTree as TreePrefixTree

    assert PrefixTree is TreePrefixTree


def test_package_version_value():
    assert __version__ == "3.1.0"
    assert w2re.__version__ == __version__


def test_application_name_value():
    assert APPLICATION_NAME == "w2re"
    assert w2re.APPLICATION_NAME == "w2re"


def test_changelog_url_is_https_and_points_to_changelog():
    assert CHANGELOG_URL.startswith("https://")
    assert CHANGELOG_URL.endswith("CHANGELOG.md")
    assert "radeklat/words-to-regular-expression" in CHANGELOG_URL
