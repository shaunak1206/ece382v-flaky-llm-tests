import re
import types

import pytest

import w2re
from w2re import (
    APPLICATION_NAME,
    CHANGELOG_URL,
    BaseFormatter,
    PrefixTree,
    PythonFormatter,
    __version__,
    iterable_to_regexp,
)
from w2re import utils as w2re_utils
from w2re.formatters import PythonFormatter as FormattersPythonFormatter

WORDS = ["hello", "help", "helicopter", "world", "word"]
NON_WORDS = ["hel", "wor", "helloo", "worlds", "", "x", "helicopters"]


def _fullmatches(pattern, word):
    return re.fullmatch(pattern, word) is not None


# --- iterable_to_regexp -----------------------------------------------------

def test_iterable_to_regexp_returns_str():
    assert isinstance(iterable_to_regexp(WORDS), str)


def test_iterable_to_regexp_result_compiles_as_python_regex():
    re.compile(iterable_to_regexp(WORDS))


@pytest.mark.parametrize("word", WORDS)
def test_iterable_to_regexp_matches_every_input_word(word):
    pattern = iterable_to_regexp(WORDS)
    assert _fullmatches(pattern, word)


@pytest.mark.parametrize("non_word", NON_WORDS)
def test_iterable_to_regexp_rejects_non_input_strings(non_word):
    pattern = iterable_to_regexp(WORDS)
    assert not _fullmatches(pattern, non_word)


def test_iterable_to_regexp_default_formatter_is_python_formatter():
    assert iterable_to_regexp(WORDS) == iterable_to_regexp(
        WORDS, formatter=PythonFormatter
    )


def test_iterable_to_regexp_accepts_generator():
    generator = (word for word in WORDS)
    assert iterable_to_regexp(generator) == iterable_to_regexp(WORDS)


def test_iterable_to_regexp_accepts_tuple():
    assert iterable_to_regexp(tuple(WORDS)) == iterable_to_regexp(WORDS)


def test_iterable_to_regexp_duplicates_do_not_change_result():
    assert iterable_to_regexp(["cat", "cat", "dog"]) == iterable_to_regexp(
        ["cat", "dog"]
    )


def test_iterable_to_regexp_escapes_special_characters():
    pattern = iterable_to_regexp(["a.b", "a+b"])
    assert _fullmatches(pattern, "a.b")
    assert _fullmatches(pattern, "a+b")
    assert not _fullmatches(pattern, "axb")
    assert not _fullmatches(pattern, "aab")


def test_iterable_to_regexp_single_word_matches_only_that_word():
    pattern = iterable_to_regexp(["cat"])
    assert _fullmatches(pattern, "cat")
    assert not _fullmatches(pattern, "ca")
    assert not _fullmatches(pattern, "cats")


def test_iterable_to_regexp_with_explicit_formatter_is_valid_regex():
    pattern = iterable_to_regexp(WORDS, formatter=PythonFormatter)
    re.compile(pattern)
    for word in WORDS:
        assert _fullmatches(pattern, word)


# --- PythonFormatter --------------------------------------------------------

def test_python_formatter_is_base_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)


def test_python_formatter_is_same_object_across_import_paths():
    assert PythonFormatter is FormattersPythonFormatter


def test_python_formatter_is_reexported_by_utils_module():
    assert w2re_utils.PythonFormatter is PythonFormatter


def test_prefix_tree_to_regexp_with_python_formatter_matches_words():
    pattern = PrefixTree(WORDS).to_regexp(PythonFormatter)
    assert isinstance(pattern, str)
    for word in WORDS:
        assert _fullmatches(pattern, word)


# --- w2re/__init__.py ------------------------------------------------------

def test_package_exports_iterable_to_regexp_from_utils():
    assert w2re.iterable_to_regexp is w2re_utils.iterable_to_regexp


def test_package_version_is_semver_string():
    assert isinstance(__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__) is not None


def test_package_application_name():
    assert APPLICATION_NAME == "w2re"


def test_package_changelog_url_is_https_and_points_to_changelog():
    assert CHANGELOG_URL.startswith("https://")
    assert CHANGELOG_URL.endswith("CHANGELOG.md")


def test_package_exports_formatter_classes():
    assert isinstance(BaseFormatter, type)
    assert isinstance(PythonFormatter, type)
    assert isinstance(w2re.PythonWordMatchFormatter, type)
