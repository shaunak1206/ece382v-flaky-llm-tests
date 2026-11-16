import inspect
import re

import pytest

import w2re
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
from w2re import formatters as formatters_module
from w2re import utils as utils_module

WORDS = ["cat", "car", "cart", "dog", "dodge"]
NON_MATCHING = ["ca", "cats", "do", "dot", "bird", "", "cart2"]


def _fully_matches(pattern, word):
    return re.fullmatch(pattern, word) is not None


# ---------------------------------------------------------------------------
# iterable_to_regexp
# ---------------------------------------------------------------------------

def test_iterable_to_regexp_returns_string():
    result = iterable_to_regexp(WORDS)
    assert isinstance(result, str)
    assert result


@pytest.mark.parametrize("word", WORDS)
def test_iterable_to_regexp_matches_every_input_word(word):
    pattern = iterable_to_regexp(WORDS)
    assert _fully_matches(pattern, word)


@pytest.mark.parametrize("word", NON_MATCHING)
def test_iterable_to_regexp_rejects_words_not_in_input(word):
    pattern = iterable_to_regexp(WORDS)
    assert not _fully_matches(pattern, word)


def test_iterable_to_regexp_single_word_matches_only_that_word():
    pattern = iterable_to_regexp(["hello"])
    assert _fully_matches(pattern, "hello")
    assert not _fully_matches(pattern, "hell")
    assert not _fully_matches(pattern, "hello!")


def test_iterable_to_regexp_accepts_generator_same_as_list():
    from_list = iterable_to_regexp(list(WORDS))
    from_generator = iterable_to_regexp(word for word in WORDS)
    assert from_generator == from_list


def test_iterable_to_regexp_accepts_tuple_same_as_list():
    assert iterable_to_regexp(tuple(WORDS)) == iterable_to_regexp(list(WORDS))


def test_iterable_to_regexp_duplicates_do_not_change_result():
    assert iterable_to_regexp(["cat", "cat", "car"]) == iterable_to_regexp(["cat", "car"])


def test_iterable_to_regexp_is_deterministic():
    assert iterable_to_regexp(WORDS) == iterable_to_regexp(WORDS)


def test_iterable_to_regexp_consumes_whole_iterable():
    consumed = []

    def source():
        for word in WORDS:
            consumed.append(word)
            yield word

    pattern = iterable_to_regexp(source())
    assert consumed == WORDS
    assert _fully_matches(pattern, "dodge")


# ---------------------------------------------------------------------------
# PythonFormatter (and formatter parameter of iterable_to_regexp)
# ---------------------------------------------------------------------------

def test_python_formatter_is_a_class():
    assert inspect.isclass(PythonFormatter)


def test_python_formatter_is_subclass_of_base_formatter():
    assert issubclass(PythonFormatter, BaseFormatter)


def test_python_formatter_is_same_object_across_modules():
    assert utils_module.PythonFormatter is PythonFormatter
    assert formatters_module.PythonFormatter is PythonFormatter
    assert w2re.PythonFormatter is PythonFormatter


def test_iterable_to_regexp_default_formatter_is_python_formatter():
    signature = inspect.signature(iterable_to_regexp)
    assert signature.parameters["formatter"].default is PythonFormatter


def test_iterable_to_regexp_explicit_python_formatter_matches_default():
    assert iterable_to_regexp(WORDS, PythonFormatter) == iterable_to_regexp(WORDS)


def test_iterable_to_regexp_with_word_match_formatter_returns_string():
    result = iterable_to_regexp(WORDS, PythonWordMatchFormatter)
    assert isinstance(result, str)
    assert result


# ---------------------------------------------------------------------------
# w2re/__init__.py public API and metadata
# ---------------------------------------------------------------------------

def test_package_exports_iterable_to_regexp_from_utils():
    assert w2re.iterable_to_regexp is utils_module.iterable_to_regexp


def test_package_exports_stream_to_regexp_from_utils():
    assert w2re.stream_to_regexp is utils_module.stream_to_regexp


def test_package_exports_prefix_tree_class():
    assert w2re.PrefixTree is PrefixTree
    assert inspect.isclass(PrefixTree)


def test_package_version_string():
    assert __version__ == "3.1.0"


def test_package_application_name():
    assert APPLICATION_NAME == "w2re"


def test_package_changelog_url_is_https_and_points_to_changelog():
    assert CHANGELOG_URL.startswith("https://")
    assert CHANGELOG_URL.endswith("CHANGELOG.md")
