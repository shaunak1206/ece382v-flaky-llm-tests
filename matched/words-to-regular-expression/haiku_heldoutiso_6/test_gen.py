import re
import types

import pytest

import w2re
import w2re.utils
from w2re import (
    APPLICATION_NAME,
    CHANGELOG_URL,
    BaseFormatter,
    PrefixTree,
    PythonFormatter,
    PythonWordMatchFormatter,
    __version__,
    iterable_to_regexp,
)
from w2re.formatters import PythonFormatter as FormattersPythonFormatter

WORDS = ["hello", "help", "world", "word"]
NON_MATCHES = ["hell", "helloo", "wor", "", "xyz", "Hello", " hello"]


def _fullmatches(pattern, text):
    return re.fullmatch(pattern, text) is not None


def test_iterable_to_regexp_returns_str():
    result = iterable_to_regexp(WORDS)
    assert isinstance(result, str)
    assert result


def test_default_formatter_matches_explicit_python_formatter():
    assert iterable_to_regexp(WORDS) == iterable_to_regexp(WORDS, PythonFormatter)


def test_result_compiles_as_python_regex():
    compiled = re.compile(iterable_to_regexp(WORDS))
    assert compiled is not None


@pytest.mark.parametrize("word", WORDS)
def test_pattern_fully_matches_every_input_word(word):
    pattern = iterable_to_regexp(WORDS)
    assert _fullmatches(pattern, word)


@pytest.mark.parametrize("text", NON_MATCHES)
def test_pattern_rejects_strings_not_in_input(text):
    pattern = iterable_to_regexp(WORDS)
    assert not _fullmatches(pattern, text)


def test_prefix_words_are_all_matched_and_extensions_rejected():
    words = ["a", "ab", "abc"]
    pattern = iterable_to_regexp(words)
    for word in words:
        assert _fullmatches(pattern, word)
    assert not _fullmatches(pattern, "abcd")
    assert not _fullmatches(pattern, "ad")


def test_duplicate_inputs_do_not_change_matching():
    with_duplicates = ["cat", "cat", "car", "car", "cart"]
    without_duplicates = ["cat", "car", "cart"]
    pattern_dup = iterable_to_regexp(with_duplicates)
    pattern_unique = iterable_to_regexp(without_duplicates)
    samples = ["cat", "car", "cart", "ca", "carts", "dog"]
    for sample in samples:
        assert _fullmatches(pattern_dup, sample) == _fullmatches(pattern_unique, sample)


def test_input_order_does_not_change_matched_language():
    forward = iterable_to_regexp(WORDS)
    backward = iterable_to_regexp(list(reversed(WORDS)))
    samples = WORDS + NON_MATCHES
    for sample in samples:
        assert _fullmatches(forward, sample) == _fullmatches(backward, sample)


def test_accepts_generator_input():
    generator = (word for word in WORDS)
    pattern = iterable_to_regexp(generator)
    for word in WORDS:
        assert _fullmatches(pattern, word)
    assert not _fullmatches(pattern, "hell")


def test_accepts_tuple_and_set_input():
    from_tuple = iterable_to_regexp(tuple(WORDS))
    from_set = iterable_to_regexp(set(WORDS))
    for word in WORDS:
        assert _fullmatches(from_tuple, word)
        assert _fullmatches(from_set, word)
    assert not _fullmatches(from_tuple, "xyz")
    assert not _fullmatches(from_set, "xyz")


def test_single_word_input_matches_only_that_word():
    pattern = iterable_to_regexp(["solo"])
    assert _fullmatches(pattern, "solo")
    assert not _fullmatches(pattern, "sol")
    assert not _fullmatches(pattern, "solos")


def test_regex_special_characters_are_treated_literally():
    words = ["a.b", "a+b", "(x)"]
    pattern = iterable_to_regexp(words)
    for word in words:
        assert _fullmatches(pattern, word)
    assert not _fullmatches(pattern, "axb")
    assert not _fullmatches(pattern, "aab")
    assert not _fullmatches(pattern, "x")


def test_unicode_words_are_matched():
    words = ["café", "naïve", "日本"]
    pattern = iterable_to_regexp(words)
    for word in words:
        assert _fullmatches(pattern, word)
    assert not _fullmatches(pattern, "cafe")


def test_python_formatter_is_a_base_formatter_subclass():
    assert isinstance(PythonFormatter, type)
    assert issubclass(PythonFormatter, BaseFormatter)


def test_utils_module_exposes_the_same_python_formatter_class():
    assert w2re.utils.PythonFormatter is FormattersPythonFormatter


def test_word_match_formatter_produces_text_matching_standalone_word():
    pattern = iterable_to_regexp(WORDS, PythonWordMatchFormatter)
    assert isinstance(pattern, str)
    assert re.search(pattern, "say hello there") is not None


def test_package_exports_expected_public_names():
    assert hasattr(w2re, "iterable_to_regexp")
    assert hasattr(w2re, "stream_to_regexp")
    assert w2re.iterable_to_regexp is w2re.utils.iterable_to_regexp
    assert w2re.PrefixTree is PrefixTree
    assert w2re.BaseFormatter is BaseFormatter


def test_version_is_semantic_version_string():
    assert isinstance(__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_application_name_constant():
    assert APPLICATION_NAME == "w2re"


def test_changelog_url_is_https_and_points_to_changelog():
    assert CHANGELOG_URL.startswith("https://")
    assert CHANGELOG_URL.endswith("CHANGELOG.md")
