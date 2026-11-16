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
from w2re import utils
from w2re.formatters import (
    BaseFormatter as FormattersBaseFormatter,
    PythonFormatter as FormattersPythonFormatter,
    PythonWordMatchFormatter as FormattersPythonWordMatchFormatter,
)


SAMPLE_WORDS = ["apple", "apply", "application", "banana", "band"]
NON_MATCHING_WORDS = ["app", "appl", "bananas", "ban", "bandana", "x", "applicationz"]


def _fullmatches(pattern, word):
    return re.fullmatch(pattern, word) is not None


def test_iterable_to_regexp_returns_nonempty_string():
    result = iterable_to_regexp(SAMPLE_WORDS)
    assert isinstance(result, str)
    assert result


def test_iterable_to_regexp_result_compiles_as_python_regex():
    pattern = iterable_to_regexp(SAMPLE_WORDS)
    re.compile(pattern)


@pytest.mark.parametrize("word", SAMPLE_WORDS)
def test_iterable_to_regexp_matches_every_input_word(word):
    pattern = iterable_to_regexp(SAMPLE_WORDS)
    assert _fullmatches(pattern, word)


@pytest.mark.parametrize("word", NON_MATCHING_WORDS)
def test_iterable_to_regexp_rejects_words_not_in_input(word):
    pattern = iterable_to_regexp(SAMPLE_WORDS)
    assert not _fullmatches(pattern, word)


def test_iterable_to_regexp_accepts_generator_input():
    pattern = iterable_to_regexp(word for word in SAMPLE_WORDS)
    assert all(_fullmatches(pattern, word) for word in SAMPLE_WORDS)
    assert not _fullmatches(pattern, "app")


def test_iterable_to_regexp_accepts_tuple_and_set_input():
    from_tuple = iterable_to_regexp(tuple(SAMPLE_WORDS))
    from_set = iterable_to_regexp(set(SAMPLE_WORDS))
    for pattern in (from_tuple, from_set):
        assert all(_fullmatches(pattern, word) for word in SAMPLE_WORDS)
        assert not _fullmatches(pattern, "ban")


def test_iterable_to_regexp_duplicate_words_do_not_change_matching():
    pattern = iterable_to_regexp(SAMPLE_WORDS + SAMPLE_WORDS)
    assert all(_fullmatches(pattern, word) for word in SAMPLE_WORDS)
    assert not _fullmatches(pattern, "app")


def test_iterable_to_regexp_single_word():
    pattern = iterable_to_regexp(["hello"])
    assert _fullmatches(pattern, "hello")
    assert not _fullmatches(pattern, "hell")
    assert not _fullmatches(pattern, "hellos")


def test_iterable_to_regexp_escapes_regex_metacharacters():
    words = ["a.b", "c+d", "(x)", "e*"]
    pattern = iterable_to_regexp(words)
    for word in words:
        assert _fullmatches(pattern, word)
    assert not _fullmatches(pattern, "axb")
    assert not _fullmatches(pattern, "cd")
    assert not _fullmatches(pattern, "x")


def test_iterable_to_regexp_handles_unicode_words():
    words = ["žluť", "kůň", "ž"]
    pattern = iterable_to_regexp(words)
    for word in words:
        assert _fullmatches(pattern, word)
    assert not _fullmatches(pattern, "žlu")


def test_iterable_to_regexp_default_formatter_equals_explicit_python_formatter():
    default_result = iterable_to_regexp(SAMPLE_WORDS)
    explicit_result = iterable_to_regexp(SAMPLE_WORDS, PythonFormatter)
    assert default_result == explicit_result


def test_python_formatter_is_base_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)


def test_python_word_match_formatter_is_base_formatter_subclass():
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_iterable_to_regexp_with_python_word_match_formatter_still_matches_words():
    pattern = iterable_to_regexp(SAMPLE_WORDS, PythonWordMatchFormatter)
    re.compile(pattern)
    assert all(_fullmatches(pattern, word) for word in SAMPLE_WORDS)
    assert not _fullmatches(pattern, "app")


def test_package_exports_are_same_objects_as_submodules():
    assert BaseFormatter is FormattersBaseFormatter
    assert PythonFormatter is FormattersPythonFormatter
    assert PythonWordMatchFormatter is FormattersPythonWordMatchFormatter
    assert w2re.iterable_to_regexp is utils.iterable_to_regexp
    assert w2re.stream_to_regexp is utils.stream_to_regexp
    assert w2re.PrefixTree is PrefixTree


def test_package_version_is_semver_string():
    assert isinstance(__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_package_application_name_constant():
    assert APPLICATION_NAME == "w2re"


def test_package_changelog_url_points_to_changelog():
    assert CHANGELOG_URL.startswith("https://")
    assert CHANGELOG_URL.endswith("CHANGELOG.md")
