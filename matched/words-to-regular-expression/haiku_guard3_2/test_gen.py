import inspect
import re

import w2re
import w2re.utils as w2re_utils
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
from w2re.formatters import PythonFormatter as FormattersPythonFormatter


WORDS = ["hello", "help", "helium", "world", "word"]
NON_WORDS = ["hel", "he", "wor", "x", "helloo", "", "Hello", "worlds"]


def _matches(pattern, word):
    return re.compile(pattern).fullmatch(word) is not None


def test_iterable_to_regexp_returns_str():
    result = iterable_to_regexp(WORDS)
    assert isinstance(result, str)
    assert result != ""


def test_iterable_to_regexp_every_input_word_fully_matches():
    pattern = iterable_to_regexp(WORDS)
    for word in WORDS:
        assert _matches(pattern, word), word


def test_iterable_to_regexp_non_members_do_not_fully_match():
    pattern = iterable_to_regexp(WORDS)
    for word in NON_WORDS:
        assert not _matches(pattern, word), word


def test_iterable_to_regexp_prefix_of_word_is_not_matched():
    pattern = iterable_to_regexp(["hello", "help"])
    assert _matches(pattern, "hello")
    assert _matches(pattern, "help")
    assert not _matches(pattern, "hel")
    assert not _matches(pattern, "hell")


def test_iterable_to_regexp_single_word():
    pattern = iterable_to_regexp(["cat"])
    assert _matches(pattern, "cat")
    assert not _matches(pattern, "ca")
    assert not _matches(pattern, "cats")


def test_iterable_to_regexp_escapes_special_characters():
    pattern = iterable_to_regexp(["a.b", "c+d", "(e)"])
    assert _matches(pattern, "a.b")
    assert _matches(pattern, "c+d")
    assert _matches(pattern, "(e)")
    assert not _matches(pattern, "axb")
    assert not _matches(pattern, "cccd")
    assert not _matches(pattern, "e")


def test_iterable_to_regexp_unicode_words():
    words = ["žluť", "žlutý", "kůň"]
    pattern = iterable_to_regexp(words)
    for word in words:
        assert _matches(pattern, word), word
    assert not _matches(pattern, "žlu")


def test_iterable_to_regexp_accepts_generator_same_as_list():
    from_list = iterable_to_regexp(list(WORDS))
    from_generator = iterable_to_regexp(word for word in WORDS)
    assert from_generator == from_list


def test_iterable_to_regexp_accepts_set_input_with_matching_semantics():
    pattern = iterable_to_regexp(set(WORDS))
    for word in WORDS:
        assert _matches(pattern, word), word
    for word in NON_WORDS:
        assert not _matches(pattern, word), word


def test_iterable_to_regexp_is_deterministic_across_calls():
    assert iterable_to_regexp(WORDS) == iterable_to_regexp(WORDS)


def test_iterable_to_regexp_reversed_input_matches_same_words():
    pattern = iterable_to_regexp(list(reversed(WORDS)))
    for word in WORDS:
        assert _matches(pattern, word), word
    for word in NON_WORDS:
        assert not _matches(pattern, word), word


def test_iterable_to_regexp_duplicates_do_not_change_matching():
    pattern = iterable_to_regexp(WORDS + WORDS)
    for word in WORDS:
        assert _matches(pattern, word), word
    for word in NON_WORDS:
        assert not _matches(pattern, word), word


def test_iterable_to_regexp_default_formatter_is_python_formatter():
    assert iterable_to_regexp(WORDS) == iterable_to_regexp(WORDS, PythonFormatter)


def test_iterable_to_regexp_explicit_python_formatter_matches_words():
    pattern = iterable_to_regexp(WORDS, formatter=PythonFormatter)
    for word in WORDS:
        assert _matches(pattern, word), word
    for word in NON_WORDS:
        assert not _matches(pattern, word), word


def test_python_formatter_is_formatter_base_subclass():
    assert inspect.isclass(PythonFormatter)
    assert issubclass(PythonFormatter, BaseFormatter)


def test_python_formatter_is_same_object_across_modules():
    assert PythonFormatter is FormattersPythonFormatter
    assert w2re_utils.PythonFormatter is FormattersPythonFormatter


def test_python_word_match_formatter_is_formatter_base_subclass():
    assert inspect.isclass(PythonWordMatchFormatter)
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_package_exports_public_api_names():
    assert w2re.iterable_to_regexp is iterable_to_regexp
    assert w2re.stream_to_regexp is stream_to_regexp
    assert w2re.PrefixTree is PrefixTree
    assert w2re.BaseFormatter is BaseFormatter
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.PythonWordMatchFormatter is PythonWordMatchFormatter


def test_version_is_semver_string_and_expected_value():
    assert isinstance(__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__) is not None
    assert __version__ == "3.1.0"


def test_application_name_and_changelog_url_constants():
    assert APPLICATION_NAME == "w2re"
    assert isinstance(CHANGELOG_URL, str)
    assert CHANGELOG_URL.startswith("https://")
    assert CHANGELOG_URL.endswith("CHANGELOG.md")
