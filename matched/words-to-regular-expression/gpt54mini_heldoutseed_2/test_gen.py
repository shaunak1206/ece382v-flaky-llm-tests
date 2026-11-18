import re
from io import StringIO

import w2re
from w2re import PythonFormatter, iterable_to_regexp, stream_to_regexp
from w2re.formatters import BaseFormatter


def test_init_exports_public_api_objects():
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.iterable_to_regexp is iterable_to_regexp
    assert w2re.stream_to_regexp is stream_to_regexp
    assert w2re.BaseFormatter is BaseFormatter


def test_init_version_and_metadata_constants():
    assert w2re.__version__ == "3.1.0"
    assert w2re.APPLICATION_NAME == "w2re"
    assert "github.com/radeklat/words-to-regular-expression" in w2re.CHANGELOG_URL
    assert w2re.CHANGELOG_URL.endswith("CHANGELOG.md")


def test_python_formatter_is_a_base_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)


def test_python_formatter_can_be_instantiated():
    formatter = PythonFormatter()
    assert isinstance(formatter, BaseFormatter)


def test_iterable_to_regexp_returns_a_string():
    regexp = iterable_to_regexp(["cat", "dog"])
    assert isinstance(regexp, str)
    assert regexp


def test_iterable_to_regexp_matches_all_given_words():
    words = ["cat", "dog", "mouse"]
    regexp = iterable_to_regexp(words)
    compiled = re.compile(rf"^(?:{regexp})$")
    for word in words:
        assert compiled.fullmatch(word)


def test_iterable_to_regexp_does_not_match_unlisted_word():
    regexp = iterable_to_regexp(["cat", "dog"])
    compiled = re.compile(rf"^(?:{regexp})$")
    assert compiled.fullmatch("bird") is None


def test_iterable_to_regexp_handles_shared_prefixes():
    words = ["car", "cart", "carbon"]
    regexp = iterable_to_regexp(words)
    compiled = re.compile(rf"^(?:{regexp})$")
    for word in words:
        assert compiled.fullmatch(word)
    assert compiled.fullmatch("care") is None


def test_iterable_to_regexp_single_word_matches_only_that_word():
    regexp = iterable_to_regexp(["hello"])
    compiled = re.compile(rf"^(?:{regexp})$")
    assert compiled.fullmatch("hello")
    assert compiled.fullmatch("hell") is None
    assert compiled.fullmatch("hello!") is None


def test_iterable_to_regexp_escapes_literal_regex_characters():
    words = ["a.b", "a+b", "a?b", "a*b", "a(b)"]
    regexp = iterable_to_regexp(words)
    compiled = re.compile(rf"^(?:{regexp})$")
    for word in words:
        assert compiled.fullmatch(word)
    assert compiled.fullmatch("axb") is None


def test_iterable_to_regexp_accepts_any_iterable_not_just_lists():
    words = (word for word in ["alpha", "beta", "gamma"])
    regexp = iterable_to_regexp(words)
    compiled = re.compile(rf"^(?:{regexp})$")
    for word in ["alpha", "beta", "gamma"]:
        assert compiled.fullmatch(word)


def test_iterable_to_regexp_ignores_empty_strings_in_input():
    regexp = iterable_to_regexp(["", "a", "", "ab"])
    compiled = re.compile(rf"^(?:{regexp})$")
    assert compiled.fullmatch("a")
    assert compiled.fullmatch("ab")
    assert compiled.fullmatch("") is None


def test_stream_to_regexp_matches_non_empty_lines_only():
    stream = StringIO("cat\n\ndog\n")
    regexp = stream_to_regexp(stream)
    compiled = re.compile(rf"^(?:{regexp})$")
    assert compiled.fullmatch("cat")
    assert compiled.fullmatch("dog")
    assert compiled.fullmatch("") is None
    assert compiled.fullmatch("bird") is None
