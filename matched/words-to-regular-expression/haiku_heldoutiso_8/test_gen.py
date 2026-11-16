import inspect
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
    stream_to_regexp,
)
from w2re import formatters as formatters_module
from w2re.prefix_tree.tree import PrefixTree as TreeFromModule


WORDS = ["apple", "apply", "apt", "banana", "band"]
NON_MEMBERS = ["app", "appl", "applesx", "ban", "bananas", "", "xyz"]


def _compiled(pattern: str):
    return re.compile(pattern)


def test_iterable_to_regexp_returns_string():
    result = iterable_to_regexp(WORDS)
    assert isinstance(result, str)
    assert result


def test_iterable_to_regexp_output_compiles():
    _compiled(iterable_to_regexp(WORDS))


def test_iterable_to_regexp_matches_every_input_word():
    pattern = _compiled(iterable_to_regexp(WORDS))
    for word in WORDS:
        assert pattern.fullmatch(word), f"{word!r} should match"


def test_iterable_to_regexp_rejects_non_members():
    pattern = _compiled(iterable_to_regexp(WORDS))
    for candidate in NON_MEMBERS:
        assert pattern.fullmatch(candidate) is None, f"{candidate!r} should not match"


def test_iterable_to_regexp_single_word():
    pattern = _compiled(iterable_to_regexp(["hello"]))
    assert pattern.fullmatch("hello")
    assert pattern.fullmatch("hell") is None
    assert pattern.fullmatch("helloo") is None


def test_iterable_to_regexp_accepts_generator():
    gen = (w for w in WORDS)
    pattern = _compiled(iterable_to_regexp(gen))
    for word in WORDS:
        assert pattern.fullmatch(word)


def test_iterable_to_regexp_accepts_tuple_and_set():
    from_tuple = _compiled(iterable_to_regexp(tuple(WORDS)))
    from_set = _compiled(iterable_to_regexp(set(WORDS)))
    for word in WORDS:
        assert from_tuple.fullmatch(word)
        assert from_set.fullmatch(word)


def test_iterable_to_regexp_duplicates_do_not_change_matching():
    pattern = _compiled(iterable_to_regexp(WORDS + WORDS))
    for word in WORDS:
        assert pattern.fullmatch(word)
    assert pattern.fullmatch("app") is None


def test_iterable_to_regexp_is_order_independent_for_matching():
    pattern = _compiled(iterable_to_regexp(list(reversed(WORDS))))
    for word in WORDS:
        assert pattern.fullmatch(word)
    for candidate in NON_MEMBERS:
        assert pattern.fullmatch(candidate) is None


def test_iterable_to_regexp_shared_prefixes():
    words = ["test", "testing", "tester"]
    pattern = _compiled(iterable_to_regexp(words))
    for word in words:
        assert pattern.fullmatch(word)
    assert pattern.fullmatch("tes") is None
    assert pattern.fullmatch("testers") is None


def test_iterable_to_regexp_escapes_special_characters():
    words = ["a.b", "c++"]
    pattern = _compiled(iterable_to_regexp(words))
    assert pattern.fullmatch("a.b")
    assert pattern.fullmatch("c++")
    assert pattern.fullmatch("axb") is None
    assert pattern.fullmatch("c") is None


def test_iterable_to_regexp_supports_unicode_words():
    words = ["naïve", "café", "日本"]
    pattern = _compiled(iterable_to_regexp(words))
    for word in words:
        assert pattern.fullmatch(word)
    assert pattern.fullmatch("cafe") is None


def test_iterable_to_regexp_default_formatter_equals_explicit_python_formatter():
    default_result = iterable_to_regexp(WORDS)
    explicit_result = iterable_to_regexp(WORDS, PythonFormatter)
    assert default_result == explicit_result


def test_iterable_to_regexp_matches_prefix_tree_directly():
    via_utils = iterable_to_regexp(WORDS, PythonFormatter)
    via_tree = PrefixTree(WORDS).to_regexp(PythonFormatter)
    assert via_utils == via_tree


def test_iterable_to_regexp_default_formatter_parameter_is_python_formatter():
    sig = inspect.signature(iterable_to_regexp)
    assert sig.parameters["formatter"].default is PythonFormatter


def test_python_formatter_is_base_formatter_subclass():
    assert issubclass(PythonFormatter, BaseFormatter)


def test_python_formatter_is_same_object_across_import_paths():
    assert PythonFormatter is formatters_module.PythonFormatter
    assert PrefixTree is TreeFromModule


def test_python_word_match_formatter_is_base_formatter_subclass():
    assert issubclass(PythonWordMatchFormatter, BaseFormatter)


def test_iterable_to_regexp_with_word_match_formatter_returns_compilable_string():
    result = iterable_to_regexp(WORDS, PythonWordMatchFormatter)
    assert isinstance(result, str)
    _compiled(result)


def test_package_exports_public_api():
    assert callable(iterable_to_regexp)
    assert callable(stream_to_regexp)
    assert w2re.iterable_to_regexp is iterable_to_regexp


def test_version_is_semver_like_string():
    assert isinstance(__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_application_name_constant():
    assert APPLICATION_NAME == "w2re"


def test_changelog_url_constant():
    assert CHANGELOG_URL.startswith("https://")
    assert CHANGELOG_URL.endswith("CHANGELOG.md")


def test_iterable_to_regexp_handles_larger_input():
    words = [f"word{i:04d}" for i in range(500)]
    pattern = _compiled(iterable_to_regexp(words))
    for word in words:
        assert pattern.fullmatch(word)
    assert pattern.fullmatch("word") is None
    assert pattern.fullmatch("word9999") is None
