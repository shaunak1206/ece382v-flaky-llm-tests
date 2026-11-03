import re

import pytest

import w2re
import w2re.formatters
import w2re.prefix_tree.tree
import w2re.utils
from w2re import (
    APPLICATION_NAME,
    CHANGELOG_URL,
    BaseFormatter,
    PythonFormatter,
    PythonWordMatchFormatter,
    __version__,
    iterable_to_regexp,
)
from w2re.formatters import ALL_FORMATTERS
from w2re.prefix_tree.primitives import PrefixTreeNode

EMPTY_MATCH = r"\A\Z"

CAT_DOG_WORDS = ["cat", "car", "dog", "do", "dot"]
CAT_DOG_PROBES = [
    "", "c", "ca", "cat", "car", "cart", "cats", "carr",
    "d", "do", "dog", "dogs", "dot", "doot", "cdog", "x",
]


class _TaggingFormatter(BaseFormatter):
    _DESCRIPTION = "tagging test formatter"
    _CODE = "tag"

    @staticmethod
    def wrap_regexp(root_node):
        return "custom:" + root_node.to_regexp()


def _assert_language(regexp, accepted, rejected):
    for word in accepted:
        assert re.fullmatch(regexp, word) is not None, (regexp, word)
    for word in rejected:
        assert re.fullmatch(regexp, word) is None, (regexp, word)


def test_empty_iterable_matches_only_empty_string():
    regexp = iterable_to_regexp([])
    assert regexp == EMPTY_MATCH
    _assert_language(regexp, [""], ["a", " ", "cat"])


def test_empty_string_words_are_ignored():
    assert iterable_to_regexp(["", "cat", ""]) == iterable_to_regexp(["cat"])
    assert iterable_to_regexp([""]) == EMPTY_MATCH


def test_result_is_str_and_compiles():
    regexp = iterable_to_regexp(["cat", "dog", "fish"])
    assert isinstance(regexp, str)
    re.compile(regexp)


def test_single_word_regexp_is_the_word():
    regexp = iterable_to_regexp(["cat"])
    assert regexp == "cat"
    _assert_language(regexp, ["cat"], ["ca", "cats", "cot", "catcat", ""])


def test_all_inserted_words_match():
    words = ["cat", "car", "cart", "dog", "do", "dot", "doghouse"]
    regexp = iterable_to_regexp(words)
    _assert_language(regexp, words, [])


def test_language_equals_word_set():
    regexp = iterable_to_regexp(CAT_DOG_WORDS)
    for probe in CAT_DOG_PROBES:
        assert (re.fullmatch(regexp, probe) is not None) == (probe in CAT_DOG_WORDS), probe


def test_shared_prefix_and_optional_suffix():
    regexp = iterable_to_regexp(["car", "cart"])
    _assert_language(regexp, ["car", "cart"], ["ca", "carts", "cartt", "cr", ""])


def test_single_letter_words_form_character_class():
    regexp = iterable_to_regexp(["a", "b", "c"])
    _assert_language(regexp, ["a", "b", "c"], ["d", "ab", "abc", "", "A"])


def test_regex_metacharacters_are_escaped():
    regexp = iterable_to_regexp(["a.b", "a+b"])
    _assert_language(regexp, ["a.b", "a+b"], ["axb", "aab", "ab", "a.b.", "a+b+"])


def test_repeated_block_is_compressed_but_exact():
    regexp = iterable_to_regexp(["abcabc"])
    _assert_language(regexp, ["abcabc"], ["abcab", "abc", "abcabcabc", "", "abcabd"])


def test_duplicate_words_do_not_change_output():
    assert iterable_to_regexp(["cat", "dog", "cat"]) == iterable_to_regexp(["cat", "dog"])


def test_insertion_order_does_not_change_language():
    for ordering in (list(reversed(CAT_DOG_WORDS)), sorted(CAT_DOG_WORDS),
                     CAT_DOG_WORDS[2:] + CAT_DOG_WORDS[:2]):
        regexp = iterable_to_regexp(ordering)
        for probe in CAT_DOG_PROBES:
            assert (re.fullmatch(regexp, probe) is not None) == (probe in CAT_DOG_WORDS), probe


def test_generator_input_is_consumed_once():
    seen = []

    def gen():
        for word in ["cat", "dog"]:
            seen.append(word)
            yield word

    regexp = iterable_to_regexp(gen())
    assert seen == ["cat", "dog"]
    _assert_language(regexp, ["cat", "dog"], ["cow"])


def test_tuple_and_set_inputs_match_list_language():
    words = ["cat", "dog", "do"]
    for container in (tuple(words), set(words), frozenset(words)):
        regexp = iterable_to_regexp(container)
        _assert_language(regexp, words, ["c", "cats", "dogs", ""])


def test_default_formatter_is_python_formatter():
    words = ["cat", "dog"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words, PythonFormatter)


def test_word_match_formatter_adds_word_boundaries():
    regexp = iterable_to_regexp(["cat", "dog"], PythonWordMatchFormatter)
    assert regexp.startswith(r"(?:\W+|\A)(")
    assert regexp.endswith(r")(?=\W+|\Z)")
    match = re.search(regexp, "the cat sat")
    assert match is not None and match.group(1) == "cat"
    assert re.search(regexp, "concatenate") is None
    assert re.search(regexp, "hotdog") is None


def test_word_match_formatter_empty_input():
    assert iterable_to_regexp([], PythonWordMatchFormatter) == EMPTY_MATCH
    assert iterable_to_regexp([""], PythonWordMatchFormatter) == EMPTY_MATCH


def test_custom_formatter_receives_root_node():
    assert iterable_to_regexp(["cat"], _TaggingFormatter) == "custom:cat"
    assert iterable_to_regexp([], _TaggingFormatter) == "custom:"


def test_python_formatter_wraps_nodes():
    assert PythonFormatter.wrap_regexp(PrefixTreeNode()) == EMPTY_MATCH
    node = PrefixTreeNode()
    node.add("cat")
    assert PythonFormatter.wrap_regexp(node) == "cat"


def test_formatter_metadata():
    assert PythonFormatter.description() == "Python regular expression"
    assert PythonFormatter.code() == "py"
    assert PythonWordMatchFormatter.description() == "Python word matching regular expression"
    assert PythonWordMatchFormatter.code() == "pyw"


def test_base_formatter_requires_subclass_overrides():
    with pytest.raises(AssertionError):
        BaseFormatter.description()
    with pytest.raises(AssertionError):
        BaseFormatter.code()


def test_all_formatters_registry():
    assert ALL_FORMATTERS == (PythonFormatter, PythonWordMatchFormatter)


def test_package_exports_same_objects():
    assert w2re.iterable_to_regexp is w2re.utils.iterable_to_regexp
    assert w2re.PythonFormatter is w2re.formatters.PythonFormatter
    assert w2re.PythonWordMatchFormatter is w2re.formatters.PythonWordMatchFormatter
    assert w2re.BaseFormatter is w2re.formatters.BaseFormatter
    assert w2re.PrefixTree is w2re.prefix_tree.tree.PrefixTree


def test_version_and_metadata_constants():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
    assert APPLICATION_NAME == "w2re"
    assert CHANGELOG_URL.startswith("https://github.com/radeklat/words-to-regular-expression/")
    assert CHANGELOG_URL.endswith("CHANGELOG.md")
