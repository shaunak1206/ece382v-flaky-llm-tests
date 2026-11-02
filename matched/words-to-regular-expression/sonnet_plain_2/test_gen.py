import io
import os
import re

import pytest

import w2re
from w2re import (
    BaseFormatter,
    PythonFormatter,
    PythonWordMatchFormatter,
    iterable_to_regexp,
    stream_to_regexp,
)
from w2re.prefix_tree.primitives import PrefixTreeNode


def _assert_matches_exactly(words, candidates=()):
    """All words must fully match the regexp; all candidates must not."""
    pattern = re.compile(iterable_to_regexp(list(words)))
    for word in words:
        assert pattern.fullmatch(word), "{!r} should match {}".format(word, pattern.pattern)
    for candidate in candidates:
        assert not pattern.fullmatch(candidate), \
            "{!r} should not match {}".format(candidate, pattern.pattern)


def test_empty_iterable_gives_empty_string_matcher():
    regexp = iterable_to_regexp([])
    assert regexp == r"\A\Z"
    pattern = re.compile(regexp)
    assert pattern.match("")
    assert not pattern.match("a")


def test_iterable_of_empty_strings_gives_empty_string_matcher():
    assert iterable_to_regexp(["", ""]) == r"\A\Z"


def test_single_word_is_returned_verbatim():
    assert iterable_to_regexp(["abc"]) == "abc"


def test_single_character_word():
    assert iterable_to_regexp(["x"]) == "x"


def test_two_adjacent_letters_form_character_class():
    assert iterable_to_regexp(["a", "b"]) == "[ab]"


def test_consecutive_letters_and_digits_collapse_to_ranges():
    assert iterable_to_regexp(["a", "b", "c"]) == "[a-c]"
    assert iterable_to_regexp(["1", "2", "3", "4"]) == "[1-4]"
    assert iterable_to_regexp(["a", "b", "c", "x", "y"]) == "[a-cxy]"


def test_common_prefix_is_factored_out():
    regexp = iterable_to_regexp(["cat", "car"])
    assert regexp == "ca[rt]"
    _assert_matches_exactly(["cat", "car"], ["ca", "cab", "cart", "at"])


def test_word_that_is_prefix_of_another_word():
    assert iterable_to_regexp(["cat", "cats"]) == "cats?"
    assert iterable_to_regexp(["cats", "cat"]) == "cats?"
    _assert_matches_exactly(["cat", "cats"], ["ca", "catss", "cas"])


def test_duplicates_do_not_change_result():
    assert iterable_to_regexp(["dog", "dog", "dog"]) == iterable_to_regexp(["dog"])
    assert iterable_to_regexp(["a", "b", "a"]) == iterable_to_regexp(["a", "b"])


def test_result_matches_same_language_regardless_of_input_order():
    words = ["apple", "apply", "ape", "banana", "band", "bandana", "can"]
    non_words = ["app", "appl", "ban", "bana", "bandan", "ca", "cans", "", "apples"]
    forward = re.compile(iterable_to_regexp(words))
    backward = re.compile(iterable_to_regexp(list(reversed(words))))
    for text in words + non_words:
        assert bool(forward.fullmatch(text)) == bool(backward.fullmatch(text))
        assert bool(forward.fullmatch(text)) == (text in words)


def test_regex_special_characters_are_escaped():
    words = ["a.b", "a+b", "a*b", "(x)", "q?", "$1", "^z", "a|b", "{1}"]
    _assert_matches_exactly(words, ["axb", "aab", "x", "q", "a", "1", "z", "b"])


def test_brackets_backslash_and_hyphen_inside_character_class():
    words = ["[", "]", "\\", "-", "a"]
    _assert_matches_exactly(words, ["", "[]", "b", "--", "ab"])


def test_repeated_substrings_are_compressed_but_still_match():
    assert iterable_to_regexp(["aaa"]) == "a{3}"
    _assert_matches_exactly(["aaa"], ["aa", "aaaa", ""])
    _assert_matches_exactly(["abababab"], ["ababab", "abababa", "ababababab"])
    _assert_matches_exactly(["...", "x..."], ["..", "....", "x.."])


def test_accepts_any_iterable_of_strings():
    from_generator = iterable_to_regexp(w for w in ["red", "reed"])
    from_tuple = iterable_to_regexp(("red", "reed"))
    from_set = iterable_to_regexp({"red", "reed"})
    for regexp in (from_generator, from_tuple, from_set):
        pattern = re.compile(regexp)
        assert pattern.fullmatch("red")
        assert pattern.fullmatch("reed")
        assert not pattern.fullmatch("re")
        assert not pattern.fullmatch("reedy")


def test_unicode_words():
    _assert_matches_exactly(["café", "cafe", "naïve"], ["caf", "naive"])


def test_default_formatter_is_python_formatter():
    words = ["one", "two", "three"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words, PythonFormatter)
    assert iterable_to_regexp(words) == iterable_to_regexp(words, formatter=PythonFormatter)


def test_word_match_formatter_wraps_with_word_boundaries():
    regexp = iterable_to_regexp(["cat"], PythonWordMatchFormatter)
    assert regexp == r"(?:\W+|\A)(cat)(?=\W+|\Z)"
    pattern = re.compile(regexp)
    assert pattern.search("a cat sat").group(1) == "cat"
    assert pattern.search("cat").group(1) == "cat"
    assert pattern.search("cat.").group(1) == "cat"
    assert pattern.search("concatenate") is None
    assert pattern.search("cats") is None


def test_word_match_formatter_with_multiple_words_and_empty_input():
    pattern = re.compile(iterable_to_regexp(["cat", "dog"], PythonWordMatchFormatter))
    assert [m.group(1) for m in pattern.finditer("cat, dog and bird")] == ["cat", "dog"]
    assert iterable_to_regexp([], PythonWordMatchFormatter) == r"\A\Z"


def test_python_formatter_metadata_and_wrap_regexp():
    assert PythonFormatter.code() == "py"
    assert PythonFormatter.description() == "Python regular expression"
    assert PythonWordMatchFormatter.code() == "pyw"
    assert PythonWordMatchFormatter.description() == "Python word matching regular expression"

    assert PythonFormatter.wrap_regexp(PrefixTreeNode()) == r"\A\Z"
    node = PrefixTreeNode()
    node.add("hi")
    assert PythonFormatter.wrap_regexp(node) == "hi"


def test_base_formatter_is_abstract():
    with pytest.raises(NotImplementedError):
        BaseFormatter.wrap_regexp(PrefixTreeNode())
    with pytest.raises(AssertionError):
        BaseFormatter.description()
    with pytest.raises(AssertionError):
        BaseFormatter.code()


def test_stream_to_regexp_ignores_blank_lines():
    text = os.linesep.join(["cat", "", "car", ""])
    regexp = stream_to_regexp(io.StringIO(text))
    assert regexp == "ca[rt]"
    assert stream_to_regexp(io.StringIO("")) == r"\A\Z"


def test_package_level_exports_and_metadata():
    assert w2re.iterable_to_regexp is iterable_to_regexp
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.PrefixTree is not None
    assert w2re.APPLICATION_NAME == "w2re"
    assert isinstance(w2re.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", w2re.__version__)
    assert w2re.CHANGELOG_URL.startswith("https://")
