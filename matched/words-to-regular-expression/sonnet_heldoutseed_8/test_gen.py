import io
import os
import re

import pytest

import w2re
from w2re import PythonFormatter, iterable_to_regexp, stream_to_regexp
from w2re.formatters import BaseFormatter


def _assert_matches_exactly(words, non_words=()):
    regexp = iterable_to_regexp(words)
    compiled = re.compile(regexp)
    for word in words:
        assert compiled.fullmatch(word), f"{word!r} should match {regexp!r}"
    for word in non_words:
        assert not compiled.fullmatch(word), f"{word!r} should not match {regexp!r}"
    return regexp


def test_returns_string():
    assert isinstance(iterable_to_regexp(["cat", "dog"]), str)


def test_result_is_valid_regexp():
    re.compile(iterable_to_regexp(["cat", "car", "dog"]))


def test_single_word():
    _assert_matches_exactly(["hello"], ["hell", "helloo", "hallo", ""])


def test_words_with_common_prefix():
    _assert_matches_exactly(
        ["car", "cart", "cards", "care"],
        ["ca", "c", "cars", "cartt", "card", "carts"],
    )


def test_word_that_is_prefix_of_another():
    _assert_matches_exactly(["a", "ab", "abc"], ["", "b", "abcd", "ac", "bc"])


def test_words_with_common_suffix_and_prefix():
    _assert_matches_exactly(
        ["running", "jumping", "runner", "jumper"],
        ["run", "jump", "runnerr", "jumpingg", "runping"],
    )


def test_no_common_prefix():
    _assert_matches_exactly(["apple", "banana", "cherry"], ["app", "bananas", "cherr", "date"])


def test_duplicates_do_not_change_matching():
    _assert_matches_exactly(["cat", "cat", "car", "cat"], ["ca", "cats"])


def test_input_order_does_not_change_matched_set():
    words = ["alpha", "alphabet", "beta", "bet", "gamma"]
    candidates = words + ["alph", "be", "gam", "alphabets", "", "delta"]
    first = re.compile(iterable_to_regexp(words))
    second = re.compile(iterable_to_regexp(list(reversed(words))))
    for candidate in candidates:
        assert bool(first.fullmatch(candidate)) == bool(second.fullmatch(candidate))
        assert bool(first.fullmatch(candidate)) == (candidate in words)


@pytest.mark.parametrize(
    "words",
    [
        ["a.b", "a+b", "a*b"],
        ["(x)", "[y]", "{z}"],
        ["a|b", "c^d", "e$f"],
        ["back\\slash", "q?"],
    ],
)
def test_special_characters_are_escaped(words):
    regexp = iterable_to_regexp(words)
    compiled = re.compile(regexp)
    for word in words:
        assert compiled.fullmatch(word)
    # the dot must be literal, not a wildcard
    if "a.b" in words:
        assert not compiled.fullmatch("axb")
        assert not compiled.fullmatch("aab")
    assert not compiled.fullmatch("a")
    assert not compiled.fullmatch("b")


def test_unicode_words():
    _assert_matches_exactly(["žluťoučký", "kůň", "日本語", "日本"], ["kuň", "日", "žluťouč"])


def test_accepts_generator_input():
    words = ["tree", "trees", "tread"]
    regexp = iterable_to_regexp(w for w in words)
    compiled = re.compile(regexp)
    for word in words:
        assert compiled.fullmatch(word)
    assert not compiled.fullmatch("tre")


def test_default_formatter_is_python_formatter():
    words = ["one", "only", "two", "twin"]
    assert iterable_to_regexp(words) == iterable_to_regexp(words, PythonFormatter)


def test_deterministic_output():
    words = ["red", "green", "grey", "blue", "reddish"]
    assert iterable_to_regexp(list(words)) == iterable_to_regexp(list(words))


def test_larger_word_list_matches_exactly():
    words = [f"{a}{b}{c}" for a in "abc" for b in "xyz" for c in "12"]
    non_words = ["abx", "a1", "ax12", "dx1", "ay3", "", "cz"]
    _assert_matches_exactly(words, non_words)


def test_stream_to_regexp_matches_same_words():
    words = ["cat", "car", "dog"]
    stream = io.StringIO(os.linesep.join(words) + os.linesep)
    compiled = re.compile(stream_to_regexp(stream))
    for word in words:
        assert compiled.fullmatch(word)
    assert not compiled.fullmatch("ca")
    assert not compiled.fullmatch("")


def test_package_exports():
    assert w2re.PythonFormatter is PythonFormatter
    assert issubclass(PythonFormatter, BaseFormatter)
    assert w2re.iterable_to_regexp is iterable_to_regexp
    assert w2re.stream_to_regexp is stream_to_regexp
    assert w2re.APPLICATION_NAME == "w2re"
    assert re.fullmatch(r"\d+\.\d+\.\d+", w2re.__version__)
    assert w2re.CHANGELOG_URL.startswith("https://")
