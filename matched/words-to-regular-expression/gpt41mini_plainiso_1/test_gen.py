import io
import re
import pytest

from w2re.utils import iterable_to_regexp
from w2re import PythonFormatter, BaseFormatter


def test_iterable_to_regexp_basic_words():
    words = ["cat", "car", "dog"]
    regexp = iterable_to_regexp(words, PythonFormatter)
    # Result should match all three words
    pattern = re.compile(regexp)
    for w in words:
        assert pattern.fullmatch(w)
    # It should not match a non-listed word
    assert not pattern.fullmatch("cow")


def test_iterable_to_regexp_empty_iterable():
    regexp = iterable_to_regexp([], PythonFormatter)
    # An empty iterable should produce a regexp that matches nothing
    pattern = re.compile(regexp)
    assert pattern.match("") is None
    assert pattern.match("anything") is None


def test_iterable_to_regexp_single_word():
    word = ["hello"]
    regexp = iterable_to_regexp(word, PythonFormatter)
    pattern = re.compile(regexp)
    assert pattern.fullmatch("hello")
    assert not pattern.fullmatch("hell")
    assert not pattern.fullmatch("hello!")


def test_iterable_to_regexp_with_similar_prefixes():
    words = ["test", "testing", "tester"]
    regexp = iterable_to_regexp(words, PythonFormatter)
    pattern = re.compile(regexp)
    for w in words:
        assert pattern.fullmatch(w)
    assert not pattern.fullmatch("testa")
    assert not pattern.fullmatch("tes")
    assert not pattern.fullmatch("testingg")


def test_iterable_to_regexp_with_duplicates():
    words = ["repeat", "repeat", "repeat"]
    regexp = iterable_to_regexp(words, PythonFormatter)
    pattern = re.compile(regexp)
    assert pattern.fullmatch("repeat")
    assert not pattern.fullmatch("repea")


def test_iterable_to_regexp_with_formatter_baseformatter(tmp_path):
    # Use BaseFormatter directly and see if output is str and nonempty
    words = ["a", "ab", "abc"]
    regexp = iterable_to_regexp(words, BaseFormatter)
    assert isinstance(regexp, str)
    assert len(regexp) > 0
    pattern = re.compile(regexp)
    for w in words:
        assert pattern.fullmatch(w)


def test_iterable_to_regexp_returns_string_type():
    words = ["x", "y", "z"]
    regexp = iterable_to_regexp(words)
    assert isinstance(regexp, str)
    assert regexp != ""


def test_pythonformatter_str_and_repr_methods():
    fmt = PythonFormatter()
    s = str(fmt)
    r = repr(fmt)
    assert "PythonFormatter" in s
    assert "PythonFormatter" in r


def test_pythonformatter_to_regexp_method_basic_usage():
    fmt = PythonFormatter()
    # The to_regexp method accepts a PrefixTree instance.
    # We create one with simple words.
    from w2re.prefix_tree.tree import PrefixTree
    pt = PrefixTree(["a", "b"])
    out = fmt.to_regexp(pt)
    assert isinstance(out, str)
    pattern = re.compile(out)
    assert pattern.fullmatch("a")
    assert pattern.fullmatch("b")
    assert not pattern.fullmatch("c")


def test_pythonformatter_to_regexp_method_empty_prefix_tree():
    from w2re.prefix_tree.tree import PrefixTree
    fmt = PythonFormatter()
    pt = PrefixTree([])
    out = fmt.to_regexp(pt)
    assert isinstance(out, str)
    pattern = re.compile(out)
    # Should match nothing
    assert pattern.match("anything") is None


def test_pythonformatter_cover_escape_characters():
    fmt = PythonFormatter()
    from w2re.prefix_tree.tree import PrefixTree
    words = ["abc.def", "abc?def", "abc*def"]
    pt = PrefixTree(words)
    pattern_str = fmt.to_regexp(pt)
    # It should escape these special characters correctly so the pattern matches the literal strings
    pattern = re.compile(pattern_str)
    for w in words:
        assert pattern.fullmatch(w)


def test_iterable_to_regexp_with_non_string_words():
    # iterable contains int - should raise error because input str is required.
    words = ["a", "b", 123]
    with pytest.raises(TypeError):
        iterable_to_regexp(words, PythonFormatter)


def test_iterable_to_regexp_with_unicode_strings():
    words = ["café", "résumé", "naïve"]
    regexp = iterable_to_regexp(words, PythonFormatter)
    pattern = re.compile(regexp)
    for w in words:
        assert pattern.fullmatch(w)
    assert not pattern.fullmatch("cafe")


def test_iterable_to_regexp_with_mixed_case():
    words = ["Dog", "dog", "DOG"]
    regexp = iterable_to_regexp(words, PythonFormatter)
    pattern = re.compile(regexp)
    for w in words:
        assert pattern.fullmatch(w)
    assert not pattern.fullmatch("doGg")


def test_iterable_to_regexp_with_long_words():
    word = ["a" * 1000]
    regexp = iterable_to_regexp(word, PythonFormatter)
    pattern = re.compile(regexp)
    assert pattern.fullmatch(word[0])
    assert not pattern.fullmatch(word[0] + "a")


def test_iterable_to_regexp_returns_same_for_repeated_calls():
    words = ["repeat", "repeat", "repeat"]
    regexp1 = iterable_to_regexp(words, PythonFormatter)
    regexp2 = iterable_to_regexp(words, PythonFormatter)
    assert regexp1 == regexp2


def test_iterable_to_regexp_with_formatter_type_inheritance():
    class DummyFormatter(PythonFormatter):
        def to_regexp(self, prefix_tree):
            return "dummy"
    words = ["test"]
    regexp = iterable_to_regexp(words, DummyFormatter)
    assert regexp == "dummy"
