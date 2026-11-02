import io
import pytest

from w2re.utils import iterable_to_regexp
from w2re.formatters import PythonFormatter
from w2re import iterable_to_regexp as top_iterable_to_regexp
from w2re import __version__, APPLICATION_NAME, CHANGELOG_URL


def test_iterable_to_regexp_empty_iterable():
    """Empty iterable should produce a regexp matching nothing."""
    result = iterable_to_regexp([])
    # The regex for empty set in prefix tree is usually one that matches nothing.
    # Since exact output may vary, check that it does not match anything
    import re
    pat = re.compile(result)
    assert not pat.match("anything")
    assert isinstance(result, str)
    assert result != ""


def test_iterable_to_regexp_single_word():
    """Single word iterable returns a regexp matching exactly that word."""
    word = "apple"
    result = iterable_to_regexp([word])
    import re
    pat = re.compile(result)
    assert pat.fullmatch(word)
    # Does not match other words
    assert not pat.fullmatch("appl")
    assert not pat.fullmatch("applepie")
    assert not pat.fullmatch("banana")


def test_iterable_to_regexp_multiple_words_with_common_prefix():
    """Words sharing prefix produce a regexp matching any of them."""
    words = ["foo", "fool", "foot", "for"]
    result = iterable_to_regexp(words)
    import re
    pat = re.compile(result)
    for w in words:
        assert pat.fullmatch(w)
    # Should not match close but different words
    assert not pat.fullmatch("fo")
    assert not pat.fullmatch("fooo")
    assert not pat.fullmatch("football")


def test_iterable_to_regexp_multiple_words_without_common_prefix():
    words = ["cat", "dog", "bird"]
    result = iterable_to_regexp(words)
    import re
    pat = re.compile(result)
    for w in words:
        assert pat.fullmatch(w)
    assert not pat.fullmatch("do")
    assert not pat.fullmatch("cats")


def test_iterable_to_regexp_handles_duplicates():
    words = ["repeat", "repeat", "repeat", "rep"]
    result = iterable_to_regexp(words)
    import re
    pat = re.compile(result)
    for w in set(words):
        assert pat.fullmatch(w)
    assert not pat.fullmatch("repeats")


def test_iterable_to_regexp_non_ascii_words():
    words = ["café", "naïve", "résumé"]
    result = iterable_to_regexp(words)
    import re
    pat = re.compile(result)
    for w in words:
        assert pat.fullmatch(w)
    assert not pat.fullmatch("resume")


def test_iterable_to_regexp_with_empty_string():
    # If the iterable contains an empty string,
    # the regexp should match the empty string.
    words = ["", "abc"]
    result = iterable_to_regexp(words)
    import re
    pat = re.compile(result)
    assert pat.fullmatch("")
    assert pat.fullmatch("abc")
    assert not pat.fullmatch("a")


@pytest.mark.parametrize(
    "words",
    [
        ["a"],
        ["a", "aa"],
        ["aa", "ab", "ac"],
        ["abc", "abd", "bcd", "xyz"],
        ["apple", "applepie", "applet", "banana"],
        [""],
    ]
)
def test_iterable_to_regexp_matches_all(words):
    result = iterable_to_regexp(words)
    import re
    pat = re.compile(result)
    for w in words:
        assert pat.fullmatch(w)
    # Also test a word not in list does not match if not in list and not prefix of any
    not_words = {"zzzz", "notpresent", "a" * 50}
    for nw in not_words:
        # It can match if nw is in words, but it is not, so...
        assert not pat.fullmatch(nw)


def test_iterable_to_regexp_output_is_string():
    words = ["test"]
    result = iterable_to_regexp(words)
    assert isinstance(result, str)


def test_imported_symbols_available_and_correct():
    import w2re
    assert w2re.__version__ == '3.1.0'
    assert w2re.APPLICATION_NAME == 'w2re'
    assert w2re.CHANGELOG_URL.startswith('https://github.com/radeklat')

    # Check that imported functions exist and are callable
    assert callable(w2re.iterable_to_regexp)
    # Their behavior should match the original
    sample = ["alpha", "beta"]
    assert w2re.iterable_to_regexp(sample) == iterable_to_regexp(sample)


def test_python_formatter_class_properties():
    fmt = PythonFormatter()
    # to_regexp method exists and returns string
    if hasattr(fmt, "to_regexp"):
        # It is expected to take PrefixTree instance, we test with dummy returns
        # Not much to check here, just instance and typical attribute exist
        assert hasattr(fmt, "format_word") or hasattr(fmt, "to_regexp")
    else:
        # Defensive fallback - class must have at least one of these attributes by design
        assert False, "PythonFormatter missing expected methods"


def test_iterable_to_regexp_with_monkeypatch(monkeypatch):
    # Use monkeypatch to wrap PrefixTree.to_regexp to a stub that returns fixed string
    from w2re.prefix_tree.tree import PrefixTree

    def fake_to_regexp(self, formatter=None):
        return "fixed-regexp-string"

    monkeypatch.setattr(PrefixTree, "to_regexp", fake_to_regexp)
    # The result must be the fixed string regardless of input
    result = iterable_to_regexp(["anything"])
    assert result == "fixed-regexp-string"


def test_iterable_to_regexp_type_error_on_non_iterable():
    with pytest.raises(TypeError):
        # Passing an int should raise TypeError when iterated
        iterable_to_regexp(42)  # Not iterable


def test_iterable_to_regexp_type_error_on_iterable_of_non_strings():
    with pytest.raises(TypeError):
        # Passing iterable with non-string element should raise TypeError
        iterable_to_regexp(["valid", 123, None])


def test_iterable_to_regexp_with_large_input():
    # Test performance and correctness on large input, deterministic
    words = [f"word{i}" for i in range(1000)]
    result = iterable_to_regexp(words)
    import re
    pat = re.compile(result)
    for w in words:
        assert pat.fullmatch(w)
    assert not pat.fullmatch("word10000")


def test_iterable_to_regexp_preserves_order_independence():
    words1 = ["cat", "dog", "bird"]
    words2 = ["dog", "bird", "cat"]
    result1 = iterable_to_regexp(words1)
    result2 = iterable_to_regexp(words2)
    # The regexp strings may differ due to order but should match same words
    import re
    pat1 = re.compile(result1)
    pat2 = re.compile(result2)
    for w in words1:
        assert pat1.fullmatch(w)
        assert pat2.fullmatch(w)


@pytest.mark.parametrize("words", [[], [""], ["a"], ["a", "ab"], ["test", "testing"]])
def test_iterable_to_regexp_compile_no_error(words):
    # Just test the output compiles as regex for various inputs including edge cases
    result = iterable_to_regexp(words)
    import re
    pat = re.compile(result)
    for w in words:
        # Must match all words in the list fully or at least partially
        # The semantics of the regex may accept full match only
        assert pat.match(w)


# As requested, all tests are deterministic and independent, no global state changed.
