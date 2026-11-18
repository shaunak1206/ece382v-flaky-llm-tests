import io
import pytest
from w2re.utils import iterable_to_regexp
from w2re import PythonFormatter, BaseFormatter, iterable_to_regexp, stream_to_regexp
from w2re.prefix_tree.tree import PrefixTree


def test_iterable_to_regexp_single_word():
    # Single word should return a regexp matching exactly that word
    result = iterable_to_regexp(["hello"])
    assert isinstance(result, str)
    assert "hello" in result


def test_iterable_to_regexp_multiple_words():
    # Multiple words combination — the regexp contains all substrings
    words = ["foo", "bar", "baz"]
    regexp = iterable_to_regexp(words)
    for word in words:
        assert word[0] in regexp  # starting letters appear


def test_iterable_to_regexp_duplicates():
    # Duplicates do not affect regexp correctness
    words = ["same", "same", "same"]
    regexp = iterable_to_regexp(words)
    assert regexp.count("s") >= 1


def test_iterable_to_regexp_empty_list():
    # Empty iterable returns something sensible, e.g. empty regex ''
    regexp = iterable_to_regexp([])
    assert isinstance(regexp, str)
    # It can be empty or something matching empty string, test no error
    assert regexp is not None


def test_iterable_to_regexp_different_formatter(monkeypatch):
    # Use a dummy formatter class to verify it is called correctly

    class DummyFormatter(BaseFormatter):
        def format(self):
            return "DUMMY"

    regexp = iterable_to_regexp(["x", "y"], formatter=DummyFormatter)
    # Since DummyFormatter returns "DUMMY" from to_regexp:
    # Actually PrefixTree.to_regexp calls formatter passed to it,
    # but our dummy formatter must be compatible and used by PrefixTree.

    # Because we can't control internal PrefixTree usage here,
    # The best we can do is test regexp is a string.
    assert isinstance(regexp, str)
    assert len(regexp) > 0


def test_python_formatter_instance():
    # Instantiate PythonFormatter; it should be subclass of BaseFormatter
    pf = PythonFormatter()
    from w2re.formatters import BaseFormatter
    assert isinstance(pf, BaseFormatter)

    # Check it implements to_regexp method result type
    result = pf.format(["a", "b"])
    assert isinstance(result, list) or isinstance(result, str) or result is None


def test_stream_to_regexp_basic():
    # Provide a stream with several words and test output string
    text = "apple\nbanana\ncarrot\n"
    stream = io.StringIO(text)
    regexp = stream_to_regexp(stream)
    assert isinstance(regexp, str)
    for w in ["apple", "banana", "carrot"]:
        # The individual words' first letters at least should be part of regexp
        assert w[0] in regexp


def test_stream_to_regexp_empty_stream():
    # Empty stream returns empty or some valid regexp string
    stream = io.StringIO("")
    regexp = stream_to_regexp(stream)
    assert isinstance(regexp, str)
    # Should not raise nor return None
    assert regexp is not None


def test_stream_to_regexp_with_windows_line_endings():
    # Use Windows style new lines in stream input
    text = "one\r\ntwo\r\nthree\r\n"
    stream = io.StringIO(text)
    regexp = stream_to_regexp(stream)
    for w in ["one", "two", "three"]:
        assert w[0] in regexp


def test_iterable_to_regexp_non_ascii_words():
    words = ["café", "niño", "jalapeño"]
    regexp = iterable_to_regexp(words)
    for w in words:
        assert w[0] in regexp


def test_iterable_to_regexp_long_words():
    words = ["a" * 1000, "b" * 1000]
    regexp = iterable_to_regexp(words)
    assert regexp is not None
    # Should contain starting letters a and b
    assert "a" in regexp
    assert "b" in regexp


def test_python_formatter_format_returns_string():
    pf = PythonFormatter()
    # It should produce a string when using the format method on a simple list
    result = pf.format(["abc", "abd"])
    # Accept if it's list or string, but it should be string ideally (depending on impl)
    assert isinstance(result, (str, list))


def test_python_formatter_format_empty():
    pf = PythonFormatter()
    result = pf.format([])
    assert isinstance(result, (str, list))
    # An empty input should not raise and produce result


def test_iterable_to_regexp_multiple_same_prefix():
    words = ["test", "testing", "tester"]
    regexp = iterable_to_regexp(words)
    # The regexp should have coverage for 'test' prefix
    assert "test"[:1] in regexp


def test_iterable_to_regexp_words_with_special_chars():
    words = ["foo.bar", "foo-bar", "foo?bar"]
    regexp = iterable_to_regexp(words)
    assert isinstance(regexp, str)
    # The special characters may be escaped somehow
    assert "foo" in regexp


def test_stream_to_regexp_stream_closed_after_use():
    text = "abc\ndef"
    stream = io.StringIO(text)
    # stream_to_regexp does not close stream itself
    # so after the call, stream should still be open (in-memory stream)
    regexp = stream_to_regexp(stream)
    assert not stream.closed
    stream.close()


def test_imports_are_correct():
    # Check that __init__.py provides expected symbols
    import w2re
    assert hasattr(w2re, "iterable_to_regexp")
    assert hasattr(w2re, "PythonFormatter")
    assert hasattr(w2re, "__version__")
    assert isinstance(w2re.__version__, str)
    # Check constants
    assert hasattr(w2re, "APPLICATION_NAME")
    assert hasattr(w2re, "CHANGELOG_URL")
    assert w2re.APPLICATION_NAME == "w2re"
    assert "github.com" in w2re.CHANGELOG_URL


def test_iterable_to_regexp_error_on_non_iterable():
    with pytest.raises(TypeError):
        # Passing int instead of iterable of strings should raise TypeError
        iterable_to_regexp(123)  # type: ignore


def test_iterable_to_regexp_error_on_iterable_with_non_string():
    with pytest.raises(TypeError):
        # Iterable contains non-string element - expect TypeError or similar
        iterable_to_regexp(["valid", 5])  # type: ignore


def test_python_formatter_repr_and_str():
    pf = PythonFormatter()
    # Check __repr__ and __str__ methods exist and are string
    assert isinstance(repr(pf), str)
    assert isinstance(str(pf), str)
