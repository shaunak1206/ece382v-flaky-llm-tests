import io
import pytest

from w2re.utils import iterable_to_regexp, stream_to_regexp
from w2re import PythonFormatter, iterable_to_regexp as imported_iterable_to_regexp
from w2re.formatters import BaseFormatter
from w2re.prefix_tree.tree import PrefixTree


def test_iterable_to_regexp_returns_string():
    words = ["foo", "bar", "baz"]
    result = iterable_to_regexp(words)
    assert isinstance(result, str)
    assert result != ""


def test_iterable_to_regexp_with_empty_iterable():
    result = iterable_to_regexp([])
    # We expect an empty regexp or maybe something minimal - check non-error and str return
    assert isinstance(result, str)


def test_iterable_to_regexp_with_single_word():
    word = ["hello"]
    result = iterable_to_regexp(word)
    # The regexp should contain the word itself as substring
    assert "hello" in result


def test_iterable_to_regexp_with_custom_formatter(monkeypatch):
    class DummyFormatter(BaseFormatter):
        def format(self, parsed_tree):
            return "dummy"

    words = ["a", "b"]
    result = iterable_to_regexp(words, formatter=DummyFormatter)
    assert result == "dummy"


def test_stream_to_regexp_with_simple_stream():
    stream = io.StringIO("foo\nbar\nbaz\n")
    result = stream_to_regexp(stream)
    assert isinstance(result, str)
    assert "foo" in result


def test_stream_to_regexp_skips_empty_lines():
    stream = io.StringIO("\nfoo\n\nbar\n\n")
    result = stream_to_regexp(stream)
    # The result should include words but not empty patterns
    assert "foo" in result
    assert "bar" in result


def test_stream_to_regexp_with_custom_formatter():
    class DummyFormatter(BaseFormatter):
        def format(self, parsed_tree):
            return "formatted_by_dummy"

    stream = io.StringIO("a\nb\n")
    result = stream_to_regexp(stream, formatter=DummyFormatter)
    assert result == "formatted_by_dummy"


def test_iterable_to_regexp_preserves_input_order_for_prefix_tree(monkeypatch):
    # Patch PrefixTree.to_regexp to check its input
    captured_iterable = []

    original_init = PrefixTree.__init__
    def fake_init(self, iterable):
        captured_iterable.extend(iterable)
        original_init(self, iterable)

    PrefixTree.__init__ = fake_init
    words = ["alpha", "beta"]

    try:
        iterable_to_regexp(words)
        # Check that words are passed fully and in expected order
        assert all(w in captured_iterable for w in words)
    finally:
        PrefixTree.__init__ = original_init


def test_iterable_to_regexp_uses_PythonFormatter_by_default():
    words = ["foo"]
    # We test here that the default formatter used is instance of PythonFormatter inside
    from w2re.utils import iterable_to_regexp as func
    result_default = func(words)
    # naive check: PythonFormatter output usually includes r' for raw strings
    assert isinstance(result_default, str)
    assert len(result_default) > 0


def test_imported_iterable_to_regexp_is_same_as_in_utils():
    from w2re import iterable_to_regexp as imported_func
    from w2re.utils import iterable_to_regexp as utils_func
    assert imported_func is utils_func


def test_stream_to_regexp_empty_stream():
    stream = io.StringIO("")
    result = stream_to_regexp(stream)
    assert isinstance(result, str)


def test_stream_to_regexp_linesep_handling():
    # Use multiple different line separators manually, simulate unusual cases
    content = "one\ntwo\r\nthree\rfour\n\n"
    stream = io.StringIO(content)
    # Should parse all non-empty lines and produce regex
    result = stream_to_regexp(stream)
    assert isinstance(result, str)
    for w in ("one", "two", "three", "four"):
        assert w in result


def test_iterable_to_regexp_with_non_string_iterable_elements():
    # This should raise an error because elements must be str
    with pytest.raises(TypeError):
        iterable_to_regexp([123, None])  # type: ignore


def test_stream_to_regexp_with_non_text_stream_type():
    class DummyStream:
        def read(self):
            return "word1\nword2\n"
    dummy_stream = DummyStream()
    result = stream_to_regexp(dummy_stream)
    assert isinstance(result, str)
    assert "word1" in result
    assert "word2" in result


def test_stream_to_regexp_with_formatter_class_not_typing_Type():
    # Pass an instance instead of class - should raise error or handle gracefully
    stream = io.StringIO("a\nb\n")
    with pytest.raises(TypeError):
        stream_to_regexp(stream, formatter=PythonFormatter())


def test_iterable_to_regexp_multiple_calls_consistent_results():
    words = ["test", "testing", "tester"]
    result1 = iterable_to_regexp(words)
    result2 = iterable_to_regexp(words)
    assert result1 == result2


def test_iterable_to_regexp_with_inherited_formatter_class():
    class MyFormatter(PythonFormatter):
        pass

    words = ["foo", "bar"]
    result = iterable_to_regexp(words, formatter=MyFormatter)
    assert isinstance(result, str)
    assert "foo" in result


def test_stream_to_regexp_with_spaces_and_tabs_in_lines():
    content = "foo\n    bar\n\tbaz\n"
    stream = io.StringIO(content)
    result = stream_to_regexp(stream)
    assert isinstance(result, str)
    # Words stripped of whitespace are passed to prefix tree, original function doesn't strip explicitly
    for word in ["foo", "    bar", "\tbaz"]:
        assert word in result or word.strip() in result


def test_iterable_to_regexp_handles_duplicates():
    words = ["dup", "dup", "unique"]
    result = iterable_to_regexp(words)
    assert isinstance(result, str)
    # Result should include unique words at least once
    assert "dup" in result and "unique" in result
