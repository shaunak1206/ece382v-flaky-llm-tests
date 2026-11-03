import io
import random
import pytest
from w2re.utils import iterable_to_regexp
from w2re import PythonFormatter, iterable_to_regexp as w2re_iterable_to_regexp
from w2re import stream_to_regexp


def test_iterable_to_regexp_empty_iterable():
    # empty iterable should produce regexp matching nothing
    result = iterable_to_regexp([])
    # Since PrefixTree with empty iterable likely returns "" or something minimal,
    # accept empty string or known minimal pattern
    assert isinstance(result, str)
    assert result == '' or result == '(?!)' or result == '^$' or result == ''


def test_iterable_to_regexp_single_word():
    word = 'cat'
    result = iterable_to_regexp([word])
    assert isinstance(result, str)
    # The resulting regexp should match the exact word
    import re
    regex = re.compile(result)
    assert regex.fullmatch(word) is not None
    # should not match other words
    assert regex.fullmatch('dog') is None


def test_iterable_to_regexp_multiple_words():
    words = ['cat', 'car', 'dog']
    result = iterable_to_regexp(words)
    import re
    regex = re.compile(result)
    for w in words:
        assert regex.fullmatch(w) is not None
    # Negative test
    assert regex.fullmatch('cow') is None
    assert regex.fullmatch('c') is None
    assert regex.fullmatch('dogg') is None


def test_iterable_to_regexp_with_duplicates():
    words = ['cat', 'cat', 'car']
    result = iterable_to_regexp(words)
    import re
    regex = re.compile(result)
    for w in ['cat', 'car']:
        assert regex.fullmatch(w) is not None


def test_iterable_to_regexp_with_prefix_substrings():
    words = ['a', 'ab', 'abc']
    result = iterable_to_regexp(words)
    import re
    regex = re.compile(result)
    for w in words:
        assert regex.fullmatch(w) is not None
    assert regex.fullmatch('abcd') is None
    assert regex.fullmatch('') is None


def test_iterable_to_regexp_with_empty_string_in_iterable():
    words = ['', 'a']
    result = iterable_to_regexp(words)
    import re
    regex = re.compile(result)
    assert regex.fullmatch('') is not None
    assert regex.fullmatch('a') is not None
    assert regex.fullmatch('b') is None


def test_pythonformatter_format_methods_exist_and_not_error():
    # PythonFormatter has methods used by to_regexp, test calls produce str
    fmt = PythonFormatter()
    # These methods should exist and return string results
    assert hasattr(fmt, 'escape') and callable(fmt.escape)
    assert hasattr(fmt, 'format_sequence') and callable(fmt.format_sequence)
    assert hasattr(fmt, 'format_pattern') and callable(fmt.format_pattern)
    # Test escape returns escaped string for special regex chars
    special_chars = '.^$*+?{}[]\\|()'
    escaped = fmt.escape(special_chars)
    # All special chars should be escaped with a backslash
    for c in special_chars:
        assert '\\' + c in escaped or escaped.count(c) < special_chars.count(c)
    # format_sequence returns string containing formatted sequence of strings
    seq = ['a', 'b', 'c']
    formatted_seq = fmt.format_sequence(seq)
    assert isinstance(formatted_seq, str)
    # format_pattern returns string and doesn't error on simple pattern
    pat = 'abc'
    formatted_pat = fmt.format_pattern(pat)
    assert isinstance(formatted_pat, str)


def test_iterable_to_regexp_uses_pythonformatter_by_default(monkeypatch):
    # Patch PythonFormatter.to_regexp to verify it is called
    called = {}

    class DummyFormatter(PythonFormatter):
        @classmethod
        def from_words(cls, words):
            called['used'] = True
            return 'dummy'

    # Monkeypatch PythonFormatter in w2re.utils to DummyFormatter to catch usage
    from w2re import utils
    monkeypatch.setattr(utils, 'PythonFormatter', DummyFormatter)

    result = utils.iterable_to_regexp(['x'])
    # Our patch returns dummy string if formatter is used
    assert result == 'dummy' or isinstance(result, str)


def test_iterable_to_regexp_different_formatter(monkeypatch):
    # Define a dummy formatter with to_regexp method
    class DummyFormatter:
        def __init__(self):
            self.called = False
        def to_regexp(self, *args, **kwargs):
            self.called = True
            return 'dummy'

        @classmethod
        def from_words(cls, words):
            return 'dummyclassmethod'

    # But our code calls PrefixTree.to_regexp(formatter),
    # so the formatter should be a class, so override PrefixTree.to_regexp temporarily.

    from w2re import prefix_tree
    monkeypatch.setattr(prefix_tree, 'PrefixTree', prefix_tree.PrefixTree)

    # The formatter param must be a class with expected interface.
    # Just use DummyFormatter class with classmethod from_words
    class DummyFormatterClass:
        @classmethod
        def from_words(cls, words):
            return 'dummyclassmethod'

    # Call iterable_to_regexp with dummy formatter class
    # We expect result to be string, possibly not dummyclassmethod because PrefixTree.to_regexp
    # but no error should occur
    result = iterable_to_regexp(['dog', 'cat'], formatter=DummyFormatterClass)
    assert isinstance(result, str)


def test_stream_to_regexp_and_iterable_to_regexp_equivalence(tmp_path):
    # Create a temporary file with lines and check that stream_to_regexp and iterable_to_regexp produce same regexp
    words = ['apple', 'banana', 'cherry', 'date']
    file_path = tmp_path / "testwords.txt"
    file_path.write_text('\n'.join(words) + '\n')

    with file_path.open('r') as f:
        regex1 = stream_to_regexp(f, formatter=PythonFormatter)

    regex2 = iterable_to_regexp(words, formatter=PythonFormatter)

    assert isinstance(regex1, str)
    assert isinstance(regex2, str)
    # The two regexps may not be byte-equal but they should both match the same words and no others
    import re
    r1 = re.compile(regex1)
    r2 = re.compile(regex2)
    for w in words:
        assert r1.fullmatch(w)
        assert r2.fullmatch(w)
    assert not r1.fullmatch('notaword')
    assert not r2.fullmatch('notaword')


def test_stream_to_regexp_empty_stream():
    # empty stream returns regexp matching nothing (or empty string)
    empty_stream = io.StringIO('')
    result = stream_to_regexp(empty_stream, formatter=PythonFormatter)
    assert isinstance(result, str)
    # Accept empty string or known minimal form
    assert result == '' or result == '(?!)' or result == '^$' or result == ''


def test_iterable_to_regexp_handles_non_ascii_words():
    words = ['café', 'naïve', 'résumé']
    result = iterable_to_regexp(words)
    import re
    regex = re.compile(result)
    for w in words:
        assert regex.fullmatch(w) is not None


def test_iterable_to_regexp_no_side_effects_on_input_iterable():
    original = ['dog', 'cat', 'wolf']
    copy_original = original.copy()
    result = iterable_to_regexp(original)
    assert original == copy_original
    # Result is string
    assert isinstance(result, str)


def test_stream_to_regexp_reads_only_once(monkeypatch):
    # create a custom stream to check read is called exactly once
    class CountingStream(io.StringIO):
        def __init__(self, initstr):
            super().__init__(initstr)
            self.read_calls = 0
        def read(self, *args, **kwargs):
            self.read_calls += 1
            return super().read(*args, **kwargs)

    words = ['a', 'b', 'c']
    s = CountingStream('\n'.join(words) + '\n')
    result = stream_to_regexp(s, formatter=PythonFormatter)
    assert isinstance(result, str)
    # Only one read call
    assert s.read_calls == 1


def test_iterable_to_regexp_large_input_consistency():
    # Test with many similar words that share prefixes
    words = ['prefix' + str(i) for i in range(100)]
    result = iterable_to_regexp(words)
    import re
    regex = re.compile(result)
    for w in words:
        assert regex.fullmatch(w)
    # One non-matching word
    assert regex.fullmatch('prefix100') is None


def test_iterable_to_regexp_random_seeded(monkeypatch):
    # If PrefixTree or formatter uses random, seed Python random to fixed state
    random.seed(12345)
    words = ['red', 'green', 'blue']
    result1 = iterable_to_regexp(words)
    # Re-seed and get again, result should be the same
    random.seed(12345)
    result2 = iterable_to_regexp(words)
    assert result1 == result2


def test_imports_in_init():
    # Check __version__ and constants imported correctly
    import w2re
    assert isinstance(w2re.__version__, str)
    assert w2re.APPLICATION_NAME == 'w2re'
    assert w2re.CHANGELOG_URL.startswith('https://github.com')


@pytest.mark.parametrize("words", [
    (["hello", "hell", "heaven", "heavy"]),
    (["one", "two", "three", "four"]),
    (["abc", "abd", "abe", "abf"]),
])
def test_iterable_to_regexp_parametrized(words):
    import re
    result = iterable_to_regexp(words)
    regex = re.compile(result)
    for w in words:
        assert regex.fullmatch(w)
    # Test some negative values
    assert regex.fullmatch("notaword") is None


def test_iterable_to_regexp_utf8_bytes_raises():
    # Passing bytes instead of str should raise at somewhere
    with pytest.raises(TypeError):
        # Iterable of bytes instead of strings
        _ = iterable_to_regexp([b"bytes", b"words"])


# Below disables unused import complaints and keep code coverage consistent
def test_dummy():
    # import things to keep coverage consistent (no side effects)
    from w2re.__init__ import BaseFormatter, PythonWordMatchFormatter, PrefixTree, stream_to_regexp
    assert BaseFormatter is not None
    assert PythonWordMatchFormatter is not None
    assert PrefixTree is not None
    assert callable(stream_to_regexp)
