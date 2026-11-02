import re
from io import StringIO
from unittest.mock import patch

import w2re
from w2re import PythonFormatter, iterable_to_regexp, stream_to_regexp
from w2re.formatters import PythonFormatter as RealPythonFormatter
from w2re.utils import iterable_to_regexp as utils_iterable_to_regexp
from w2re.utils import stream_to_regexp as utils_stream_to_regexp


def test_package_version_constant():
    assert w2re.__version__ == "3.1.0"


def test_package_application_name_constant():
    assert w2re.APPLICATION_NAME == "w2re"


def test_package_changelog_url_constant():
    assert w2re.CHANGELOG_URL == (
        "https://github.com/radeklat/words-to-regular-expression/"
        "blob/develop/CHANGELOG.md"
    )


def test_init_reexports_iterable_to_regexp():
    assert w2re.iterable_to_regexp is iterable_to_regexp
    assert w2re.iterable_to_regexp is utils_iterable_to_regexp


def test_init_reexports_stream_to_regexp():
    assert w2re.stream_to_regexp is stream_to_regexp
    assert w2re.stream_to_regexp is utils_stream_to_regexp


def test_init_reexports_python_formatter():
    assert w2re.PythonFormatter is PythonFormatter
    assert w2re.PythonFormatter is RealPythonFormatter


def test_iterable_to_regexp_returns_string():
    result = iterable_to_regexp(["alpha"])
    assert isinstance(result, str)
    assert result


def test_iterable_to_regexp_matches_single_word():
    pattern = iterable_to_regexp(["alpha"])
    assert re.fullmatch(pattern, "alpha")
    assert not re.fullmatch(pattern, "beta")


def test_iterable_to_regexp_matches_multiple_words():
    words = ["cat", "car", "dog"]
    pattern = iterable_to_regexp(words)
    for word in words:
        assert re.fullmatch(pattern, word), pattern
    assert not re.fullmatch(pattern, "cow")
    assert not re.fullmatch(pattern, "carton")


def test_iterable_to_regexp_is_order_independent():
    words1 = ["cat", "car", "dog"]
    words2 = ["dog", "car", "cat"]
    assert iterable_to_regexp(words1) == iterable_to_regexp(words2)


def test_iterable_to_regexp_ignores_duplicate_words():
    words = ["alpha", "alpha", "beta"]
    pattern = iterable_to_regexp(words)
    assert re.fullmatch(pattern, "alpha")
    assert re.fullmatch(pattern, "beta")
    assert not re.fullmatch(pattern, "gamma")


def test_iterable_to_regexp_supports_generators():
    pattern = iterable_to_regexp(word for word in ["one", "two", "three"])
    assert re.fullmatch(pattern, "one")
    assert re.fullmatch(pattern, "two")
    assert re.fullmatch(pattern, "three")
    assert not re.fullmatch(pattern, "four")


def test_stream_to_regexp_matches_stream_content():
    stream = StringIO("apple\napricot\nbanana")
    pattern = stream_to_regexp(stream)
    assert re.fullmatch(pattern, "apple")
    assert re.fullmatch(pattern, "apricot")
    assert re.fullmatch(pattern, "banana")
    assert not re.fullmatch(pattern, "pear")


def test_iterable_to_regexp_uses_prefix_tree_to_regexp_with_formatter():
    with patch("w2re.utils.PrefixTree") as mock_prefix_tree:
        mock_instance = mock_prefix_tree.return_value
        mock_instance.to_regexp.return_value = "REGEXP"

        result = utils_iterable_to_regexp(["alpha", "beta"], formatter=PythonFormatter)

        assert result == "REGEXP"
        mock_prefix_tree.assert_called_once()
        mock_instance.to_regexp.assert_called_once_with(PythonFormatter)


def test_stream_to_regexp_filters_empty_lines_before_building_tree():
    with patch("w2re.utils.PrefixTree") as mock_prefix_tree:
        mock_instance = mock_prefix_tree.return_value
        mock_instance.to_regexp.return_value = "REGEXP"

        stream = StringIO("alpha\n\nbeta\n")
        result = utils_stream_to_regexp(stream, formatter=PythonFormatter)

        assert result == "REGEXP"
        mock_prefix_tree.assert_called_once()
        passed_iterable = mock_prefix_tree.call_args.args[0]
        assert list(passed_iterable) == ["alpha", "beta"]
