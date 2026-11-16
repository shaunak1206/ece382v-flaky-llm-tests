import collections

import pytest

from langsplit import splitter
from langsplit.splitter import split


@pytest.fixture
def fake_detect(monkeypatch):
    """Replace language detection with a deterministic fake."""
    def _detect(text, max_length=2):
        if text.startswith("Bonjour"):
            return "fr"
        if text.startswith("Intro"):
            return "de"
        return "en"
    monkeypatch.setattr(splitter, "detect_language", _detect)
    return _detect


@pytest.fixture
def failing_detect(monkeypatch):
    def _detect(text, max_length=2):
        raise RuntimeError("cannot detect")
    monkeypatch.setattr(splitter, "detect_language", _detect)


def test_empty_string_returned_as_is():
    assert split("") == ""


def test_none_returned_as_is():
    assert split(None) is None


def test_basic_split_two_languages():
    result = split(".:en\nHello.:fr\nBonjour")
    assert result == {"en": "Hello", "fr": "Bonjour"}
    assert list(result.keys()) == ["en", "fr"]


def test_returns_ordered_dict():
    result = split(".:en\nHello.:fr\nBonjour")
    assert isinstance(result, collections.OrderedDict)


def test_same_keys_are_combined_in_order_of_first_appearance():
    result = split(".:en\nHello .:fr\nBonjour.:en\nWorld")
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello World"
    assert result["fr"] == "Bonjour"


def test_short_text_without_end_symbol():
    result = split(".:en")
    assert isinstance(result, collections.OrderedDict)
    assert result == {"en": ""}


def test_colon_as_end_symbol():
    result = split(".:en:Hello.:fr:Bonjour")
    assert result == {"en": "Hello", "fr": "Bonjour"}


def test_empty_tokens_are_skipped():
    result = split(".:.:en\nHi")
    assert result == {"en": "Hi"}


def test_trailing_newline_gives_empty_value():
    result = split(".:en\n")
    assert result == {"en": ""}


def test_custom_separator():
    result = split("|en\nHello|fr\nBonjour", sep="|")
    assert result == {"en": "Hello", "fr": "Bonjour"}


def test_max_key_length_allows_longer_keys():
    result = split(".:eng\nHello", max_key_length=3)
    assert result == {"eng": "Hello"}


def test_key_shorter_than_min_goes_to_unknown_without_autodetect():
    result = split(".:e\nHello", autodetect=False)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]


def test_unmarked_text_autodetected(fake_detect):
    result = split("Hello world")
    assert result == {"en": "Hello world"}


def test_autodetect_treats_paragraphs_separately(fake_detect):
    result = split("Hello\n\nBonjour")
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello\n\n"
    assert result["fr"] == "Bonjour"


def test_unmarked_text_before_marked_sections(fake_detect):
    result = split("Intro text.:en\nHello")
    assert list(result.keys()) == ["de", "en"]
    assert result["de"] == "Intro text"
    assert result["en"] == "Hello"


def test_detection_failure_falls_back_to_unknown(failing_detect):
    result = split("Some text")
    assert result == {splitter.settings.UNKNOWN_LANGUAGE: "Some text"}


def test_markdown_output_roundtrip():
    text = ".:en\nHello.:fr\nBonjour"
    assert split(text, markdown=True) == text


def test_markdown_title_uses_colon_end():
    result = split(".:en\nHello.:fr\nBonjour", markdown=True, title=True)
    assert result == ".:en:Hello.:fr:Bonjour"
