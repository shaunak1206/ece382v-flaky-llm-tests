import collections

import pytest

from langsplit import splitter
from langsplit.splitter import split


@pytest.fixture
def fake_en(monkeypatch):
    """Make language detection deterministic: everything is English."""
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")


def test_empty_string_returned_as_is():
    assert split("") == ""


def test_none_returned_as_is():
    assert split(None) is None


def test_short_key_without_end_symbol():
    result = split(".:en")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "")]


def test_single_keyed_section_newline_end():
    result = split(".:en\nHello")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Hello")]


def test_single_keyed_section_colon_end():
    result = split(".:en:Hello")
    assert list(result.items()) == [("en", "Hello")]


def test_multiple_sections_preserve_order():
    result = split(".:fr\nBonjour.:en\nHello.:de\nHallo")
    assert list(result.keys()) == ["fr", "en", "de"]
    assert result["fr"] == "Bonjour"
    assert result["en"] == "Hello"
    assert result["de"] == "Hallo"


def test_same_keys_are_combined():
    result = split(".:en\nHello .:fr\nBonjour.:en\nWorld")
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello World"
    assert result["fr"] == "Bonjour"


def test_custom_separator():
    result = split("##en\nHello##fr\nBonjour", sep="##")
    assert list(result.items()) == [("en", "Hello"), ("fr", "Bonjour")]


def test_longer_key_with_max_key_length():
    result = split(".:eng\nHello", max_key_length=3)
    assert list(result.items()) == [("eng", "Hello")]


def test_unkeyed_text_uses_detection(fake_en):
    result = split("Hello world")
    assert list(result.items()) == [("en", "Hello world")]


def test_unkeyed_text_before_keyed_section(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "fr")
    result = split("Intro text.:en\nHello")
    assert list(result.keys()) == ["fr", "en"]
    assert result["fr"] == "Intro text"
    assert result["en"] == "Hello"


def test_key_too_long_for_defaults_falls_back_to_detection(fake_en):
    result = split(".:eng\nHello")
    # 'eng' is not a valid 2-letter key, so the token is detected as a whole
    assert list(result.keys()) == ["en"]
    assert result["en"] == "eng\nHello"


def test_paragraphs_detected_separately(monkeypatch):
    def fake_detect(paragraph):
        return "en" if "Hello" in paragraph else "fr"

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    result = split("Hello\n\nBonjour")
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello\n\n"
    assert result["fr"] == "Bonjour"


def test_detection_called_once_per_nonempty_paragraph(monkeypatch):
    calls = []

    def fake_detect(paragraph):
        calls.append(paragraph)
        return "en"

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    split("One\n\n\n\nTwo")
    # split on '\n\n' gives ['One', '', 'Two']; the empty one is skipped
    assert calls == ["One", "Two"]


def test_detection_failure_assigned_to_unknown_language(monkeypatch):
    def failing_detect(paragraph):
        raise ValueError("no features in text")

    monkeypatch.setattr(splitter, "detect_language", failing_detect)
    result = split("12345")
    assert list(result.items()) == [(splitter.settings.UNKNOWN_LANGUAGE, "12345")]


def test_autodetect_false_does_not_call_detection(monkeypatch):
    def boom(paragraph):
        raise AssertionError("detection must not be called")

    monkeypatch.setattr(splitter, "detect_language", boom)
    result = split("Hello world", autodetect=False)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]


def test_autodetect_false_keyed_sections_still_parsed(monkeypatch):
    def boom(paragraph):
        raise AssertionError("detection must not be called")

    monkeypatch.setattr(splitter, "detect_language", boom)
    result = split(".:en\nHello.:fr\nBonjour", autodetect=False)
    assert list(result.items()) == [("en", "Hello"), ("fr", "Bonjour")]


def test_markdown_roundtrip():
    text = ".:en\nHello.:fr\nBonjour"
    assert split(text, markdown=True) == text


def test_markdown_title_uses_colon_end():
    result = split(".:en\nHello.:fr\nBonjour", markdown=True, title=True)
    assert result == ".:en:Hello.:fr:Bonjour"


def test_convert_non_dict_returned_unchanged():
    assert splitter.convert("plain") == "plain"


def test_detect_language_maps_long_codes(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    assert splitter.detect_language("whatever") == "cn"
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "en")
    assert splitter.detect_language("whatever") == "en"
