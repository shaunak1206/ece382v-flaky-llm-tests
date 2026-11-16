import collections

import pytest

import langsplit.splitter as splitter
from langsplit.splitter import split


def _fake_detect(mapping, default="en"):
    def fake(text, max_length=2):
        return mapping.get(text, default)
    return fake


def test_empty_string_returns_empty_string():
    assert split("") == ""


def test_none_returns_none():
    assert split(None) is None


def test_basic_two_languages_with_newline_end():
    result = split(".:en\nHello.:ru\nПривет")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Hello"), ("ru", "Привет")]


def test_same_key_chunks_are_combined_and_order_preserved():
    result = split(".:en\nHello .:ru\nПривет.:en\nWorld")
    assert list(result.keys()) == ["en", "ru"]
    assert result["en"] == "Hello World"
    assert result["ru"] == "Привет"


def test_colon_as_end_symbol():
    result = split(".:en:Hello.:fr:Bonjour")
    assert dict(result) == {"en": "Hello", "fr": "Bonjour"}


def test_short_text_without_end_symbol_returns_key_only():
    result = split(".:en")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "")]


def test_short_single_char_key_without_end():
    result = split(".:e")
    assert list(result.items()) == [("e", "")]


def test_custom_separator():
    result = split("#en\nHello#ru\nПривет", sep="#")
    assert dict(result) == {"en": "Hello", "ru": "Привет"}


def test_custom_key_length():
    result = split(".:eng\nHello.:rus\nPrivet", min_key_length=3, max_key_length=3)
    assert list(result.items()) == [("eng", "Hello"), ("rus", "Privet")]


def test_key_longer_than_max_goes_to_unknown_without_autodetect():
    result = split(".:eng\nHello", autodetect=False)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]


def test_unknown_text_without_autodetect_does_not_call_detector(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("detect_language must not be called")

    monkeypatch.setattr(splitter, "detect_language", boom)
    result = split("Just some plain text", autodetect=False)
    assert list(result.keys()) == ["xx"]


def test_text_before_first_separator_is_unknown_and_first(monkeypatch):
    result = split("Introduction.:en\nHello", autodetect=False)
    assert list(result.keys()) == ["xx", "en"]
    assert result["en"] == "Hello"


def test_autodetect_assigns_language_per_paragraph(monkeypatch):
    monkeypatch.setattr(
        splitter,
        "detect_language",
        _fake_detect({"Hello": "en", "Bonjour": "fr"}),
    )
    result = split("Hello\n\nBonjour")
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello\n\n"
    assert result["fr"] == "Bonjour"


def test_autodetect_merges_paragraphs_of_same_language(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", _fake_detect({}, default="en"))
    result = split("First text\n\nSecond text")
    assert list(result.keys()) == ["en"]
    assert result["en"] == "First text\n\nSecond text"


def test_autodetect_failure_falls_back_to_unknown(monkeypatch):
    def failing(text, max_length=2):
        raise RuntimeError("cannot detect")

    monkeypatch.setattr(splitter, "detect_language", failing)
    result = split("12345 ???")
    assert dict(result) == {"xx": "12345 ???"}


def test_autodetect_skips_empty_paragraphs(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", _fake_detect({}, default="en"))
    result = split("Hello\n\n\n\nWorld")
    assert list(result.keys()) == ["en"]
    assert "Hello" in result["en"] and "World" in result["en"]


def test_markdown_output():
    result = split(".:en\nHello.:ru\nПривет", markdown=True)
    assert result == ".:en\nHello.:ru\nПривет"


def test_markdown_title_output_uses_colon():
    result = split(".:en\nHello.:ru\nПривет", markdown=True, title=True)
    assert result == ".:en:Hello.:ru:Привет"


def test_title_markdown_roundtrip():
    original = ".:en:Hello.:fr:Bonjour"
    parsed = split(original)
    assert split(original, markdown=True, title=True) == original
    assert dict(parsed) == {"en": "Hello", "fr": "Bonjour"}
