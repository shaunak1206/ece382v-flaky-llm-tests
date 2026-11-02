import collections
import importlib

import pytest

import langsplit.splitter as splitter


def test_detect_language_shortens_zh_cn_to_cn(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    assert splitter.detect_language("any text") == "cn"


def test_detect_language_truncates_long_language_code(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "eng")
    assert splitter.detect_language("any text", max_length=2) == "en"


def test_detect_language_keeps_short_code_when_within_max_length(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "en")
    assert splitter.detect_language("hello", max_length=2) == "en"


def test_convert_returns_input_when_not_dict():
    assert splitter.convert("plain text") == "plain text"


def test_convert_basic_dict_uses_default_separator_and_newline():
    data = {"en": " Hello", "fr": " Bonjour"}
    assert splitter.convert(data) == ".:en\nHello.:fr\nBonjour"


def test_convert_title_uses_colon_end_and_strips_leading_whitespace():
    data = {"en": " Hello", "fr": " World"}
    assert splitter.convert(data, title=True) == ".:en:Hello.:fr:World"


def test_convert_removes_leading_end_markers_from_result():
    data = {"en": "\n\nHello"}
    assert splitter.convert(data) == "en\nHello"


def test_split_returns_input_for_falsey_text():
    assert splitter.split("") == ""
    assert splitter.split(None) is None


def test_split_handles_short_prefixed_text_without_end_symbol():
    result = splitter.split(".:en")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ["en"]
    assert result["en"] == ""


def test_split_explicit_language_token_without_autodetect():
    text = ".:enHello.:frBonjour"
    result = splitter.split(text, autodetect=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello"
    assert result["fr"] == "Bonjour"


def test_split_with_autodetect_assigns_unknown_chunk_to_detected_language(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    result = splitter.split("Hello world", autodetect=True)
    assert list(result.keys()) == ["en"]
    assert result["en"] == "Hello world"


def test_split_with_autodetect_multiple_paragraphs_preserves_paragraph_separator(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    text = "First paragraph\n\nSecond paragraph"
    result = splitter.split(text, autodetect=True)
    assert list(result.keys()) == ["en"]
    assert result["en"] == "First paragraph\n\nSecond paragraph"


def test_split_with_autodetect_uses_multiple_detected_languages(monkeypatch):
    mapping = {"Hola": "es", "Hello": "en"}

    def fake_detect(paragraph):
        return mapping[paragraph]

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    result = splitter.split("Hola.:Hello", autodetect=True)
    assert list(result.keys()) == ["es", "en"]
    assert result["es"] == "Hola"
    assert result["en"] == "Hello"


def test_split_falls_back_to_unknown_language_when_detection_fails(monkeypatch):
    def fake_detect(paragraph):
        raise RuntimeError("boom")

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    result = splitter.split("Unclear text", autodetect=True)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == "Unclear text"


def test_split_returns_markdown_when_requested(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    assert splitter.split("Hello", autodetect=True, markdown=True) == ".:en\nHello"


def test_split_returns_title_markdown_when_requested(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    assert splitter.split("Hello", autodetect=True, markdown=True, title=True) == ".:en:Hello"
