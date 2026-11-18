import collections

import pytest

import langsplit.splitter as splitter


def test_detect_language_truncates_detected_code(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "en")
    assert splitter.detect_language("hello") == "en"


def test_detect_language_handles_shorter_max_length(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "eng")
    assert splitter.detect_language("hello", max_length=1) == "e"


def test_detect_language_maps_zh_cn_to_cn(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    assert splitter.detect_language("你好") == "cn"


def test_convert_dict_uses_default_separator_and_strips_leading_whitespace():
    data = collections.OrderedDict([("en", " hello"), ("fr", " monde")])
    assert splitter.convert(data) == "en.hello.:fr.monde"


def test_convert_dict_title_mode_uses_colon_as_end():
    data = collections.OrderedDict([("en", " hello"), ("fr", " monde")])
    assert splitter.convert(data, title=True) == "en:hello.:fr:monde"


def test_convert_returns_non_dict_unchanged():
    text = "plain text"
    assert splitter.convert(text) == text


def test_split_empty_text_returns_input_as_is():
    assert splitter.split("") == ""


def test_split_special_short_prefixed_text_without_end_symbol():
    result = splitter.split(".:en")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "")]


def test_split_explicit_language_blocks_preserve_order_and_text():
    text = ".:enHello.:frBonjour"
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello"
    assert result["fr"] == "Bonjour"


def test_split_explicit_language_blocks_with_newlines_are_kept():
    text = ".:enHello\n.:frBonjour"
    result = splitter.split(text)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello\n"
    assert result["fr"] == "Bonjour"


def test_split_explicit_blocks_markdown_roundtrip():
    text = ".:enHello.:frBonjour"
    result = splitter.split(text, markdown=True)
    assert result == "en.Hello.:fr.Bonjour"


def test_split_explicit_blocks_markdown_title_roundtrip():
    text = ".:enHello.:frBonjour"
    result = splitter.split(text, markdown=True, title=True)
    assert result == "en:Hello.:fr:Bonjour"


def test_split_autodetect_single_paragraph_uses_detect_language(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    result = splitter.split("Hello world")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ["en"]
    assert result["en"] == "Hello world"


def test_split_autodetect_multiple_paragraphs_inserts_paragraph_separator(monkeypatch):
    calls = []

    def fake_detect(paragraph):
        calls.append(paragraph)
        return "en" if "Hello" in paragraph else "fr"

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    result = splitter.split("Hello\n\nBonjour")
    assert calls == ["Hello", "Bonjour"]
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello\n\n"
    assert result["fr"] == "Bonjour"


def test_split_autodetect_unknown_language_when_detection_raises(monkeypatch):
    def boom(_):
        raise RuntimeError("no language")

    monkeypatch.setattr(splitter, "detect_language", boom)
    result = splitter.split("???")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == "???"


def test_split_autodetect_ignores_empty_paragraphs(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    result = splitter.split("\n\nHello\n\n")
    assert list(result.keys()) == ["en"]
    assert result["en"] == "Hello"


def test_split_non_autodetect_unknown_chunk_kept_under_unknown_language():
    result = splitter.split("just text", autodetect=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == "ext"


def test_split_custom_separator_and_ends(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    text = "##en|Hello##fr|Bonjour"
    result = splitter.split(text, sep="##", ends=["|"], autodetect=False)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello"
    assert result["fr"] == "Bonjour"
