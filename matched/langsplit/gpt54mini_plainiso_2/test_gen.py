import collections
import pytest

import langsplit.splitter as splitter


def test_detect_language_truncates_long_code(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "eng")
    assert splitter.detect_language("hello", max_length=2) == "en"


def test_detect_language_maps_zh_cn_to_cn(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    assert splitter.detect_language("你好", max_length=2) == "cn"


def test_detect_language_returns_short_code_unchanged(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "en")
    assert splitter.detect_language("hello", max_length=2) == "en"


def test_convert_non_dict_returns_input_unchanged():
    text = "plain text"
    assert splitter.convert(text) == text


def test_convert_dict_uses_default_separator_and_strips_leading_spaces():
    data = {"en": " hello", "fr": " bonjour"}
    result = splitter.convert(data)
    assert result == "en\nhello.:fr\nbonjour"


def test_convert_dict_title_mode_uses_colon_newline():
    data = {"en": " hello", "fr": " bonjour"}
    result = splitter.convert(data, title=True)
    assert result == "en:\nhello.:fr:\nbonjour"


def test_convert_removes_initial_separator_tokens():
    data = collections.OrderedDict([("en", "hello")])
    result = splitter.convert(data, sep=".:", ends=["\n", "\r", "\r\n", ":"])
    assert result == "en\nhello"


def test_split_returns_input_when_empty():
    assert splitter.split("") == ""


def test_split_short_prefixed_token_without_end_symbol_returns_ordered_dict():
    result = splitter.split(".:enhello")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "hello")]


def test_split_explicit_language_segments_are_collected_in_order():
    text = ".:enHello.:frBonjour"
    result = splitter.split(text, autodetect=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello"
    assert result["fr"] == "Bonjour"


def test_split_aggregates_repeated_language_keys():
    text = ".:enHello.:enWorld"
    result = splitter.split(text, autodetect=False)
    assert list(result.keys()) == ["en"]
    assert result["en"] == "HelloWorld"


def test_split_autodetects_language_for_unknown_chunks(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    text = "hello world"
    result = splitter.split(text, autodetect=True, pargraph_sep="\n\n")
    assert list(result.keys()) == ["en"]
    assert result["en"] == "hello world"


def test_split_autodetects_each_paragraph_separately(monkeypatch):
    calls = []

    def fake_detect(paragraph):
        calls.append(paragraph)
        return "en" if "hello" in paragraph else "fr"

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    text = "hello\n\nbonjour"
    result = splitter.split(text, autodetect=True, pargraph_sep="\n\n")
    assert calls == ["hello", "bonjour"]
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "hello\n\n"
    assert result["fr"] == "bonjour"


def test_split_preserves_unknown_language_when_detection_fails(monkeypatch):
    def fake_detect(_paragraph):
        raise ValueError("cannot detect")

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    text = "undetectable"
    result = splitter.split(text, autodetect=True)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == "undetectable"


def test_split_markdown_roundtrip_from_explicit_languages():
    text = ".:enHello.:frBonjour"
    result = splitter.split(text, autodetect=False, markdown=True)
    assert result == "en\nHello.:fr\nBonjour"


def test_split_title_markdown_roundtrip_from_explicit_languages():
    text = ".:enHello.:frBonjour"
    result = splitter.split(text, autodetect=False, markdown=True, title=True)
    assert result == "en:\nHello.:fr:\nBonjour"
