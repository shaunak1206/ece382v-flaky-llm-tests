import collections

import pytest

import langsplit.splitter as splitter


def test_detect_language_returns_short_code_for_long_code(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "english")
    assert splitter.detect_language("hello") == "en"


def test_detect_language_keeps_short_code(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "fr")
    assert splitter.detect_language("bonjour") == "fr"


def test_detect_language_maps_zh_cn_to_cn(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    assert splitter.detect_language("中文") == "cn"


def test_convert_dict_uses_default_separator_and_ends():
    data = collections.OrderedDict([("en", " hello"), ("fr", " bonjour")])
    result = splitter.convert(data)
    assert result == "en\nhello.:fr\nbonjour"


def test_convert_dict_with_title_uses_colon_end():
    data = collections.OrderedDict([("en", " hello"), ("fr", " bonjour")])
    result = splitter.convert(data, title=True)
    assert result == "en:hello.:fr:bonjour"


def test_convert_str_returns_input_unchanged():
    text = "plain text"
    assert splitter.convert(text) == text


def test_split_empty_text_returns_input():
    assert splitter.split("") == ""


def test_split_short_prefixed_text_returns_ordered_dict_without_detection():
    result = splitter.split(".:en")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "")]


def test_split_explicit_language_token_single_chunk():
    text = ".:enHello world"
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Hello world")]


def test_split_multiple_explicit_language_tokens_preserve_order():
    text = ".:enHello.:frBonjour"
    result = splitter.split(text)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello"
    assert result["fr"] == "Bonjour"


def test_split_autodetect_single_chunk_uses_detect_language(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda text, max_length=2: "en")
    result = splitter.split("Just some text")
    assert list(result.items()) == [("en", "Just some text")]


def test_split_autodetect_multiple_paragraphs_inserts_separator(monkeypatch):
    calls = []

    def fake_detect(text, max_length=2):
        calls.append(text)
        return "en" if "first" in text else "fr"

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    text = "first paragraph\n\nsecond paragraph"
    result = splitter.split(text)
    assert calls == ["first paragraph", "second paragraph"]
    assert result["en"] == "first paragraph\n\n"
    assert result["fr"] == "second paragraph"


def test_split_autodetect_appends_unknown_language_on_detection_error(monkeypatch):
    def fake_detect(text, max_length=2):
        raise RuntimeError("no language")

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    result = splitter.split("undetectable text")
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == "undetectable text"


def test_split_markdown_roundtrip_uses_convert(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda text, max_length=2: "en")
    result = splitter.split("hello", markdown=True)
    assert result == "en\nhello"


def test_split_markdown_roundtrip_with_title_uses_colon(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda text, max_length=2: "fr")
    result = splitter.split("bonjour", markdown=True, title=True)
    assert result == "fr:bonjour"


def test_split_explicit_language_with_extra_text_and_paragraphs():
    text = ".:enHello\n\nWorld.:frBonjour"
    result = splitter.split(text)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello\n\nWorld"
    assert result["fr"] == "Bonjour"


def test_split_non_autodetect_branch_keeps_text_after_language_prefix(monkeypatch):
    text = ".:enHello world"
    result = splitter.split(text, autodetect=False)
    assert list(result.items()) == [("en", "Hello world")]


def test_split_returns_empty_ordered_dict_for_non_empty_text_with_no_language_and_disabled_autodetect():
    result = splitter.split("plain text", autodetect=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == []
