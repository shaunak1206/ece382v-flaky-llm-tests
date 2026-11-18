import collections

import pytest

import langsplit.splitter as splitter


def test_split_returns_empty_string_for_empty_input():
    assert splitter.split("") == ""


def test_split_returns_none_for_none_input():
    assert splitter.split(None) is None


def test_split_short_special_case_returns_ordered_dict():
    result = splitter.split(".:en")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "")]


def test_split_short_special_case_with_extra_text_after_code():
    result = splitter.split(".:enHello")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Hello")]


def test_split_explicit_language_tokens_are_grouped_by_key():
    text = ".:enHello.:enWorld"
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "HelloWorld")]


def test_split_explicit_language_tokens_preserve_order_of_first_appearance():
    text = ".:frBonjour.:enHello.:frMonde"
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ["fr", "en"]
    assert result["fr"] == "BonjourMonde"
    assert result["en"] == "Hello"


def test_split_unknown_chunk_with_autodetect_false_goes_to_unknown_language():
    text = "plain text with no separator"
    result = splitter.split(text, autodetect=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [(splitter.settings.UNKNOWN_LANGUAGE, "plain text with no separator")]


def test_split_autodetect_uses_detect_language_for_unknown_chunks(monkeypatch):
    calls = []

    def fake_detect_language(text, max_length=2):
        calls.append((text, max_length))
        return "en"

    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)

    result = splitter.split("Hello world", autodetect=True)
    assert list(result.items()) == [("en", "Hello world")]
    assert calls == [("Hello world", 2)]


def test_split_autodetect_handles_exception_by_placing_text_in_unknown_language(monkeypatch):
    def fake_detect_language(text, max_length=2):
        raise ValueError("nope")

    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)

    result = splitter.split("Bonjour")
    assert list(result.items()) == [(splitter.settings.UNKNOWN_LANGUAGE, "Bonjour")]


def test_split_paragraph_separator_is_preserved_between_detected_paragraphs(monkeypatch):
    def fake_detect_language(text, max_length=2):
        return "en" if text == "First" else "fr"

    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)

    result = splitter.split("First\n\nSecond", autodetect=True)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "First\n\n"
    assert result["fr"] == "Second"


def test_split_markdown_false_returns_ordered_dict_even_with_detected_chunks(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda text, max_length=2: "en")
    result = splitter.split("Some text", markdown=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Some text")]


def test_split_markdown_true_converts_back_to_markdown(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda text, max_length=2: "en")
    result = splitter.split("Some text", markdown=True)
    assert result == ".:en\nSome text"


def test_split_markdown_true_with_title_uses_colon_separator(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda text, max_length=2: "en")
    result = splitter.split("Some text", markdown=True, title=True)
    assert result == ".:en:Some text"


def test_convert_dict_uses_input_order_and_strips_leading_whitespace():
    data = collections.OrderedDict([("en", "  Hello"), ("fr", " Monde")])
    result = splitter.convert(data)
    assert result == ".:en\nHello.:fr\nMonde"


def test_detect_language_shortens_zh_cn_and_zh_tw(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    assert splitter.detect_language("你好") == "cn"

    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-tw")
    assert splitter.detect_language("你好") == "cn"
