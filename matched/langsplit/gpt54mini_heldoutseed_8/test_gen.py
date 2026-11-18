import collections

import pytest

import langsplit.splitter as splitter


def test_detect_language_uses_short_code_for_zh_cn(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    assert splitter.detect_language("中文文本") == "cn"


def test_detect_language_truncates_long_code(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "english")
    assert splitter.detect_language("anything") == "en"


def test_detect_language_returns_code_as_is_for_short_language(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "fr")
    assert splitter.detect_language("bonjour") == "fr"


def test_convert_dict_uses_default_separators_and_strips_leading_whitespace():
    data = collections.OrderedDict([("en", " Hello"), ("fr", " Bonjour")])
    result = splitter.convert(data)
    assert result == "en\nHello.:fr\nBonjour"


def test_convert_dict_title_mode_uses_colon_separator():
    data = collections.OrderedDict([("en", " Hello"), ("fr", " Bonjour")])
    result = splitter.convert(data, title=True)
    assert result == "en:Hello.:fr:Bonjour"


def test_convert_non_dict_returns_input_unchanged():
    original = "not a dict"
    assert splitter.convert(original) == original


def test_split_none_or_empty_returns_input_unchanged():
    assert splitter.split("") == ""
    assert splitter.split(None) is None


def test_split_special_case_short_prefixed_text_returns_ordered_dict():
    result = splitter.split(".:en")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "")]


def test_split_explicit_language_chunks_are_grouped_in_order():
    text = ".:en:Hello.:fr:Bonjour"
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello"
    assert result["fr"] == "Bonjour"


def test_split_accumulates_multiple_chunks_for_same_language():
    text = ".:en:Hello.:fr:Bonjour.:en:World"
    result = splitter.split(text)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "HelloWorld"
    assert result["fr"] == "Bonjour"


def test_split_ignores_empty_tokens_between_separators():
    text = ".:en:Hello.:.:fr:Bonjour"
    result = splitter.split(text)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello"
    assert result["fr"] == "Bonjour"


def test_split_autodetect_false_uses_unknown_language_bucket():
    text = ".:plain text without explicit language"
    result = splitter.split(text, autodetect=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == "in text without explicit language"


def test_split_autodetect_uses_detect_language_for_unknown_chunks(monkeypatch):
    monkeypatch.setattr(
        splitter,
        "detect_language",
        lambda paragraph: "en" if "Hello" in paragraph else "fr",
    )
    text = ".:Hello world\n\nBonjour tout le monde"
    result = splitter.split(text, sep=".:")
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello world"
    assert result["fr"] == "Bonjour tout le monde"


def test_split_autodetect_preserves_paragraph_separator_between_detected_paragraphs(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    text = ".:Paragraph one\n\nParagraph two"
    result = splitter.split(text, sep=".:")
    assert list(result.keys()) == ["en"]
    assert result["en"] == "Paragraph one\n\nParagraph two"


def test_split_autodetect_logs_and_uses_unknown_language_when_detection_fails(monkeypatch):
    def boom(_text):
        raise RuntimeError("failed")

    monkeypatch.setattr(splitter, "detect_language", boom)

    messages = []

    def fake_info(msg):
        messages.append(msg)

    monkeypatch.setattr(splitter.logger, "info", fake_info)

    text = ".:Unclear paragraph"
    result = splitter.split(text, sep=".:")
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == "Unclear paragraph"
    assert messages == ["Language not detected: Unclear paragraph"]


def test_split_markdown_true_converts_result_back_to_markdown(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    text = ".:Hello world"
    result = splitter.split(text, sep=".:", markdown=True)
    assert result == "en\nHello world"


def test_split_markdown_title_mode_uses_title_separator(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    text = ".:Hello world"
    result = splitter.split(text, sep=".:", markdown=True, title=True)
    assert result == "en:Hello world"
