import collections

import pytest

import langsplit.splitter as splitter


def test_detect_language_truncates_long_code(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "eng")
    assert splitter.detect_language("hello", max_length=2) == "en"


def test_detect_language_handles_short_code(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "en")
    assert splitter.detect_language("hello", max_length=2) == "en"


def test_detect_language_shorter_codes_mapping_zh_cn(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    assert splitter.detect_language("你好", max_length=2) == "cn"


def test_convert_returns_non_dict_input_unchanged():
    original = "plain text"
    assert splitter.convert(original) == original


def test_convert_dict_uses_dot_sep_by_default():
    data = collections.OrderedDict([("en", " Hello"), ("fr", " Bonjour")])
    assert splitter.convert(data) == "en\nHello.:fr\nBonjour"


def test_convert_title_uses_colon_as_end():
    data = collections.OrderedDict([("en", " Hello"), ("fr", " Bonjour")])
    assert splitter.convert(data, title=True) == "en:Hello.:fr:Bonjour"


def test_convert_strips_leading_separators_created_by_empty_values():
    data = collections.OrderedDict([("en", "Hello"), ("fr", "\nBonjour")])
    result = splitter.convert(data)
    assert result.startswith("en\nHello")
    assert ".:fr\nBonjour" in result


def test_split_returns_input_when_empty():
    assert splitter.split("") == ""


def test_split_simple_explicit_language_tags():
    text = ".:enHello.:frBonjour"
    result = splitter.split(text)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello"
    assert result["fr"] == "Bonjour"


def test_split_merges_repeated_language_keys_in_order():
    text = ".:enHello.:frBonjour.:enWorld"
    result = splitter.split(text)
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "HelloWorld"
    assert result["fr"] == "Bonjour"


def test_split_without_autodetect_keeps_unknown_text_under_xx():
    text = "hello world"
    result = splitter.split(text, autodetect=False)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == "hello world"


def test_split_autodetect_single_paragraph_uses_detect_language(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    result = splitter.split("hello world", autodetect=True)
    assert list(result.keys()) == ["en"]
    assert result["en"] == "hello world"


def test_split_autodetect_multiple_paragraphs_preserves_paragraph_separator(monkeypatch):
    calls = []

    def fake_detect(paragraph):
        calls.append(paragraph)
        return "en"

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    result = splitter.split("first\n\nsecond", autodetect=True)

    assert calls == ["first", "second"]
    assert list(result.keys()) == ["en"]
    assert result["en"] == "first\n\nsecond"


def test_split_autodetect_falls_back_to_unknown_on_detection_error(monkeypatch):
    def fake_detect(paragraph):
        raise RuntimeError("boom")

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    result = splitter.split("undetectable text", autodetect=True)

    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == "undetectable text"


def test_split_returns_ordered_dict_for_special_short_input():
    result = splitter.split(".:en")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ["en"]
    assert result["en"] == ""


def test_split_markdown_round_trip_with_title_flag():
    text = ".:enHello.:frBonjour"
    markdown = splitter.split(text, markdown=True, title=True)
    assert markdown == "en:Hello.:fr:Bonjour"


def test_split_with_custom_separator_and_ends():
    text = "|en!Hello|fr!Bonjour"
    result = splitter.split(text, sep="|", ends=["\n", "!"])
    assert list(result.keys()) == ["en", "fr"]
    assert result["en"] == "Hello"
    assert result["fr"] == "Bonjour"
