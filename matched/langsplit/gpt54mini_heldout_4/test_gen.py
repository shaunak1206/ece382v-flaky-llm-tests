import collections

import pytest

from langsplit import splitter


def test_detect_language_truncates_detected_code(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "eng")
    assert splitter.detect_language("hello") == "en"


def test_detect_language_uses_shorter_codes_mapping(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    assert splitter.detect_language("你好") == "cn"


def test_detect_language_respects_max_length(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "fr")
    assert splitter.detect_language("bonjour", max_length=1) == "f"


def test_convert_returns_original_for_non_dict():
    text = "plain text"
    assert splitter.convert(text) == text


def test_convert_joins_language_segments_with_default_separator():
    data = collections.OrderedDict([("en", "Hello"), ("fr", "Bonjour")])
    assert splitter.convert(data) == ".:en\nHello.:fr\nBonjour"


def test_convert_title_mode_uses_title_separator():
    data = collections.OrderedDict([("en", "Hello"), ("fr", "Bonjour")])
    assert splitter.convert(data, title=True) == ".:en:Hello.:fr:Bonjour"


def test_convert_lstrips_text_values_before_joining():
    data = collections.OrderedDict([("en", "  Hello"), ("fr", "\nBonjour")])
    assert splitter.convert(data) == ".:en\nHello.:fr\nBonjour"


def test_split_returns_input_for_empty_text():
    assert splitter.split("") == ""


def test_split_handles_short_prefixed_fragment_without_newline():
    result = splitter.split(".:enHello")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Hello")]


def test_split_detects_language_for_single_paragraph(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    result = splitter.split("Hello world")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Hello world")]


def test_split_detects_multiple_paragraphs_and_preserves_separator(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    text = "Hello\n\nWorld"
    result = splitter.split(text)
    assert list(result.keys()) == ["en"]
    assert result["en"] == "Hello\n\nWorld"


def test_split_collects_unknown_language_when_detection_fails(monkeypatch):
    def boom(_):
        raise ValueError("no language")

    monkeypatch.setattr(splitter, "detect_language", boom)
    result = splitter.split("??")
    assert list(result.items()) == [("xx", "??")]


def test_split_parses_explicit_language_tags_without_autodetect():
    text = ".:en\nHello.:fr\nBonjour"
    result = splitter.split(text, autodetect=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Hello"), ("fr", "Bonjour")]


def test_split_preserves_order_of_explicit_tags():
    text = ".:fr\nBonjour.:en\nHello.:de\nHallo"
    result = splitter.split(text, autodetect=False)
    assert list(result.keys()) == ["fr", "en", "de"]


def test_split_uses_custom_separator_and_endings_for_markdown():
    text = "||en|Hello||fr|Bonjour"
    result = splitter.split(text, sep="||", ends=["|"], autodetect=False, markdown=True)
    assert result == "||en\nHello||fr\nBonjour"


def test_split_markdown_title_mode_uses_colon_after_language():
    text = "||en|Hello||fr|Bonjour"
    result = splitter.split(text, sep="||", ends=["|"], autodetect=False, markdown=True, title=True)
    assert result == "||en:Hello||fr:Bonjour"
