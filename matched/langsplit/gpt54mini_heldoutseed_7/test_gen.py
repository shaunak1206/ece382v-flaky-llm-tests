import collections

import pytest

import langsplit.splitter as splitter


def test_detect_language_truncates_detected_code(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "eng")
    assert splitter.detect_language("hello world") == "en"


def test_detect_language_maps_shorter_codes(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    assert splitter.detect_language("中文") == "cn"


def test_convert_dict_to_markdown_default_format():
    data = collections.OrderedDict([("en", "hello"), ("fr", "bonjour")])
    assert splitter.convert(data) == ".:en\nhello.:fr\nbonjour"


def test_convert_dict_to_markdown_title_format():
    data = collections.OrderedDict([("en", "hello"), ("fr", "bonjour")])
    assert splitter.convert(data, title=True) == ".:en:hello.:fr:bonjour"


def test_convert_str_passthrough():
    assert splitter.convert("already markdown") == "already markdown"


def test_convert_strips_leading_separator_tokens():
    data = collections.OrderedDict([("en", "hello")])
    # The output starts with ".:en\n...", and the cleanup loop should remove the leading ".:"
    assert splitter.convert(data).startswith("en\nhello")
    assert not splitter.convert(data).startswith(".:")


def test_split_returns_empty_input_unchanged():
    assert splitter.split("") == ""


def test_split_handles_short_prefix_without_end_symbol():
    result = splitter.split(".:en")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "")]


def test_split_explicit_language_and_chunk():
    result = splitter.split(".:enHello")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Hello")]


def test_split_merges_repeated_language_keys():
    result = splitter.split(".:enHello.:enWorld")
    assert list(result.items()) == [("en", "HelloWorld")]


def test_split_with_markdown_roundtrip(monkeypatch):
    # Use an explicit language marker so no autodetection is needed.
    result = splitter.split(".:enHello.:frBonjour", markdown=True)
    assert result == ".:en\nHello.:fr\nBonjour"


def test_split_with_title_markdown_roundtrip():
    result = splitter.split(".:enHello.:frBonjour", markdown=True, title=True)
    assert result == ".:en:Hello.:fr:Bonjour"


def test_split_autodetect_single_paragraph_uses_detect_language(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda text: "en")
    result = splitter.split("Plain paragraph without markers", autodetect=True)
    assert list(result.items()) == [("en", "Plain paragraph without markers")]


def test_split_autodetect_multiple_paragraphs_preserves_separator(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda text: "en")
    result = splitter.split("First paragraph\n\nSecond paragraph", autodetect=True)
    assert list(result.items()) == [("en", "First paragraph\n\nSecond paragraph")]


def test_split_autodetect_falls_back_to_unknown_language(monkeypatch):
    def boom(text):
        raise ValueError("no language")

    monkeypatch.setattr(splitter, "detect_language", boom)
    result = splitter.split("Undetectable text", autodetect=True)
    assert list(result.items()) == [("xx", "Undetectable text")]


def test_split_non_autodetect_uses_rest_of_chunk_after_language_tag():
    result = splitter.split(".:enHello", autodetect=False)
    assert list(result.items()) == [("en", "Hello")]
