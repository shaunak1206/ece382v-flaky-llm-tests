import collections

import pytest

import langsplit.splitter as splitter


def test_detect_language_truncates_to_max_length(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "english")
    assert splitter.detect_language("hello", max_length=2) == "en"


def test_detect_language_maps_zh_cn_to_cn(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    assert splitter.detect_language("你好", max_length=2) == "cn"


def test_detect_language_maps_zh_tw_to_cn(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-tw")
    assert splitter.detect_language("你好", max_length=2) == "cn"


def test_convert_dict_basic_uses_default_separator_and_end():
    data = collections.OrderedDict([("en", "Hello"), ("fr", "Bonjour")])
    assert splitter.convert(data) == ".:en\nHello.:fr\nBonjour"


def test_convert_dict_title_uses_colon_end():
    data = collections.OrderedDict([("en", "Hello"), ("fr", "Bonjour")])
    assert splitter.convert(data, title=True) == ".:en:Hello.:fr:Bonjour"


def test_convert_strips_leading_whitespace_from_values():
    data = collections.OrderedDict([("en", "  Hello"), ("fr", "\nBonjour")])
    assert splitter.convert(data) == ".:en\nHello.:fr\nBonjour"


def test_convert_non_dict_returns_input_unchanged():
    value = "plain text"
    assert splitter.convert(value) == value


def test_split_empty_text_returns_input_unchanged():
    assert splitter.split("") == ""


def test_split_short_prefixed_text_returns_single_item_ordered_dict():
    result = splitter.split(".:en")
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "")]


def test_split_explicit_languages_preserves_order_and_concatenates_chunks():
    text = ".:en\nHello.:fr\nBonjour"
    result = splitter.split(text, autodetect=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Hello"), ("fr", "Bonjour")]


def test_split_autodetect_single_chunk_uses_detected_language(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    result = splitter.split("Hello world", autodetect=True)
    assert list(result.items()) == [("en", "Hello world")]


def test_split_autodetect_multiple_paragraphs_keeps_paragraph_separator(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    text = "First paragraph\n\nSecond paragraph"
    result = splitter.split(text, autodetect=True)
    assert list(result.items()) == [("en", "First paragraph\n\nSecond paragraph")]


def test_split_autodetect_logs_and_uses_unknown_language_on_detection_error(monkeypatch, caplog):
    def raise_error(_paragraph):
        raise ValueError("boom")

    monkeypatch.setattr(splitter, "detect_language", raise_error)
    with caplog.at_level("INFO"):
        result = splitter.split("Undetectable text", autodetect=True)

    assert list(result.items()) == [("xx", "Undetectable text")]
    assert any("Language not detected" in record.message for record in caplog.records)


def test_split_markdown_false_returns_ordered_dict(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    result = splitter.split("Hello", autodetect=True, markdown=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Hello")]


def test_split_markdown_true_converts_back_to_markdown(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    result = splitter.split("Hello", autodetect=True, markdown=True)
    assert result == ".:en\nHello"


def test_split_markdown_true_title_mode_uses_colon_separator(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    result = splitter.split("Hello", autodetect=True, markdown=True, title=True)
    assert result == ".:en:Hello"


def test_split_handles_unknown_prefix_without_autodetect():
    # This exercises the branch that treats an unrecognized prefix as unknown.
    result = splitter.split("foo bar", autodetect=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("xx", "foo bar")]


def test_split_ignores_empty_tokens_between_separators(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    text = ".:en\nHello.:.:fr\nBonjour"
    result = splitter.split(text, autodetect=True)
    assert list(result.items()) == [("en", "Hello"), ("fr", "Bonjour")]
