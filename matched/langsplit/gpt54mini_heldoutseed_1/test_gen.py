import collections

import pytest

from langsplit import splitter


def test_detect_language_uses_short_code_for_zh_cn(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    assert splitter.detect_language("你好") == "cn"


def test_detect_language_truncates_long_code_to_max_length(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "eng")
    assert splitter.detect_language("hello", max_length=2) == "en"


def test_convert_returns_input_when_not_dict():
    assert splitter.convert("plain text") == "plain text"


def test_convert_dict_joins_with_default_separator_and_strips_leading_spaces():
    data = {"en": " hello", "fr": " monde"}
    out = splitter.convert(data)
    assert out == ".:en\nhello.:fr\nmonde"


def test_convert_title_uses_colon_separator():
    data = {"en": " hello"}
    out = splitter.convert(data, title=True)
    assert out == ".:en:hello"


def test_convert_removes_repeated_leading_separators_from_output():
    data = {"en": "text"}
    out = splitter.convert(data, sep=".:", ends=["\n", "\r", "\r\n", ":"], title=False)
    assert out == ".:en\ntext"


def test_split_returns_empty_string_for_empty_text():
    assert splitter.split("") == ""


def test_split_special_short_prefixed_token_returns_ordered_dict():
    out = splitter.split(".:en")
    assert isinstance(out, collections.OrderedDict)
    assert list(out.items()) == [("en", "")]


def test_split_explicit_language_token_single_chunk():
    out = splitter.split(".:enHello")
    assert isinstance(out, collections.OrderedDict)
    assert list(out.items()) == [("en", "Hello")]


def test_split_combines_multiple_explicit_language_chunks_in_order():
    out = splitter.split(".:enHello.:frBonjour")
    assert list(out.items()) == [("en", "Hello"), ("fr", "Bonjour")]


def test_split_keeps_repeated_explicit_language_chunks_together():
    out = splitter.split(".:enHello.:enWorld")
    assert list(out.items()) == [("en", "HelloWorld")]


def test_split_autodetect_single_paragraph_uses_detect_language(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    out = splitter.split("hello world", autodetect=True)
    assert list(out.items()) == [("en", "hello world")]


def test_split_autodetect_multiple_paragraphs_preserves_separator(monkeypatch):
    calls = []

    def fake_detect(paragraph):
        calls.append(paragraph)
        return "en" if "one" in paragraph else "fr"

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    out = splitter.split("one\n\ntwo", autodetect=True)
    assert calls == ["one", "two"]
    assert list(out.items()) == [("en", "one\n\n"), ("fr", "two")]


def test_split_autodetect_routes_detection_errors_to_unknown_language(monkeypatch, caplog):
    def fake_detect(_paragraph):
        raise ValueError("boom")

    monkeypatch.setattr(splitter, "detect_language", fake_detect)
    with caplog.at_level("INFO"):
        out = splitter.split("unhandled text", autodetect=True)

    assert list(out.items()) == [("xx", "unhandled text")]
    assert any("Language not detected" in rec.message for rec in caplog.records)


def test_split_without_autodetect_keeps_unknown_bucket(monkeypatch):
    out = splitter.split("plain text", autodetect=False)
    assert list(out.items()) == [("xx", "text")]


def test_split_markdown_round_trip_with_title_false(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    out = splitter.split("hello", autodetect=True, markdown=True)
    assert out == ".:en\nhello"


def test_split_markdown_round_trip_with_title_true(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda paragraph: "en")
    out = splitter.split("hello", autodetect=True, markdown=True, title=True)
    assert out == ".:en:hello"
