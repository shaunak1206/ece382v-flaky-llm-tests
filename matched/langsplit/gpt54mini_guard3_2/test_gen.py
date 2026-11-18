import collections

import pytest

from langsplit import splitter


def test_split_empty_string_returns_input():
    assert splitter.split("") == ""


def test_split_none_like_falsy_value_returns_input():
    assert splitter.split(None) is None


def test_split_short_prefixed_text_returns_ordered_dict_without_detection():
    text = ".:en"
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "")]


def test_split_explicit_language_token_single_segment():
    text = ".:enHello"
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Hello")]


def test_split_merges_same_language_across_multiple_tokens():
    text = ".:enHello.:enWorld"
    result = splitter.split(text)
    assert list(result.items()) == [("en", "HelloWorld")]


def test_split_preserves_language_order_of_first_appearance():
    text = ".:frBonjour.:enHello.:frSalut"
    result = splitter.split(text)
    assert list(result.items()) == [("fr", "BonjourSalut"), ("en", "Hello")]


def test_split_uses_unknown_language_for_unmarked_text_when_autodetect_disabled():
    text = "plain text"
    result = splitter.split(text, autodetect=False)
    assert list(result.items()) == [(splitter.settings.UNKNOWN_LANGUAGE, "lain text")]


def test_split_autodetect_uses_patched_language_detector(monkeypatch):
    calls = []

    def fake_detect_language(text, max_length=2):
        calls.append((text, max_length))
        return "en"

    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)
    result = splitter.split("hello world", autodetect=True)
    assert list(result.items()) == [("en", "hello world")]
    assert calls == [("hello world", 2)]


def test_split_autodetect_multiple_paragraphs_in_same_chunk(monkeypatch):
    def fake_detect_language(text, max_length=2):
        return "fr" if "bonjour" in text else "en"

    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)
    text = "bonjour\n\nhello"
    result = splitter.split(text, autodetect=True, pargraph_sep="\n\n")
    assert list(result.items()) == [("fr", "bonjour\n\n"), ("en", "hello")]


def test_split_autodetect_falls_back_to_unknown_on_detection_error(monkeypatch):
    def fake_detect_language(text, max_length=2):
        raise ValueError("cannot detect")

    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)
    text = "mystery"
    result = splitter.split(text, autodetect=True)
    assert list(result.items()) == [(splitter.settings.UNKNOWN_LANGUAGE, "mystery")]


def test_split_markdown_false_returns_ordered_dict_with_explicit_language():
    text = ".:enHello.:frSalut"
    result = splitter.split(text, markdown=False)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [("en", "Hello"), ("fr", "Salut")]


def test_split_markdown_true_uses_convert_format():
    text = ".:enHello.:frSalut"
    result = splitter.split(text, markdown=True)
    assert result == "en\nHello.:fr\nSalut"


def test_convert_returns_input_unchanged_for_non_dict():
    text = "already formatted"
    assert splitter.convert(text) == text


def test_convert_with_title_uses_last_end_symbol():
    data = collections.OrderedDict([("en", "Hello"), ("fr", "Salut")])
    result = splitter.convert(data, title=True)
    assert result == "en:Hello.:fr:Salut"


def test_convert_strips_leading_whitespace_from_values():
    data = collections.OrderedDict([("en", "   Hello"), ("fr", "\tSalut")])
    result = splitter.convert(data)
    assert result == "en\nHello.:fr\nSalut"


def test_split_handles_custom_separators_and_end_symbols():
    text = "|en:Hello|fr:Salut"
    result = splitter.split(text, sep="|", ends=[":", "\n"], autodetect=False)
    assert list(result.items()) == [("en", "Hello"), ("fr", "Salut")]


def test_split_with_empty_token_segments_ignored(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda text, max_length=2: "en")
    text = ".:enHello.:.:frWorld"
    result = splitter.split(text, autodetect=True)
    assert list(result.items()) == [("en", "Hello"), ("fr", "World")]
