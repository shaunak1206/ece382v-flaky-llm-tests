import collections

import pytest

from langsplit import splitter
from langsplit.splitter import split


def _fake_detect(paragraph):
    if 'Hello' in paragraph:
        return 'en'
    if 'Bonjour' in paragraph:
        return 'fr'
    raise ValueError('cannot detect')


def test_empty_string_returns_empty_string():
    assert split('') == ''


def test_none_returns_none():
    assert split(None) is None


def test_short_text_without_end_symbol():
    result = split('.:en')
    assert isinstance(result, collections.OrderedDict)
    assert result == {'en': ''}


def test_short_text_with_partial_key():
    result = split('.:e')
    assert dict(result) == {'e': ''}


def test_basic_split_two_languages():
    result = split('.:en\nHello.:fr\nBonjour')
    assert isinstance(result, collections.OrderedDict)
    assert dict(result) == {'en': 'Hello', 'fr': 'Bonjour'}


def test_key_order_is_preserved():
    result = split('.:fr\nBonjour.:en\nHello.:de\nHallo')
    assert list(result.keys()) == ['fr', 'en', 'de']


def test_same_keys_are_combined():
    result = split('.:en\nHello .:fr\nBonjour .:en\nWorld')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello World'
    assert result['fr'] == 'Bonjour '


def test_colon_as_end_symbol():
    result = split('.:en:Hello.:fr:Bonjour')
    assert dict(result) == {'en': 'Hello', 'fr': 'Bonjour'}


def test_custom_separator():
    result = split('|en\nHello|fr\nBonjour', sep='|')
    assert dict(result) == {'en': 'Hello', 'fr': 'Bonjour'}


def test_max_key_length_three():
    result = split('.:eng\nHello.:fra\nBonjour', max_key_length=3)
    assert dict(result) == {'eng': 'Hello', 'fra': 'Bonjour'}


def test_multiline_content_is_kept():
    result = split('.:en\nLine one\nLine two.:fr\nLigne un')
    assert result['en'] == 'Line one\nLine two'
    assert result['fr'] == 'Ligne un'


def test_markdown_roundtrip():
    text = '.:en\nHello.:fr\nBonjour'
    assert split(text, markdown=True) == text


def test_markdown_title_uses_colon_end():
    result = split('.:en\nHello.:fr\nBonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_autodetect_assigns_languages_per_paragraph(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language', _fake_detect)
    result = split('Hello\n\nBonjour')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello\n\n'
    assert result['fr'] == 'Bonjour'


def test_autodetect_failure_falls_back_to_unknown(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language', _fake_detect)
    result = split('Something unrecognizable')
    assert dict(result) == {splitter.settings.UNKNOWN_LANGUAGE: 'Something unrecognizable'}


def test_text_before_first_separator_is_detected(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language', _fake_detect)
    result = split('Hello there.:fr\nBonjour')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello there'
    assert result['fr'] == 'Bonjour'


def test_autodetect_disabled_does_not_call_detector(monkeypatch):
    def boom(_):
        raise AssertionError('detector should not be called')

    monkeypatch.setattr(splitter, 'detect_language', boom)
    result = split('Hello world', autodetect=False)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]


def test_autodetect_disabled_still_parses_explicit_keys():
    result = split('.:en\nHello.:fr\nBonjour', autodetect=False)
    assert dict(result) == {'en': 'Hello', 'fr': 'Bonjour'}
