import collections

import pytest

from langsplit import splitter
from langsplit.splitter import split


@pytest.fixture
def fake_detect(monkeypatch):
    """Deterministic language detection: 'Bonjour' -> fr, 'Hallo' -> de, else en."""
    def _fake(text, max_length=2):
        if 'Bonjour' in text:
            return 'fr'
        if 'Hallo' in text:
            return 'de'
        return 'en'
    monkeypatch.setattr(splitter, 'detect_language', _fake)
    return _fake


def test_empty_string_returned_as_is():
    assert split('') == ''


def test_none_returned_as_is():
    assert split(None) is None


def test_short_text_without_end_symbol():
    result = split('.:en')
    assert isinstance(result, collections.OrderedDict)
    assert result == {'en': ''}


def test_short_text_with_custom_max_key_length():
    result = split('.:eng', max_key_length=3)
    assert dict(result) == {'eng': ''}


def test_basic_two_languages():
    result = split('.:en\nHello.:fr\nBonjour')
    assert isinstance(result, collections.OrderedDict)
    assert dict(result) == {'en': 'Hello', 'fr': 'Bonjour'}


def test_key_order_is_preserved():
    result = split('.:fr\nBonjour.:en\nHello.:de\nHallo')
    assert list(result.keys()) == ['fr', 'en', 'de']


def test_same_keys_are_combined():
    result = split('.:en\nHello .:fr\nBonjour.:en\nWorld')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello World'
    assert result['fr'] == 'Bonjour'


def test_colon_as_end_symbol():
    result = split('.:en:Hello.:fr:Bonjour')
    assert dict(result) == {'en': 'Hello', 'fr': 'Bonjour'}


def test_key_with_empty_text():
    result = split('.:en\n')
    assert dict(result) == {'en': ''}


def test_custom_separator():
    result = split('|en\nHello|fr\nBonjour', sep='|')
    assert dict(result) == {'en': 'Hello', 'fr': 'Bonjour'}


def test_custom_key_length():
    result = split('.:eng\nHello.:fra\nBonjour', max_key_length=3)
    assert dict(result) == {'eng': 'Hello', 'fra': 'Bonjour'}


def test_too_short_key_goes_to_unknown_without_autodetect():
    result = split('.:e\nHello there', autodetect=False)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]


def test_unmarked_text_without_autodetect_is_unknown():
    result = split('Hello world', autodetect=False)
    assert list(result.keys()) == ['xx']


def test_autodetect_splits_paragraphs(fake_detect):
    result = split('Hello\n\nBonjour')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello\n\n'
    assert result['fr'] == 'Bonjour'


def test_autodetect_unmarked_prefix_before_marked_part(fake_detect):
    result = split('Hallo Welt.:en\nHello')
    assert list(result.keys()) == ['de', 'en']
    assert result['de'] == 'Hallo Welt'
    assert result['en'] == 'Hello'


def test_autodetect_failure_falls_back_to_unknown(monkeypatch):
    def failing(text, max_length=2):
        raise ValueError('cannot detect')
    monkeypatch.setattr(splitter, 'detect_language', failing)
    result = split('Something here')
    assert dict(result) == {'xx': 'Something here'}


def test_markdown_output():
    result = split('.:en\nHello.:fr\nBonjour', markdown=True)
    assert result == '.:en\nHello.:fr\nBonjour'


def test_markdown_title_output():
    result = split('.:en\nHello.:fr\nBonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_markdown_roundtrip_with_title_input():
    result = split('.:en:Hello.:fr:Bonjour', markdown=True)
    assert result == '.:en\nHello.:fr\nBonjour'
