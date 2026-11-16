import collections

import pytest

from langsplit import splitter
from langsplit.splitter import split


@pytest.fixture
def fake_detect(monkeypatch):
    """Replace the (non-deterministic) language detection with a simple fake."""
    def _detect(text, max_length=2):
        if 'Hello' in text:
            return 'en'
        if 'Bonjour' in text:
            return 'fr'
        return 'de'
    monkeypatch.setattr(splitter, 'detect_language', _detect)
    return _detect


def test_empty_string_returned_as_is():
    assert split('') == ''


def test_none_returned_as_is():
    assert split(None) is None


def test_basic_two_languages():
    result = split('.:en\nHello .:ru\nПривет')
    assert result == {'en': 'Hello ', 'ru': 'Привет'}


def test_returns_ordered_dict_preserving_order():
    result = split('.:ru\nПривет .:en\nHello')
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ['ru', 'en']


def test_same_keys_are_combined():
    result = split('.:en\nHello .:ru\nПривет .:en\nWorld')
    assert list(result.keys()) == ['en', 'ru']
    assert result['en'] == 'Hello World'
    assert result['ru'] == 'Привет '


def test_colon_as_end_symbol():
    result = split('.:en:Hello .:ru:Привет')
    assert result == {'en': 'Hello ', 'ru': 'Привет'}


def test_key_without_end_symbol_special_case():
    result = split('.:en')
    assert isinstance(result, collections.OrderedDict)
    assert result == {'en': ''}


def test_key_with_end_but_empty_text():
    result = split('.:en\n')
    assert result == {'en': ''}


def test_custom_separator():
    result = split('##en\nHello ##ru\nПривет', sep='##')
    assert result == {'en': 'Hello ', 'ru': 'Привет'}


def test_custom_max_key_length():
    result = split('.:eng\nHello .:rus\nПривет', max_key_length=3)
    assert result == {'eng': 'Hello ', 'rus': 'Привет'}


def test_markdown_round_trip():
    text = '.:en\nHello .:ru\nПривет'
    assert split(text, markdown=True) == text


def test_markdown_title_uses_colon_end():
    result = split('.:en\nHello .:ru\nПривет', markdown=True, title=True)
    assert result == '.:en:Hello .:ru:Привет'


def test_autodetect_without_markers_splits_paragraphs(fake_detect):
    result = split('Hello\n\nBonjour')
    assert result == {'en': 'Hello\n\n', 'fr': 'Bonjour'}
    assert list(result.keys()) == ['en', 'fr']


def test_autodetect_for_unmarked_part_before_marked_part(fake_detect):
    result = split('Intro text .:en\nHello')
    assert list(result.keys()) == ['de', 'en']
    assert result['de'] == 'Intro text '
    assert result['en'] == 'Hello'


def test_autodetect_failure_falls_back_to_unknown_language(monkeypatch):
    def boom(text, max_length=2):
        raise RuntimeError('no detection')
    monkeypatch.setattr(splitter, 'detect_language', boom)
    result = split('Some text here')
    assert result == {splitter.settings.UNKNOWN_LANGUAGE: 'Some text here'}


def test_autodetect_disabled_uses_unknown_language_key(monkeypatch):
    def fail(text, max_length=2):
        raise AssertionError('detect_language should not be called')
    monkeypatch.setattr(splitter, 'detect_language', fail)
    result = split('Hello world', autodetect=False)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]


def test_too_short_key_is_treated_as_unknown(fake_detect):
    result = split('.:e\nHello')
    assert 'e' not in result
    assert 'en' in result
