import logging

import pytest

from langsplit import splitter
from langsplit.splitter import convert, detect_language, split


@pytest.fixture
def fake_detect(monkeypatch):
    """Replace language detection with a deterministic fake that records calls."""
    calls = []

    def _install(func):
        def recording(paragraph, *args, **kwargs):
            calls.append(paragraph)
            return func(paragraph, *args, **kwargs)
        monkeypatch.setattr(splitter, 'detect_language', recording)
        return calls

    return _install


def test_split_empty_input_is_returned_unchanged():
    assert split('') == ''
    assert split(None) is None


def test_split_short_text_with_key_but_no_end_symbol_returns_empty_value():
    result = split('.:en')
    assert list(result.keys()) == ['en']
    assert result['en'] == ''


def test_split_parses_two_languages_in_order():
    result = split('.:en\nHello.:fr\nBonjour')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello'
    assert result['fr'] == 'Bonjour'


def test_split_merges_same_language_keeping_first_appearance_order():
    result = split('.:en\nOne.:fr\nUn.:en\nTwo')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'OneTwo'
    assert result['fr'] == 'Un'


def test_split_accepts_colon_as_end_symbol():
    result = split('.:en:Hi.:fr:Salut')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hi'
    assert result['fr'] == 'Salut'


def test_split_with_custom_separator():
    result = split('##en\nHello##fr\nBonjour', sep='##')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello'
    assert result['fr'] == 'Bonjour'


def test_split_with_longer_key_length_parses_three_letter_key():
    result = split('.:eng\nHello', max_key_length=3)
    assert list(result.keys()) == ['eng']
    assert result['eng'] == 'Hello'


def test_split_uses_detection_for_untagged_text(fake_detect):
    calls = fake_detect(lambda paragraph, *a, **k: 'en')
    result = split('Hello world.')
    assert list(result.keys()) == ['en']
    assert result['en'] == 'Hello world.'
    assert calls == ['Hello world.']


def test_split_detects_each_paragraph_separately(fake_detect):
    fake_detect(lambda paragraph, *a, **k: 'en' if 'First' in paragraph else 'fr')
    result = split('First para.\n\nSecond para.')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'First para.\n\n'
    assert result['fr'] == 'Second para.'


def test_split_falls_back_to_unknown_language_when_detection_fails(fake_detect):
    def failing(paragraph, *a, **k):
        raise ValueError('no features in text')

    fake_detect(failing)
    result = split('Hello.')
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == 'Hello.'


def test_split_without_autodetect_does_not_call_detection(fake_detect):
    calls = fake_detect(lambda paragraph, *a, **k: 'en')
    result = split('Hello', autodetect=False)
    assert calls == []
    assert splitter.settings.UNKNOWN_LANGUAGE in result


def test_split_markdown_round_trip():
    text = '.:en\nHello.:fr\nBonjour'
    assert split(text, markdown=True) == text


def test_split_markdown_with_title_uses_colon_end():
    result = split('.:en\nHi.:fr\nSalut', markdown=True, title=True)
    assert result == '.:en:Hi.:fr:Salut'


def test_convert_builds_markdown_from_dict():
    assert convert({'en': 'Hello', 'fr': 'Bonjour'}) == '.:en\nHello.:fr\nBonjour'


def test_convert_strips_leading_whitespace_of_values():
    assert convert({'en': '   Hi'}) == '.:en\nHi'


def test_convert_returns_non_dict_input_unchanged():
    assert convert('plain text') == 'plain text'
    assert convert(None) is None


def test_detect_language_shortens_long_code_via_mapping(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'zh-cn')
    assert detect_language('some text') == 'cn'


def test_detect_language_keeps_code_within_max_length(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'zh-cn')
    assert detect_language('some text', max_length=5) == 'zh-cn'
