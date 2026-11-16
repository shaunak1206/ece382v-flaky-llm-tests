import collections

import pytest

from langsplit import splitter
from langsplit.splitter import split


EN_RU = '.:en\nHello .:ru\nПривет'


def test_empty_string_returned_as_is():
    assert split('') == ''


def test_none_returned_as_is():
    assert split(None) is None


def test_basic_split_two_languages():
    result = split(EN_RU)
    assert result == {'en': 'Hello ', 'ru': 'Привет'}


def test_result_is_ordered_dict_with_sequence_preserved():
    result = split(EN_RU)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ['en', 'ru']


def test_order_follows_first_appearance():
    result = split('.:ru\nПривет .:en\nHello')
    assert list(result.keys()) == ['ru', 'en']


def test_same_keys_are_combined():
    result = split('.:en\nHello .:ru\nПривет .:en\nWorld')
    assert list(result.keys()) == ['en', 'ru']
    assert result['en'] == 'Hello World'
    assert result['ru'] == 'Привет '


def test_key_only_without_end_symbol():
    result = split('.:en')
    assert result == collections.OrderedDict([('en', '')])


def test_key_with_colon_as_end():
    result = split('.:en:Hello')
    assert result == {'en': 'Hello'}


def test_custom_separator():
    result = split('##en\nHi ##ru\nПривет', sep='##')
    assert result == {'en': 'Hi ', 'ru': 'Привет'}


def test_longer_key_allowed_with_max_key_length():
    result = split('.:eng\nHi .:rus\nПривет', max_key_length=3)
    assert result == {'eng': 'Hi ', 'rus': 'Привет'}


def test_short_key_with_max_key_length_three_no_end():
    result = split('.:eng', max_key_length=3)
    assert result == {'eng': ''}


def test_too_long_key_goes_to_unknown_without_autodetect():
    result = split('.:eng\nHi', autodetect=False)
    assert list(result.keys()) == ['xx']


def test_too_short_key_goes_to_unknown_without_autodetect():
    result = split('.:e\nHi', autodetect=False)
    assert list(result.keys()) == ['xx']


def test_unknown_text_before_keys_without_autodetect():
    result = split('intro .:en\nHi', autodetect=False)
    assert list(result.keys()) == ['xx', 'en']
    assert result['en'] == 'Hi'


def test_autodetect_used_for_text_without_keys(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language', lambda p: 'en')
    result = split('Hello world')
    assert result == {'en': 'Hello world'}


def test_autodetect_treats_paragraphs_separately(monkeypatch):
    def fake_detect(paragraph):
        return 'en' if 'Hello' in paragraph else 'ru'

    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    result = split('Hello\n\nПривет')
    assert list(result.keys()) == ['en', 'ru']
    assert result['en'] == 'Hello\n\n'
    assert result['ru'] == 'Привет'


def test_autodetect_same_language_paragraphs_keep_separator(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language', lambda p: 'en')
    result = split('Hello\n\nWorld')
    assert result == {'en': 'Hello\n\nWorld'}


def test_autodetect_failure_falls_back_to_unknown(monkeypatch):
    def failing(paragraph):
        raise RuntimeError('no language')

    monkeypatch.setattr(splitter, 'detect_language', failing)
    result = split('12345')
    assert result == {'xx': '12345'}


def test_markdown_true_returns_string():
    result = split(EN_RU, markdown=True)
    assert isinstance(result, str)
    assert result == '.:en\nHello .:ru\nПривет'


def test_markdown_title_uses_colon_end():
    result = split(EN_RU, markdown=True, title=True)
    assert result == '.:en:Hello .:ru:Привет'
