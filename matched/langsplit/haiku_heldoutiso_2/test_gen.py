import collections

import langdetect
import pytest

from langsplit import splitter
from langsplit.splitter import convert, detect_language, settings, split


def test_split_empty_text_is_returned_unchanged():
    assert split('') == ''
    assert split(None) is None


def test_split_parses_language_sections():
    result = split('.:en:Hello.:fr:Bonjour')
    assert dict(result) == {'en': 'Hello', 'fr': 'Bonjour'}


def test_split_returns_ordered_dict():
    result = split('.:en:Hello.:fr:Bonjour')
    assert isinstance(result, collections.OrderedDict)


def test_split_merges_repeated_keys_keeping_first_seen_order():
    result = split('.:fr:A.:en:B.:fr:C')
    assert list(result.items()) == [('fr', 'AC'), ('en', 'B')]


def test_split_accepts_newline_as_key_end_marker():
    result = split('.:en\nHello')
    assert dict(result) == {'en': 'Hello'}


def test_split_accepts_three_letter_keys_when_max_key_length_is_three():
    result = split('.:eng:Hi', max_key_length=3)
    assert dict(result) == {'eng': 'Hi'}


def test_split_respects_min_key_length_for_single_letter_keys():
    result = split('.:e:Hi', min_key_length=1, max_key_length=1, autodetect=False)
    assert dict(result) == {'e': 'Hi'}


def test_split_uses_custom_separator():
    result = split('|en:Hi|fr:Yo', sep='|')
    assert dict(result) == {'en': 'Hi', 'fr': 'Yo'}


def test_split_short_text_with_only_separator_and_key():
    result = split('.:en')
    assert dict(result) == {'en': ''}


def test_split_autodetects_unkeyed_text(monkeypatch):
    monkeypatch.setattr(langdetect, 'detect', lambda text: 'fr')
    result = split('Bonjour tout le monde')
    assert dict(result) == {'fr': 'Bonjour tout le monde'}


def test_split_autodetects_each_paragraph_separately(monkeypatch):
    def fake_detect(text):
        return 'en' if text.startswith('Hello') else 'fr'

    monkeypatch.setattr(langdetect, 'detect', fake_detect)
    result = split('Hello world\n\nBonjour le monde')
    assert list(result.items()) == [
        ('en', 'Hello world\n\n'),
        ('fr', 'Bonjour le monde'),
    ]


def test_split_detection_failure_puts_text_under_unknown_language(monkeypatch):
    def boom(text):
        raise RuntimeError('no features in text')

    monkeypatch.setattr(langdetect, 'detect', boom)
    result = split('some text')
    assert dict(result) == {settings.UNKNOWN_LANGUAGE: 'some text'}


def test_split_markdown_output_uses_newline_end_marker():
    result = split('.:en:Hello.:fr:Bonjour', markdown=True)
    assert result == '.:en\nHello.:fr\nBonjour'


def test_split_markdown_title_output_uses_colon_end_marker():
    result = split('.:en:Hello.:fr:Bonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_convert_dict_uses_newline_end_by_default():
    languages = collections.OrderedDict([('en', 'Hello'), ('fr', 'Bonjour')])
    assert convert(languages) == '.:en\nHello.:fr\nBonjour'


def test_convert_title_uses_colon_end():
    languages = collections.OrderedDict([('en', 'Hello'), ('fr', 'Bonjour')])
    assert convert(languages, title=True) == '.:en:Hello.:fr:Bonjour'


def test_convert_returns_non_dict_input_unchanged():
    assert convert('plain text') == 'plain text'
    assert convert(None) is None


def test_convert_strips_leading_whitespace_of_values():
    languages = collections.OrderedDict([('en', '  Hi ')])
    assert convert(languages) == '.:en\nHi'


@pytest.mark.parametrize('code', ['zh-cn', 'zh-tw'])
def test_detect_language_shortens_chinese_codes(monkeypatch, code):
    monkeypatch.setattr(langdetect, 'detect', lambda text: code)
    assert detect_language('some text') == 'cn'


def test_detect_language_returns_plain_short_code(monkeypatch):
    monkeypatch.setattr(langdetect, 'detect', lambda text: 'en')
    assert detect_language('Hello') == 'en'


def test_detect_language_truncates_to_max_length(monkeypatch):
    monkeypatch.setattr(langdetect, 'detect', lambda text: 'en')
    assert detect_language('Hello', max_length=1) == 'e'


def test_settings_defaults():
    assert settings.UNKNOWN_LANGUAGE == 'xx'
    assert settings.SEP == '.:'
    assert settings.ENDS == ['\n', '\r', '\r\n', ':']


def test_language_codes_include_english():
    assert 'en' in splitter.LANGUAGE_CODES
