import logging

import langdetect
import pytest

from langsplit import splitter


def _fail_detector(text, max_length=2):
    raise AssertionError('language detector should not be called')


def test_empty_text_is_returned_unchanged():
    assert splitter.split('') == ''
    assert splitter.split(None) is None


def test_single_keyed_block():
    result = splitter.split('.:en\nHello')
    assert result == {'en': 'Hello'}


def test_multiple_languages_keep_first_appearance_order():
    result = splitter.split('.:en\nHello.:fr\nBonjour')
    assert list(result.keys()) == ['en', 'fr']
    assert result == {'en': 'Hello', 'fr': 'Bonjour'}


def test_chunks_with_same_key_are_combined():
    text = '.:en\nHello.:fr\nBonjour.:en\nWorld'
    result = splitter.split(text)
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'HelloWorld'
    assert result['fr'] == 'Bonjour'


def test_colon_is_accepted_as_key_end():
    result = splitter.split('.:en:Hello.:fr:Bonjour')
    assert result == {'en': 'Hello', 'fr': 'Bonjour'}


def test_custom_separator():
    result = splitter.split('##en\nHello##fr\nBonjour', sep='##')
    assert result == {'en': 'Hello', 'fr': 'Bonjour'}


def test_custom_ends():
    result = splitter.split('.:en|Hello.:fr|Bonjour', ends=['|'])
    assert result == {'en': 'Hello', 'fr': 'Bonjour'}


def test_max_key_length_three_accepts_three_letter_keys():
    result = splitter.split('.:eng\nHello', max_key_length=3)
    assert result == {'eng': 'Hello'}


def test_short_key_without_content_returns_empty_value():
    result = splitter.split('.:en')
    assert isinstance(result, dict)
    assert result == {'en': ''}


def test_empty_tokens_are_skipped():
    result = splitter.split('.:.:en\nHi')
    assert result == {'en': 'Hi'}


def test_unkeyed_text_is_autodetected(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language',
                        lambda text, max_length=2: 'fr')
    result = splitter.split('Bonjour le monde')
    assert result == {'fr': 'Bonjour le monde'}


def test_autodetect_paragraphs_are_detected_separately(monkeypatch):
    def fake_detect(text, max_length=2):
        return 'en' if 'Hello' in text else 'fr'

    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    result = splitter.split('Hello\n\nBonjour')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello\n\n'
    assert result['fr'] == 'Bonjour'


def test_detection_failure_falls_back_to_unknown_language(monkeypatch, caplog):
    def fake_detect(text, max_length=2):
        if 'XXX' in text:
            raise langdetect.lang_detect_exception.LangDetectException(0, 'fail')
        return 'en'

    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    with caplog.at_level(logging.INFO):
        result = splitter.split('XXX unknown text')
    assert result == {splitter.settings.UNKNOWN_LANGUAGE: 'XXX unknown text'}
    assert 'Language not detected' in caplog.text


def test_keyed_blocks_do_not_call_detector(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language', _fail_detector)
    result = splitter.split('.:en\nHello.:fr\nBonjour', autodetect=False)
    assert result == {'en': 'Hello', 'fr': 'Bonjour'}


def test_markdown_output_round_trips_blocks():
    result = splitter.split('.:en\nHello.:fr\nBonjour', markdown=True)
    assert result == '.:en\nHello.:fr\nBonjour'


def test_markdown_title_output_uses_colon_end():
    result = splitter.split('.:en\nHello.:fr\nBonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_convert_dict_strips_leading_whitespace_of_values():
    result = splitter.convert({'en': '  Hello', 'fr': 'Bonjour'})
    assert result == '.:en\nHello.:fr\nBonjour'


def test_convert_dict_with_title_uses_colon_end():
    assert splitter.convert({'en': 'Hello'}, title=True) == '.:en:Hello'


def test_convert_non_dict_is_returned_unchanged():
    assert splitter.convert('plain text') == 'plain text'


@pytest.mark.parametrize('detected, expected', [
    ('zh-cn', 'cn'),
    ('zh-tw', 'cn'),
    ('fr', 'fr'),
])
def test_detect_language_shortens_codes(monkeypatch, detected, expected):
    monkeypatch.setattr(langdetect, 'detect', lambda text: detected)
    assert splitter.detect_language('some text') == expected


def test_split_returns_ordered_dict_instance():
    import collections
    result = splitter.split('.:en\nHello')
    assert isinstance(result, collections.OrderedDict)
