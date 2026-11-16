import collections
import logging

import langdetect
import pytest

from langsplit import splitter
from langsplit.splitter import convert, detect_language, settings, split


@pytest.fixture
def fake_detect(monkeypatch):
    """Replace language detection with a recorder that always returns 'fr'."""
    calls = []

    def _fake(paragraph, max_length=2):
        calls.append(paragraph)
        return 'fr'

    monkeypatch.setattr(splitter, 'detect_language', _fake)
    return calls


def test_empty_string_is_returned_as_is():
    assert split('') == ''


def test_none_is_returned_as_is():
    assert split(None) is None


def test_two_explicit_languages_are_split():
    result = split('.:en:Hello.:fr:Bonjour')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_result_is_ordered_dict():
    result = split('.:en:Hello.:fr:Bonjour')
    assert isinstance(result, collections.OrderedDict)


def test_same_language_chunks_are_combined_in_first_appearance_order():
    result = split('.:fr:A.:en:B.:fr:C')
    assert list(result.items()) == [('fr', 'AC'), ('en', 'B')]


@pytest.mark.parametrize('end', ['\n', '\r', ':'])
def test_single_character_end_symbols_separate_key(end):
    result = split('.:en{}Hello'.format(end))
    assert list(result.items()) == [('en', 'Hello')]


def test_short_text_with_only_separator_and_key_gives_empty_value():
    result = split('.:en')
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [('en', '')]


def test_custom_separator_is_used():
    result = split('|en:Hello|fr:Bonjour', sep='|')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_custom_ends_are_used():
    result = split('.:en=Hello', ends=['='])
    assert list(result.items()) == [('en', 'Hello')]


def test_key_length_three_is_recognized_when_configured():
    result = split('.:eng:Hi', min_key_length=3, max_key_length=3)
    assert list(result.items()) == [('eng', 'Hi')]


def test_key_longer_than_max_length_is_not_a_key(fake_detect):
    result = split('.:eng:Hi')
    # 'eng' is not a valid 2-char key, so the whole token goes to detection.
    assert fake_detect == ['eng:Hi']
    assert list(result.items()) == [('fr', 'eng:Hi')]


def test_unmarked_text_is_autodetected(fake_detect):
    result = split('.:en:Hello.:Bonjour le monde')
    assert fake_detect == ['Bonjour le monde']
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour le monde')]


def test_paragraphs_are_detected_separately_and_rejoined(monkeypatch):
    def _fake(paragraph, max_length=2):
        return 'es' if 'Hola' in paragraph else 'fr'

    monkeypatch.setattr(splitter, 'detect_language', _fake)
    result = split('.:en:Hello.:Bonjour\n\nHola')
    assert list(result.items()) == [
        ('en', 'Hello'),
        ('fr', 'Bonjour\n\n'),
        ('es', 'Hola'),
    ]


def test_detection_failure_goes_to_unknown_key(monkeypatch, caplog):
    def _failing(paragraph, max_length=2):
        raise Exception('cannot detect')

    monkeypatch.setattr(splitter, 'detect_language', _failing)
    with caplog.at_level(logging.INFO):
        result = split('.:Hello there')
    assert list(result.items()) == [(settings.UNKNOWN_LANGUAGE, 'Hello there')]
    assert 'Language not detected' in caplog.text


def test_autodetect_false_does_not_call_detection(fake_detect):
    result = split('.:en:Hello.:plain text', autodetect=False)
    assert fake_detect == []
    assert settings.UNKNOWN_LANGUAGE in result
    assert result['en'] == 'Hello'


def test_empty_parts_between_separators_are_ignored():
    result = split('.:en:Hi.:.:fr:Yo')
    assert list(result.items()) == [('en', 'Hi'), ('fr', 'Yo')]


def test_markdown_output_uses_newline_end():
    result = split('.:en:Hello.:fr:Bonjour', markdown=True)
    assert result == '.:en\nHello.:fr\nBonjour'


def test_markdown_title_output_uses_colon_end():
    result = split('.:en:Hello.:fr:Bonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_convert_strips_leading_whitespace_of_values():
    assert convert({'en': '   Hi '}) == '.:en\nHi'


def test_convert_passes_non_dict_through():
    assert convert('plain text') == 'plain text'


def test_convert_empty_dict_gives_empty_string():
    assert convert({}) == ''


def test_detect_language_shortens_zh_cn(monkeypatch):
    monkeypatch.setattr(langdetect, 'detect', lambda text: 'zh-cn')
    assert detect_language('anything') == 'cn'


def test_detect_language_shortens_zh_tw(monkeypatch):
    monkeypatch.setattr(langdetect, 'detect', lambda text: 'zh-tw')
    assert detect_language('anything') == 'cn'


def test_detect_language_keeps_two_letter_code(monkeypatch):
    monkeypatch.setattr(langdetect, 'detect', lambda text: 'fr')
    assert detect_language('Bonjour tout le monde') == 'fr'


def test_detect_language_on_clear_english_text():
    langdetect.DetectorFactory.seed = 0
    text = (
        'This is a fairly long English sentence about the weather '
        'and how it changes across the seasons of the year.'
    )
    assert detect_language(text) == 'en'
