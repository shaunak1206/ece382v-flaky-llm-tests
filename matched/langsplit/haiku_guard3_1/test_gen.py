import collections
import random

import pytest

from langsplit import splitter as splitter_module
from langsplit.splitter import split, settings


def _fake_detector(default_code='de', failing=None, mapping=None):
    """Build a deterministic stand-in for detect_language.

    Paragraphs listed in `failing` raise; paragraphs containing a key of
    `mapping` return the mapped code; anything else returns `default_code`.
    """
    failing = set(failing or [])
    mapping = dict(mapping or {})

    def fake(paragraph, max_length=2):
        if paragraph in failing:
            raise ValueError('cannot detect')
        for needle, code in mapping.items():
            if needle in paragraph:
                return code
        return default_code

    return fake


def test_empty_text_is_returned_unchanged():
    assert split('') == ''
    assert split(None) is None


def test_key_only_text_returns_empty_value_for_that_key():
    result = split('.:en')
    assert isinstance(result, collections.OrderedDict)
    assert result == {'en': ''}


def test_splits_text_by_language_keys_on_newline():
    result = split('.:en\nHello.:fr\nBonjour')
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello'
    assert result['fr'] == 'Bonjour'


def test_repeated_keys_are_combined_and_keep_first_seen_order():
    result = split('.:fr\nA.:en\nB.:fr\nC')
    assert list(result.keys()) == ['fr', 'en']
    assert result['fr'] == 'AC'
    assert result['en'] == 'B'


def test_key_terminated_by_colon():
    result = split('.:en:Hi.:fr:Salut')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hi'
    assert result['fr'] == 'Salut'


def test_custom_separator_is_used():
    result = split('|en\nHi|fr\nYo', sep='|')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hi'
    assert result['fr'] == 'Yo'


def test_custom_ends_are_used_for_key_termination():
    result = split('.:en;Hi', ends=[';'])
    assert list(result.keys()) == ['en']
    assert result['en'] == 'Hi'


def test_custom_key_length_accepts_three_letter_keys():
    result = split('.:eng\nHi', min_key_length=3, max_key_length=3)
    assert list(result.keys()) == ['eng']
    assert result['eng'] == 'Hi'


def test_key_shorter_than_min_length_is_not_treated_as_key(monkeypatch):
    monkeypatch.setattr(splitter_module, 'detect_language', _fake_detector('de'))
    result = split('.:e\nHi')
    assert list(result.keys()) == ['de']
    assert result['de'] == 'e\nHi'


def test_empty_tokens_from_repeated_separators_are_skipped():
    result = split('.:.:en\nHi')
    assert list(result.keys()) == ['en']
    assert result['en'] == 'Hi'


def test_unmarked_text_is_assigned_by_detector(monkeypatch):
    monkeypatch.setattr(splitter_module, 'detect_language', _fake_detector('de'))
    result = split('Hallo Welt')
    assert list(result.keys()) == ['de']
    assert result['de'] == 'Hallo Welt'


def test_paragraphs_are_detected_separately_and_separator_follows_previous(monkeypatch):
    fake = _fake_detector(default_code='en', mapping={'Hallo': 'de'})
    monkeypatch.setattr(splitter_module, 'detect_language', fake)
    result = split('Hallo\n\nHello')
    assert list(result.keys()) == ['de', 'en']
    assert result['de'] == 'Hallo\n\n'
    assert result['en'] == 'Hello'


def test_empty_paragraphs_are_skipped(monkeypatch):
    fake = _fake_detector(default_code='en', mapping={'Hallo': 'de'})
    monkeypatch.setattr(splitter_module, 'detect_language', fake)
    result = split('Hallo\n\n\n\nHello')
    assert list(result.keys()) == ['de', 'en']
    assert result['de'] == 'Hallo\n\n'
    assert result['en'] == 'Hello'


def test_detection_failure_falls_back_to_unknown_language(monkeypatch):
    fake = _fake_detector(default_code='en', failing=['bad'])
    monkeypatch.setattr(splitter_module, 'detect_language', fake)
    result = split('bad\n\nHello')
    assert list(result.keys()) == [settings.UNKNOWN_LANGUAGE, 'en']
    assert result[settings.UNKNOWN_LANGUAGE] == 'bad\n\n'
    assert result['en'] == 'Hello'


def test_autodetect_false_stores_unmarked_text_under_unknown_language():
    result = split('Hello world', autodetect=False)
    assert list(result.keys()) == [settings.UNKNOWN_LANGUAGE]


def test_markdown_output_round_trips_input():
    text = '.:en\nHello.:fr\nBonjour'
    assert split(text, markdown=True) == text


def test_markdown_title_uses_colon_as_end():
    result = split('.:en\nHello.:fr\nBonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_real_langdetect_assigns_english_to_english_text():
    # langdetect uses the random module internally, so seed it here.
    random.seed(0)
    text = 'This is a simple English sentence written for testing purposes.'
    result = split(text)
    assert list(result.keys()) == ['en']
    assert result['en'] == text
