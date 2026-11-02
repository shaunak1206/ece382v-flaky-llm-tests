import collections

import pytest

from langsplit import splitter
from langsplit.splitter import split


def _fake_detect(paragraph):
    """Deterministic stand-in for language detection."""
    if 'bonjour' in paragraph.lower():
        return 'fr'
    return 'en'


@pytest.fixture
def fake_detect(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language', _fake_detect)


def test_empty_string_returned_unchanged():
    assert split('') == ''


def test_none_returned_unchanged():
    assert split(None) is None


def test_key_only_without_end_symbol_returns_single_entry():
    result = split('.:en')
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [('en', '')]


def test_short_key_only_text_uses_available_characters():
    result = split('.:e')
    assert list(result.items()) == [('e', '')]


def test_newline_end_splits_into_languages():
    result = split('.:en\nHello\n.:fr\nBonjour')
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [('en', 'Hello\n'), ('fr', 'Bonjour')]


def test_colon_end_single_line():
    result = split('.:en:Hello')
    assert list(result.items()) == [('en', 'Hello')]


def test_carriage_return_end():
    result = split('.:en\rHello')
    assert list(result.items()) == [('en', 'Hello')]


def test_same_key_parts_are_combined_in_first_seen_order():
    result = split('.:en\nHello .:fr\nBonjour .:en\nWorld')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello World'
    assert result['fr'] == 'Bonjour '


def test_empty_tokens_are_skipped():
    result = split('.:.:en\nHi')
    assert list(result.items()) == [('en', 'Hi')]


def test_custom_separator():
    result = split('##en\nHello##fr\nBonjour', sep='##')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_custom_key_length():
    result = split('.:eng\nHello', min_key_length=2, max_key_length=3)
    assert list(result.items()) == [('eng', 'Hello')]


def test_markdown_round_trip_with_newline_ends():
    text = '.:en\nHello\n.:fr\nBonjour'
    assert split(text, markdown=True) == text


def test_markdown_title_uses_colon_end():
    result = split('.:en\nHello\n.:fr\nBonjour', markdown=True, title=True)
    assert result == '.:en:Hello\n.:fr:Bonjour'


def test_autodetect_assigns_paragraphs_to_detected_languages(fake_detect):
    result = split('Hello there\n\nBonjour tout le monde')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello there\n\n'
    assert result['fr'] == 'Bonjour tout le monde'


def test_autodetect_same_language_paragraphs_keep_separator(fake_detect):
    result = split('Hello world\n\nSecond para')
    assert list(result.items()) == [('en', 'Hello world\n\nSecond para')]


def test_autodetect_custom_paragraph_separator(fake_detect):
    result = split('Hello world|Bonjour monde', pargraph_sep='|')
    assert result['en'] == 'Hello world|'
    assert result['fr'] == 'Bonjour monde'


def test_autodetect_text_before_first_separator_merges_with_keyed_part(fake_detect):
    result = split('hello .:en\nHi')
    assert list(result.items()) == [('en', 'hello Hi')]


def test_detection_failure_falls_back_to_unknown_language(monkeypatch):
    def boom(paragraph):
        raise ValueError('cannot detect')

    monkeypatch.setattr(splitter, 'detect_language', boom)
    result = split('Some text')
    assert list(result.items()) == [(splitter.settings.UNKNOWN_LANGUAGE, 'Some text')]


def test_real_detector_failure_on_digits_goes_to_unknown():
    result = split('12345 6789')
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == '12345 6789'


def test_autodetect_disabled_puts_text_under_unknown_key():
    result = split('plain text', autodetect=False)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    # The first len(name)+1 characters of the chunk are dropped in this branch.
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == 'in text'


def test_autodetect_disabled_still_parses_explicit_keys():
    result = split('.:en\nHello\n.:fr\nBonjour', autodetect=False)
    assert list(result.items()) == [('en', 'Hello\n'), ('fr', 'Bonjour')]
