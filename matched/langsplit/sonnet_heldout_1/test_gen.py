import collections

import pytest

from langsplit import splitter
from langsplit.splitter import split


@pytest.fixture
def fake_detect(monkeypatch):
    """Replace language detection with a deterministic fake."""
    def _detect(text, max_length=2):
        if text.startswith('Hello'):
            return 'en'
        if text.startswith('Bonjour'):
            return 'fr'
        if text.startswith('Boom'):
            raise ValueError('cannot detect')
        return 'de'
    monkeypatch.setattr(splitter, 'detect_language', _detect)
    return _detect


def test_empty_string_returned_as_is():
    assert split('') == ''


def test_none_returned_as_is():
    assert split(None) is None


def test_basic_two_languages():
    result = split('.:en\nHello.:ru\nПривет')
    assert result == {'en': 'Hello', 'ru': 'Привет'}


def test_returns_ordered_dict_preserving_order():
    result = split('.:ru\nПривет.:en\nHello.:de\nHallo')
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ['ru', 'en', 'de']


def test_same_key_parts_are_combined():
    result = split('.:en\nHello .:ru\nПривет.:en\nWorld')
    assert list(result.keys()) == ['en', 'ru']
    assert result['en'] == 'Hello World'
    assert result['ru'] == 'Привет'


def test_short_text_without_end_symbol():
    result = split('.:en')
    assert isinstance(result, collections.OrderedDict)
    assert dict(result) == {'en': ''}


def test_key_with_only_end_char_gives_empty_value():
    result = split('.:en\n')
    assert dict(result) == {'en': ''}


def test_colon_as_end_symbol():
    result = split('.:en:Hello.:ru:Привет')
    assert result == {'en': 'Hello', 'ru': 'Привет'}


def test_custom_separator():
    result = split('#en\nHello#es\nHola', sep='#')
    assert result == {'en': 'Hello', 'es': 'Hola'}


def test_custom_key_length():
    result = split('.:eng\nHello.:rus\nPrivet', min_key_length=3, max_key_length=3)
    assert result == {'eng': 'Hello', 'rus': 'Privet'}


def test_markdown_output_roundtrip():
    text = '.:en\nHello.:ru\nПривет'
    assert split(text, markdown=True) == text


def test_markdown_title_uses_colon_end():
    result = split('.:en\nHello.:ru\nПривет', markdown=True, title=True)
    assert result == '.:en:Hello.:ru:Привет'


def test_autodetect_unmarked_paragraphs(fake_detect):
    result = split('Hello there\n\nBonjour tout le monde')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello there\n\n'
    assert result['fr'] == 'Bonjour tout le monde'


def test_autodetect_text_before_marked_section(fake_detect):
    result = split('Hello world.:ru\nПривет')
    assert list(result.keys()) == ['en', 'ru']
    assert result['en'] == 'Hello world'
    assert result['ru'] == 'Привет'


def test_autodetect_failure_goes_to_unknown_language(fake_detect):
    result = split('Boom something')
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == 'Boom something'


def test_autodetect_disabled_uses_unknown_language(fake_detect):
    result = split('Hello world', autodetect=False)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    # detection must not have been used for the key
    assert 'en' not in result


def test_custom_paragraph_separator(fake_detect):
    result = split('Hello one|Bonjour deux', pargraph_sep='|')
    assert result['en'] == 'Hello one|'
    assert result['fr'] == 'Bonjour deux'
