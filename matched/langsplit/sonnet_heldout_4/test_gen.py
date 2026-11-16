import collections

import pytest

from langsplit import splitter
from langsplit.splitter import split


@pytest.fixture
def fake_detect(monkeypatch):
    """Replace language detection with a deterministic fake."""
    def _detect(text, max_length=2):
        if 'Bonjour' in text:
            return 'fr'
        return 'en'
    monkeypatch.setattr(splitter, 'detect_language', _detect)
    return _detect


def test_empty_string_returned_as_is():
    assert split('') == ''


def test_none_returned_as_is():
    assert split(None) is None


def test_basic_split_two_languages():
    result = split('.:en\nHello.:fr\nBonjour')
    assert result == {'en': 'Hello', 'fr': 'Bonjour'}


def test_result_is_ordered_dict_with_order_preserved():
    result = split('.:fr\nBonjour.:en\nHello.:de\nHallo')
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ['fr', 'en', 'de']


def test_same_key_texts_are_combined():
    result = split('.:en\nHello.:fr\nBonjour.:en\nWorld')
    assert result['en'] == 'HelloWorld'
    assert result['fr'] == 'Bonjour'
    assert list(result.keys()) == ['en', 'fr']


def test_title_style_colon_end():
    result = split('.:en:Hello.:fr:Bonjour')
    assert result == {'en': 'Hello', 'fr': 'Bonjour'}


def test_short_text_without_end_symbol():
    result = split('.:en')
    assert isinstance(result, collections.OrderedDict)
    assert result == {'en': ''}


def test_custom_separator():
    result = split('#en\nHello#fr\nBonjour', sep='#')
    assert result == {'en': 'Hello', 'fr': 'Bonjour'}


def test_custom_max_key_length():
    result = split('.:eng\nHello.:fra\nBonjour', max_key_length=3)
    assert result == {'eng': 'Hello', 'fra': 'Bonjour'}


def test_markdown_output_roundtrip():
    text = '.:en\nHello.:fr\nBonjour'
    assert split(text, markdown=True) == text


def test_markdown_title_output_uses_colon():
    result = split('.:en\nHello.:fr\nBonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_autodetect_splits_paragraphs(fake_detect):
    result = split('Hello\n\nBonjour')
    assert result == {'en': 'Hello\n\n', 'fr': 'Bonjour'}
    assert list(result.keys()) == ['en', 'fr']


def test_autodetect_text_before_separator(fake_detect):
    result = split('Bonjour tout.:en\nHello')
    assert result == {'fr': 'Bonjour tout', 'en': 'Hello'}


def test_short_key_falls_back_to_detection(fake_detect):
    result = split('.:e\nHello')
    assert result == {'en': 'e\nHello'}


def test_detection_failure_assigns_unknown_language(monkeypatch):
    def boom(text, max_length=2):
        raise RuntimeError('cannot detect')
    monkeypatch.setattr(splitter, 'detect_language', boom)

    result = split('Hello world')
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == 'Hello world'


def test_autodetect_disabled_uses_unknown_key(monkeypatch):
    def fail(text, max_length=2):
        raise AssertionError('detect_language must not be called')
    monkeypatch.setattr(splitter, 'detect_language', fail)

    result = split('Hello world', autodetect=False)
    # Without a key, the first len(name)+1 characters are dropped.
    assert result == {'xx': 'lo world'}


def test_carriage_return_end_leaves_newline_in_chunk():
    result = split('.:en\r\nHello')
    assert result == {'en': '\nHello'}
