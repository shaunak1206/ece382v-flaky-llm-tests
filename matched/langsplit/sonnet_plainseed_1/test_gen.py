import collections

import pytest

from langsplit import splitter
from langsplit.splitter import split, convert


def _fake_detect(paragraph):
    """Deterministic stand-in for the (random) language detector."""
    if paragraph.startswith('Hello'):
        return 'en'
    if paragraph.startswith('Bonjour'):
        return 'fr'
    return 'de'


@pytest.fixture
def fake_detect(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language', _fake_detect)


def test_split_empty_string_returns_empty_string():
    assert split('') == ''


def test_split_none_returns_none():
    assert split(None) is None


def test_split_two_labelled_languages_newline_end():
    result = split('.:en\nHello.:fr\nBonjour')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_split_returns_ordered_dict():
    result = split('.:en\nHello.:fr\nBonjour')
    assert isinstance(result, collections.OrderedDict)


def test_split_colon_as_end_symbol():
    result = split('.:en:Hello.:fr:Bonjour')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_split_carriage_return_end_symbol():
    result = split('.:en\rHello.:fr\rBonjour')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_split_combines_repeated_keys_in_first_seen_order():
    result = split('.:en\nHello .:fr\nBonjour.:en\nWorld')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello World'
    assert result['fr'] == 'Bonjour'


def test_split_only_prefix_without_end_symbol_returns_key_only():
    result = split('.:en')
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [('en', '')]


def test_split_prefix_with_trailing_newline_gives_empty_value():
    result = split('.:en\n')
    assert list(result.items()) == [('en', '')]


def test_split_ignores_empty_tokens_between_separators():
    result = split('.:en\nHello.:.:fr\nBonjour')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_split_custom_separator():
    result = split('#en\nHello#fr\nBonjour', sep='#')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_split_longer_key_with_max_key_length():
    result = split('.:eng\nHello.:fra\nBonjour', max_key_length=3, min_key_length=3)
    assert list(result.items()) == [('eng', 'Hello'), ('fra', 'Bonjour')]


def test_split_unlabelled_text_without_autodetect_goes_to_unknown_key():
    result = split('Hello world', autodetect=False)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]


def test_split_markdown_roundtrip():
    original = '.:en\nHello.:fr\nBonjour'
    assert split(original, markdown=True) == original


def test_split_markdown_title_uses_colon_end():
    result = split('.:en\nHello.:fr\nBonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_split_autodetect_splits_paragraphs_by_language(fake_detect):
    result = split('Hello world\n\nBonjour le monde')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello world\n\n'
    assert result['fr'] == 'Bonjour le monde'


def test_split_autodetect_custom_paragraph_separator(fake_detect):
    result = split('Hello|Bonjour', pargraph_sep='|')
    assert list(result.items()) == [('en', 'Hello|'), ('fr', 'Bonjour')]


def test_split_autodetect_unlabelled_prefix_before_labelled_part(fake_detect):
    result = split('Intro text .:en\nHello')
    assert list(result.items()) == [('de', 'Intro text '), ('en', 'Hello')]


def test_split_autodetect_skips_empty_paragraphs(fake_detect):
    result = split('Hello one\n\n\n\nHello two')
    assert list(result.keys()) == ['en']
    assert result['en'] == 'Hello one\n\nHello two'


def test_split_autodetect_failure_falls_back_to_unknown(monkeypatch):
    def boom(paragraph):
        raise ValueError('no features')

    monkeypatch.setattr(splitter, 'detect_language', boom)
    result = split('Hello world')
    assert list(result.items()) == [(splitter.settings.UNKNOWN_LANGUAGE, 'Hello world')]


def test_convert_non_dict_is_returned_unchanged():
    assert convert('plain text') == 'plain text'
    assert convert(None) is None
