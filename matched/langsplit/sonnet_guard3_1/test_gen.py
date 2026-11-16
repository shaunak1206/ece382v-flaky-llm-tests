import collections

import pytest

from langsplit import splitter


def _fake_detect(paragraph, max_length=2):
    return 'en' if 'Hello' in paragraph else 'fr'


@pytest.fixture
def fake_detect(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language', _fake_detect)


def test_empty_string_returned_as_is():
    assert splitter.split('') == ''


def test_none_returned_as_is():
    assert splitter.split(None) is None


def test_short_key_without_end():
    result = splitter.split('.:en')
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [('en', '')]


def test_single_keyed_section():
    result = splitter.split('.:en\nHello')
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [('en', 'Hello')]


def test_multiple_keyed_sections_preserve_order():
    result = splitter.split('.:en\nHello.:fr\nBonjour')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_same_keys_are_combined_in_first_seen_order():
    result = splitter.split('.:fr\nA.:en\nB.:fr\nC')
    assert list(result.keys()) == ['fr', 'en']
    assert result['fr'] == 'AC'
    assert result['en'] == 'B'


def test_colon_as_end_symbol():
    result = splitter.split('.:en:Hello.:fr:Bonjour')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_custom_separator():
    result = splitter.split('|en\nHello|fr\nBonjour', sep='|')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_longer_key_with_max_key_length():
    result = splitter.split('.:eng\nHello', min_key_length=2, max_key_length=3)
    assert list(result.items()) == [('eng', 'Hello')]


def test_autodetect_not_called_when_keys_present(monkeypatch):
    calls = []

    def recording_detect(paragraph, max_length=2):
        calls.append(paragraph)
        return 'en'

    monkeypatch.setattr(splitter, 'detect_language', recording_detect)
    result = splitter.split('.:en\nHello.:fr\nBonjour')
    assert calls == []
    assert list(result.keys()) == ['en', 'fr']


def test_autodetect_splits_paragraphs(fake_detect):
    result = splitter.split('Hello there\n\nBonjour')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello there\n\n'
    assert result['fr'] == 'Bonjour'


def test_autodetect_skips_empty_paragraphs(fake_detect):
    result = splitter.split('Hello\n\n\n\nWorld')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Hello\n\n'
    assert result['fr'] == 'World'


def test_autodetect_for_text_before_first_separator(fake_detect):
    result = splitter.split('Intro text.:en\nHello')
    assert list(result.keys()) == ['fr', 'en']
    assert result['fr'] == 'Intro text'
    assert result['en'] == 'Hello'


def test_detection_failure_falls_back_to_unknown_language(monkeypatch):
    def failing_detect(paragraph, max_length=2):
        raise ValueError('no features in text')

    monkeypatch.setattr(splitter, 'detect_language', failing_detect)
    result = splitter.split('Some text here')
    assert list(result.items()) == [(splitter.settings.UNKNOWN_LANGUAGE, 'Some text here')]


def test_detection_failure_keeps_paragraph_separator(monkeypatch):
    def failing_detect(paragraph, max_length=2):
        raise ValueError('no features in text')

    monkeypatch.setattr(splitter, 'detect_language', failing_detect)
    result = splitter.split('One\n\nTwo')
    assert list(result.items()) == [('xx', 'One\n\nTwo')]


def test_overlong_key_goes_to_autodetect(fake_detect):
    result = splitter.split('Hello\nworld.:en\nHi')
    assert list(result.keys()) == ['en']
    assert result['en'] == 'Hello\nworldHi'


def test_autodetect_disabled_puts_text_under_unknown_key(monkeypatch):
    calls = []

    def recording_detect(paragraph, max_length=2):
        calls.append(paragraph)
        return 'en'

    monkeypatch.setattr(splitter, 'detect_language', recording_detect)
    result = splitter.split('Hello world', autodetect=False)
    assert calls == []
    assert list(result.keys()) == ['xx']


def test_autodetect_disabled_keeps_keyed_text():
    result = splitter.split('.:en\nHello', autodetect=False)
    assert list(result.items()) == [('en', 'Hello')]


def test_markdown_output():
    result = splitter.split('.:en\nHello.:fr\nBonjour', markdown=True)
    assert result == '.:en\nHello.:fr\nBonjour'


def test_markdown_title_output_uses_colon_end():
    result = splitter.split('.:en\nHello.:fr\nBonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_short_key_case_ignores_markdown_flag():
    result = splitter.split('.:en', markdown=True)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [('en', '')]
