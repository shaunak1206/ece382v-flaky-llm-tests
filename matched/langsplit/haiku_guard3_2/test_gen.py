import pytest

from langsplit import splitter
from langsplit.splitter import convert, detect_language, split


def test_split_empty_text_is_returned_unchanged():
    assert split('') == ''
    assert split(None) is None


def test_split_two_known_languages():
    result = split('.:en\nHello.:fr\nBonjour')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_split_repeated_language_concatenates_in_first_seen_order():
    result = split('.:fr\nA.:en\nB.:fr\nC')
    assert list(result.keys()) == ['fr', 'en']
    assert result['fr'] == 'AC'
    assert result['en'] == 'B'


def test_split_accepts_colon_as_end_symbol():
    result = split('.:en:Hello.:fr:Bonjour')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_split_short_text_with_key_only_takes_early_return_path():
    result = split('.:en')
    assert list(result.items()) == [('en', '')]


def test_split_with_custom_separator():
    result = split('##en\nHi##fr\nYo', sep='##')
    assert list(result.items()) == [('en', 'Hi'), ('fr', 'Yo')]


def test_split_with_custom_key_length():
    result = split('.:eng\nHi.:fra\nSalut', min_key_length=3, max_key_length=3)
    assert list(result.items()) == [('eng', 'Hi'), ('fra', 'Salut')]


def test_split_key_length_mismatch_is_not_treated_as_key():
    # The key 'en' has length 2, but min/max are 3, so no language key is found.
    result = split('.:en\nHi', min_key_length=3, max_key_length=3, autodetect=False)
    assert list(result.keys()) == ['xx']


def test_split_unknown_text_without_autodetect_uses_unknown_key():
    result = split('.:Hello', autodetect=False)
    assert list(result.keys()) == ['xx']


def test_split_autodetect_uses_detected_language(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language', lambda text: 'en')
    result = split('.:Hello there')
    assert list(result.items()) == [('en', 'Hello there')]


def test_split_detection_failure_falls_back_to_unknown_key(monkeypatch):
    def failing_detect(text):
        raise ValueError('no features in text')

    monkeypatch.setattr(splitter, 'detect_language', failing_detect)
    result = split('.:Hello')
    assert list(result.items()) == [('xx', 'Hello')]


def test_split_detects_each_paragraph_separately(monkeypatch):
    def fake_detect(text):
        return 'en' if text == 'Hello' else 'fr'

    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    result = split('.:Hello\n\nBonjour')
    assert list(result.items()) == [('en', 'Hello\n\n'), ('fr', 'Bonjour')]


def test_split_unknown_pieces_merge_into_detected_language(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language', lambda text: 'en')
    result = split('.:en\nHi.:Hello')
    assert list(result.items()) == [('en', 'HiHello')]


def test_split_markdown_round_trip():
    text = '.:en\nHello.:fr\nBonjour'
    assert split(text, markdown=True) == text


def test_split_markdown_title_uses_colon_end():
    result = split('.:en\nHello.:fr\nBonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_convert_plain_dict_produces_markdown():
    assert convert({'en': 'Hello', 'fr': 'Bonjour'}) == '.:en\nHello.:fr\nBonjour'


def test_convert_title_uses_last_end_symbol():
    assert convert({'en': 'Hello'}, title=True) == '.:en:Hello'


def test_convert_strips_leading_end_markers():
    # The leading '\r\n' matches a two-character end and is removed.
    assert convert({'a': 'x'}, sep='\r\n', ends=['\r\n']) == 'a\r\nx'


def test_convert_left_strips_values_and_strips_result():
    assert convert({'en': '  Hi  '}) == '.:en\nHi'


def test_convert_non_dict_is_returned_as_is():
    assert convert('plain') == 'plain'
    assert convert(None) is None


def test_detect_language_shortens_regional_codes(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'zh-tw')
    assert detect_language('some text') == 'cn'

    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'en')
    assert detect_language('some text') == 'en'


def test_detect_language_respects_max_length(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'zh-cn')
    assert detect_language('some text', max_length=5) == 'zh-cn'

    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'en')
    assert detect_language('some text', max_length=1) == 'e'
