import collections
import logging

import pytest

from langsplit import splitter


@pytest.fixture
def fake_detect(monkeypatch):
    """Replace language detection so tests do not depend on langdetect's model."""
    def _install(func):
        monkeypatch.setattr(splitter, 'detect_language', func)
        return func
    return _install


# ---------- convert ----------

def test_convert_non_dict_is_returned_unchanged():
    assert splitter.convert('plain text') == 'plain text'
    assert splitter.convert(None) is None


def test_convert_empty_dict_returns_empty_string():
    assert splitter.convert({}) == ''


def test_convert_default_uses_newline_as_end():
    result = splitter.convert({'en': 'Hello', 'fr': 'Bonjour'})
    assert result == '.:en\nHello.:fr\nBonjour'


def test_convert_title_uses_colon_as_end():
    result = splitter.convert({'en': 'Hello', 'fr': 'Bonjour'}, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_convert_strips_leading_whitespace_from_values():
    result = splitter.convert({'en': '   Hi'})
    assert result == '.:en\nHi'


# ---------- detect_language ----------

def test_detect_language_shortens_chinese_variants(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'zh-tw')
    assert splitter.detect_language('anything') == 'cn'

    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'zh-cn')
    assert splitter.detect_language('anything') == 'cn'


def test_detect_language_respects_max_length(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'zh-cn')
    # 'zh-cn' is longer than 5? No: it has length 5, so it is not shortened.
    assert splitter.detect_language('anything', max_length=5) == 'zh-cn'


def test_language_codes_contains_english():
    assert isinstance(splitter.LANGUAGE_CODES, list)
    assert 'en' in splitter.LANGUAGE_CODES


# ---------- split: empty and special cases ----------

def test_split_empty_text_is_returned_as_is():
    assert splitter.split('') == ''
    assert splitter.split(None) is None


def test_split_short_key_without_end_symbol():
    result = splitter.split('.:en')
    assert dict(result) == {'en': ''}


# ---------- split: explicit language keys ----------

def test_split_explicit_keys(fake_detect):
    result = splitter.split('.:en\nHello.:fr\nBonjour')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_split_merges_repeated_keys_keeping_first_seen_order(fake_detect):
    result = splitter.split('.:en\nA.:fr\nB.:en\nC')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'AC'
    assert result['fr'] == 'B'


def test_split_key_longer_than_max_length_is_not_a_key(fake_detect):
    fake_detect(lambda text, max_length=2: 'en')
    # 'eng' is three characters, so it is not treated as a key with max_key_length=2.
    result = splitter.split('.:eng\nHi')
    assert list(result.keys()) == ['en']
    assert result['en'] == 'eng\nHi'


def test_split_custom_separator(fake_detect):
    result = splitter.split('##en\nHi##fr\nYo', sep='##')
    assert list(result.items()) == [('en', 'Hi'), ('fr', 'Yo')]


def test_split_custom_ends(fake_detect):
    result = splitter.split('.:en:Hello', ends=[':'])
    assert dict(result) == {'en': 'Hello'}


def test_split_custom_key_lengths(fake_detect):
    result = splitter.split('.:eng\nHi', min_key_length=3, max_key_length=3)
    assert dict(result) == {'eng': 'Hi'}


# ---------- split: autodetection ----------

def test_split_autodetects_plain_text(fake_detect):
    fake_detect(lambda text, max_length=2: 'en')
    result = splitter.split('Hello world')
    assert dict(result) == {'en': 'Hello world'}


def test_split_detects_each_paragraph_separately(fake_detect):
    fake_detect(lambda text, max_length=2: 'en' if 'one' in text else 'fr')
    result = splitter.split('Para one\n\nPara two')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'] == 'Para one\n\n'
    assert result['fr'] == 'Para two'


def test_split_detection_failure_goes_to_unknown_language(fake_detect, caplog):
    def failing(text, max_length=2):
        raise ValueError('no features in text')

    fake_detect(failing)
    with caplog.at_level(logging.INFO):
        result = splitter.split('Mystery')

    assert dict(result) == {splitter.settings.UNKNOWN_LANGUAGE: 'Mystery'}
    assert 'Language not detected: Mystery' in caplog.text


def test_split_with_autodetect_disabled_uses_unknown_key():
    result = splitter.split('Plain text', autodetect=False)
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]


# ---------- split: markdown output ----------

def test_split_markdown_returns_string_round_trip(fake_detect):
    result = splitter.split('.:en\nHello.:fr\nBonjour', markdown=True)
    assert result == '.:en\nHello.:fr\nBonjour'


def test_split_markdown_title_round_trip(fake_detect):
    result = splitter.split('.:en:Hello.:fr:Bonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'
