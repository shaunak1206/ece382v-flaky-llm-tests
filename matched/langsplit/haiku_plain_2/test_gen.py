"""Unit tests for langsplit.splitter: split, convert and detect_language."""
import collections

import langdetect
import pytest

from langsplit import splitter
from langsplit.splitter import convert, detect_language, settings, split


def _recording_detector(answers, default=None):
    """Return (fake_detect_language, calls).

    A paragraph containing one of the `answers` keys gets that language code.
    Anything else gets `default`, or raises ValueError when no default is set.
    Every paragraph passed in is recorded in `calls`.
    """
    calls = []

    def fake(paragraph):
        calls.append(paragraph)
        for needle, lang in answers.items():
            if needle in paragraph:
                return lang
        if default is None:
            raise ValueError('undetectable: ' + paragraph)
        return default

    return fake, calls


def _od(*pairs):
    return collections.OrderedDict(pairs)


def test_split_falsy_input_is_returned_unchanged():
    assert split('') == ''
    assert split(None) is None


def test_split_known_keys_build_ordered_dict():
    result = split('.:en\nHello.:fr\nBonjour')
    assert isinstance(result, collections.OrderedDict)
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_split_colon_terminator_recognized():
    result = split('.:en:Hello.:fr:Bonjour')
    assert result == _od(('en', 'Hello'), ('fr', 'Bonjour'))


def test_split_merges_repeated_keys_in_first_seen_order():
    result = split('.:fr\nA.:en\nB.:fr\nC')
    assert list(result.items()) == [('fr', 'AC'), ('en', 'B')]


def test_split_key_only_short_text_returns_empty_value():
    # Early-return branch: a single-line '.:xx' fragment with no body.
    assert split('.:en') == _od(('en', ''))


def test_split_custom_separator():
    result = split('@@en\nHello@@fr\nBonjour', sep='@@')
    assert list(result.items()) == [('en', 'Hello'), ('fr', 'Bonjour')]


def test_split_custom_key_length_accepts_three_letter_keys():
    result = split('.:eng\nHello', min_key_length=3, max_key_length=3)
    assert result == _od(('eng', 'Hello'))


def test_split_single_letter_prefix_is_not_a_key(monkeypatch):
    fake, calls = _recording_detector({}, default='de')
    monkeypatch.setattr(splitter, 'detect_language', fake)
    result = split('.:e\nHi')
    # 'e' is shorter than min_key_length, so the whole token is auto-detected.
    assert result == _od(('de', 'e\nHi'))
    assert calls == ['e\nHi']


def test_split_known_keys_skip_detection(monkeypatch):
    fake, calls = _recording_detector({}, default='en')
    monkeypatch.setattr(splitter, 'detect_language', fake)
    split('.:en\nHello.:fr\nBonjour')
    assert calls == []


def test_split_unknown_leading_text_is_detected_and_ordered_first(monkeypatch):
    fake, _ = _recording_detector({}, default='en')
    monkeypatch.setattr(splitter, 'detect_language', fake)
    result = split('Hello there.:fr\nBonjour')
    assert list(result.items()) == [('en', 'Hello there'), ('fr', 'Bonjour')]


def test_split_paragraphs_detected_separately_and_separator_kept(monkeypatch):
    fake, calls = _recording_detector({'Hello': 'en', 'Bonjour': 'fr'})
    monkeypatch.setattr(splitter, 'detect_language', fake)
    result = split('Hello world\n\nBonjour le monde')
    assert calls == ['Hello world', 'Bonjour le monde']
    # The paragraph separator stays with the language it followed.
    assert result == _od(('en', 'Hello world\n\n'), ('fr', 'Bonjour le monde'))


def test_split_custom_paragraph_separator_is_reinserted(monkeypatch):
    fake, calls = _recording_detector({'Hello': 'en', 'Bonjour': 'fr'})
    monkeypatch.setattr(splitter, 'detect_language', fake)
    result = split('Hello|Bonjour', pargraph_sep='|')
    assert calls == ['Hello', 'Bonjour']
    assert result == _od(('en', 'Hello|'), ('fr', 'Bonjour'))


def test_split_failed_detection_goes_to_unknown_bucket(monkeypatch):
    fake, _ = _recording_detector({})  # every call raises
    monkeypatch.setattr(splitter, 'detect_language', fake)
    result = split('Garbage')
    assert result == _od((settings.UNKNOWN_LANGUAGE, 'Garbage'))


def test_split_failed_detection_paragraphs_keep_separator(monkeypatch):
    fake, _ = _recording_detector({})  # every call raises
    monkeypatch.setattr(splitter, 'detect_language', fake)
    result = split('A\n\nB')
    assert result == _od((settings.UNKNOWN_LANGUAGE, 'A\n\nB'))


def test_split_crlf_terminator_recognized():
    result = split('.:en\r\nHello.:fr\r\nBonjour')
    assert list(result.keys()) == ['en', 'fr']
    assert result['en'].strip() == 'Hello'
    assert result['fr'].strip() == 'Bonjour'


def test_split_markdown_round_trip():
    text = '.:en\nHello.:fr\nBonjour'
    as_dict = split(text)
    as_markdown = split(text, markdown=True)
    assert as_markdown == text
    assert split(as_markdown) == as_dict


def test_split_markdown_strips_leading_whitespace_of_values():
    assert split('.:en\n   Hello', markdown=True) == '.:en\nHello'


def test_split_markdown_title_mode_uses_colon_end():
    text = '.:en:Hello.:fr:Bonjour'
    assert split(text, markdown=True, title=True) == text


def test_convert_dict_uses_newline_end():
    assert convert({'en': 'Hello', 'fr': 'Bonjour'}) == '.:en\nHello.:fr\nBonjour'


def test_convert_title_uses_colon_end():
    assert convert({'en': 'Hello', 'fr': 'Bonjour'}, title=True) == '.:en:Hello.:fr:Bonjour'


def test_convert_lstrips_values_and_passes_through_non_dict():
    assert convert({'en': '   Hi'}) == '.:en\nHi'
    assert convert('plain text') == 'plain text'
    assert convert(None) is None


def test_detect_language_returns_two_letter_code(monkeypatch):
    monkeypatch.setattr(langdetect, 'detect', lambda text: 'en')
    assert detect_language('Hello') == 'en'


def test_detect_language_maps_chinese_variants_to_cn(monkeypatch):
    monkeypatch.setattr(langdetect, 'detect', lambda text: 'zh-cn')
    assert detect_language('你好') == 'cn'
    monkeypatch.setattr(langdetect, 'detect', lambda text: 'zh-tw')
    assert detect_language('你好') == 'cn'


def test_detect_language_propagates_detector_errors(monkeypatch):
    def boom(text):
        raise ValueError('no features in text')

    monkeypatch.setattr(langdetect, 'detect', boom)
    with pytest.raises(ValueError):
        detect_language('')
