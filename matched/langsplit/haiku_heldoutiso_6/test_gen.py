import collections

import langdetect
import pytest

from langsplit import splitter


def make_fake_detect(mapping):
    """Return a stand-in for detect_language: the first needle found in the
    text picks the language code; any text with no match raises."""
    def fake(text, max_length=2):
        for needle, code in mapping.items():
            if needle in text:
                return code[:max_length]
        raise ValueError('no language detected')
    return fake


def test_empty_text_is_returned_unchanged():
    assert splitter.split('') == ''


def test_tagged_sections_with_newline_end():
    result = splitter.split('.:en\nHello\n.:fr\nBonjour')
    assert list(result.items()) == [('en', 'Hello\n'), ('fr', 'Bonjour')]


def test_tagged_section_with_colon_end():
    result = splitter.split('.:en:Hello')
    assert dict(result) == {'en': 'Hello'}


def test_same_keys_are_combined():
    text = '.:en\nA\n.:fr\nB\n.:en\nC'
    result = splitter.split(text)
    assert list(result.items()) == [('en', 'A\nC'), ('fr', 'B\n')]


def test_key_order_follows_first_appearance():
    text = '.:fr\nA\n.:en\nB\n.:fr\nC'
    result = splitter.split(text)
    assert list(result.keys()) == ['fr', 'en']
    assert result['fr'] == 'A\nC'
    assert result['en'] == 'B\n'


def test_custom_separator():
    result = splitter.split('##en\nHi\n##fr\nYo', sep='##')
    assert list(result.items()) == [('en', 'Hi\n'), ('fr', 'Yo')]


def test_empty_tokens_are_skipped():
    result = splitter.split('.:.:en\nHi')
    assert dict(result) == {'en': 'Hi'}


def test_short_key_only_text_returns_empty_value():
    result = splitter.split('.:en')
    assert result == collections.OrderedDict([('en', '')])


def test_longer_keys_with_max_key_length_three():
    result = splitter.split('.:eng\nHi', max_key_length=3)
    assert dict(result) == {'eng': 'Hi'}


def test_untagged_text_is_autodetected(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language',
                        make_fake_detect({'Hello': 'en'}))
    result = splitter.split('Hello world')
    assert dict(result) == {'en': 'Hello world'}


def test_paragraphs_are_detected_separately(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language',
                        make_fake_detect({'Hello': 'en', 'Bonjour': 'fr'}))
    result = splitter.split('Hello\n\nBonjour')
    assert list(result.items()) == [('en', 'Hello\n\n'), ('fr', 'Bonjour')]


def test_detection_failure_goes_to_unknown_language(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language',
                        make_fake_detect({}))
    result = splitter.split('Hello')
    assert dict(result) == {splitter.settings.UNKNOWN_LANGUAGE: 'Hello'}


def test_untagged_prefix_mixed_with_tagged_section(monkeypatch):
    monkeypatch.setattr(splitter, 'detect_language',
                        make_fake_detect({'Plain': 'en'}))
    result = splitter.split('Plain\n.:fr\nBonjour')
    assert list(result.items()) == [('en', 'Plain\n'), ('fr', 'Bonjour')]


def test_markdown_output_round_trips():
    text = '.:en\nHello\n.:fr\nBonjour'
    assert splitter.split(text, markdown=True) == text


def test_markdown_title_uses_colon_end():
    text = '.:en\nHello\n.:fr\nBonjour'
    assert splitter.split(text, markdown=True, title=True) == \
        '.:en:Hello\n.:fr:Bonjour'


def test_convert_returns_non_dict_input_unchanged():
    assert splitter.convert('abc') == 'abc'


def test_convert_strips_leading_whitespace_of_values():
    assert splitter.convert({'en': '   Hi'}) == '.:en\nHi'


def test_convert_with_custom_separator():
    assert splitter.convert({'en': 'a'}, sep='##') == '##en\na'


def test_detect_language_shortens_zh_codes(monkeypatch):
    monkeypatch.setattr(langdetect, 'detect', lambda text: 'zh-cn')
    assert splitter.detect_language('anything') == 'cn'

    monkeypatch.setattr(langdetect, 'detect', lambda text: 'zh-tw')
    assert splitter.detect_language('anything') == 'cn'


def test_detect_language_keeps_two_letter_code(monkeypatch):
    monkeypatch.setattr(langdetect, 'detect', lambda text: 'en')
    assert splitter.detect_language('anything') == 'en'
