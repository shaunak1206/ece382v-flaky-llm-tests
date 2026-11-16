import collections

import pytest

from langsplit import splitter as splitter_module
from langsplit.splitter import split


@pytest.fixture
def fake_detect(monkeypatch):
    """Replace language detection with a deterministic fake.

    Paragraphs containing 'bad' raise an error, paragraphs containing
    'Hello' are detected as 'en', and everything else as 'de'.
    """
    def _fake(text, max_length=2):
        if 'bad' in text:
            raise ValueError('cannot detect')
        if 'Hello' in text:
            return 'en'
        return 'de'

    monkeypatch.setattr(splitter_module, 'detect_language', _fake)
    return _fake


def test_empty_string_is_returned_unchanged():
    assert split('') == ''


def test_none_is_returned_unchanged():
    assert split(None) is None


def test_simple_split_into_two_languages():
    result = split('.:en:Hello.:fr:Bonjour')
    assert result == {'en': 'Hello', 'fr': 'Bonjour'}


def test_result_is_ordered_dict():
    result = split('.:en:Hello.:fr:Bonjour')
    assert isinstance(result, collections.OrderedDict)


def test_newline_end_separator_is_recognized():
    result = split('.:en\nHello.:fr\nBonjour')
    assert result == {'en': 'Hello', 'fr': 'Bonjour'}


def test_repeated_key_contents_are_merged():
    result = split('.:en:Hi.:fr:Salut.:en:Bye')
    assert result == {'en': 'HiBye', 'fr': 'Salut'}


def test_keys_ordered_by_first_appearance():
    result = split('.:fr:A.:en:B.:fr:C')
    assert list(result.keys()) == ['fr', 'en']
    assert result['fr'] == 'AC'


def test_custom_sep_is_used():
    result = split('##en:Hi##fr:Salut', sep='##')
    assert result == {'en': 'Hi', 'fr': 'Salut'}


def test_custom_ends_are_used():
    result = split('.:en;Hello.:fr;Bonjour', ends=[';'])
    assert result == {'en': 'Hello', 'fr': 'Bonjour'}


def test_key_length_range_allows_three_char_keys():
    result = split('.:eng:Hi', min_key_length=2, max_key_length=3)
    assert result == {'eng': 'Hi'}


def test_long_key_is_not_treated_as_language(fake_detect):
    # 'eng' is longer than max_key_length=2, so the token is not a keyed chunk.
    result = split('.:eng:Hi')
    assert result == {'de': 'eng:Hi'}


def test_short_prefix_without_end_symbol_is_special_cased():
    result = split('.:en')
    assert isinstance(result, collections.OrderedDict)
    assert dict(result) == {'en': ''}


def test_empty_value_after_key_is_kept():
    result = split('.:en:')
    assert result == {'en': ''}


def test_keyed_text_without_autodetect():
    result = split('.:en:Hi.:fr:Salut', autodetect=False)
    assert result == {'en': 'Hi', 'fr': 'Salut'}


def test_autodetect_splits_unkeyed_text_by_paragraph(fake_detect):
    result = split('Hallo Welt\n\nHello there')
    assert result == {'de': 'Hallo Welt\n\n', 'en': 'Hello there'}


def test_empty_paragraphs_are_skipped(fake_detect):
    result = split('Hallo\n\n\n\nBye')
    assert result == {'de': 'Hallo\n\nBye'}


def test_detection_failure_goes_to_unknown_language(fake_detect):
    result = split('Hallo\n\nbad')
    assert result == {'de': 'Hallo\n\n', 'xx': 'bad'}


def test_markdown_output_uses_newline_end_by_default():
    result = split('.:en:Hello.:fr:Bonjour', markdown=True)
    assert result == '.:en\nHello.:fr\nBonjour'


def test_markdown_output_with_title_uses_colon_end():
    result = split('.:en\nHello.:fr\nBonjour', markdown=True, title=True)
    assert result == '.:en:Hello.:fr:Bonjour'


def test_markdown_round_trip_returns_same_dict():
    source = '.:en:Hello.:fr:Bonjour.:en:Again'
    markdown = split(source, markdown=True)
    assert split(markdown) == split(source)


def test_empty_text_with_markdown_returns_empty_string():
    assert split('', markdown=True) == ''
