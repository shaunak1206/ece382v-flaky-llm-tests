import collections
import pytest
import logging

from langsplit import splitter


def test_detect_language_returns_short_code_for_default_length():
    # langdetect will detect 'en' from english text
    text = "This is a test."
    code = splitter.detect_language(text)
    assert isinstance(code, str)
    assert len(code) == 2


def test_detect_language_shorter_codes_mapping():
    # manually test shortcut mapping by patching langdetect.detect
    original_detect = splitter.langdetect.detect
    splitter.langdetect.detect = lambda text: 'zh-cn'
    code = splitter.detect_language("dummy text")
    assert code == 'cn'  # shortened, and truncated to 2 chars
    splitter.langdetect.detect = original_detect


def test_convert_with_dict_single_key_default_ends():
    d = {'en': ' Hello world'}
    result = splitter.convert(d, sep='.:', ends=['\n', '\r', '\r\n', ':'], title=False)
    assert result.startswith('en')
    assert 'Hello world' in result


def test_convert_with_dict_title_true_uses_last_end():
    d = {'en': 'text'}
    result = splitter.convert(d, title=True)
    # ends[-1] is ':', should end the language part with ':'
    assert result.startswith('en:')
    assert 'text' in result


def test_convert_with_non_dict_returns_input():
    input_text = "some string"
    assert splitter.convert(input_text) == input_text


def test_split_with_empty_text_returns_empty():
    assert splitter.split('') == ''


def test_split_with_only_sep_and_lang_key_short_text():
    # text with '.:en' but too short for end symbol, returns OrderedDict with key 'en' and empty text
    input_text = '.:entext'
    # The string length is > len(sep)+max_key_length+min_end (5+2+1=8) == 9, 8 < 9 triggers else branch
    # We need total length shorter to trigger this if branch:
    short_input = '.:en'
    result = splitter.split(short_input)
    assert isinstance(result, collections.OrderedDict)
    # key should be 'en'
    assert 'en' in result


def test_split_text_with_known_language_key_and_end():
    # use 'en' as lang key, followed by '\n' as end symbol from ends[0]
    text = f".:en\nHello world.:fr\nBonjour"
    result = splitter.split(text)
    assert 'en' in result
    assert 'fr' in result
    # Text values should contain expected strings, no leading space
    assert result['en'].startswith('Hello')
    assert result['fr'].startswith('Bonjour')


def test_split_text_with_unknown_language_autodetect_false():
    # text has no language prefix, autodetect=False puts all under 'xx'
    text = "This text has no language prefix."
    result = splitter.split(text, autodetect=False)
    assert isinstance(result, collections.OrderedDict)
    # all text should be under 'xx' key
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    combined = ''.join(result.values())
    assert 'This text has no language prefix.' in combined


def test_split_with_multiple_paragraphs_autodetect_true(monkeypatch):
    # Patch detect_language to return 'en' reliably to avoid dependency on langdetect
    def fake_detect_language(text, max_length=2):
        return 'en'
    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)
    text = ".:xx\nParagraph1\n\nParagraph2"
    result = splitter.split(text, autodetect=True, pargraph_sep='\n\n')
    assert 'en' in result
    # Chunks joined with paragraph separator
    assert result['en'] == "Paragraph1\n\nParagraph2"


def test_split_with_ends_variation():
    text = ".:en:Hello world.:fr\nBonjour"
    result = splitter.split(text)
    assert 'en' in result
    assert result['en'].startswith("Hello world")
    assert 'fr' in result


def test_split_markdown_true_returns_string():
    text = ".:en\nHello world.:fr\nBonjour"
    result = splitter.split(text, markdown=True)
    assert isinstance(result, str)
    # Contains language keys with separator and ends
    assert '.:en' in result or '.:fr' in result


def test_split_title_true_and_markdown_true():
    text = ".:en:Hello world.:fr:Bonjour"
    result = splitter.split(text, markdown=True, title=True)
    assert isinstance(result, str)
    # Should use ends[-1] which is ':' as end in convert
    assert result.startswith('en:')


def test_split_text_with_unknown_language_and_handling_exception(monkeypatch, caplog):
    # Patch detect_language to raise exception to test logging and fallback to UNKNOWN_LANGUAGE
    def raise_exception(text, max_length=2):
        raise RuntimeError("fail detect")
    monkeypatch.setattr(splitter, 'detect_language', raise_exception)
    text = "some text without language marker"
    caplog.set_level(logging.INFO)
    result = splitter.split(text)
    # Because autodetect default True, falls back to UNKNOWN_LANGUAGE key
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    # Log should capture 'Language not detected'
    assert any("Language not detected" in rec.message for rec in caplog.records)


def test_split_text_preserves_order():
    text = ".:fr\nBonjour.:en\nHello.:fr\nBonsoir"
    result = splitter.split(text)
    keys = list(result.keys())
    # Order should be fr, en (fr second time ignored)
    assert keys == ['fr', 'en']
    assert result['fr'].startswith("Bonjour")
    assert "Bonsoir" in result['fr']


def test_split_respects_min_and_max_key_length():
    # Provide keys of varying length and test min_key_length=2, max_key_length=3
    text = ".:eng\nHello.:fr\nBonjour.:e\nx"
    result = splitter.split(text, min_key_length=2, max_key_length=3)
    # 'eng' should be recognized, 'fr' recognized, 'e' too short ignored (unknown)
    assert 'eng' in result
    assert 'fr' in result
    # 'xx' should contain leftover 'x'
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert 'x' in result[splitter.settings.UNKNOWN_LANGUAGE]


def test_split_multiline_single_token_with_missing_end():
    # Trigger the early return for a single line, starting with sep, with length < sep+max_key+min(end)
    short_text = ".:en"
    result = splitter.split(short_text)
    assert isinstance(result, collections.OrderedDict)
    assert 'en' in result
    # The value should be empty string
    assert result['en'] == ''


def test_split_token_with_end_symbol_in_key_position():
    # text where symbol in ends appears inside first max_key_length+1 chars, position within key length
    text = ".:en:Hello.:fr\nBonjour"
    result = splitter.split(text)
    assert 'en' in result
    assert 'fr' in result
    assert result['en'].startswith("Hello")
    assert result['fr'].startswith("Bonjour")


def test_split_multiple_empty_tokens():
    text = ".:en\nHello..:fr\nBonjour"
    result = splitter.split(text)
    assert 'en' in result
    assert 'fr' in result
    assert result['en'].startswith('Hello')
    assert result['fr'].startswith('Bonjour')


def test_convert_removes_leading_ends():
    # Make sure convert removes leading ends[:2] pattern
    d = {'en': ':text'}
    # It will add sep+lang+end before text, text is ':text' starts with ':' which is in ends
    result = splitter.convert(d)
    # The result should not start with ':' twice
    assert not result.startswith('::') and not result.startswith('\n\n')


def test_split_with_pargraph_sep_and_autodetect():
    # Use text with 2 paragraphs separated by default '\n\n'
    text = ".:xx\nPara1\n\nPara2"
    result = splitter.split(text, autodetect=True)
    # Result should contain detected languages for paragraphs or xx key
    assert isinstance(result, collections.OrderedDict)
    combined_text = ''.join(result.values())
    assert 'Para1' in combined_text and 'Para2' in combined_text


def test_split_returns_original_if_not_string():
    # Passing dict to split should return input directly
    # Actually code expects string but test behavior with dict
    input_data = {'en': 'text'}
    result = splitter.split(input_data)  # It tries to split string, so should return as-is
    assert result == input_data
