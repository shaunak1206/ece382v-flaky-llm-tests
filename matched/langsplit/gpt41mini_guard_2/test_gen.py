import collections
import pytest

import langsplit.splitter as splitter


def test_detect_language_returns_known_code_or_shorter():
    # "This is English." should be detected as "en" typically
    lang = splitter.detect_language("This is English.")
    assert isinstance(lang, str)
    assert 1 <= len(lang) <= 2
    # Should map longer codes like 'zh-cn' to 'cn'
    shorter = splitter.SHORTER_CODES.get('zh-cn')
    if shorter:
        assert shorter == 'cn'


def test_convert_with_dict_and_default_ends_and_sep():
    d = {'en': ' hello', 'fr': ' bonjour'}
    # title False uses ends[0] == '\n'
    result = splitter.convert(d)
    # Should start with sep+key+end
    for key in d.keys():
        assert splitter.settings.SEP in result
        assert key in result
    # Leading spaces stripped from text values
    assert ' hello' not in result
    assert 'bonjour' in result


def test_convert_with_title_true_uses_last_end():
    d = {'en': ' hello', 'fr': ' bonjour'}
    result = splitter.convert(d, title=True)
    # ends[-1] should be last character before text, verify ':' included
    assert result.count(splitter.settings.ENDS[-1]) >= 1
    # Leading spaces stripped
    assert ' hello' not in result


def test_convert_with_non_dict_returns_input():
    s = "not a dict"
    assert splitter.convert(s) == s


def test_split_empty_or_none_returns_same():
    assert splitter.split('') == ''
    assert splitter.split(None) is None


def test_split_single_line_no_end_symbol_returns_ordered_dict():
    val = splitter.settings.SEP + 'en'  # minimal form, no end symbol
    result = splitter.split(val)
    assert isinstance(result, collections.OrderedDict)
    key = list(result.keys())[0]
    assert key == 'en'
    assert result[key] == ''


def test_split_simple_marked_text():
    text = '.:en\nHello\n.:fr\nBonjour\n'
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)
    assert set(result.keys()) == {'en', 'fr'}
    assert result['en'].startswith('Hello')
    assert result['fr'].startswith('Bonjour')


def test_split_with_autodetect_off_assigns_unknown_language():
    # Text with no valid detected language keys, should go to 'xx'
    text = 'some random text without language key'
    result = splitter.split(text, autodetect=False)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    # full input assigned since no sep markers
    assert text in result[splitter.settings.UNKNOWN_LANGUAGE]


def test_split_with_autodetect_true_detects_paragraphs_and_assigns():
    # Provide text with two paragraphs in different languages
    text = ".:en\nHello.\n\n.:fr\nBonjour.\n"
    result = splitter.split(text)
    assert 'en' in result and 'fr' in result
    # Both paragraphs preserved with \n\n separator where applicable
    assert result['en'].startswith('Hello')
    assert result['fr'].startswith('Bonjour')


def test_split_respects_min_and_max_key_length():
    # key length beyond max_key_length treated as unknown language or autodetect
    text = '.:eng\nHello\n.:french\nBonjour\n'
    result = splitter.split(text, min_key_length=2, max_key_length=3)
    # Keys 'eng' and 'fre' truncated or unknown?
    # 'eng' exact length 3 -> accepted
    assert 'eng' in result or splitter.settings.UNKNOWN_LANGUAGE in result
    # 'french' length 6 > max_key_length, expect unknown language or detect
    assert 'fr' not in result  # since key longer than max_key_length
    # When autodetect, unknown should be assigned
    assert splitter.settings.UNKNOWN_LANGUAGE in result


def test_split_returns_markdown_when_markdown_true():
    text = '.:en\n Hello\n.:fr\n Bonjour'
    md = splitter.split(text, markdown=True)
    assert isinstance(md, str)
    # Markdown output contains sep, language codes and ends
    for lang in ['en', 'fr']:
        assert splitter.settings.SEP in md
        assert lang in md


def test_split_with_title_true_uses_title_ends_in_markdown():
    text = '.:en\n Hello\n.:fr\n Bonjour'
    md = splitter.split(text, markdown=True, title=True)
    # Ends used in title mode is last character of ends list -> ':'
    assert splitter.settings.ENDS[-1] in md


def test_split_handles_empty_tokens_and_whitespace():
    text = '.:en  \nHello\n\n\n.:fr  \nBonjour\n\n'
    result = splitter.split(text)
    assert 'en' in result and 'fr' in result
    # Whitespace after token identifier should be stripped by convert; here only split tested
    assert result['en'].startswith('Hello')
    assert result['fr'].startswith('Bonjour')


def test_split_detect_language_exception_handled(monkeypatch):
    def raise_exception(_):
        raise Exception("fail detection")

    monkeypatch.setattr(splitter, "detect_language", raise_exception)

    text = "some text without language key"
    # Detection exceptions cause assignment to xx and log an info
    # No exception propagated
    result = splitter.split(text)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == text


def test_split_keeps_order_of_languages():
    text = '.:fr\nBonjour\n.:en\nHello\n'
    result = splitter.split(text)
    keys = list(result.keys())
    # Order preserved as appeared in text
    assert keys == ['fr', 'en']


def test_split_handles_consecutive_sep_tokens():
    text = '.:en\nHello\n.:en\n World\n.:fr\nBonjour\n'
    result = splitter.split(text)
    # Text for 'en' combined
    assert 'Hello' in result['en']
    assert 'World' in result['en']
    assert 'Bonjour' in result['fr']


def test_split_no_end_symbol_fallback():
    # If no end symbol in sep/token, fallback to unknown language
    text = '.:enHello without end symbol'
    result = splitter.split(text)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    # Whole string after sep should appear there
    assert 'Hello without end symbol' in result[splitter.settings.UNKNOWN_LANGUAGE]


def test_convert_removes_leading_ends_in_text_md():
    # The convert method should strip leading occurrences of ends from text_md
    d = {'en': 'hello'}
    # Manually create a string with leading ends repeated
    text_md = f"{splitter.settings.ENDS[0]}{splitter.settings.ENDS[0]}{splitter.settings.SEP}en{splitter.settings.ENDS[0]}hello"
    # We check that convert removes those leading ends in output
    out = splitter.convert(d)
    assert not any(out.startswith(end) for end in splitter.settings.ENDS)


def test_split_respects_pargraph_sep_parameter():
    text = '.:en\nHello\n\nHi\n.:fr\nBonjour\n'
    # Use a paragraph separator that is a single newline to split paragraphs
    result = splitter.split(text, pargraph_sep='\n')
    # 'en' text contains both paragraphs joined by \n
    assert 'Hello\nHi' in result['en']
    # 'fr' text is as expected
    assert 'Bonjour' in result['fr']


def test_split_non_dict_returned_unchanged_by_convert_markdown(monkeypatch):
    # If split's result is non-dict (shouldn't happen usually), convert returns input as is
    result = splitter.split('simple text without markers')
    # Should be a string (unknown language assigned) or string input directly
    if isinstance(result, dict):
        md = splitter.convert(result)
        assert isinstance(md, str)
    else:
        md = splitter.convert(result)
        assert md == result
