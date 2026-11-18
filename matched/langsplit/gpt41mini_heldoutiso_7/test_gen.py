import collections
import pytest

import langsplit.splitter as splitter

def test_split_empty_returns_empty():
    assert splitter.split('') == ''

def test_split_single_sep_no_text_returns_unknown():
    # text like '.:en' without end symbol length less than sep+max_key_length+min_end_len
    text = '.:en'
    result = splitter.split(text)
    expected = collections.OrderedDict([('en', '')])
    assert result == expected

def test_split_single_sep_with_text_assigned_to_key():
    text = '.:enHello'
    result = splitter.split(text)
    expected = collections.OrderedDict([('en', 'Hello')])
    assert result == expected

def test_split_basic_with_keys_and_texts():
    text = '.:en:Hello.:fr:Bonjour.:es:Hóla'
    result = splitter.split(text)
    expected = collections.OrderedDict([
        ('en', 'Hello'),
        ('fr', 'Bonjour'),
        ('es', 'Hóla'),
    ])
    assert result == expected

def test_split_with_multiple_paragraphs_autodetect_false_assigns_unknown():
    # When autodetect=False and chunk has unknown key it should assign rest after name +1 to unknown
    text = '.:xx:part1\n\npart2'
    result = splitter.split(text, autodetect=False)
    # 'xx' is length 2, so should assign chunk[len(name)+1:] which is from after "xx" + 1 char = after "xx:"
    expected = collections.OrderedDict([('xx', 'part1\n\npart2')])
    assert result == expected

def test_split_with_unknown_key_and_autodetect_true_calls_detect_language(monkeypatch):
    texts_detected = []
    def fake_detect_language(text_inner, max_length=2):
        texts_detected.append(text_inner)
        return 'en'
    monkeypatch.setattr(splitter, 'detect_language', fake_detect_language)
    # This chunk has no valid key, so should autodetect per paragraph
    text = '.:unk:hello\n\nworld\n\n'
    result = splitter.split(text)
    assert 'en' in result
    assert 'hello' in result['en']
    assert 'world' in result['en']
    assert texts_detected == ['hello', 'world']

def test_split_detect_language_called_and_handles_exception(monkeypatch):
    def raise_exc(text_inner, max_length=2):
        raise Exception('fail')
    monkeypatch.setattr(splitter, 'detect_language', raise_exc)
    text = '.:unk:para1\n\npara2'
    result = splitter.split(text)
    # Because detect_language raises, text should go to unknown language
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert 'para1' in result[splitter.settings.UNKNOWN_LANGUAGE]
    assert 'para2' in result[splitter.settings.UNKNOWN_LANGUAGE]

def test_convert_with_dict_and_title_false_formats_properly():
    d = {'en': ' hello world', 'fr': ' bonjour '}
    text_md = splitter.convert(d, title=False)
    # Should strip leading spaces on text, use ends[0] = '\n'
    expected = '.:en\nhello world.:fr\nbonjour'
    assert text_md == expected

def test_convert_with_dict_and_title_true_uses_last_end():
    d = {'en': 'hello', 'fr': 'bonjour'}
    text_md = splitter.convert(d, title=True)
    # ends[-1] = ':' used as end
    expected = '.:en:hello.:fr:bonjour'
    assert text_md == expected

def test_convert_with_non_dict_returns_as_is():
    val = 'some string'
    assert splitter.convert(val) == val

def test_detect_language_shortens_code_correctly():
    # Provide codes longer than max_length=2, e.g. 'zh-cn' shortened to 'cn'
    assert splitter.detect_language('some text', max_length=2) in splitter.SHORTER_CODES.values() or len(splitter.detect_language('some text', max_length=2)) == 2

def test_detect_language_returns_first_n_chars_for_long_codes(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'abcde')
    # SHORTER_CODES does not contain 'abcde' so slice returned is first max_length chars
    assert splitter.detect_language('foo', max_length=3) == 'abc'

def test_split_respects_min_and_max_key_length_and_parses_keys():
    text = '.:en:foo.:abc:def.:abcd:ghi'
    # max_key_length=3, min_key_length=2 so 'en' accepted, 'abc' accepted, 'abcd' too long -> unknown
    result = splitter.split(text, min_key_length=2, max_key_length=3)
    # 'en' and 'abc' keys recognized, rest assigned to UNKNOWN_LANGUAGE
    assert 'en' in result and 'abc' in result
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert result['en'] == 'foo'
    assert result['abc'] == 'def.'
    # 'abcd:ghi' chunk - 'abcd' length 4 > max_key_length -> unknown
    assert 'ghi' in result[splitter.settings.UNKNOWN_LANGUAGE]

def test_split_order_preserves_language_sequence():
    text = '.:fr:bonjour.:en:hello.:fr:comment ça va?:.es:hola'
    result = splitter.split(text)
    keys = list(result.keys())
    assert keys == ['fr', 'en', 'es']

def test_split_returns_markdown_when_markdown_true():
    text = '.:en:hello.:fr:bonjour'
    result_dict = splitter.split(text, markdown=True)
    # The returned is a str due to convert call
    assert isinstance(result_dict, str)
    # It should contain the language codes with proper separators and ends
    assert '.:en' in result_dict and '.:fr' in result_dict

def test_split_title_true_changes_conversion_behavior():
    text = '.:en:hello.:fr:bonjour'
    result = splitter.split(text, markdown=True, title=True)
    # ends ending with ':' so output should have language codes ending with ':'
    assert '.:en:' in result and '.:fr:' in result

def test_split_handles_text_starting_with_sep_but_too_short_to_parse():
    text = '.:e'  # shorter than sep+max_key_length+min end length (3+2+1=6)
    result = splitter.split(text)
    expected = collections.OrderedDict([('e', '')])
    assert result == expected

def test_split_with_text_without_separators_assigns_to_unknown(monkeypatch):
    # Text with no sep and autodetect True: whole text split(['text']) gives one token with no key -> autodetect whole text
    called = []
    def fake_detect(text, max_length=2):
        called.append(text)
        return 'en'
    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    text = 'plain text without separators'
    result = splitter.split(text)
    assert 'en' in result
    assert called == [text]

def test_split_handles_multiple_empty_tokens():
    text = '.:en:one..:.fr:two'
    result = splitter.split(text)
    assert result == collections.OrderedDict([('en', 'one'), ('fr', 'two')])

def test_split_preserves_paragraph_separators_between_paragraphs():
    text = '.:xx:first paragraph\n\nsecond paragraph\n\nthird'
    # This tests that paragraphs split by "\n\n" keep the separator inserted between paragraphs
    result = splitter.split(text)
    val = result.get('xx') or ''
    assert '\n\n' in val

def test_split_non_dict_returned_from_convert():
    val = ['not', 'a', 'dict']
    converted = splitter.convert(val)
    assert converted == val
