import collections
import pytest
from langsplit import splitter

def test_detect_language_short_code():
    # 'en' is two letter code, should return 'en'
    assert splitter.detect_language('This is English.') == 'en'

def test_detect_language_long_code_shortened():
    # For 'zh-cn', should return 'cn' (shorter code)
    # Provide simple chinese text to detect 'zh-cn'
    text = '这是中文。'
    code = splitter.detect_language(text)
    # Since short code is 'cn', detect_language should return 'cn'
    assert code == 'cn'

def test_convert_with_dict_basic():
    d = {'en': 'hello', 'fr': 'bonjour'}
    res = splitter.convert(d, sep='.:', ends=['\n'])
    # Should start with 'en\nhello' and contain 'fr\nbonjour'
    assert res.startswith('en\nhello')
    assert 'fr\nbonjour' in res

def test_convert_with_dict_title_true():
    d = {'en': 'hello', 'fr': 'bonjour'}
    res = splitter.convert(d, sep='.:', ends=['\n', '\r', '\r\n', ':'], title=True)
    # Ends should use last in ends list (':')
    assert res.startswith('en:hello')
    assert 'fr:bonjour' in res

def test_convert_non_dict_returns_same():
    data = "Not a dict"
    assert splitter.convert(data) == data

def test_split_empty_text_returns_empty():
    assert splitter.split('') == ''

def test_split_single_token_with_sep_and_short_length():
    # Text like '.:en' without end symbol, triggers special case
    text = '.:en'
    result = splitter.split(text, max_key_length=2)
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ['en']
    assert result['en'] == ''

def test_split_specified_keys_and_texts():
    text = '.:en\nHello world.:fr\nBonjour le monde.'
    result = splitter.split(text, sep='.:', ends=['\n'])
    # Keys detected: 'en' and 'fr'
    assert 'en' in result
    assert 'fr' in result
    assert 'Hello world' in result['en']
    assert 'Bonjour le monde' in result['fr']

def test_split_with_autodetect_false_assigns_unknown_language():
    text = '.:en\nHello world.:xx\nBonjour'
    result = splitter.split(text, sep='.:', ends=['\n'], autodetect=False)
    # Since autodetect=False, unknown languages with key 'xx' assigned but text excludes key length+1 chars
    # 'xx' key present and text contains "Bonjour"
    assert 'xx' in result
    assert 'Bonjour' in result['xx']

def test_split_autodetect_with_known_languages():
    # Provide paragraphs with English and French text without keys to trigger autodetect
    text = 'Hello world\n\nBonjour le monde'

    result = splitter.split(text, sep='.:', ends=['\n'], autodetect=True, pargraph_sep='\n\n')
    # The keys will likely be 'en' and 'fr' or unknown but containing the text paragraphs
    # Check at least contains keys and corresponding texts
    keys = list(result.keys())
    assert any(isinstance(k, str) and k != splitter.settings.UNKNOWN_LANGUAGE for k in keys)
    joined_texts = ''.join(result.values())
    # Both original texts appear in combined values (ignoring language key accuracy)
    assert 'Hello world' in joined_texts
    assert 'Bonjour le monde' in joined_texts

def test_split_with_markdown_true_returns_string():
    text = '.:en\nHello world.:fr\nBonjour'
    res = splitter.split(text, sep='.:', ends=['\n'], markdown=True)
    # Should return a string representing converted markdown format
    assert isinstance(res, str)
    assert 'en' in res
    assert 'fr' in res

def test_split_with_title_true_in_markdown():
    text = '.:en\nHello world.:fr\nBonjour'
    res = splitter.split(text, sep='.:', ends=['\n', '\r', '\r\n', ':'], markdown=True, title=True)
    # Should return string with ':' end for each language segment
    assert all(sep in res for sep in ['.:', ':'])

def test_split_handles_multiple_consecutive_seps():
    text = '.:en\nHello.:fr\nBonjour.:en\nWorld.'
    result = splitter.split(text, sep='.:', ends=['\n'])
    # The texts for 'en' should be concatenated
    assert result['en'].count('Hello') == 1
    assert result['en'].count('World.') == 1
    assert result['fr'].count('Bonjour') == 1

def test_split_detect_language_exception_handled(monkeypatch):
    # Monkeypatch detect_language to raise exception for test
    def raise_exc(_):
        raise Exception("fail")

    monkeypatch.setattr(splitter, 'detect_language', raise_exc)

    text = 'some text without language markers'
    # Because of exception, language should be unknown 'xx'
    res = splitter.split(text)
    assert splitter.settings.UNKNOWN_LANGUAGE in res
    assert text in res[splitter.settings.UNKNOWN_LANGUAGE]

def test_split_text_without_sep_returns_autodetect_split():
    text = "Hello world\n\nBonjour le monde"
    # No language markers, should trigger autodetect and split by paragraphs
    res = splitter.split(text, sep='.:', ends=['\n'], autodetect=True, pargraph_sep='\n\n')
    assert isinstance(res, collections.OrderedDict)
    combined = ''.join(res.values())
    assert 'Hello world' in combined and 'Bonjour le monde' in combined

def test_split_with_min_key_length_larger_than_key_does_not_assign_key():
    # Provide key shorter than min_key_length to cause unknown assignment
    text = '.:e\nHello world'
    res = splitter.split(text, sep='.:', ends=['\n'], min_key_length=2)
    # Should assign to unknown language because key 'e' is length 1 < min_key_length 2
    assert splitter.settings.UNKNOWN_LANGUAGE in res
    combined = ''.join(res.values())
    assert 'Hello world' in combined

def test_split_returns_ordered_dict_preserving_order():
    text = '.:fr\nBonjour.:en\nHello'
    res = splitter.split(text)
    assert isinstance(res, collections.OrderedDict)
    keys = list(res.keys())
    # Order of first lang encountered preserved
    assert keys == ['fr', 'en']

def test_split_ignores_empty_tokens_between_seps():
    text = '.:en\nHello\n.:.:fr\nBonjour'
    res = splitter.split(text)
    # Should still detect en and fr keys even with empty token between .:.: sequence
    assert 'en' in res
    assert 'fr' in res

def test_convert_strips_leading_ends_from_result():
    d = {'en': ' test'}
    # Provide ends such that first characters in output are in ends list to strip
    res = splitter.convert(d, sep='.:', ends=['\n', '\r'], title=False)
    # Result should NOT start with '\n' or '\r'
    assert not res.startswith('\n')
    assert not res.startswith('\r')

def test_split_with_non_string_input_returns_same():
    # Input of None or number returns same value (not string)
    for val in [None, 42, 3.14, ['list'], {'dict': 1}]:
        assert splitter.split(val) == val
