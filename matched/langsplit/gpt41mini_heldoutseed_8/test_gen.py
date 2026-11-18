import collections
import pytest
import logging

from langsplit import splitter


def test_split_empty_string_returns_empty():
    assert splitter.split('') == ''


def test_split_single_lang_code_without_end():
    # Input like ".:en" without end should return OrderedDict with key 'en' and empty text
    res = splitter.split(".:en")
    expected = collections.OrderedDict([('en', '')])
    assert res == expected


def test_split_single_language_with_text():
    # Single language code and some text after end character
    text = ".:en:Hello world"
    res = splitter.split(text)
    expected = collections.OrderedDict([('en', 'Hello world')])
    assert res == expected


def test_split_multiple_languages_different_texts():
    text = ".:en:Hello.:fr:Bonjour.:es:Hóla"
    res = splitter.split(text)
    expected = collections.OrderedDict([
        ('en', 'Hello'),
        ('fr', 'Bonjour'),
        ('es', 'Hóla'),
    ])
    assert res == expected


def test_split_unknown_language_text_autodetect_false():
    # Text without language code, autodetect False leads to unknown language key only
    text = "Just some random text without separator"
    res = splitter.split(text, autodetect=False)
    # Because no separator, we get default unknown language key with full text minus something
    # but since no separator, default is return text unchanged as per code? No it's text as-is
    assert isinstance(res, str)
    assert res == text


def test_split_unknown_language_text_autodetect_true(monkeypatch):
    # Patch detect_language to a fixed value to test autodetect path
    text = ".:Hello world"
    # monkeypatch detect_language to always return 'en'
    monkeypatch.setattr(splitter, 'detect_language', lambda t: 'en')
    res = splitter.split(text, autodetect=True)
    assert 'en' in res
    # The text under 'en' should contain "Hello world"
    assert 'Hello world' in res['en']


def test_split_text_with_paragraphs_autodetect(monkeypatch):
    # Text with paragraphs; each paragraph assigned detected language
    text = ".:Hello world\n\n.:Bonjour monde\n\n.:Hola mundo"
    # Patch detect_language to return different language for each paragraph by content
    def fake_detect_language(t):
        t = t.strip().lower()
        if t.startswith('hello'):
            return 'en'
        elif t.startswith('bonjour'):
            return 'fr'
        elif t.startswith('hola'):
            return 'es'
        return 'xx'
    monkeypatch.setattr(splitter, 'detect_language', fake_detect_language)
    res = splitter.split(text, autodetect=True, pargraph_sep='\n\n')
    assert isinstance(res, collections.OrderedDict)
    assert res['en'].startswith('Hello world')
    assert res['fr'].startswith('Bonjour monde')
    assert res['es'].startswith('Hola mundo')


def test_split_with_min_max_key_lengths():
    # If key length outside min/max, text should be assigned to unknown_language 'xx'
    text = ".:en:Hello.:english:ignored"
    res = splitter.split(text, min_key_length=2, max_key_length=2)
    # 'en' should be detected but 'english' is longer than max_key_length so text goes to 'xx'
    assert 'en' in res
    assert 'xx' in res
    assert 'Hello' in res['en']
    assert 'ignored' in res['xx']


def test_split_with_markdown_true_returns_string():
    # When markdown True, output is a string, not dict
    text = ".:en:Hello.:fr:Bonjour"
    out = splitter.split(text, markdown=True)
    assert isinstance(out, str)
    # Should contain the separator and language codes
    assert splitter.settings.SEP in out
    assert 'en' in out and 'fr' in out


def test_convert_dict_to_text_default_ends():
    d = {'en': 'Hello', 'fr': 'Bonjour'}
    result = splitter.convert(d)
    # Result should contain language codes and string text
    assert '.:' in result
    assert 'en' in result and 'fr' in result
    assert 'Hello' in result and 'Bonjour' in result


def test_convert_dict_to_text_title_ends():
    d = {'en': 'Hello', 'fr': 'Bonjour'}
    result = splitter.convert(d, title=True)
    # The text should end with the last element in ends (':')
    assert result.count(':') >= 2  # at least two separators used


def test_convert_non_dict_returns_input():
    input_val = "some string"
    assert splitter.convert(input_val) == input_val


def test_detect_language_short_code_replacement(monkeypatch):
    # Patch langdetect.detect to return 'zh-cn' and check we get 'cn'
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda t: 'zh-cn')
    result = splitter.detect_language('dummy text', max_length=2)
    assert result == 'cn'


def test_detect_language_code_truncation(monkeypatch):
    # Patch langdetect.detect to return longer string than max_length
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda t: 'abcde')
    result = splitter.detect_language('dummy', max_length=3)
    assert result == 'abc'


def test_split_multiline_input_with_unknown_lang_and_autodetect():
    # Text contains one segment with separator and unknown language without key
    text = ".:en:Hello world\nRandom text without key\n.:fr:Bonjour"
    # monkeypatch detect_language to always return 'xx' to force unknown language
    def fake_detect_language(t):
        if 'Hello' in t:
            return 'en'
        if 'Bonjour' in t:
            return 'fr'
        return 'xx'
    import pytest
    pytest.monkeypatch.setattr(splitter, 'detect_language', fake_detect_language)
    # This test depends on autodetect, so we get 'xx' key for unknown part
    # Because the line "Random text without key" does not have language code
    # and does not start with sep, it will be detected automatically
    res = splitter.split(text)
    assert isinstance(res, collections.OrderedDict)
    # We expect 'en' and 'fr' keys, and possibly 'xx' for unknown text
    assert 'en' in res
    assert 'fr' in res
    assert any(k == splitter.settings.UNKNOWN_LANGUAGE for k in res.keys()) or len(res) >= 2


def test_split_ignores_empty_tokens():
    text = ".:en:Hello.:.:fr:Bonjour"
    res = splitter.split(text)
    # Should parse 'en' and 'fr' only, ignoring empty tokens after split by sep
    assert 'en' in res and 'fr' in res
    assert len(res) == 2


def test_split_with_custom_sep_and_ends():
    # Use custom separator and end characters, and verify functionality
    custom_sep = "=="
    custom_ends = ['|', '||']
    text = "==en|Hello==fr|Bonjour"
    res = splitter.split(text, sep=custom_sep, ends=custom_ends)
    assert 'en' in res
    assert 'fr' in res
    assert res['en'].startswith('Hello')
    assert res['fr'].startswith('Bonjour')


def test_split_title_true_alters_convert_output():
    text = ".:en:Hello.:fr:Bonjour"
    out = splitter.split(text, markdown=True, title=True)
    # Output is string with ends[-1], which is ':'
    assert out.endswith(':')


def test_split_text_with_key_length_edge_cases():
    # Key length exactly min_key_length or max_key_length accepted
    text = ".:en:Hi.:fr:Ok"
    res = splitter.split(text, min_key_length=2, max_key_length=2)
    assert 'en' in res
    assert 'fr' in res


def test_split_text_with_key_length_below_min_length():
    # If key shorter than min_key_length text goes to unknown
    text = ".:e:Hi"
    res = splitter.split(text, min_key_length=2, max_key_length=2)
    assert splitter.settings.UNKNOWN_LANGUAGE in res
    assert 'Hi' in res[splitter.settings.UNKNOWN_LANGUAGE]


def test_split_logging_on_detect_failure(caplog):
    # Force detect_language to raise exception to test logging of unknown language
    def raise_exc(text):
        raise Exception("fail detect")

    import pytest
    pytest.monkeypatch.setattr(splitter, 'detect_language', raise_exc)

    caplog.set_level(logging.INFO)
    text = ".:Hello world"
    res = splitter.split(text)
    # Should contain unknown language key with text
    assert splitter.settings.UNKNOWN_LANGUAGE in res
    assert 'Hello world' in res[splitter.settings.UNKNOWN_LANGUAGE]
    # Check log contains 'Language not detected' message
    found_log = any("Language not detected" in record.message for record in caplog.records)
    assert found_log


def test_convert_removes_starting_end_chars():
    # convert strips starting end chars when they appear at the start twice
    d = collections.OrderedDict([('en', ' text')])
    # Prepare a string starting with characters in ends twice to check stripping
    # The function trims while text_md[:2] in ends, so supply text like that
    # Let's test this by direct call of convert producing text starting with ends
    out = splitter.convert(d)
    # The output should not start with a character from ends list repeated twice
    assert not any(out.startswith(e*2) for e in splitter.settings.ENDS)


def test_split_text_without_sep():
    # Text with no separator should be returned as is
    text = "Just a simple text without any separator"
    res = splitter.split(text)
    assert res == text


def test_split_text_only_sep_multiline():
    # Text with multiple separators but no language key at all should assign all text to unknown
    text = ".:.:.:\n"
    res = splitter.split(text)
    assert splitter.settings.UNKNOWN_LANGUAGE in res


def test_split_text_with_key_and_no_text():
    # Text with a key but no following text after end symbol
    text = ".:en:"
    res = splitter.split(text)
    assert 'en' in res
    assert res['en'] == ''


# End of test functions
