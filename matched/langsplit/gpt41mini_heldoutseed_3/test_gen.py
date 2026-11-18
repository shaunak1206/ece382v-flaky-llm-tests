import collections
import logging
import pytest
from langsplit import splitter

def test_detect_language_returns_known_code():
    # Basic check for detect_language return value length and type
    lang = splitter.detect_language('This is an English sentence.')
    assert isinstance(lang, str)
    assert len(lang) == 2

def test_detect_language_shorter_code_map():
    # Simulate zh-cn detection mapping to cn
    # We patch langdetect.detect to return 'zh-cn'
    original_detect = splitter.langdetect.detect
    splitter.langdetect.detect = lambda text: 'zh-cn'
    try:
        code = splitter.detect_language('fake text')
        assert code == 'cn'
    finally:
        splitter.langdetect.detect = original_detect

def test_convert_with_dict_no_title_default_ends():
    data = {'en': ' Hello', 'fr': 'Bonjour'}
    text_md = splitter.convert(data)
    # It should prepend sep and lang and ends[0] = '\n'
    parts = text_md.split(splitter.settings.SEP)
    # first part may be empty if text_md starts with sep, strip to test
    assert any(part.startswith('en\nHello') or part.startswith('fr\nBonjour') for part in parts)

def test_convert_with_dict_title_uses_last_end():
    data = {'en': 'Hi there'}
    text_md = splitter.convert(data, title=True)
    # with title=True ends[-1] = ':'
    assert text_md.startswith(f"{splitter.settings.SEP}en{splitter.settings.ENDS[-1]}")

def test_convert_with_non_dict_returns_input():
    input_text = "No dict"
    output = splitter.convert(input_text)
    assert output == input_text

def test_split_returns_input_on_empty():
    # empty string returns itself
    assert splitter.split('') == ''
    assert splitter.split(None) is None

def test_split_special_case_short_sep_prefix():
    # text starting with sep and short length triggers special case
    sep = splitter.settings.SEP
    # Length less than sep + max_key_length + min(len(end))
    max_key_length = 2
    min_end_length = min(len(e) for e in splitter.settings.ENDS)
    special_text = sep + "en"  # no end symbol and very short
    ret = splitter.split(special_text, sep=sep, max_key_length=max_key_length)
    # returns OrderedDict with one key 'en' with empty string as value
    assert isinstance(ret, collections.OrderedDict)
    assert 'en' in ret
    assert ret['en'] == ''

def test_split_basic_lang_key_split():
    text = f"{splitter.settings.SEP}en{splitter.settings.ENDS[0]}Hello{splitter.settings.SEP}fr{splitter.settings.ENDS[0]}Bonjour"
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)
    assert 'en' in result and 'Bonjour' not in result['en']
    assert 'fr' in result
    assert result['en'].startswith('Hello')
    assert result['fr'].startswith('Bonjour')

def test_split_combines_chunks_same_lang():
    text = f"{splitter.settings.SEP}en{splitter.settings.ENDS[0]}Hello{splitter.settings.SEP}en{splitter.settings.ENDS[0]} World"
    result = splitter.split(text)
    # The texts for 'en' should be concatenated
    assert 'en' in result
    assert result['en'] == 'Hello World'

def test_split_unknown_language_key():
    # A chunk with no recognized key and autodetect off assigns to unknown_language
    text = "Random text without sep"
    result = splitter.split(text, autodetect=False)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert result[splitter.settings.UNKNOWN_LANGUAGE].strip() == text.strip()

def test_split_autodetect_detects_language():
    # Text without explicit lang keys but with autodetect=True returns detected language keys
    text = "This is English.\n\nCeci est français."
    result = splitter.split(text, autodetect=True, pargraph_sep='\n\n')
    # Should have keys e.g. 'en' and 'fr' or at least some detected languages
    assert isinstance(result, collections.OrderedDict)
    assert any(lang in result for lang in ('en', 'fr', splitter.settings.UNKNOWN_LANGUAGE))

def test_split_autodetect_with_paragraph_separator():
    # Test that paragraphs get separated by pargraph_sep and languages detected separately
    text = "Hello world.\n\nBonjour le monde.\n\nHallo Welt."
    result = splitter.split(text, autodetect=True, pargraph_sep='\n\n')
    # We expect multiple keys with text corresponding to individual paragraphs
    assert len(result) >= 1
    all_text = ''.join(result.values())
    assert "Hello world" in all_text or "Bonjour le monde" in all_text or "Hallo Welt" in all_text

def test_split_markdown_true_returns_string():
    text = f"{splitter.settings.SEP}en{splitter.settings.ENDS[0]}Hello"
    result = splitter.split(text, markdown=True)
    # result should be a string starting with language code sep
    assert isinstance(result, str)
    assert result.startswith(f"{splitter.settings.SEP}en")

def test_split_title_true_with_markdown():
    text = f"{splitter.settings.SEP}en{splitter.settings.ENDS[0]}Hello"
    result = splitter.split(text, markdown=True, title=True)
    # The ends used should be ends[-1]
    assert result.startswith(f"{splitter.settings.SEP}en{splitter.settings.ENDS[-1]}")

def test_split_with_nonstandard_ends():
    # Provide custom ends and verify split uses those
    custom_ends = ['|', '#']
    sep = splitter.settings.SEP
    text = f"{sep}en|Hello{sep}fr#Bonjour"
    result = splitter.split(text, ends=custom_ends)
    assert 'en' in result and 'fr' in result
    assert result['en'].startswith('Hello')
    assert result['fr'].startswith('Bonjour')

def test_split_with_min_max_key_length():
    # Keys shorter than min_key_length or longer than max_key_length assigned to unknown
    sep = splitter.settings.SEP
    text = f"{sep}e{splitter.settings.ENDS[0]}Hello{sep}eng{splitter.settings.ENDS[0]}World{sep}fr{splitter.settings.ENDS[0]}Bonjour"
    result = splitter.split(text, min_key_length=2, max_key_length=2)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert 'fr' in result
    # 'e' and 'eng' keys too short or too long, assigned unknown language
    unknown_text = result[splitter.settings.UNKNOWN_LANGUAGE]
    assert 'Hello' in unknown_text or 'World' in unknown_text

def test_split_with_empty_tokens_skipped():
    # Double separator leads to empty token: should be skipped without error
    text = f"{splitter.settings.SEP}en{splitter.settings.ENDS[0]}Hello{splitter.settings.SEP}{splitter.settings.SEP}fr{splitter.settings.ENDS[0]}Bonjour"
    result = splitter.split(text)
    assert 'en' in result and 'fr' in result

def test_split_detect_language_exception_handling(monkeypatch):
    # Patch detect_language to raise error, text should go to unknown language and log info called
    def raise_error(text): raise Exception("fail")
    monkeypatch.setattr(splitter, "detect_language", raise_error)
    # Use a text with no explicit language keys to force autodetect usage
    text = "Random text paragraph."
    result = splitter.split(text, autodetect=True)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    # The result should contain the full text
    assert "Random text paragraph." in result[splitter.settings.UNKNOWN_LANGUAGE]

def test_split_non_autodetect_uses_unknown_key_text():
    # With autodetect false, chunk assigned to unknown_language (except for key len conditions)
    sep = splitter.settings.SEP
    text = f"{sep}zz{splitter.settings.ENDS[0]}Text here"
    result = splitter.split(text, autodetect=False)
    assert 'zz' in result or splitter.settings.UNKNOWN_LANGUAGE in result
    # If zz is length 2, it should be the key (so not unknown language)
    if len('zz') >= 2:
        assert 'zz' in result
    else:
        assert splitter.settings.UNKNOWN_LANGUAGE in result

def test_detect_language_returns_original_for_short_code():
    # If langdetect returns a short code <= max_length, returned code is unchanged
    original_detect = splitter.langdetect.detect
    splitter.langdetect.detect = lambda text: 'fr'  # length == 2
    try:
        code = splitter.detect_language("Bonjour", max_length=2)
        assert code == 'fr'
    finally:
        splitter.langdetect.detect = original_detect

def test_split_with_multiline_text_and_unrecognized_keys():
    # Text with unknown keys, mixing recognized and unknown
    sep = splitter.settings.SEP
    text = f"{sep}en{splitter.settings.ENDS[0]}Hello\nWorld{sep}zz{splitter.settings.ENDS[0]}UnknownKeyText"
    result = splitter.split(text)
    assert 'en' in result
    # zz key length is 2, so should be recognized as is
    assert 'zz' in result
    assert 'Hello\nWorld' in result['en']
    assert 'UnknownKeyText' in result['zz']

def test_split_with_multiple_paragraphs_and_autodetect():
    # Check paragraphs separated and languages detected separately
    text = f"{splitter.settings.SEP}unknown:Some paragraph.\n\n{splitter.settings.SEP}unknown:Another paragraph."
    # Forcing unknown language by invalid keys
    result = splitter.split(text, autodetect=True, pargraph_sep='\n\n')
    assert isinstance(result, collections.OrderedDict)
    total_text = ''.join(result.values())
    assert 'Some paragraph' in total_text
    assert 'Another paragraph' in total_text

def test_split_handles_text_without_separators():
    # Should assign whole text to unknown language if no separators found
    text = "This is a text with no separator or language code."
    result = splitter.split(text)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert text in result[splitter.settings.UNKNOWN_LANGUAGE]
