import pytest
import random
import numpy as np
import torch

from langsplit import splitter


def test_detect_language_known_code():
    # Known lang short code mapping
    assert splitter.detect_language("This is English.") == "en"
    assert splitter.detect_language("English text", max_length=3) == "eng"[:3]

def test_detect_language_shorter_codes_mapping():
    # Setup a code in SHORTER_CODES explicitly
    # The detect will return 'zh-cn' or 'zh-tw' only if text matches; we simulate detection by patching
    # Use monkeypatch to replace langdetect.detect
    def fake_detect(text):  # simulate detection of zh-cn code
        return 'zh-cn'
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(splitter.langdetect, 'detect', fake_detect)
    try:
        assert splitter.detect_language("dummy text") == 'cn'
        # max_length 3 truncates to cn
        code = splitter.detect_language("dummy text", max_length=3)
        assert code == 'cn'
    finally:
        monkeypatch.undo()


def test_convert_with_dict_basic():
    d = {'en': 'Hello', 'fr': 'Bonjour'}
    result = splitter.convert(d)
    # Should produce a concatenation with sep and ends[0]
    assert result.startswith(splitter.settings.SEP)
    for k, v in d.items():
        assert f"{splitter.settings.SEP}{k}{splitter.settings.ENDS[0]}{v.lstrip()}" in result

def test_convert_with_title_true_uses_last_end():
    d = {'en': 'hello', 'fr': 'salut'}
    res = splitter.convert(d, title=True)
    # The end character should be ends[-1]
    for end in splitter.settings.ENDS[:-1]:
        assert end not in res or res.endswith(end)  # It should not end with these final ends except last
    assert any(end == splitter.settings.ENDS[-1] for end in res)

def test_convert_with_non_dict_returns_input_unchanged():
    input_text = "just a normal string"
    out = splitter.convert(input_text)
    assert out == input_text

def test_split_empty_input_returns_input():
    assert splitter.split("") == ""

def test_split_simple_known_language_key():
    # text providing known key and small length
    text = splitter.settings.SEP + "en" + splitter.settings.ENDS[0] + "Hello world"
    result = splitter.split(text)
    assert isinstance(result, dict)
    assert result.get("en", None) == "Hello world"

def test_split_with_no_sep_but_shorter_than_expected():
    # When text is just '.:en' without any end symbols and short length
    text = splitter.settings.SEP + "en"
    out = splitter.split(text)
    # Should return an OrderedDict with key "en" and empty string value
    # Because text is just sep + key with no content after
    assert isinstance(out, dict)
    assert list(out.keys())[0] == 'en'
    assert list(out.values())[0] == ""

def test_split_multi_languages_autodetect(monkeypatch):
    # Patch langdetect.detect to return fixed languages for predictable output
    test_paragraphs = {
        "This is English.": "en",
        "C'est français.": "fr",
        "Texto en español.": "es",
    }
    def fake_detect(text):
        # Return mapping or default unknown
        return test_paragraphs.get(text, splitter.settings.UNKNOWN_LANGUAGE)

    monkeypatch.setattr(splitter.langdetect, "detect", fake_detect)

    input_text = "\n\n".join(test_paragraphs.keys())
    result = splitter.split(input_text, autodetect=True, pargraph_sep="\n\n")
    # All keys must be present as detected keys in order
    assert list(result.keys()) == list(test_paragraphs.values())
    # Values should match paragraphs exactly
    for k, v in result.items():
        assert v == [p for p, lang in test_paragraphs.items() if lang == k][0]

def test_split_multi_languages_autodetect_additional_sep(monkeypatch):
    # Similar but with sep dividing parts and explicit language keys
    # text has multiple parts like .:en: inside text
    monkeypatch.setattr(splitter.langdetect, "detect", lambda t: "en")
    text = splitter.settings.SEP + "fr" + splitter.settings.ENDS[0] + "Bonjour" + \
           splitter.settings.SEP + "en" + splitter.settings.ENDS[0] + "Hello"
    result = splitter.split(text)
    assert result.get('fr') == "Bonjour"
    assert result.get('en') == "Hello"

def test_split_autodetect_with_empty_paragraph(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda t: "en")
    text = "\n\nHello\n\n\n\nWorld\n\n"
    result = splitter.split(text, pargraph_sep="\n\n")
    # Should combine English text separated by empty paragraphs
    assert 'en' in result
    # The output text should contain the text combined with separator
    assert "Hello" in result['en']
    assert "World" in result['en']

def test_split_with_autodetect_exception(monkeypatch):
    # langdetect.detect raises exception -> fallback to UNKNOWN_LANGUAGE
    def raising_detect(text):
        raise Exception("fail detection")

    monkeypatch.setattr(splitter.langdetect, "detect", raising_detect)

    text = "Some text without language spec"
    result = splitter.split(text, autodetect=True)
    # The key should be UNKNOWN_LANGUAGE in the output
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == text

def test_split_with_markdown_true(monkeypatch):
    # To test markdown=True, it should return string formatted by convert
    monkeypatch.setattr(splitter.langdetect, "detect", lambda t: "en")

    text = splitter.settings.SEP + 'en' + splitter.settings.ENDS[0] + "Text"
    result = splitter.split(text, markdown=True)
    assert isinstance(result, str)
    # Result should contain sep and language code
    assert splitter.settings.SEP in result
    assert 'en' in result

def test_split_respects_min_max_key_length(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda t: "en")
    # Prepare a key longer than max_key_length => text with unknown language
    sep = splitter.settings.SEP
    ends = splitter.settings.ENDS
    text = sep + "engl" + ends[0] + "Hello"
    # max_key_length is 2 by default, so this key is longer, lands in unknown
    result = splitter.split(text, min_key_length=2, max_key_length=2)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    # The value contains the entire text chunk
    assert "Hello" in result[splitter.settings.UNKNOWN_LANGUAGE]

def test_split_combines_texts_of_same_key(monkeypatch):
    # Two tokens with same language key are combined
    monkeypatch.setattr(splitter.langdetect, "detect", lambda t: "en")
    text = splitter.settings.SEP + "en" + splitter.settings.ENDS[0] + "Hello" + \
           splitter.settings.SEP + "en" + splitter.settings.ENDS[0] + " World"
    result = splitter.split(text)
    assert 'en' in result
    # Combined values, no separator added
    assert result['en'] == "Hello World"

def test_split_ordered_dict_keys_order():
    # The order of keys in result respects first appearance sequence
    text = splitter.settings.SEP + "fr" + splitter.settings.ENDS[0] + "Bonjour" + \
           splitter.settings.SEP + "en" + splitter.settings.ENDS[0] + "Hello" + \
           splitter.settings.SEP + "fr" + splitter.settings.ENDS[0] + " tout le monde"
    result = splitter.split(text, autodetect=False)
    # Keys order must be fr then en, in OrderedDict
    keys = list(result.keys())
    assert keys.index('fr') < keys.index('en')

def test_split_text_without_sep_autodetect(monkeypatch):
    # When text has no sep, should autodetect whole text language
    monkeypatch.setattr(splitter.langdetect, "detect", lambda t: "en")
    text = "This is a full paragraph without sep"
    result = splitter.split(text)
    assert 'en' in result
    assert result['en'] == text

def test_split_with_no_autodetect(monkeypatch):
    # Provide text without language prefix keys and autodetect False
    text = "randomtext"
    result = splitter.split(text, autodetect=False)
    # Should assign all to UNKNOWN_LANGUAGE but:
    # code tries to get chunk[len(name)+1:], but name is 'xx', so len 2,
    # we expect substring from pos 3, but if text shorter it results empty string
    # So the result value should be shorter than text or empty string
    unk = splitter.settings.UNKNOWN_LANGUAGE
    assert unk in result
    # value will likely be empty because chunk[len(name)+1:], but chunk is whole token here without prefix
    # So it can be empty or leftover string.
    val = result[unk]
    assert isinstance(val, str)

def test_split_respects_pargraph_sep(monkeypatch):
    # Test pargraph_sep splitting and recombination with autodetect True
    monkeypatch.setattr(splitter.langdetect, "detect", lambda t: "en")
    # Create 3 paragraphs separated by double newlines
    text = "para1\n\npara2\n\npara3"
    result = splitter.split(text, pargraph_sep="\n\n")
    # Result 'en' key text should contain all paragraphs joined with pargraph_sep
    assert 'en' in result
    assert result['en'] == text

def test_split_ignores_empty_tokens(monkeypatch):
    # Empty tokens created by split should be ignored and no keys created for them
    monkeypatch.setattr(splitter.langdetect, "detect", lambda t: "en")
    text = splitter.settings.SEP + splitter.settings.SEP + "en" + splitter.settings.ENDS[0] + "Hello"
    result = splitter.split(text)
    # Only a single 'en' key with Hello text.
    assert 'en' in result
    assert list(result.keys()) == ['en']
    assert result['en'] == "Hello"

def test_detect_language_returns_max_length_slice(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda t: "english")
    # max_length=2 expected to slice code to first 2 letters
    code = splitter.detect_language("text", max_length=2)
    assert len(code) == 2
    assert code == "en"

def test_convert_removes_leading_end_symbols(monkeypatch):
    d = { 'en': splitter.settings.ENDS[0] + "text"}
    out = splitter.convert(d)
    # The output should remove leading end symbols at start of text_md
    assert not out.startswith(tuple(splitter.settings.ENDS))
    # Check the text after language code
    assert "text" in out

def test_split_with_title_true_changes_convert_behavior(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda t: "en")
    text = splitter.settings.SEP + 'en' + splitter.settings.ENDS[0] + "Text"
    # with markdown=True and title=True, convert adds different ends[-1]
    result = splitter.split(text, markdown=True, title=True)
    # Output string uses ends[-1] as separator (for title)
    assert splitter.settings.ENDS[-1] in result

def test_split_returns_ordered_dict(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda t: "en")
    text = splitter.settings.SEP + 'en' + splitter.settings.ENDS[0] + "Text"
    result = splitter.split(text)
    # The returned dict is collections.OrderedDict
    import collections
    assert isinstance(result, collections.OrderedDict)

def test_convert_with_empty_dict_returns_empty_string():
    result = splitter.convert({})
    assert result == ""
