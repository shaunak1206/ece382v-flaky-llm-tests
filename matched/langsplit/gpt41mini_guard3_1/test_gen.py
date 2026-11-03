import random
import collections
import pytest
import numpy as np

from langsplit import splitter

def test_detect_language_basic_seeded():
    random.seed(12345)
    np.random.seed(12345)
    # Example known language: English text
    lang = splitter.detect_language("This is a test sentence.")
    # The code should be either 'en' or shortened 'en' truncated to 2 chars
    assert isinstance(lang, str)
    assert len(lang) <= 2
    assert lang.islower()

def test_detect_language_shorter_codes_override():
    random.seed(123)
    np.random.seed(123)
    # zh-cn should be converted to cn trimmed length 2
    # directly test SHORTER_CODES mapping via detect_language
    # Inject a text that langdetect returns 'zh-cn'
    # We'll monkeypatch langdetect.detect
    def fake_detect(_):
        return 'zh-cn'
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(splitter.langdetect, "detect", fake_detect)
    try:
        code = splitter.detect_language("irrelevant")
        assert code == "cn"
    finally:
        monkeypatch.undo()

def test_convert_dict_basic_title_false():
    random.seed(1)
    np.random.seed(1)
    d = {'en': ' hello world', 'fr': ' bonjour'}
    result = splitter.convert(d, sep='.:', ends=['\n', '\r', '\r\n', ':'], title=False)
    # No title => use ends[0], which is '\n'
    # Result should contain language and text properly formatted
    assert result.startswith(".:en\n") or result.startswith(".:en\r") or result.startswith(".:en\r\n") or result.startswith(".:en:")
    # Should contain stripped text for each language, no leading spaces
    assert "hello" in result and "bonjour" in result
    # The result should not start with any ends characters stripped off
    for end in splitter.settings.ENDS:
        assert not result.startswith(end * 2)

def test_convert_dict_with_title_true_uses_last_end():
    random.seed(2)
    np.random.seed(2)
    d = {'en': 'test'}
    result = splitter.convert(d, title=True)
    # When title=True, end should be ends[-1] == ':'
    assert result.endswith(':test') or result.startswith(".:en:")

def test_convert_non_dict_returns_input_unchanged():
    random.seed(3)
    np.random.seed(3)
    s = "not a dict"
    assert splitter.convert(s) == s

def test_split_empty_text_returns_empty():
    random.seed(4)
    np.random.seed(4)
    assert splitter.split('') == ''

@pytest.mark.parametrize("input_text,expected_key,expected_value", [
    ('.:en', 'en', ''),
    ('.:frtext', 'fr', 'text'),
])
def test_split_special_single_line_short_prefix(input_text, expected_key, expected_value):
    random.seed(5)
    np.random.seed(5)
    result = splitter.split(input_text, sep='.:', max_key_length=2, min_key_length=2)
    assert isinstance(result, collections.OrderedDict)
    assert expected_key in result
    assert result[expected_key] == expected_value

def test_split_known_language_key_text_simple(monkeypatch):
    random.seed(6)
    np.random.seed(6)
    text = ".:en\nHello\n.:fr\nBonjour\n"

    # Monkeypatch detect_language to always return "en" or "fr" based on text presence
    def fake_detect_language(t):
        if "Bonjour" in t:
            return "fr"
        return "en"
    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)
    result = splitter.split(text, sep='.:', ends=splitter.settings.ENDS)

    # Should contain keys 'en' and 'fr'
    assert isinstance(result, collections.OrderedDict)
    assert set(result.keys()).issuperset({'en', 'fr'})
    # Text associated with each key contains original text snippets without leading spaces
    assert "Hello" in result['en']
    assert "Bonjour" in result['fr']

def test_split_autodetect_false_assigns_unknown(monkeypatch):
    random.seed(7)
    np.random.seed(7)
    input_text = ".:entext without end symbol"
    # Since autodetect=False and no proper end, text goes to UNKNOWN_LANGUAGE
    result = splitter.split(input_text, sep='.:', ends=splitter.settings.ENDS, autodetect=False)
    assert isinstance(result, collections.OrderedDict)
    # Should have UNKNOWN_LANGUAGE key
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    # The text is the chunk minus name +1 but since name unknown, it's added entirely
    assert result[splitter.settings.UNKNOWN_LANGUAGE].startswith("text without end symbol")

def test_split_paragraphs_autodetect(monkeypatch):
    random.seed(8)
    np.random.seed(8)
    # Provide multi-paragraph text without language prefix
    par1 = "Hello world."
    par2 = "Bonjour tout le monde."
    input_text = par1 + "\n\n" + par2
    # patch detect_language to return fixed values for paragraphs
    def fake_detect_language(text):
        if "Bonjour" in text:
            return "fr"
        else:
            return "en"
    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)

    result = splitter.split(input_text, pargraph_sep='\n\n', autodetect=True)
    # The result should be OrderedDict with keys en and fr
    assert isinstance(result, collections.OrderedDict)
    assert set(result.keys()) == {'en', 'fr'}
    # The paragraph separator should be preserved in values
    assert result['en'] == par1 + "\n\n"
    assert result['fr'] == par2

def test_split_markdown_true_returns_string(monkeypatch):
    random.seed(9)
    np.random.seed(9)
    d = collections.OrderedDict([('en', 'Text EN'), ('fr', 'Texte FR')])
    # convert is called when markdown=True
    # We test that split returns string if markdown True
    # Compose text that will parse back to d
    text = splitter.convert(d)
    result = splitter.split(text, markdown=True)
    assert isinstance(result, str)
    # The converted markdown should contain both languages codes and texts
    for lang in d.keys():
        assert lang in result
        assert d[lang].strip() in result

def test_split_unknown_language_exception_logs(monkeypatch, caplog):
    random.seed(10)
    np.random.seed(10)
    # Arrange chunk that triggers exception in detect_language
    def raise_exc(_):
        raise Exception("fail")
    monkeypatch.setattr(splitter, "detect_language", raise_exc)
    text = "some text without language prefix"

    with caplog.at_level("INFO"):
        result = splitter.split(text, autodetect=True)
        # The text should be assigned to UNKNOWN_LANGUAGE key as fallback
        assert splitter.settings.UNKNOWN_LANGUAGE in result
        # There should be at least one log message about language not detected
        found = any("Language not detected" in rec.message for rec in caplog.records)
        assert found

def test_split_result_order_preserved_for_lang_seq(monkeypatch):
    random.seed(11)
    np.random.seed(11)
    input_text = ".:fr: bonjour.:en: hello.:xx: unknown"

    # patch detect_language that returns xx to raise exception, so unknown entries get logged
    def fake_detect(text):
        if "bonjour" in text:
            return "fr"
        if "hello" in text:
            return "en"
        raise Exception("fail")
    monkeypatch.setattr(splitter, "detect_language", fake_detect)

    result = splitter.split(input_text, sep=".:", ends=[":"])
    # The keys returned must be in the order of appearance
    assert list(result.keys()) == ['fr', 'en', 'xx']

def test_split_with_varied_ends(monkeypatch):
    random.seed(12)
    np.random.seed(12)
    # Test split with custom ends list with multi-character ending
    text = ".:en\r\nHello\r\n.:fr\nBonjour\n"
    monkeypatch.setattr(splitter, "detect_language", lambda txt: "en" if "Hello" in txt else "fr")
    result = splitter.split(text, sep=".:", ends=['\r\n', '\n'], markdown=False)
    assert set(result.keys()).issuperset({'en','fr'})
    assert result['en'].startswith("Hello")
    assert result['fr'].startswith("Bonjour")

def test_split_long_key_and_chunk(monkeypatch):
    random.seed(13)
    np.random.seed(13)
    # Text with language key longer than max_key_length, so should be assigned to UNKNOWN_LANGUAGE
    text = ".:english: Text with long language key"
    monkeypatch.setattr(splitter, "detect_language", lambda s: "xx")
    result = splitter.split(text, max_key_length=2)
    # Should assign the entire chunk to UNKNOWN_LANGUAGE since no valid lang key matched
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    # The resulting text should contain our entire chunk minus separator
    assert "Text with long language key" in result[splitter.settings.UNKNOWN_LANGUAGE]

def test_split_combines_text_for_same_key(monkeypatch):
    random.seed(14)
    np.random.seed(14)
    # Same lang key repeated, chunks should concatenate
    text = ".:en:Hello.:en: World."
    monkeypatch.setattr(splitter, "detect_language", lambda s: "en")
    result = splitter.split(text, sep='.:', ends=[':'])
    # result['en'] should be concatenation of both texts without separators
    assert result['en'] == "Hello World."

def test_split_handles_no_tokens(monkeypatch):
    random.seed(15)
    np.random.seed(15)
    # Text w/o sep token -> autodetect done on whole text, force detect_language return xx
    text = "Plain text without sep"
    monkeypatch.setattr(splitter, "detect_language", lambda s: "xx")
    result = splitter.split(text, sep='.:')
    assert isinstance(result, collections.OrderedDict)
    # The only key should be 'xx' with entire text
    assert list(result.keys()) == ['xx']
    assert result['xx'] == text

def test_split_ignores_empty_tokens(monkeypatch):
    random.seed(16)
    np.random.seed(16)
    # Include empty tokens from multiple sep in input
    text = " .:en:Hello.:.:fr:Salut"
    monkeypatch.setattr(splitter, "detect_language", lambda s: "en" if "Hello" in s else "fr")
    result = splitter.split(text, sep='.:', ends=[':'])
    # Keys both present
    assert set(result.keys()) == {'en', 'fr'}
    # Texts present and no empty concatenations
    assert "Hello" in result['en']
    assert "Salut" in result['fr']

def test_split_min_max_key_length(monkeypatch):
    random.seed(17)
    np.random.seed(17)
    # Text with keys below min_key_length are assigned to UNKNOWN_LANGUAGE
    text = ".:e:one.:eng:two three"
    monkeypatch.setattr(splitter, "detect_language", lambda s: "en")
    result = splitter.split(text, sep='.:', ends=[':'], min_key_length=2, max_key_length=3)
    # '.:e:' invalid key length, chunk assigned to UNKNOWN_LANGUAGE
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    # '.:eng:' key ok, should be present in result
    assert "eng" in result or "xx" in result
    # Unknown language key text contains "one"
    assert "one" in result[splitter.settings.UNKNOWN_LANGUAGE]

def test_split_strip_leading_spaces(monkeypatch):
    random.seed(18)
    np.random.seed(18)
    # Leading spaces in chunks are stripped in convert, so test that split returns raw strings with leading spaces
    text = ".:en:   Hello"
    monkeypatch.setattr(splitter, "detect_language", lambda s: "en")
    result = splitter.split(text, sep='.:', ends=[':'])
    # Leading spaces should still be present because convert strips after split, split returns raw data
    assert result['en'].startswith("   ")

def test_split_with_custom_paragraph_separator(monkeypatch):
    random.seed(19)
    np.random.seed(19)
    # Using custom paragraph separator to split autodetect paragraphs
    par1 = "Hola mundo"
    par2 = "Bonjour le monde"
    input_text = par1 + "<sep>" + par2
    def fake_detect_language(text):
        if "Bonjour" in text:
            return "fr"
        else:
            return "es"
    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)
    result = splitter.split(input_text, pargraph_sep="<sep>", autodetect=True)
    # Keys are 'es' and 'fr'
    assert set(result.keys()) == {'es','fr'}
    assert result['es'] == par1 + "<sep>"
    assert result['fr'] == par2
