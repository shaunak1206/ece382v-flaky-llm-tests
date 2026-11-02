import collections
import pytest
from langsplit import splitter

def test_detect_language_known_language():
    # 'en' is a known short language code with length 2
    lang = splitter.detect_language("This is a test")
    assert isinstance(lang, str)
    assert len(lang) == 2
    assert lang == "en"

def test_detect_language_shorter_code_conversion():
    # 'zh-cn' shortens to 'cn' (2 letters)
    code = 'zh-cn'
    # forcibly test the SHORTER_CODES map effect
    short_code = splitter.SHORTER_CODES.get(code)
    assert short_code == 'cn'
    # detecting language 'zh-cn' manually for testing detect_language logic
    # We simulate detect returning 'zh-cn'
    # We'll monkeypatch langdetect.detect
    original_detect = splitter.langdetect.detect
    splitter.langdetect.detect = lambda _: 'zh-cn'
    try:
        lang = splitter.detect_language("测试")
        assert lang == 'cn'
    finally:
        splitter.langdetect.detect = original_detect

def test_detect_language_default_no_conversion():
    # If langdetect returns a code <= max_length, it returns as is
    original_detect = splitter.langdetect.detect
    splitter.langdetect.detect = lambda _: 'de'
    try:
        lang = splitter.detect_language("Dies ist ein Test", max_length=2)
        assert lang == 'de'
    finally:
        splitter.langdetect.detect = original_detect

def test_convert_returns_str_for_non_dict_input():
    assert splitter.convert("just a string") == "just a string"
    assert splitter.convert(123) == 123

def test_convert_basic_dict_without_title():
    d = {'en': ' Hello', 'de': ' Hallo'}
    result = splitter.convert(d, sep='.:', ends=['\n', ':'], title=False)
    # Each language block ends with ends[0], which is '\n' here.
    # Result should include the sep + lang + end + text.lstrip()
    expected = ''
    for k, v in d.items():
        expected += f".:{k}\n{v.lstrip()}"
    # The function strips leading sep if present
    expected = expected.lstrip('.:')
    assert result == expected.strip()

def test_convert_with_title_true_ends_last_used():
    d = {'en': 'text here'}
    result = splitter.convert(d, sep='.:', ends=['\n', ':'], title=True)
    # ends[-1] = ':' should be used
    expected = '.:en:text here'
    assert result.strip() == expected.lstrip('.:')

def test_convert_removes_leading_end_codes():
    # Setup ends to include '.:', so it is at beginning
    d = {'en': 'text'}
    # forcibly produce leading ends (2 chars)
    # The function strips while text_md[:2] in ends
    # Let's create a dict that will produce leading ends to strip
    output = splitter.convert(d, sep='.:', ends=['.:', ':'])
    # It should not start with ':.', so check stripped properly
    assert not output.startswith('.:') and not output.startswith(':')

def test_split_empty_input_returns_empty():
    res = splitter.split("")
    assert res == ""

def test_split_single_line_sep_starts_short_text():
    # Text like '.:en' but shorter than threshold
    text = ".:en"
    # Since len(text.split('\n')) == 1 and text.startswith(sep) and len(text) < len(sep) + max_key_length + min(len(end) for end in ends)
    result = splitter.split(text, sep='.:', ends=['\n', ':'])
    # Result must be an OrderedDict with key 'en' and empty text
    assert isinstance(result, collections.OrderedDict)
    assert 'en' in result
    assert result['en'] == ''

def test_split_assigns_unknown_language_when_key_len_out_of_range():
    # Compose text with a key shorter than min_key_length
    text = ".:e:This is text"  # 'e' is length 1 < min_key_length=2
    result = splitter.split(text, sep='.:', min_key_length=2, max_key_length=2)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    # The chunk should be assigned to UNKNOWN_LANGUAGE
    assert result[splitter.settings.UNKNOWN_LANGUAGE].endswith("This is text")

def test_split_assigns_known_language_within_key_length():
    # Compose text with language key length 2 (default min_key_length=2/max_key_length=2)
    text = ".:en:Hello world"
    result = splitter.split(text, sep='.:', min_key_length=2, max_key_length=2)
    assert 'en' in result
    assert result['en'] == 'Hello world'

def test_split_multiple_languages_and_order_preserved():
    txt = ".:en:Hello.:de:Hallo.:fr:Bonjour"
    result = splitter.split(txt)
    # Result must be OrderedDict preserving order en, de, fr
    expected_keys = ['en', 'de', 'fr']
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == expected_keys
    assert result['en'] == 'Hello'
    assert result['de'] == 'Hallo'
    assert result['fr'] == 'Bonjour'

def test_split_text_without_sep_autodetect_true_detects_language(monkeypatch):
    # Text with no explicit language key, triggers autodetect = True by default
    text = "This is in English.\n\nDas ist auf Deutsch."
    # Patch detect_language to return fixed values per paragraph
    def fake_detect_language(t):
        if 'English' in t:
            return 'en'
        elif 'Deutsch' in t:
            return 'de'
        else:
            return splitter.settings.UNKNOWN_LANGUAGE
    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)
    result = splitter.split(text, sep='.:')
    assert isinstance(result, collections.OrderedDict)
    # Both 'en' and 'de' keys should appear with the correct paragraphs
    assert 'en' in result and 'de' in result
    # Paragraphs are joined with \n\n
    assert 'English' in result['en']
    assert 'Deutsch' in result['de']

def test_split_text_without_sep_autodetect_false_assigns_unknown(monkeypatch):
    # No sep keys, autodetect=False must assign all to unknown language
    text = "Some random text without language key."
    result = splitter.split(text, sep='.:', autodetect=False)
    assert isinstance(result, collections.OrderedDict)
    # The result should have the unknown language key only
    assert list(result.keys()) == [splitter.settings.UNKNOWN_LANGUAGE]
    assert result[splitter.settings.UNKNOWN_LANGUAGE].strip() == text.strip()

def test_split_markdown_true_returns_string(monkeypatch):
    # text with explicit language keys, returns string (markdown=true)
    text = ".:en:Hello\n\n.:fr:Bonjour"
    result = splitter.split(text, sep='.:', markdown=True)
    assert isinstance(result, str)
    # The string should contain the separators and language codes
    assert '.:en' in result
    assert '.:fr' in result

def test_split_markdown_title_true_ends_is_colon(monkeypatch):
    text = ".:en:Hello\n\n.:fr:Bonjour"
    # markdown and title are true
    result = splitter.split(text, sep='.:', markdown=True, title=True)
    assert isinstance(result, str)
    # Because title=True, the ends used by convert will be ends[-1] = ':'
    assert all(s in result for s in ['.:en:', '.:fr:'])
    # Text contents appear after the lang code and colon
    assert 'Hello' in result and 'Bonjour' in result

def test_split_with_paragraph_separator_and_autodetect(monkeypatch):
    # Use paragraph separator '\n\n'
    text = "Hello.\n\nHallo.\n\nBonjour."
    # patch detect_language to return based on keyword
    def detect_fake(t):
        if 'Hello' in t:
            return 'en'
        if 'Hallo' in t:
            return 'de'
        if 'Bonjour' in t:
            return 'fr'
        return splitter.settings.UNKNOWN_LANGUAGE
    monkeypatch.setattr(splitter, 'detect_language', detect_fake)

    result = splitter.split(text, autodetect=True, pargraph_sep='\n\n')
    assert isinstance(result, collections.OrderedDict)
    # All languages detected as keys
    assert set(result.keys()) == {'en', 'de', 'fr'}
    # Each paragraph is assigned
    assert result['en'].startswith('Hello')
    assert result['de'].startswith('Hallo')
    assert result['fr'].startswith('Bonjour')
    # Paragraph separator included between paragraphs if multiple paragraphs per language
    for lang_text in result.values():
        # Paragraph separator only where multiple paragraphs per language
        # Since one paragraph each, no paragraph_sep in text
        assert '\n\n' not in lang_text

def test_split_with_exception_in_detect_logging(monkeypatch, caplog):
    # Patch detect_language to raise exception for some text to test logging fallback
    def detect_raises(t):
        raise RuntimeError("failed detection")
    monkeypatch.setattr(splitter, "detect_language", detect_raises)
    caplog.set_level("INFO", logger="langsplit.splitter")
    text = "No lang info here"

    result = splitter.split(text, autodetect=True)
    # All text assigned to unknown language as fallback
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert result[splitter.settings.UNKNOWN_LANGUAGE].strip() == text.strip()
    # Logging happened with info level
    assert any("Language not detected" in rec.message for rec in caplog.records)

def test_split_multiple_tokens_with_same_lang_combined():
    # Compose text with multiple segments with the same language key
    text = ".:en:Hello.:en: World.:de:Hallo"
    result = splitter.split(text)
    assert 'en' in result and 'de' in result
    # The texts with 'en' should be concatenated in order
    assert result['en'] == "Hello World."
    assert result['de'] == "Hallo"

def test_split_unknown_language_key_not_detected(apply_monkeypatch=False):
    # Text with a language key not in known range - should assign to unknown
    key = 'zzzz'  # length 4 exceeds max_key_length default of 2
    text = f".:{key}:This text"
    result = splitter.split(text)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    # The key zz is not recognized, so whole text assigned to unknown lang
    unknown_text = result[splitter.settings.UNKNOWN_LANGUAGE]
    assert 'This text' in unknown_text

def test_split_leading_sep_skipped_and_order_preserved():
    # Text starting with multiple sep
    text = ".:.:" + ".:en:Hello" + ".:de:Hallo"
    result = splitter.split(text)
    # Should skip empty tokens and preserve order
    assert list(result.keys()) == ['en', 'de']
    assert result['en'] == 'Hello'
    assert result['de'] == 'Hallo'
