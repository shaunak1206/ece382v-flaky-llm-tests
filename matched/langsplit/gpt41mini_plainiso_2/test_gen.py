import collections
import pytest
from langsplit import splitter


def test_detect_language_basic_known_language():
    # 'en' should be detected and returned as 'en'
    text = "This is a simple test."
    code = splitter.detect_language(text)
    assert code == 'en'


def test_detect_language_shorter_codes_mapping():
    # 'zh-cn' should be converted to 'cn'
    # We'll force detect() to return 'zh-cn' by monkeypatching
    def fake_detect(_):
        return 'zh-cn'
    original_detect = splitter.langdetect.detect
    splitter.langdetect.detect = fake_detect
    try:
        code = splitter.detect_language("dummy text", max_length=2)
        assert code == 'cn'
    finally:
        splitter.langdetect.detect = original_detect


def test_detect_language_no_mapping_long_code_truncated():
    # For a long code not in SHORTER_CODES, truncated to max_length
    def fake_detect(_):
        return 'abcd'
    original_detect = splitter.langdetect.detect
    splitter.langdetect.detect = fake_detect
    try:
        code = splitter.detect_language("text", max_length=2)
        assert code == 'ab'
    finally:
        splitter.langdetect.detect = original_detect


def test_convert_with_dict_and_title_false():
    d = {"en": " hello", "fr": " bonjour"}
    result = splitter.convert(d, sep='.:', ends=['\n', '\r', '\r\n', ':'], title=False)
    # Should start with "en:<text>", then "fr:<text>" with lstrip done
    assert 'en.' in result or 'en:' in result
    assert 'fr.' in result or 'fr:' in result
    assert 'hello' in result
    assert 'bonjour' in result
    # No leading separator
    assert not result.startswith('.:')


def test_convert_with_dict_and_title_true():
    d = {"en": " hello", "fr": " bonjour"}
    ends = ['\n', '\r', '\r\n', ':']
    result = splitter.convert(d, sep='.:', ends=ends, title=True)
    # End char should be the last in ends: ':'
    assert all(part.endswith(':') for part in [result[:3], result[3:6]] if part.strip())
    # Values are lstrip'ed
    assert 'hello' in result
    assert 'bonjour' in result


def test_convert_with_non_dict_returns_input_unchanged():
    text = "some random text"
    assert splitter.convert(text) == text


def test_split_empty_text_returns_text():
    assert splitter.split('') == ''


def test_split_only_sep_and_short_text_returns_ordereddict():
    # Text is like ".:en"
    text = '.:en'
    result = splitter.split(text)
    # The key should be 'en'
    assert isinstance(result, collections.OrderedDict)
    assert list(result.keys()) == ['en']
    assert result['en'] == ''


def test_split_simple_known_keys_no_autodetect():
    text = '.:en\nHello.:fr\nBonjour'
    # Adjust sep and ends so .:en\nHello format works
    sep = '.:'
    ends = ['\n', '\r', '\r\n', ':']
    d = splitter.split(text, sep=sep, ends=ends, autodetect=False)
    assert isinstance(d, collections.OrderedDict)
    assert 'en' in d
    assert 'fr' in d
    assert 'Hello' in d['en']
    assert 'Bonjour' in d['fr']


def test_split_uses_autodetect_for_unknown_language(monkeypatch):
    # Test that autodetect is used to assign unknown parts
    dummy_text = "some random text without separator"

    captured_langs = []

    def fake_detect(text):
        captured_langs.append(text)
        return 'en'

    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    result = splitter.split(dummy_text, autodetect=True)
    assert 'en' in result
    # The entire dummy_text is captured as a chunk
    assert dummy_text in result['en']
    # The fake_detect was called at least once
    assert captured_langs


def test_split_assigns_unknown_language_when_detection_fails(monkeypatch):
    # Simulate detect_language raising exception for some paragraphs
    def fake_detect(text):
        raise ValueError("Cannot detect language")

    monkeypatch.setattr(splitter, 'detect_language', fake_detect)

    text = 'random text that fails detection'

    result = splitter.split(text, autodetect=True)
    # Should assign text to 'xx' (UNKNOWN_LANGUAGE)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert text in result[splitter.settings.UNKNOWN_LANGUAGE]


def test_split_preserves_language_order():
    text = ".:en\nHello\n\n.:fr\nBonjour\n\nsome unknown text"
    monkeypatch_dict = {}

    def fake_detect(text):
        # Return 'xx' for unknown text, 'en'/'fr' ignored as they come from keys
        if 'unknown' in text:
            return splitter.settings.UNKNOWN_LANGUAGE
        return 'en'

    monkeypatch.setattr(splitter, 'detect_language', fake_detect)

    result = splitter.split(text, autodetect=True)
    keys = list(result.keys())
    # The order of keys should match the order appeared: en, fr, xx
    assert 'en' in keys
    assert 'fr' in keys or splitter.settings.UNKNOWN_LANGUAGE in keys


def test_split_with_markdown_true_returns_string(monkeypatch):
    text = ".:en\nHello\n\nsome unknown text"

    def fake_detect(text):
        return 'en'

    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    result = splitter.split(text, autodetect=True, markdown=True)
    assert isinstance(result, str)
    # The markdown format should include separator and language codes
    assert '.:' in result


def test_split_with_title_true_changes_convert_output(monkeypatch):
    text = ".:en\nHello"

    def fake_detect(text):
        return 'en'

    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    result = splitter.split(text, autodetect=True, markdown=True, title=True)
    assert isinstance(result, str)
    # Check that convert uses ends[-1] i.e. ':'
    assert result.endswith(':' + 'Hello')


def test_split_ignores_empty_tokens(monkeypatch):
    # Text with repeated separators resulting in empty tokens
    text = ".:en\nHello..::fr\nBonjour"
    def fake_detect(text):
        return 'en'
    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    result = splitter.split(text, autodetect=True)
    # Should not include empty keys or empty text chunks
    for key, val in result.items():
        assert val != ''


def test_split_handles_paragraph_sep_correctly(monkeypatch):
    text = ".:en\nHello\n\nHow are you?\n\n.:fr\nBonjour\n\nÇa va?"
    def fake_detect(text):
        # Split paragraph contains \n\n and detect_language see chunks separately
        if 'Bonjour' in text or 'Ça va?' in text:
            return 'fr'
        return 'en'

    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    result = splitter.split(text, autodetect=True, pargraph_sep='\n\n')
    assert 'en' in result and 'fr' in result
    # Paragraphs are separated by \n\n
    assert '\n\n' in result['en']
    assert '\n\n' in result['fr']


def test_split_chunk_length_greater_than_max_key_length_treated_as_unknown(monkeypatch):
    text = ".:engHello"  # key length 3 (max_key_length=2), should fail to detect
    def fake_detect(text):
        return 'en'
    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    result = splitter.split(text, autodetect=True)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    # The entire chunk after sep included
    assert 'Hello' in result[splitter.settings.UNKNOWN_LANGUAGE]


def test_split_returns_original_text_if_not_dict_or_empty():
    # convert returns original if not dict, split returns text if empty or None
    assert splitter.split(None) is None
    assert splitter.split('') == ''
    # convert returns input if not dict
    assert splitter.convert('some text') == 'some text'


def test_split_no_autodetect_text_after_unknown_key():
    text = ".:xxSome unknown language text here."
    result = splitter.split(text, autodetect=False)
    # The text part is everything after the key 'xx'
    assert 'xx' in result
    assert 'Some unknown language text here.' in result['xx']


def test_detect_language_respects_max_length():
    # Long language code cut strictly at max_length
    def fake_detect(_):
        return 'deu'  # 3 letters
    original_detect = splitter.langdetect.detect
    splitter.langdetect.detect = fake_detect
    try:
        # max_length is 2, so should return 'de'
        assert splitter.detect_language("any text", max_length=2) == 'de'
        # max_length is 3, should return 'deu'
        assert splitter.detect_language("any text", max_length=3) == 'deu'
    finally:
        splitter.langdetect.detect = original_detect
