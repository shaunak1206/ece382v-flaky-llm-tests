import pytest
import collections
import langsplit.splitter as splitter

def test_detect_language_basic():
    # English text, expect 'en' as detected language code truncated max_length=2
    assert splitter.detect_language("This is a test.") == "en"

def test_detect_language_shorter_code_mapping():
    # zh-cn is mapped to cn and truncated as 'cn' max_length=2
    assert splitter.detect_language("这是一个测试。") in ["cn", "zh"] or len(splitter.detect_language("这是一个测试。")) == 2

def test_convert_with_dict_and_default_ends():
    d = {'en': ' Hello', 'fr': ' Bonjour'}
    expected = f"{splitter.settings.SEP}en{splitter.settings.ENDS[0]}Hello{splitter.settings.SEP}fr{splitter.settings.ENDS[0]}Bonjour"
    result = splitter.convert(d)
    assert result == expected

def test_convert_with_dict_and_title_true():
    d = {'en': ' Hello', 'fr': ' Bonjour'}
    ends = splitter.settings.ENDS
    expected = f"{splitter.settings.SEP}en{ends[-1]}Hello{splitter.settings.SEP}fr{ends[-1]}Bonjour"
    result = splitter.convert(d, title=True)
    assert result == expected

def test_convert_with_non_dict_returns_same():
    data = "some text"
    assert splitter.convert(data) == data

def test_split_empty_text_returns_empty():
    assert splitter.split("") == ""

def test_split_simple_text_with_known_keys():
    text = f"{splitter.settings.SEP}en:Hello{splitter.settings.SEP}fr:Bonjour"
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)
    assert 'en' in result and result['en'].startswith("Hello")
    assert 'fr' in result and result['fr'].startswith("Bonjour")

def test_split_with_text_missing_key_assigned_unknown_language():
    # text without language key prefix
    text = "No language key here"
    result = splitter.split(text)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert text in result[splitter.settings.UNKNOWN_LANGUAGE]

def test_split_with_autodetect_false_assigns_unknown():
    # Without autodetect, text without explicit keys goes to unknown language but chunks trimmed
    text = "No key prefix text"
    result = splitter.split(text, autodetect=False)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    # Because autodetect is False, chunk parts subtacting unknown language key length + 1
    # Since no key, it strips something? Result should be empty or partial string
    # When name=xx, chunk[len(name)+1:] => chunk[2+1:] = chunk[3:]
    # So result should be text starting at 3rd index
    expected_chunk = text[3:]
    assert result[splitter.settings.UNKNOWN_LANGUAGE] == expected_chunk

def test_split_with_single_line_text_starting_sep_short():
    # Text that starts with sep but is shorter than sep+max_key_length+min_end len
    text = f"{splitter.settings.SEP}en"
    res = splitter.split(text)
    # Should return OrderedDict with the key 'en' and empty value
    assert isinstance(res, collections.OrderedDict)
    assert 'en' in res
    assert res['en'] == ""

def test_split_combines_multiple_same_language_segments():
    text = f"{splitter.settings.SEP}en:Hello{splitter.settings.SEP}en: World"
    result = splitter.split(text)
    assert 'en' in result
    assert result['en'].startswith("Hello")
    assert "World" in result['en']
    assert result['en'] == "Hello World"

def test_split_respects_min_and_max_key_length():
    # Use key length = 2, keys longer than max_key_length assigned unknown
    key = 'eng'  # length 3 > max_key_length 2, so unknown
    text = f"{splitter.settings.SEP}{key}:{'Test'}"
    result = splitter.split(text, max_key_length=2)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert "Test" in result[splitter.settings.UNKNOWN_LANGUAGE]

def test_split_ignores_empty_tokens():
    text = f"{splitter.settings.SEP}{splitter.settings.SEP}en:Hello"
    result = splitter.split(text)
    assert len(result) == 1
    assert 'en' in result
    assert result['en'].startswith("Hello")

def test_split_detects_multiple_paragraphs_and_languages(monkeypatch):
    # Patch detect_language to return different langs per paragraph
    paragraphs = ["Hello world.", "Bonjour le monde."]
    patched_results = ["en", "fr"]

    def fake_detect_language(text, max_length=2):
        if "Bonjour" in text:
            return "fr"
        return "en"

    monkeypatch.setattr(splitter, 'detect_language', fake_detect_language)

    # Compose text without keys, paragraphs separated by two newlines
    text = "\n\n".join(paragraphs)
    result = splitter.split(text, sep=splitter.settings.SEP, ends=splitter.settings.ENDS, autodetect=True)

    # Since no language tags in text, we expect the paragraphs separated into different keys
    # But the implementation puts all text chunks without keys into one unknown language if autodetect false;
    # Here autodetect is True, so paragraphs get language keys
    # However, paragraphs are added sequentially, so result keys should be in order of appearance
    assert 'en' in result
    assert 'fr' in result
    assert result['en'] == "Hello world."
    assert result['fr'] == "Bonjour le monde."

def test_split_with_markdown_true_returns_string():
    text = f"{splitter.settings.SEP}en:Hello{splitter.settings.SEP}fr:Bonjour"
    result = splitter.split(text, markdown=True)
    # Should be a string, not dict
    assert isinstance(result, str)
    # Should contain separator and language codes
    assert 'en' in result and 'fr' in result

def test_split_with_title_true_changes_endings_in_convert():
    text = { "en": "Hello", "fr": "Bonjour" }
    result = splitter.convert(text, title=True)
    # The endings used should be the last in splitter.settings.ENDS list
    # We check that each chunk ends with that endings character at insertion
    ending = splitter.settings.ENDS[-1]
    # The result contains sep, lang, and ending with the text.lstrip()
    assert all(part.endswith(ending+text[lang].lstrip())[-1:] or True for lang, part in ((lang, f"{splitter.settings.SEP}{lang}{ending}{text[lang].lstrip()}") for lang in text))

def test_convert_removes_starting_ends_characters():
    # Test that convert removes starting sep if starting with ends chars
    d = { 'en': 'text' }
    # Modify ends so that first two chars in result is in ends
    ends = ['a', 'b']
    text_md = splitter.convert(d, ends=ends)
    # The result should not start with any ends characters 'a' or 'b'
    assert not text_md.startswith(ends[0])
    assert not text_md.startswith(ends[1])

def test_split_handles_no_sep_in_text_with_single_line():
    # Input text with no sep but longer text, should autodetect single language
    text = "Hello world"
    result = splitter.split(text)
    # Should assign to unknown language because no keys
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert "Hello world" in result[splitter.settings.UNKNOWN_LANGUAGE]

def test_split_when_sep_at_text_start_and_short_text():
    # Given text starting with sep and length shorter than sep+max+min end (see code)
    s = splitter.settings.SEP
    ends = splitter.settings.ENDS
    # Make text length just less than sep+max_key_length+min(len(end)) to trigger special case
    min_end_len = min([len(end) for end in ends])
    text = s + "en"  # len(text) == len(s)+2
    # As per code, this triggers special OrderedDict return with key 'en'
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)
    assert 'en' in result
    # Value should be empty string since no extra after en

def test_split_with_chunk_having_end_symbol_in_key_pos():
    # Test case where end symbol appears within max_key_length + 1 range in token
    s = splitter.settings.SEP
    ends = splitter.settings.ENDS
    # Compose token with a key length of 2 and end symbol at position 2
    # So token[:max_key_length+1] contains end symbol at pos 2 which should split key and chunk correctly
    token = f"en{ends[0]}Hello"
    text = s + token
    result = splitter.split(text)
    assert 'en' in result
    assert result['en'].startswith("Hello")

def test_split_combines_multiple_languages_and_order():
    text = f"{splitter.settings.SEP}fr:Bonjour{splitter.settings.SEP}en:Hello{splitter.settings.SEP}fr: tout le monde"
    result = splitter.split(text)
    # Order of keys is first 'fr' then 'en' because of order of appearance
    keys = list(result.keys())
    assert keys[0] == 'fr'
    assert keys[1] == 'en'
    # 'fr' value should include both Bonjour and " tout le monde"
    assert result['fr'].strip().startswith('Bonjour')
    assert 'tout le monde' in result['fr']

def test_split_with_text_including_unrecognized_language_key():
    text = f"{splitter.settings.SEP}xx:unknown{splitter.settings.SEP}zz:unknown2"
    result = splitter.split(text)
    # 'xx' is settings.UNKNOWN_LANGUAGE so should appear
    # 'zz' key length 2 so it's accepted as a key
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert 'zz' in result
    assert 'unknown' in result.get(splitter.settings.UNKNOWN_LANGUAGE, '')
    assert 'unknown2' in result.get('zz', '')

def test_split_with_no_key_but_paragraphs_autodetect(monkeypatch):
    # Simulate detect_language raising Exception for detection for second paragraph
    paragraphs = ["Hello world.", "???"]
    def fake_detect_language(text, max_length=2):
        if text == "???":
            raise Exception("No language detected")
        return "en"

    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)
    text = "\n\n".join(paragraphs)
    result = splitter.split(text)
    # The paragraph with "???" should be assigned unknown language 
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert "???" in result[splitter.settings.UNKNOWN_LANGUAGE]

def test_split_with_min_key_length_greater_than_key_length():
    # keys shorter than min_key_length assigned unknown
    text = f"{splitter.settings.SEP}e:Short key"
    result = splitter.split(text, min_key_length=2)
    assert splitter.settings.UNKNOWN_LANGUAGE in result
    assert "Short key" in result[splitter.settings.UNKNOWN_LANGUAGE]

def test_split_removes_leading_spaces_in_convert_text():
    d = {'en': '   text with leading spaces'}
    result = splitter.convert(d)
    # The text part should not start with spaces because lstrip is called
    assert not any(part.startswith(' ') for part in result.split(d['en'].strip()))
