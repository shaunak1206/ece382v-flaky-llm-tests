import random
import langsplit.splitter as splitter
import pytest

# Seed random for every test function at the start, per requirements
def seed_random():
    random.seed(12345)

def test_detect_language_returns_known_code_or_shorter():
    seed_random()
    # inputs with langdetect known languages, force known language codes to be checked
    # we test basic properties: output length and from SHORTER_CODES mapping or truncated
    code = splitter.detect_language("This is a test")
    # output length is max 2
    assert len(code) <= 2
    # if code is in SHORTER_CODES keys longer than two letters, code shorter version must appear
    if '_' in code or len(code) > 2:
        # must be in SHORTER_CODES values or has length max_length
        assert code in splitter.SHORTER_CODES.values() or len(code) <= 2

def test_convert_dict_converts_and_strips_correctly():
    seed_random()
    input_dict = {'en': ' hello', 'fr': ' world'}
    sep = '.:'
    ends = ['\n', '\r', '\r\n', ':']
    converted = splitter.convert(input_dict, sep=sep, ends=ends, title=False)
    # should start with 'en' or like 'en\n' separator and contain languages keys and stripped text
    assert all(lang in converted for lang in input_dict.keys())
    assert all(input_dict[lang].lstrip() in converted for lang in input_dict)
    # Check starts properly: converted should not start with ends elements doubled
    for e in ends:
        assert not converted.startswith(e * 2)

def test_convert_not_dict_returns_input_unchanged():
    seed_random()
    s = "some string"
    assert splitter.convert(s) == s
    # also test with list
    lst = ['a', 'b']
    assert splitter.convert(lst) == lst

def test_split_empty_text_returns_text():
    seed_random()
    # input empty string returns empty string
    assert splitter.split('') == ''
    # input None returns None
    assert splitter.split(None) is None

def test_split_single_line_short_text_with_sep_prefix_returns_ordereddict():
    seed_random()
    sep = splitter.settings.SEP
    short_text = f"{sep}en:mytext"
    # length shorter than len(sep)+max_key_length+min([len(e) for e in ends])
    result = splitter.split(short_text)
    assert isinstance(result, (dict,list))
    # we expect OrderedDict with key 'en' starting the string after sep
    # keys are string of max_key_length length, so 'en'
    keys = list(result.keys())
    assert len(keys) == 1
    assert keys[0] == short_text[len(sep):len(sep)+2]
    # The value should be whatever after key + max_key_length
    expected_value = short_text[len(sep)+2:]
    assert result[keys[0]] == expected_value

def test_split_separates_and_detects_languages_with_autodetect(monkeypatch):
    seed_random()
    # monkeypatch detect_language to avoid dependency on langdetect and randomness
    called_langs = []
    def fake_detect_language(text, max_length=2):
        called_langs.append(text)
        return 'en'
    monkeypatch.setattr(splitter, 'detect_language', fake_detect_language)

    text = "This is paragraph one.\n\nThis is paragraph two."
    sep = splitter.settings.SEP

    # Make text have no explicit language keys, so chunk processed by autodetect
    input_text = sep + text

    result = splitter.split(input_text, autodetect=True, pargraph_sep='\n\n')

    # It should call fake_detect_language twice (two paragraphs)
    assert len(called_langs) == 2
    # Result keys should contain 'en'
    assert 'en' in result
    # The combined text should be original text (joined with '\n\n')
    assert text in result['en']

def test_split_separates_and_assigns_unknown_language_when_detection_raises(monkeypatch):
    seed_random()
    def fake_detect_language(text, max_length=2):
        raise Exception("fail")
    monkeypatch.setattr(splitter, 'detect_language', fake_detect_language)

    text = "Unknown language text.\n\nAnother paragraph."

    sep = splitter.settings.SEP
    input_text = sep + text

    result = splitter.split(input_text, autodetect=True, pargraph_sep='\n\n')

    # Should contain the UNKNOWN_LANGUAGE key from settings
    unk = splitter.settings.UNKNOWN_LANGUAGE
    assert unk in result
    # The text should be preserved in UNKNOWN_LANGUAGE
    assert text in result[unk]

def test_split_with_markdown_true_invokes_convert(monkeypatch):
    seed_random()
    called = {}
    def fake_convert(LanguagesDict, sep, ends, title):
        called['called'] = True
        # return joined string of keys as test output
        return ','.join(LanguagesDict.keys())
    monkeypatch.setattr(splitter, 'convert', fake_convert)

    text = splitter.settings.SEP + 'en:hello world'

    output = splitter.split(text, markdown=True)
    assert called.get('called') == True
    # output is a string with keys joined
    assert 'en' in output

def test_split_with_title_converts_to_last_element_of_ends(monkeypatch):
    seed_random()
    # Use text with multiple keys, title True means uses ends[-1]
    d = {'en': ' hello', 'fr': ' world'}
    converted = splitter.convert(d, title=True)
    # should end with ends[-1] used in all sep patterns (we test convert directly)
    for end in splitter.settings.ENDS[:-1]:
        assert end not in converted[:len(end)*2]

def test_split_with_autodetect_false_assigns_to_unknown_language(monkeypatch):
    seed_random()
    text = "sometext"

    sep = splitter.settings.SEP
    input_text = sep + text

    result = splitter.split(input_text, autodetect=False)

    unk = splitter.settings.UNKNOWN_LANGUAGE
    # Should contain UNKNOWN_LANGUAGE key
    assert unk in result
    # Text without the first len(name)+1 characters because of this line: result[name] += chunk[len(name)+1:]
    # We check that text without unknown lang prefix is contained in value
    assert result[unk] == text[len(unk)+1:]

def test_split_does_not_duplicate_language_keys_order(monkeypatch):
    seed_random()
    # Fake detect_language to sequence over different codes
    detected = ['en', 'fr', 'en', 'xx']
    call_index = 0
    def fake_detect_language(text, max_length=2):
        nonlocal call_index
        val = detected[call_index % len(detected)]
        call_index += 1
        return val
    monkeypatch.setattr(splitter, 'detect_language', fake_detect_language)

    sep = splitter.settings.SEP
    paragraphs = ["para1", "para2", "para3", "para4"]
    input_text = sep + ("\n\n".join(paragraphs))

    result = splitter.split(input_text, autodetect=True, pargraph_sep='\n\n')

    # lang_seq should not duplicate_keys - keys should be in order of first appearance of detected languages
    keys = list(result.keys())
    # detected sequence is ['en','fr','en','xx'] but keys contain each unique lang once in order
    assert keys == ['en','fr','xx']

def test_split_handles_no_sep_input_returns_input(monkeypatch):
    seed_random()
    text = "Just some random text with no separator."
    result = splitter.split(text)
    # if no sep found, should autodetect paragraph and assign key accordingly or return string
    # actually, there is no separator, so it should trigger autodetect
    assert isinstance(result, dict)
    # Keys should contain either known lang or UNKNOWN_LANGUAGE
    keys = list(result.keys())
    known_or_unknown = all(k in splitter.LANGUAGE_CODES or k == splitter.settings.UNKNOWN_LANGUAGE or len(k)==2 for k in keys)
    assert known_or_unknown

def test_split_with_custom_sep_and_ends(monkeypatch):
    seed_random()
    # Use fake detect_language to always return 'xx'
    monkeypatch.setattr(splitter, 'detect_language', lambda t, max_length=2: 'xx')
    text = "==en*hello==fr*world"
    # Using custom sep and ends
    result = splitter.split(text, sep='==', ends=['*'])
    # keys expected to be 'en' and 'fr'
    assert 'en' in result
    assert 'fr' in result
    # values should contain hello and world respectively
    assert 'hello' in result['en']
    assert 'world' in result['fr']

def test_convert_strips_leading_seps_and_ends(monkeypatch):
    seed_random()
    data = {'en': 'hello', 'fr': ' world'}
    sep = '.:'
    ends = ['\n', '\r', '\r\n', ':']
    out = splitter.convert(data, sep=sep, ends=ends, title=False)
    # output should not begin with any ends repeated twice
    for e in ends:
        assert not out.startswith(e * 2)
    # output should contain both language keys
    assert all(key in out for key in data.keys())
    # Values are stripped of leading whitespace
    assert ' world' not in out
    assert 'world' in out

def test_split_text_with_multiple_explicit_lang_keys(monkeypatch):
    seed_random()
    text = ".:en:Hello.:fr:Bonjour"
    result = splitter.split(text)
    # Should parse both languages separately and preserve values
    assert 'en' in result
    assert 'fr' in result
    assert 'Hello' in result['en']
    assert 'Bonjour' in result['fr']

def test_split_handles_tokens_without_end_symbol(monkeypatch):
    seed_random()
    # token with no end symbol after key (simulate edge case)
    text = ".:enHello"
    result = splitter.split(text)
    # In this case, key might be unknown because no separator found
    # So it should autodetect or assign unknown
    assert isinstance(result, dict) or isinstance(result, str)
    # Result keys possible to be UNKNOWN_LANGUAGE
    # If dict, UNKNOWN_LANGUAGE key present or 'en' present
    if isinstance(result, dict):
        keys = list(result.keys())
        assert any(k == 'en' or k == splitter.settings.UNKNOWN_LANGUAGE for k in keys)

def test_split_preserves_order_of_languages(monkeypatch):
    seed_random()
    # Fake detect_language to return alternated languages
    langs = ['fr', 'en', 'es']
    i = 0
    def fake_detect_language(text, max_length=2):
        nonlocal i
        val = langs[i % len(langs)]
        i += 1
        return val
    monkeypatch.setattr(splitter, 'detect_language', fake_detect_language)

    paragraphs = ["p1", "p2", "p3", "p4"]
    sep = splitter.settings.SEP
    text = sep + ("\n\n".join(paragraphs))

    result = splitter.split(text)

    # keys in result should be languages in order first detected: fr, en, es
    keys = list(result.keys())
    assert keys == ['fr', 'en', 'es']

def test_split_with_title_true_passes_to_convert(monkeypatch):
    seed_random()
    # patch convert to record arguments
    called = {}
    def fake_convert(LanguagesDict, sep, ends, title):
        called['title'] = title
        return "converted"
    monkeypatch.setattr(splitter, 'convert', fake_convert)

    text = splitter.settings.SEP + "en:hello"
    output = splitter.split(text, markdown=True, title=True)
    assert called.get('title') is True
    assert output == "converted"
