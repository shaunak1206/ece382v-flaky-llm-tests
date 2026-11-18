import pytest
import collections
import langsplit.splitter as splitter

def test_split_empty_returns_empty():
    assert splitter.split('') == ''

def test_split_single_sep_key_and_text():
    s = splitter.split('.:enhello')
    expected = collections.OrderedDict([('en', 'hello')])
    assert s == expected

def test_split_multiple_languages_basic():
    text = '.:en:Hello.:fr:Bonjour.:es:Hóla'
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)
    assert result['en'].lstrip().startswith('Hello')
    assert result['fr'].lstrip().startswith('Bonjour')
    assert result['es'].lstrip().startswith('Hóla')

def test_split_unknown_language_key_length_too_short():
    # If key length is less than min_key_length it goes to unknown language
    text = '.:eHello'
    res = splitter.split(text, min_key_length=2)
    assert splitter.settings.UNKNOWN_LANGUAGE in res
    assert res[splitter.settings.UNKNOWN_LANGUAGE].startswith('Hello')

def test_split_unknown_language_key_length_too_long():
    # If key length is longer than max_key_length it goes to unknown language
    text = '.:engHello'
    res = splitter.split(text, max_key_length=2)
    assert splitter.settings.UNKNOWN_LANGUAGE in res
    assert res[splitter.settings.UNKNOWN_LANGUAGE].startswith('Hello')

def test_split_detect_language_autodetect_false():
    # chunk assigned to unknown language without autodetect tries to assign chunk as is, skipping detection
    text = '.:Hello world'
    result = splitter.split(text, autodetect=False)
    # All placed under unknown language except separator
    key = splitter.settings.UNKNOWN_LANGUAGE
    assert key in result
    assert 'Hello world' in result[key]

def test_split_detect_language_autodetect_true(monkeypatch):
    # Patch detect_language to return 'en'
    def fake_detect(text):
        return 'en'
    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    text = '.:Hello world\n\n.:Bonjour\n\n.:Hola'
    r = splitter.split(text, autodetect=True)
    # Should include detected languages
    assert 'en' in r
    assert r['en'].startswith('Hello')
    assert 'xx' not in r  # no unknown because detection succeeded

def test_split_detect_language_with_multiple_paragraphs(monkeypatch):
    # Monkeypatch detect_language to distinguish paragraphs by keywords
    def fake_detect(text):
        if 'Hola' in text:
            return 'es'
        if 'Bonjour' in text:
            return 'fr'
        return 'en'
    monkeypatch.setattr(splitter, 'detect_language', fake_detect)
    text = '.:Hello world\n\nBonjour\n\nHola'
    r = splitter.split(text, autodetect=True)
    assert r['en'].startswith('Hello world')
    assert 'fr' in r and 'Bonjour' in r['fr']
    assert 'es' in r and 'Hola' in r['es']

def test_split_returns_ordered_dict_preserves_order():
    text = '.:fr:Bonjour\n.:en:Hello\n.:es:Hóla'
    result = splitter.split(text)
    keys = list(result.keys())
    assert keys == ['fr', 'en', 'es']

def test_split_markdown_true_returns_string():
    text = '.:en:Hello\n.:fr:Bonjour\n.:es:Hóla'
    res = splitter.split(text, markdown=True)
    assert isinstance(res, str)
    # Should contain separators and languages
    assert '.:en' in res
    assert '.:fr' in res
    assert '.:es' in res

def test_convert_dictionary_and_title_false():
    d = collections.OrderedDict([('en', 'Hello'), ('fr', 'Bonjour')])
    out = splitter.convert(d, title=False)
    # Ends[0] is '\n', starter separator is '.:'
    # The string should combine languages with sep, end and text
    assert out.startswith('en\nHello') or 'en' in out
    assert 'fr' in out

def test_convert_dictionary_and_title_true():
    d = collections.OrderedDict([('en', 'Hello'), ('fr', 'Bonjour')])
    out = splitter.convert(d, title=True)
    # ends[-1] is ':'
    assert ':en:' not in out  # Actually format is '{sep}{lang}{end}{text}' and end = ends[-1]
    assert out.count(':') >= 2  # should have colons as ends

def test_convert_non_dict_returns_input():
    non_dict = 'a string'
    assert splitter.convert(non_dict) == non_dict

def test_detect_language_short_and_long_codes(monkeypatch):
    # Patch langdetect.detect to return both short and long forms
    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'zh-cn')
    assert splitter.detect_language('some text') == 'cn'

    monkeypatch.setattr(splitter.langdetect, 'detect', lambda text: 'fr')
    assert splitter.detect_language('some text') == 'fr'

def test_split_handles_unusual_end_symbols():
    s = '.:en\rHello.:fr\nBonjour'
    # Force ends to a subset including '\r' and '\n'
    ends = ['\r', '\n', ':']
    result = splitter.split(s, ends=ends)
    assert 'en' in result or 'fr' in result

def test_split_with_text_without_sep_and_shorter_than_sep_plus_key(monkeypatch):
    text = '.:e'
    # This is shorter than sep + max_key_length + min end length
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)

def test_split_with_duplicate_language_keys_combines_text():
    text = '.:en:Hello .:en: World'
    result = splitter.split(text)
    assert 'en' in result
    assert 'Hello' in result['en']
    assert 'World' in result['en']

def test_split_with_text_starting_with_sep_and_length_equal_edge(monkeypatch):
    # text starts with sep + 2-char key + minimum end length
    min_end_len = min(len(e) for e in splitter.settings.ENDS)
    text = splitter.settings.SEP + 'en' + splitter.settings.ENDS[0]
    result = splitter.split(text)
    assert isinstance(result, collections.OrderedDict)

def test_split_markdown_and_title_combined():
    text = '.:en:Hello\n.:fr:Bonjour'
    res = splitter.split(text, markdown=True, title=True)
    # result should be a str and contain colon ends as per title=True
    assert isinstance(res, str)
    assert ':en:' in res or '.:en:' in res
    assert ':fr:' in res or '.:fr:' in res

def test_split_with_nonstandard_sep_and_ends():
    sep = '-->'
    ends = [';', '|']
    text = '-->en;Hello-->fr|Bonjour'
    result = splitter.split(text, sep=sep, ends=ends)
    assert 'en' in result
    assert 'fr' in result
    assert 'Hello' in result['en']
    assert 'Bonjour' in result['fr']
