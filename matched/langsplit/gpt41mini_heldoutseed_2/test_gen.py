import pytest
import collections
from langsplit import splitter

def test_split_empty_string_returns_empty():
    assert splitter.split("") == ""

def test_split_simple_lang_key_and_text():
    text = ".:en:Hello"
    expected = collections.OrderedDict([("en", "Hello")])
    assert splitter.split(text) == expected

def test_split_text_without_sep_returns_unknown_key_autodetect(monkeypatch):
    # monkeypatch detect_language to return 'en'
    monkeypatch.setattr(splitter, "detect_language", lambda text: "en")
    text = "Hello world"
    expected = collections.OrderedDict([("en", "Hello world")])
    assert splitter.split(text) == expected

def test_split_text_with_multiple_languages_and_sep():
    text = ".:en:Hello.:fr:Bonjour"
    expected = collections.OrderedDict([
        ("en", "Hello"),
        ("fr", "Bonjour"),
    ])
    result = splitter.split(text)
    assert result == expected

def test_split_text_with_unknown_language_key_len_out_of_bounds(monkeypatch):
    # If key length is smaller or larger than min_key_length or max_key_length,
    # text goes to unknown language key (xx)
    text = ".:e:Hello"  # key length = 1 < min_key_length (2)
    monkeypatch.setattr(splitter, "detect_language", lambda text: "en")
    result = splitter.split(text)
    # Should detect language since autodetect True
    expected = collections.OrderedDict([('en', 'Hello')])
    assert result == expected

def test_split_text_multiple_paragraphs_autodetect(monkeypatch):
    # Provide a chunk with paragraphs separated by '\n\n'
    paragraphs = ["Hello there.", "How are you?"]
    text = ".:xx:" + "\n\n".join(paragraphs)
    called = []
    def fake_detect_language(paragraph):
        called.append(paragraph)
        return "en"
    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)
    result = splitter.split(text)
    assert "en" in result
    assert result["en"] == "\n\n".join(paragraphs)
    # detect_language called for each paragraph
    assert called == paragraphs

def test_split_with_markdown_true_returns_string():
    text = ".:en:Hello\n\n.:fr:Bonjour"
    result = splitter.split(text, markdown=True)
    # result should be a string that includes .:en and .:fr
    assert isinstance(result, str)
    assert ".:en" in result
    assert ".:fr" in result
    assert "Hello" in result
    assert "Bonjour" in result

def test_split_with_title_true_converts_ends_correctly():
    d = collections.OrderedDict([("en", "Hello"), ("fr", "Bonjour")])
    result = splitter.convert(d, title=True)
    # ends used in convert for title=True is ends[-1], which is ':'
    # the keys should be separated by '.:en:' etc.
    assert result.startswith(".:en:")
    assert ":Bonjour" in result

def test_convert_with_non_dict_returns_same_object():
    val = "Just a string"
    assert splitter.convert(val) == val

def test_convert_removes_leading_sep_and_ends_from_result():
    d = collections.OrderedDict([("en", " Hello"), ("fr", " Bonjour")])
    result = splitter.convert(d)
    # Should not start with any ends and strip leading whitespace from texts
    for end in splitter.settings.ENDS:
        assert not result.startswith(end*2)
    assert "Hello" in result
    assert "Bonjour" in result

def test_detect_language_returns_shorter_code_if_needed(monkeypatch):
    # Provide detect returning zh-cn, expecting cn shortened code
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "zh-cn")
    code = splitter.detect_language("some text", max_length=2)
    assert code == "cn"
    # For code shorter than max_length, returns code as is
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "fr")
    code2 = splitter.detect_language("another text", max_length=2)
    assert code2 == "fr"

def test_detect_language_returns_code_trimmed_to_max_length(monkeypatch):
    monkeypatch.setattr(splitter.langdetect, "detect", lambda text: "eng")
    code = splitter.detect_language("hello", max_length=2)
    assert code == "en"

def test_split_handles_token_with_end_symbol_in_key(monkeypatch):
    # .:en:Hello. Should split "en" as lang and "Hello." as text
    text = ".:en:Hello."
    expected = collections.OrderedDict([("en", "Hello.")])
    assert splitter.split(text) == expected

def test_split_token_without_sep_and_length_less_than_sep_key_end(monkeypatch):
    # Text starts with sep but too short to contain key + end symbol (should return dict token[len(sep):])
    text = ".:e"
    expected = collections.OrderedDict([("e", "")])
    assert splitter.split(text) == expected

def test_split_with_autodetect_false_adds_to_unknown_language():
    text = "Hello world"
    result = splitter.split(text, autodetect=False)
    # Since autodetect False and no language, content should be added to key 'xx'
    assert "xx" in result
    assert "Hello world" in result['xx']

def test_split_with_non_ascii_text(monkeypatch):
    text = ".:fr:Bonjour à tous\n\n.:en:Hello everyone"
    result = splitter.split(text)
    assert "fr" in result
    assert "en" in result
    assert "Bonjour" in result["fr"]
    assert "Hello" in result["en"]

def test_split_concatenates_texts_for_same_language():
    text = ".:en:Hello.:en: world"
    expected = collections.OrderedDict([("en", "Hello world")])
    result = splitter.split(text)
    assert result == expected

def test_split_respects_min_max_key_length(monkeypatch):
    # Key length outside min and max key length = 2
    text = ".:eng:Hello"
    monkeypatch.setattr(splitter, "detect_language", lambda text: "en")
    result = splitter.split(text)
    # 'eng' key is length 3, outside max_key_length, so detected
    assert "en" in result
    assert result["en"] == "Hello"

def test_split_handles_multiple_unknown_language_chunks(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda text: "en")
    text = "some random text\n\n\nmore text"
    result = splitter.split(text)
    assert "en" in result
    assert "some random text" in result["en"]
    assert "more text" in result["en"]

def test_split_returns_ordered_dict_with_lang_seq_order(monkeypatch):
    text = ".:fr:Bonjour.:en:Hello"
    result = splitter.split(text)
    # The keys order should be as in text input: first 'fr', then 'en'
    keys = list(result.keys())
    assert keys == ["fr", "en"]

def test_split_handles_token_with_key_and_end_symbol_at_start(monkeypatch):
    # The case where token[:max_key_length+1] includes the end symbol at position within key length boundaries
    text = ".:en:Hello"
    result = splitter.split(text)
    assert "en" in result
    assert result["en"] == "Hello"

def test_split_does_not_fail_on_empty_token_between_sep(monkeypatch):
    text = ".:en:Hello.:.:fr:Bonjour"
    result = splitter.split(text)
    assert "en" in result
    assert "fr" in result
    assert "Hello" in result["en"]
    assert "Bonjour" in result["fr"]

def test_split_with_custom_paragraph_separator(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda text: "en")
    chunk = "para1\npara2"
    text = ".:xx:" + chunk
    result = splitter.split(text, pargraph_sep='\n')
    expected_text = "para1\npara2"
    assert "en" in result
    assert result["en"] == expected_text
