import pytest
import random
import numpy as np
import torch

from langsplit import splitter


def test_split_empty_string_returns_empty_string():
    result = splitter.split("")
    assert result == ""


def test_split_single_sep_prefix_short_text():
    # text shorter than sep + max_key_length + min length of ends
    # sep = '.:', max_key_length=2, min length of ends = 1
    text = ".:en"
    # Expect OrderedDict with key "en" and empty string as value
    result = splitter.split(text, max_key_length=2, min_key_length=1)
    # Should be OrderedDict with one key "en" and value ""
    assert isinstance(result, dict)
    assert list(result.keys()) == ["en"]
    assert result["en"] == ""


def test_detect_language_short_and_long_codes():
    # "zh-cn" (long code) converts to "cn"
    assert splitter.detect_language("这是中文") == "cn"
    # "en" stays "en"
    assert splitter.detect_language("This is English") == "en"
    # max_length implemented properly
    # If code is longer than max_length, use SHORTER_CODES if present
    d1 = splitter.detect_language("これは日本語です", max_length=2)
    d2 = splitter.detect_language("これは日本語です", max_length=1)
    # Outputs are strings with length <= max_length
    assert len(d1) <= 2
    assert len(d2) <= 1


def test_convert_with_dict_title_false_default_sep_and_ends():
    d = {"en": "Hello", "fr": "Bonjour"}
    out = splitter.convert(d)
    # Output should include sep and language codes, text values with lstrip
    assert ".:en\nHello" not in out  # lstrip removes leading spaces if any, text no leading spaces here
    # Because sep = '.:', ends = ['\n', '\r', '\r\n', ':']
    # The end used when title=False is ends[0] == '\n'
    expected = ".:en\nHello.:fr\nBonjour"
    # The while loop removes leading ends from start (only if first two characters in ends)
    # Since string starts with '.:', no removal; so output starts with '.:en\n'
    assert out.startswith(".:en\n")
    assert "en" in out
    assert "fr" in out
    assert "Hello" in out
    assert "Bonjour" in out


def test_convert_with_non_dict_returns_input():
    sample = "plain text"
    result = splitter.convert(sample)
    assert result == sample


def test_split_simple_two_languages_explicit_keys():
    # text with two languages separated by sep and ends chars
    # Use a text that matches pattern: ".:en\ntext.:fr\ntexte"
    text = ".:en\nHello world.:fr\nBonjour le monde"
    result = splitter.split(text, autodetect=False)
    # Should be OrderedDict with keys 'en' and 'fr' with corresponding texts
    assert isinstance(result, dict)
    assert "en" in result
    assert "fr" in result
    assert result["en"].startswith("Hello")
    assert result["fr"].startswith("Bonjour")


def test_split_autodetect_different_paragraphs(monkeypatch):
    # This tests autodetect fragments and language detection being called per paragraph.
    # We patch detect_language to produce a predictable output to avoid flakiness
    def fake_detect_language(text, max_length=2):
        if "Hola" in text:
            return "es"
        if "Hello" in text:
            return "en"
        return "xx"

    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)

    text = ".:unknownHello\n\nHola mundo\n\n"  # unknown prefix text with paragraphs separated by \n\n
    result = splitter.split(text, autodetect=True, pargraph_sep="\n\n")
    # Since autodetect True, keys should include es, en, xx but here only en and es expected
    assert isinstance(result, dict)
    keys = list(result.keys())
    # 'es' and 'en' should be detected keys
    assert "es" in keys
    assert "en" in keys
    # Their contents include the respective texts
    assert "Hola" in result["es"]
    assert "Hello" in result["en"]


def test_split_with_markdown_true_returns_string(monkeypatch):
    # Patch detect_language so test is deterministic
    monkeypatch.setattr(splitter, "detect_language", lambda t, max_length=2: "en")

    # Input with unknown keys (should trigger autodetect)
    text = "This is a test with no separator"

    # markdown True returns string, equal to convert() output
    result = splitter.split(text, markdown=True)
    assert isinstance(result, str)
    assert "en" in result or "xx" in result


def test_split_min_max_key_length_filtering(monkeypatch):
    # Provide token whose length exceeds max_key_length, so name remains UNKNOWN_LANGUAGE
    text = ".:english:Hello English"  # key length > max_key_length=2, so fallback
    monkeypatch.setattr(splitter, "detect_language", lambda t, max_length=2: "en")

    result = splitter.split(text, min_key_length=2, max_key_length=2, autodetect=True)
    # Since key length > max_key_length, fallback to unknown language with autodetect
    # The detected language is forced en, so expect to see key "en"
    assert "en" in result


def test_split_language_sequence_order(monkeypatch):
    # Purpose: check that lang_seq preserves order of first appearances
    monkeypatch.setattr(splitter, "detect_language", lambda t, max_length=2: "en")

    # Compose multi-part text with explicit keys and autodetect
    text = ".:fr\nBonjour.:en\nHello"  # Should add fr then en in order

    result = splitter.split(text, autodetect=True)
    assert list(result.keys())[0] == "fr"
    assert list(result.keys())[1] == "en"


def test_split_raises_no_exception_when_langdetect_fails(monkeypatch, caplog):
    # Patch detect_language to raise exception for testing
    def broken_detect_language(text, max_length=2):
        raise RuntimeError("fail detection")

    monkeypatch.setattr(splitter, "detect_language", broken_detect_language)

    text = "Some normal text without prefix"
    with caplog.at_level("INFO"):
        result = splitter.split(text, autodetect=True)
    # The result should have settings.UNKNOWN_LANGUAGE as key because detection failed
    unk = splitter.settings.UNKNOWN_LANGUAGE
    assert unk in result
    # The text should be included in the value
    assert "Some normal text" in result[unk]
    # The logging should include info about the failure
    assert any("Language not detected" in m for m in caplog.messages)


def test_convert_title_true_uses_last_end():
    d = {"en": "hello", "fr": "bonjour"}
    text = splitter.convert(d, title=True)
    # ends last item is ':'
    for k in d.keys():
        # The separator between lang and text ends with colon
        assert f"{k}:" in text


def test_split_with_non_default_sep_and_ends(monkeypatch):
    # Patch detect_language to always return "en"
    monkeypatch.setattr(splitter, "detect_language", lambda t, max_length=2: "en")
    sep = "--"
    ends = [';', '\r\n']
    text = "--en;Hello world--fr;Bonjour"
    result = splitter.split(text, sep=sep, ends=ends, autodetect=False)
    assert "en" in result
    assert "fr" in result
    assert result["en"].startswith("Hello")
    assert result["fr"].startswith("Bonjour")


def test_split_multiple_paragraphs_different_languages(monkeypatch):
    # Patch to alternate languages for paragraph detection
    detect_results = ["en", "fr", "es"]
    calls = -1

    def fake_detect_language(text, max_length=2):
        nonlocal calls
        calls += 1
        return detect_results[calls % len(detect_results)]

    monkeypatch.setattr(splitter, "detect_language", fake_detect_language)

    text = "Para1\n\nPara2\n\nPara3"
    result = splitter.split(text, autodetect=True, pargraph_sep="\n\n")
    keys = list(result.keys())
    # The keys appear in order of first detection
    assert keys == ["en", "fr", "es"]
    # Each key has some text including "Para"
    for k in keys:
        assert "Para" in result[k]


def test_split_does_not_duplicate_langseq(monkeypatch):
    # Patch detect_language to always 'en'
    monkeypatch.setattr(splitter, "detect_language", lambda t, max_length=2: "en")
    # Text that should produce multiple chunks for the same lang
    text = ".:en\nHello.:en\n World."
    result = splitter.split(text)
    # lang_seq has unique lang codes, so result keys length is 1, with key 'en'
    assert list(result.keys()) == ["en"]
    # The text combines both chunks
    assert "Hello" in result["en"]
    assert "World" in result["en"]


def test_split_respects_markdown_title_flags(monkeypatch):
    """Test that markdown=True and title=True produces expected string output"""
    monkeypatch.setattr(splitter, "detect_language", lambda t, max_length=2: "en")
    text = ".:en\nHello"
    result = splitter.split(text, markdown=True, title=True)
    # The output is a string that ends with ':'
    assert isinstance(result, str)
    assert result.endswith(":Hello")


def test_split_with_unicode_and_multilingual_content(monkeypatch):
    monkeypatch.setattr(splitter, "detect_language", lambda t, max_length=2: "en")
    # Use Unicode accents and wide characters
    text = ".:en\nCafé😊.:fr\nÉcole"
    result = splitter.split(text, autodetect=False)
    assert "en" in result
    assert "fr" in result
    # Accented characters preserved
    assert "Café" in result["en"]
    assert "École" in result["fr"]


def test_split_respects_autodetect_false_and_unknown_language(monkeypatch):
    # If autodetect False, unknown language, should assign rest of chunk to UNKNOWN_LANGUAGE without autodetect
    monkeypatch.setattr(splitter, "detect_language", lambda t, max_length=2: "xx")
    # Provide text with unknown keys so autodetect triggers
    text = ".:xx\nsome text"
    result = splitter.split(text, autodetect=False)
    unk = splitter.settings.UNKNOWN_LANGUAGE
    assert unk in result
    assert result[unk].strip() == "some text"


def test_split_random_seed_stability(monkeypatch):
    # This test demonstrates seeding and stability, but splitter code doesn't use random internally
    # We seed random, numpy, torch as requested in prompt to cover defensive cases
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # Just run split on some text to verify no exceptions, stable output nonempty
    text = ".:en\nHello"
    result1 = splitter.split(text)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    result2 = splitter.split(text)
    assert result1 == result2


def test_convert_removes_leading_ends(monkeypatch):
    # Leading ends appear as first 2 chars in ends, and are stripped in convert
    d = {"en": "\nHello"}
    text = splitter.convert(d)
    # Strip removes leading ends repeated while they appear at start
    # So text should NOT start with any ends element
    for end in splitter.settings.ENDS:
        # Only test if two first chars equal end (if length is 2)
        if len(end) == 2:
            assert not text.startswith(end)
    # And output should include language code and remaining text stripped from left space
    assert "en" in text
    assert "Hello" in text


def test_split_handles_chunks_without_separator(monkeypatch):
    # Provide text with token longer than max_key_length but no ends found in first max_key_length+1 chars:
    # This should fallback to UNKNOWN_LANGUAGE with autodetect enabled
    monkeypatch.setattr(splitter, "detect_language", lambda t, max_length=2: "en")
    text = ".:longkeySome text"
    result = splitter.split(text, min_key_length=2, max_key_length=2, autodetect=True)
    assert "en" in result


def test_split_ordered_dict_return(monkeypatch):
    # Confirm split returns OrderedDict when text not empty string
    monkeypatch.setattr(splitter, "detect_language", lambda t, max_length=2: "en")
    text = ".:en\nHello.:fr\nBonjour"
    result = splitter.split(text)
    import collections
    assert isinstance(result, collections.OrderedDict)

