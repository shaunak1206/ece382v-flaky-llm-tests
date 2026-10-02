"""Tests for boltons.strutils."""

import re
import uuid

import pytest

from boltons.strutils import (
    MultiReplace,
    a10n,
    args2cmd,
    args2sh,
    asciify,
    bytes2human,
    camel2under,
    cardinalize,
    complement_int_list,
    ellipsize,
    escape_shell_args,
    find_hashtags,
    format_int_list,
    gunzip_bytes,
    gzip_bytes,
    html2text,
    human_readable_list,
    indent,
    int_ranges_from_int_list,
    is_ascii,
    is_uuid,
    iter_splitlines,
    multi_replace,
    ordinalize,
    parse_int_list,
    pluralize,
    removeprefix,
    singularize,
    slugify,
    split_punct_ws,
    strip_ansi,
    under2camel,
    unit_len,
    unwrap_text,
)

ELLIPSIS = "…"


def test_case_conversion_and_slugify():
    assert camel2under("BasicParseTest") == "basic_parse_test"
    assert camel2under("HTTPServer") == "http_server"
    assert camel2under("already_snake") == "already_snake"
    assert under2camel("complex_tokenizer") == "ComplexTokenizer"
    assert under2camel("a__b") == "A_B"

    assert split_punct_ws("First post! Hi!!!!~1    ") == ["First", "post", "Hi", "1"]
    assert slugify("First post! Hi!!!!~1    ") == "first_post_hi_1"
    assert slugify("First post! Hi!!!!~1    ", delim="-") == "first-post-hi-1"
    assert slugify("First Post", lower=False) == "First_Post"
    assert slugify("Kurt Gödel's pretty cool.", ascii=True) == b"kurt_goedel_s_pretty_cool"
    assert slugify("") == ""
    assert slugify("!!!") == "_"


def test_pluralization_ordinals_and_unit_descriptions():
    assert pluralize("friend") == "friends"
    assert pluralize("enemy") == "enemies"
    assert pluralize("day") == "days"
    assert pluralize("wish") == "wishes"
    assert pluralize("box") == "boxes"
    assert pluralize("Sheep") == "Sheep"
    assert pluralize("Mouse") == "Mice"
    assert pluralize("child") == "children"

    assert cardinalize("vowel", 1) == "vowel"
    assert cardinalize("vowel", 5) == "vowels"
    assert cardinalize("vowel", 0) == "vowels"
    assert cardinalize("Wish", 3) == "Wishes"

    assert singularize("chances") == "chance"
    assert singularize("Activities") == "Activity"
    assert singularize("Glasses") == "Glass"
    assert singularize("FEET") == "FOOT"
    assert singularize("glass") == "glass"
    assert singularize("men") == "man"
    assert singularize("cat") == "cat"
    assert singularize("cats") == "cat"

    expected = [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (102, "102nd"),
        (111, "111th"),
        (1515, "1515th"),
        (3694839230, "3694839230th"),
    ]
    for number, text in expected:
        assert ordinalize(number) == text
    assert ordinalize("hi") == "hi"
    assert ordinalize(21, ext_only=True) == "st"

    assert unit_len(range(10), "number") == "10 numbers"
    assert unit_len("aeiou", "vowel") == "5 vowels"
    assert unit_len([], "worry") == "No worries"
    assert unit_len(["x"], "item") == "1 item"
    assert a10n("abbreviation") == "a10n"
    assert a10n("internationalization") == "i18n"
    assert a10n("") == ""
    assert a10n("ab") == "ab"


def test_bytes2human():
    assert bytes2human(128991) == "126K"
    assert bytes2human(100001221) == "95M"
    assert bytes2human(0, 2) == "0.00B"
    assert bytes2human(1024) == "1K"
    assert bytes2human(1024**8) == "1Y"
    assert bytes2human(1023) == "1023B"
    assert bytes2human(1536, 1) == "1.5K"
    assert bytes2human(1500, 2) == "1.46K"
    assert bytes2human(-2048) == "-2K"


def test_ascii_uuid_ansi_hashtag_and_html_helpers():
    assert asciify("Beyoncé") == b"Beyonce"
    assert asciify("plain") == b"plain"
    assert is_ascii("Beyoncé") is False
    assert is_ascii("Beyonce") is True
    assert is_ascii(b"abc") is True
    assert is_ascii(b"\xff") is False
    with pytest.raises(ValueError):
        is_ascii(5)

    assert strip_ansi("\x1b[0m\x1b[1;36mart\x1b[46;34m") == "art"
    assert strip_ansi(b"\x1b[31mred\x1b[0m") == b"red"
    assert strip_ansi(bytearray(b"\x1b[31mred")) == bytearray(b"red")

    assert is_uuid("e682ccca-5a4c-4ef2-9711-73f9ad1e15ea") is True
    assert is_uuid("0221f0d9-d4b9-11e5-a478-10ddb1c2feb9") is False
    assert is_uuid("0221f0d9-d4b9-11e5-a478-10ddb1c2feb9", version=1) is True
    assert is_uuid(uuid.uuid4()) is True
    assert is_uuid("not-a-uuid") is False
    assert is_uuid(None) is False

    assert find_hashtags("#atag http://asite/#ananchor") == ["atag"]
    assert find_hashtags("go #python and #rust") == ["python", "rust"]

    markup = '<a href="#">Test &amp;<em>(Δ&#x03b7;&#956;&#x03CE;)</em></a>'
    assert html2text(markup) == "Test &(Δημώ)"
    assert html2text("<p>Hello <b>world</b></p>") == "Hello world"


def test_line_splitting_indenting_unwrapping_and_gzip():
    assert list(iter_splitlines("\nhi\nbye\n")) == ["", "hi", "bye", ""]
    assert list(iter_splitlines("\r\nhi\rbye\r\n")) == ["", "hi", "bye", ""]
    assert list(iter_splitlines("")) == []
    assert list(iter_splitlines("no newline")) == ["no newline"]

    assert indent("a\n\nb", "  ") == "  a\n\n  b"
    assert indent("a\nb", "> ", newline="|") == "> a|> b"
    assert indent("a\n\nb", "  ", key=lambda line: True) == "  a\n  \n  b"

    text = "Short \n lines  \nwrapped\nsmall.\n\nAnother\nparagraph."
    assert unwrap_text(text) == "Short lines wrapped small.\n\nAnother paragraph."
    assert unwrap_text(text, ending=None) == ["Short lines wrapped small.", "Another paragraph."]

    payload = b"hello world " * 100
    packed = gzip_bytes(payload)
    assert packed[:2] == b"\x1f\x8b"
    assert len(packed) < len(payload)
    assert gunzip_bytes(packed) == payload
    assert gunzip_bytes(gzip_bytes(payload, level=1)) == payload
    assert gunzip_bytes(gzip_bytes(b"")) == b""


def test_multi_replace():
    text = "The foo bar cat ate a bat"
    mapping = {"foo": "zoo", "cat": "hat", "bat": "kraken"}
    assert multi_replace(text, mapping) == "The zoo bar hat ate a kraken"
    assert MultiReplace(list(mapping.items())).sub(text) == "The zoo bar hat ate a kraken"
    # replacements happen in a single pass
    assert MultiReplace({"foo": "baz", "baz": "bar"}).sub("foo bar baz") == "baz bar bar"
    assert multi_replace("a1 b22 c", [(r"\d+", "#")], regex=True) == "a# b# c"
    assert multi_replace("Hello HELLO", {"hello": "bye"}, flags=re.IGNORECASE) == "bye bye"
    assert multi_replace("a.b", {".": "-"}) == "a-b"
    assert multi_replace("unchanged", {}) == "unchanged"


def test_int_list_parsing_formatting_and_complements():
    assert parse_int_list("1,3,5-8,10-11,15") == [1, 3, 5, 6, 7, 8, 10, 11, 15]
    assert parse_int_list("3,1,2") == [1, 2, 3]
    assert parse_int_list("8-5") == [5, 6, 7, 8]
    assert parse_int_list(" 1, 2 ") == [1, 2]
    assert parse_int_list("1;3;5..7", delim=";", range_delim="..") == [1, 3, 5, 6, 7]
    assert parse_int_list("") == []

    assert format_int_list([1, 3, 5, 6, 7, 8, 10, 11, 15]) == "1,3,5-8,10-11,15"
    assert format_int_list([1, 3, 5, 6, 7, 8, 10, 11, 15], delim_space=True) == "1, 3, 5-8, 10-11, 15"
    assert format_int_list([3, 1, 2, 2]) == "1-3"
    assert format_int_list([]) == ""

    assert complement_int_list("1,3,5-8,10-11,15") == "0,2,4,9,12-14"
    assert complement_int_list("1,3,5-8,10-11,15", range_start=3) == "4,9,12-14"
    assert complement_int_list("1,3,5-8,10-11,15", range_end=20) == "0,2,4,9,12-14,16-19"
    assert complement_int_list("") == ""

    assert int_ranges_from_int_list("1,3,5-8,10-11,15") == ((1, 1), (3, 3), (5, 8), (10, 11), (15, 15))
    assert int_ranges_from_int_list("1") == ((1, 1),)
    assert int_ranges_from_int_list("") == ()


def test_shell_escaping():
    args = ["aa", "[bb]", "cc'cc", 'dd"dd']
    assert args2sh(args) == "aa '[bb]' 'cc'\"'\"'cc' 'dd\"dd'"
    assert args2cmd(args) == 'aa [bb] cc\'cc dd\\"dd'
    assert args2sh([""]) == "''"
    assert args2sh(["a b"]) == "'a b'"
    assert args2sh(["plain", "file.txt", "a_b-c"]) == "plain file.txt a_b-c"
    assert args2sh(["a", "b"], sep=",") == "a,b"
    assert args2cmd(["two words"]) == '"two words"'
    assert args2cmd([""]) == '""'
    assert escape_shell_args(["a b", "c"], style="sh") == "'a b' c"
    assert escape_shell_args(["a b", "c"], style="cmd") == '"a b" c'
    with pytest.raises(ValueError):
        escape_shell_args(["a"], style="fish")


def test_human_readable_list_removeprefix_and_ellipsize():
    assert human_readable_list([]) == ""
    assert human_readable_list(["a"]) == "a"
    assert human_readable_list(["a", "b"]) == "a and b"
    assert human_readable_list(["a", "b", "c"]) == "a, b, and c"
    assert human_readable_list(["a", "b", "c"], oxford=False) == "a, b and c"
    assert human_readable_list(["a", "b", "c"], conjunction="or") == "a, b, or c"
    assert human_readable_list(["a", "b", "c"], delimiter=";") == "a; b; and c"

    assert removeprefix("foobar", "foo") == "bar"
    assert removeprefix("foobar", "baz") == "foobar"
    assert removeprefix("foo", "foo") == ""

    assert ellipsize("Hello, World!", 16) == "Hello, World!"
    assert ellipsize("Beautiful is better than ugly. Explicit is better.", 31) == (
        "Beautiful is better than ugly" + ELLIPSIS
    )
    assert ellipsize("rates around 6.5% this week", 20) == "rates around 6.5%" + ELLIPSIS
    assert ellipsize("antidisestablishmentarianism", 10) == "antidises" + ELLIPSIS
    assert ellipsize("a long sentence here", 12, ellipsis="...") == "a long..."
    long_result = ellipsize("word " * 100, 20)
    assert long_result.endswith(ELLIPSIS)
    assert len(long_result) <= 20
    with pytest.raises(ValueError):
        ellipsize("abc", 1)
