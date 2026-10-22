import uuid

import pytest

from boltons import strutils
from boltons.strutils import (
    MultiReplace,
    a10n,
    args2cmd,
    args2sh,
    asciify,
    bytes2human,
    camel2under,
    cardinalize,
    ellipsize,
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


def test_camel2under_and_under2camel():
    assert camel2under('BasicParseTest') == 'basic_parse_test'
    assert under2camel('complex_tokenizer') == 'ComplexTokenizer'
    assert under2camel(camel2under('RoundTripName')) == 'RoundTripName'


def test_slugify_options():
    assert slugify('First post! Hi!!!!~1    ') == 'first_post_hi_1'
    assert slugify('Hello World', delim='-') == 'hello-world'
    assert slugify('Hello World', lower=False) == 'Hello_World'
    assert slugify("Kurt Gödel's pretty cool.", ascii=True) == b'kurt_goedel_s_pretty_cool'


def test_split_punct_ws_hashtags_and_a10n():
    assert split_punct_ws('First post! Hi!!!!~1    ') == ['First', 'post', 'Hi', '1']
    assert find_hashtags('#atag http://asite/#ananchor') == ['atag']
    assert a10n('abbreviation') == 'a10n'
    assert a10n('internationalization') == 'i18n'
    assert a10n('') == ''


def test_ordinalize():
    assert ordinalize(1) == '1st'
    assert ordinalize(2) == '2nd'
    assert ordinalize(3) == '3rd'
    assert ordinalize(4) == '4th'
    assert ordinalize(11) == '11th'
    assert ordinalize(12) == '12th'
    assert ordinalize(13) == '13th'
    assert ordinalize(21) == '21st'
    assert ordinalize(1515) == '1515th'
    assert ordinalize(3694839230) == '3694839230th'
    assert ordinalize('hi') == 'hi'
    assert ordinalize(2, ext_only=True) == 'nd'


def test_cardinalize_and_unit_len():
    assert cardinalize('vowel', 5) == 'vowels'
    assert cardinalize('vowel', 1) == 'vowel'
    assert cardinalize('Wish', 3) == 'Wishes'
    assert unit_len(range(10), 'number') == '10 numbers'
    assert unit_len('aeiou', 'vowel') == '5 vowels'
    assert unit_len([], 'worry') == 'No worries'


def test_pluralize_and_singularize():
    assert pluralize('friend') == 'friends'
    assert pluralize('enemy') == 'enemies'
    assert pluralize('Sheep') == 'Sheep'
    assert singularize('chances') == 'chance'
    assert singularize('Activities') == 'Activity'
    assert singularize('Glasses') == 'Glass'
    assert singularize('FEET') == 'FOOT'


def test_strip_ansi_for_str_and_bytes():
    assert strip_ansi('\x1b[0m\x1b[1;36mart\x1b[46;34m') == 'art'
    out = strip_ansi(b'\x1b[0m\x1b[1;36mart')
    assert out == b'art'
    assert isinstance(out, bytes)
    assert strip_ansi('plain') == 'plain'


def test_asciify_and_is_ascii():
    assert asciify('Beyoncé') == b'Beyonce'
    assert is_ascii('Beyonce')
    assert not is_ascii('Beyoncé')
    assert is_ascii(b'abc')
    with pytest.raises(ValueError):
        is_ascii(123)


def test_bytes2human():
    assert bytes2human(128991) == '126K'
    assert bytes2human(100001221) == '95M'
    assert bytes2human(0, 2) == '0.00B'
    assert bytes2human(1024) == '1K'
    assert bytes2human(1024 ** 8) == '1Y'


def test_html2text_and_gzip_roundtrip():
    result = html2text('<a href="#">Test &amp;<em>(Δ&#x03b7;&#956;&#x03CE;)</em></a>')
    assert result == 'Test &(Δημώ)'
    data = b'bytesahoy! ' * 100
    assert gunzip_bytes(gzip_bytes(data)) == data
    assert gunzip_bytes(gzip_bytes(data, level=1)) == data
    assert gunzip_bytes(gzip_bytes(b'')) == b''
    assert len(gzip_bytes(b'a' * 10000)) < 200


def test_iter_splitlines_and_indent():
    assert list(iter_splitlines('\nhi\nbye\n')) == ['', 'hi', 'bye', '']
    assert list(iter_splitlines('\r\nhi\rbye\r\n')) == ['', 'hi', 'bye', '']
    assert list(iter_splitlines('')) == []
    assert indent('a\n\nb', '  ') == '  a\n\n  b'
    assert indent('a\n\nb', '> ', key=lambda line: True) == '> a\n> \n> b'


def test_is_uuid():
    assert is_uuid('e682ccca-5a4c-4ef2-9711-73f9ad1e15ea')
    assert not is_uuid('0221f0d9-d4b9-11e5-a478-10ddb1c2feb9')
    assert is_uuid('0221f0d9-d4b9-11e5-a478-10ddb1c2feb9', version=1)
    assert is_uuid('0221f0d9-d4b9-11e5-a478-10ddb1c2feb9', version=0)
    assert is_uuid(uuid.uuid4())
    assert not is_uuid('not-a-uuid')


def test_shell_argument_escaping():
    args = ['aa', '[bb]', "cc'cc", 'dd"dd']
    assert args2sh(args) == "aa '[bb]' 'cc'\"'\"'cc' 'dd\"dd'"
    assert args2cmd(args) == "aa [bb] cc'cc dd\\\"dd"
    assert args2sh(['a', 'b'], sep=',') == 'a,b'
    assert strutils.escape_shell_args(['a b'], style='sh') == "'a b'"


def test_int_list_parsing_and_formatting():
    assert parse_int_list('1,3,5-8,10-11,15') == [1, 3, 5, 6, 7, 8, 10, 11, 15]
    assert parse_int_list('1,2,4-6,8') == [1, 2, 4, 5, 6, 8]
    assert format_int_list([1, 3, 5, 6, 7, 8, 10, 11, 15]) == '1,3,5-8,10-11,15'
    assert format_int_list([1, 3], delim_space=True) == '1, 3'
    assert int_ranges_from_int_list('1,3,5-8,10-11,15') == ((1, 1), (3, 3), (5, 8), (10, 11), (15, 15))
    assert int_ranges_from_int_list('1') == ((1, 1),)
    assert int_ranges_from_int_list('') == ()


def test_multi_replace_plain_and_regex():
    text = 'The foo bar cat ate a bat'
    sub_map = {'foo': 'zoo', 'cat': 'hat', 'bat': 'kraken'}
    expected = 'The zoo bar hat ate a kraken'
    assert multi_replace(text, sub_map) == expected
    assert MultiReplace(sub_map).sub(text) == expected
    assert MultiReplace({r'\d+': '#'}, regex=True).sub('a1b22c333') == 'a#b#c#'


def test_unwrap_text_and_removeprefix():
    text = "Short \n lines  \nwrapped\nsmall.\n\nAnother\nparagraph."
    assert unwrap_text(text) == 'Short lines wrapped small.\n\nAnother paragraph.'
    assert unwrap_text(text, ending=None) == ['Short lines wrapped small.', 'Another paragraph.']
    assert removeprefix('foobar', 'foo') == 'bar'
    assert removeprefix('foobar', 'x') == 'foobar'


def test_human_readable_list_and_ellipsize():
    assert human_readable_list(['a']) == 'a'
    assert human_readable_list(['a', 'b']) == 'a and b'
    assert human_readable_list(['a', 'b', 'c']) == 'a, b, and c'
    assert human_readable_list(['a', 'b', 'c'], oxford=False) == 'a, b and c'
    assert human_readable_list(['a', 'b', 'c'], conjunction='or') == 'a, b, or c'

    assert ellipsize('Hello, World!', 16) == 'Hello, World!'
    assert ellipsize('Beautiful is better than ugly. Explicit is better.', 31) == 'Beautiful is better than ugly…'
    assert ellipsize('rates around 6.5% this week', 20) == 'rates around 6.5%…'
    long_text = 'word ' * 100
    assert len(ellipsize(long_text, 40)) <= 40
