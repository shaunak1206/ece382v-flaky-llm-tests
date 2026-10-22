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


def test_case_conversion_and_slugify():
    assert camel2under('BasicParseTest') == 'basic_parse_test'
    assert camel2under('already_under') == 'already_under'
    assert under2camel('complex_tokenizer') == 'ComplexTokenizer'
    assert under2camel(camel2under('SomeClassName')) == 'SomeClassName'

    assert split_punct_ws('First post! Hi!!!!~1    ') == ['First', 'post', 'Hi', '1']
    assert slugify('First post! Hi!!!!~1    ') == 'first_post_hi_1'
    assert slugify('First post! Hi', delim='-') == 'first-post-hi'
    assert slugify('First Post', lower=False) == 'First_Post'
    assert slugify('') == ''
    assert slugify("Kurt Gödel's pretty cool.", ascii=True) == b'kurt_goedel_s_pretty_cool'


def test_ordinalize_and_unit_len():
    expected = {0: '0th', 1: '1st', 2: '2nd', 3: '3rd', 4: '4th', 10: '10th',
                11: '11th', 12: '12th', 13: '13th', 21: '21st', 22: '22nd',
                23: '23rd', 101: '101st', 111: '111th', 1515: '1515th'}
    for number, text in expected.items():
        assert ordinalize(number) == text
    assert ordinalize('hi') == 'hi'
    assert ordinalize(2, ext_only=True) == 'nd'
    assert ordinalize('7') == '7th'

    assert unit_len(range(10), 'number') == '10 numbers'
    assert unit_len('aeiou', 'vowel') == '5 vowels'
    assert unit_len([1], 'item') == '1 item'
    assert unit_len([], 'worry') == 'No worries'


def test_pluralize_singularize_cardinalize():
    assert pluralize('friend') == 'friends'
    assert pluralize('enemy') == 'enemies'
    assert pluralize('toy') == 'toys'
    assert pluralize('box') == 'boxes'
    assert pluralize('Wish') == 'Wishes'
    assert pluralize('Sheep') == 'Sheep'
    assert pluralize('child') == 'children'
    assert pluralize('FOOT') == 'FEET'

    assert singularize('chances') == 'chance'
    assert singularize('Activities') == 'Activity'
    assert singularize('Glasses') == 'Glass'
    assert singularize('Glass') == 'Glass'
    assert singularize('FEET') == 'FOOT'
    assert singularize('children') == 'child'
    assert singularize('') == ''

    assert cardinalize('vowel', 5) == 'vowels'
    assert cardinalize('vowel', 1) == 'vowel'
    assert cardinalize('vowel', 0) == 'vowels'
    assert cardinalize('Wish', 3) == 'Wishes'


def test_ansi_ascii_and_text_checks():
    assert strip_ansi('\x1b[0m\x1b[1;36mart\x1b[46;34m') == 'art'
    assert strip_ansi('plain') == 'plain'
    assert strip_ansi(b'\x1b[31mred\x1b[0m') == b'red'
    assert isinstance(strip_ansi(bytearray(b'\x1b[1mbold')), bytearray)

    assert asciify('Beyoncé') == b'Beyonce'
    assert asciify('plain') == b'plain'
    assert asciify('Ærø', ignore=True).isascii()
    assert is_ascii('Beyonce')
    assert not is_ascii('Beyoncé')
    assert is_ascii(b'abc')
    assert not is_ascii(b'\xff')
    with pytest.raises(ValueError):
        is_ascii(123)


def test_bytes2human():
    assert bytes2human(128991) == '126K'
    assert bytes2human(100001221) == '95M'
    assert bytes2human(0, 2) == '0.00B'
    assert bytes2human(1023) == '1023B'
    assert bytes2human(1024) == '1K'
    assert bytes2human(1536, 1) == '1.5K'
    assert bytes2human(1024 ** 3) == '1G'
    assert bytes2human(1024 ** 8) == '1Y'
    assert bytes2human(-2048) == '-2K'


def test_html2text_and_gzip_roundtrip():
    assert html2text('<p>Hello <b>world</b> &amp; &lt;friends&gt;</p>') == 'Hello world & <friends>'
    assert html2text('<a href="#">Test &amp;<em>(Δ&#x03b7;&#956;&#x03CE;)</em></a>') == (
        'Test &(Δημώ)')
    assert html2text('no tags') == 'no tags'

    data = b'hello world' * 100
    packed = gzip_bytes(data)
    assert packed[:2] == b'\x1f\x8b'
    assert len(packed) < len(data)
    assert gunzip_bytes(packed) == data
    assert gunzip_bytes(gzip_bytes(b'')) == b''
    assert gunzip_bytes(gzip_bytes(data, level=1)) == data


def test_iter_splitlines_and_indent():
    assert list(iter_splitlines('\nhi\nbye\n')) == ['', 'hi', 'bye', '']
    assert list(iter_splitlines('\r\nhi\rbye\r\n')) == ['', 'hi', 'bye', '']
    assert list(iter_splitlines('')) == []
    assert list(iter_splitlines('one line')) == ['one line']

    assert indent('a\n\nb', '  ') == '  a\n\n  b'
    assert indent('a\nb', '> ', newline='|') == '> a|> b'
    assert indent('a\n\nb', '  ', key=lambda line: True) == '  a\n  \n  b'


def test_uuid_hashtags_and_abbreviation():
    assert is_uuid('e682ccca-5a4c-4ef2-9711-73f9ad1e15ea')
    assert not is_uuid('0221f0d9-d4b9-11e5-a478-10ddb1c2feb9')
    assert is_uuid('0221f0d9-d4b9-11e5-a478-10ddb1c2feb9', version=1)
    assert is_uuid('0221f0d9-d4b9-11e5-a478-10ddb1c2feb9', version=0)
    assert is_uuid(uuid.UUID('e682ccca-5a4c-4ef2-9711-73f9ad1e15ea'))
    assert not is_uuid('not-a-uuid')
    assert not is_uuid(None)
    assert not is_uuid(12345)

    assert find_hashtags('#atag http://asite/#ananchor') == ['atag']
    assert find_hashtags('two #one and #two tags') == ['one', 'two']
    assert find_hashtags('nothing here') == []

    assert a10n('abbreviation') == 'a10n'
    assert a10n('internationalization') == 'i18n'
    assert a10n('') == ''
    assert a10n('ab') == 'ab'
    assert a10n('abc') == 'a1c'


def test_shell_argument_escaping():
    assert args2sh(['aa', '[bb]', "cc'cc", 'dd"dd']) == "aa '[bb]' 'cc'\"'\"'cc' 'dd\"dd'"
    assert args2sh(['']) == "''"
    assert args2sh(['a', 'b'], sep=',') == 'a,b'

    assert args2cmd(['aa', '[bb]', "cc'cc", 'dd"dd']) == "aa [bb] cc'cc dd" + '\\"' + 'dd'
    assert args2cmd(['a b']) == '"a b"'
    assert args2cmd(['']) == '""'

    assert escape_shell_args(['a b'], style='sh') == "'a b'"
    assert escape_shell_args(['a b'], style='cmd') == '"a b"'
    assert escape_shell_args(['x', 'y'], sep='+', style='sh') == 'x+y'
    with pytest.raises(ValueError):
        escape_shell_args(['x'], style='powershell')


def test_int_list_parsing_and_formatting():
    assert parse_int_list('1,3,5-8,10-11,15') == [1, 3, 5, 6, 7, 8, 10, 11, 15]
    assert parse_int_list('8-5') == [5, 6, 7, 8]
    assert parse_int_list('1,,3') == [1, 3]
    assert parse_int_list('') == []
    assert parse_int_list('1;3:5', delim=';', range_delim=':') == [1, 3, 4, 5]
    with pytest.raises(ValueError):
        parse_int_list('1,a')

    assert format_int_list([1, 3, 5, 6, 7, 8, 10, 11, 15]) == '1,3,5-8,10-11,15'
    assert format_int_list([3, 1, 2, 2, 7]) == '1-3,7'
    assert format_int_list([1, 3, 5], delim_space=True) == '1, 3, 5'
    assert format_int_list([1, 2, 3], range_delim=':') == '1:3'
    assert format_int_list([]) == ''
    assert format_int_list(parse_int_list('2,4-6,9')) == '2,4-6,9'


def test_complement_and_int_ranges():
    spec = '1,3,5-8,10-11,15'
    assert complement_int_list(spec) == '0,2,4,9,12-14'
    assert complement_int_list(spec, range_start=1) == '2,4,9,12-14'
    assert complement_int_list(spec, range_start=3) == '4,9,12-14'
    assert complement_int_list(spec, range_end=14) == '0,2,4,9,12-13'
    assert complement_int_list(spec, range_end=20) == '0,2,4,9,12-14,16-19'
    assert complement_int_list(spec, range_end=0) == ''
    assert complement_int_list('') == ''
    assert complement_int_list('1,3,5-8', range_start=2, range_end=3) == '2'

    assert int_ranges_from_int_list(spec) == ((1, 1), (3, 3), (5, 8), (10, 11), (15, 15))
    assert int_ranges_from_int_list('1') == ((1, 1),)
    assert int_ranges_from_int_list('') == ()


def test_multi_replace():
    sub_map = {'foo': 'zoo', 'cat': 'hat', 'bat': 'kraken'}
    text = 'The foo bar cat ate a bat'
    assert MultiReplace(sub_map).sub(text) == 'The zoo bar hat ate a kraken'
    assert MultiReplace(list(sub_map.items())).sub(text) == 'The zoo bar hat ate a kraken'
    assert multi_replace(text, sub_map) == 'The zoo bar hat ate a kraken'

    # replacements happen in a single pass, so results are not re-replaced
    chained = MultiReplace([('foo', 'baz'), ('baz', 'bar')])
    assert chained.sub('foo bar baz') == 'baz bar bar'

    assert multi_replace('a1b22', {r'\d+': '#'}, regex=True) == 'a#b#'
    assert multi_replace('a.b', {'.': '!'}) == 'a!b'
    assert multi_replace('A-a', {'a': 'x'}, flags=re.IGNORECASE, regex=True) == 'x-x'
    assert MultiReplace({}).sub('unchanged') == 'unchanged'


def test_text_helpers():
    text = "Short \n lines  \nwrapped\nsmall.\n\nAnother\nparagraph."
    assert unwrap_text(text) == 'Short lines wrapped small.\n\nAnother paragraph.'
    assert unwrap_text(text, ending=None) == [
        'Short lines wrapped small.', 'Another paragraph.']

    assert removeprefix('prefix-body', 'prefix-') == 'body'
    assert removeprefix('body', 'prefix-') == 'body'
    assert removeprefix('abc', '') == 'abc'

    assert human_readable_list([]) == ''
    assert human_readable_list(['a']) == 'a'
    assert human_readable_list(['a', 'b']) == 'a and b'
    assert human_readable_list(['a', 'b', 'c']) == 'a, b, and c'
    assert human_readable_list(['a', 'b', 'c'], oxford=False) == 'a, b and c'
    assert human_readable_list(['a', 'b', 'c'], conjunction='or') == 'a, b, or c'
    assert human_readable_list(['a', 'b', 'c'], delimiter=';') == 'a; b; and c'


def test_ellipsize():
    assert ellipsize('Hello, World!', 16) == 'Hello, World!'
    assert ellipsize('exact', 5) == 'exact'
    assert ellipsize('Beautiful is better than ugly. Explicit is better.', 31) == (
        'Beautiful is better than ugly…')
    assert ellipsize('rates around 6.5% this week', 20) == 'rates around 6.5%…'
    assert ellipsize('antidisestablishmentarianism', 10) == 'antidises…'
    assert ellipsize('hello wonderful world', 10, ellipsis='...') == 'hello...'

    long_text = 'word ' * 100
    result = ellipsize(long_text, 40)
    assert len(result) <= 40
    assert result.endswith('…')

    with pytest.raises(ValueError):
        ellipsize('abc', 1)
    with pytest.raises(ValueError):
        ellipsize('abcdef', 3, ellipsis='...')
