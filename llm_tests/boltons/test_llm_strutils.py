"""Unit tests for boltons.strutils."""
import uuid

import pytest

from boltons.strutils import (
    MultiReplace, a10n, args2cmd, args2sh, asciify, bytes2human, camel2under,
    cardinalize, complement_int_list, ellipsize, escape_shell_args,
    find_hashtags, format_int_list, gunzip_bytes, gzip_bytes,
    human_readable_list, html2text, indent, int_ranges_from_int_list,
    is_ascii, is_uuid, iter_splitlines, multi_replace, ordinalize,
    parse_int_list, pluralize, removeprefix, singularize, slugify,
    split_punct_ws, strip_ansi, under2camel, unit_len, unwrap_text)


@pytest.mark.parametrize('camel, under', [
    ('BasicParseTest', 'basic_parse_test'),
    ('HTTPResponse', 'http_response'),
    ('already_under', 'already_under'),
    ('x', 'x'),
])
def test_camel2under(camel, under):
    assert camel2under(camel) == under


def test_under2camel_and_roundtrip():
    assert under2camel('complex_tokenizer') == 'ComplexTokenizer'
    assert camel2under(under2camel('some_snake_name')) == 'some_snake_name'


def test_slugify_collapses_punctuation_and_whitespace():
    assert slugify('First post! Hi!!!!~1    ') == 'first_post_hi_1'
    assert slugify('Hello World', delim='-') == 'hello-world'
    assert slugify('Hello World', lower=False) == 'Hello_World'
    assert slugify('') == ''


def test_slugify_ascii_returns_bytes():
    assert slugify("Kurt Gödel's pretty cool.", ascii=True) == b'kurt_goedel_s_pretty_cool'


def test_split_punct_ws():
    assert split_punct_ws('First post! Hi!!!!~1    ') == ['First', 'post', 'Hi', '1']


def test_unit_len_pluralizes_and_handles_empty():
    assert unit_len(range(10), 'number') == '10 numbers'
    assert unit_len('a', 'vowel') == '1 vowel'
    assert unit_len([], 'worry') == 'No worries'


@pytest.mark.parametrize('number, expected', [
    (1, '1st'), (2, '2nd'), (3, '3rd'), (4, '4th'), (11, '11th'),
    (12, '12th'), (13, '13th'), (21, '21st'), (102, '102nd'),
    (111, '111th'), (3694839230, '3694839230th'), ('hi', 'hi'),
])
def test_ordinalize(number, expected):
    assert ordinalize(number) == expected


def test_ordinalize_ext_only():
    assert ordinalize(22, ext_only=True) == 'nd'
    assert ordinalize(5, ext_only=True) == 'th'


@pytest.mark.parametrize('word, plural', [
    ('friend', 'friends'),
    ('enemy', 'enemies'),
    ('day', 'days'),
    ('box', 'boxes'),
    ('church', 'churches'),
    ('child', 'children'),
    ('Sheep', 'Sheep'),
    ('MAN', 'MEN'),
    ('Wish', 'Wishes'),
])
def test_pluralize(word, plural):
    assert pluralize(word) == plural


@pytest.mark.parametrize('word, singular', [
    ('chances', 'chance'),
    ('Activities', 'Activity'),
    ('Glasses', 'Glass'),
    ('FEET', 'FOOT'),
    ('children', 'child'),
    ('cats', 'cat'),
    ('cat', 'cat'),
])
def test_singularize(word, singular):
    assert singularize(word) == singular


def test_cardinalize_uses_count():
    assert cardinalize('vowel', 1) == 'vowel'
    assert cardinalize('vowel', 5) == 'vowels'
    assert cardinalize('Wish', 3) == 'Wishes'
    assert cardinalize('vowel', 0) == 'vowels'


def test_find_hashtags_ignores_url_anchors():
    assert find_hashtags('#atag http://asite/#ananchor') == ['atag']
    assert find_hashtags('nothing here') == []
    assert find_hashtags('x #one and #two') == ['one', 'two']


def test_a10n_numeronym():
    assert a10n('internationalization') == 'i18n'
    assert a10n('abbreviation') == 'a10n'
    assert a10n('ab') == 'ab'
    assert a10n('') == ''


def test_strip_ansi_preserves_input_type():
    colored = '\x1b[0m\x1b[1;36mart\x1b[46;34m'
    assert strip_ansi(colored) == 'art'
    assert strip_ansi(colored.encode('utf-8')) == b'art'
    assert strip_ansi(bytearray(colored, 'utf-8')) == bytearray(b'art')
    assert strip_ansi('plain text') == 'plain text'


def test_asciify_and_is_ascii():
    assert asciify('Beyoncé') == b'Beyonce'
    assert asciify('plain') == b'plain'
    assert is_ascii('Beyonce') and not is_ascii('Beyoncé')
    assert is_ascii(b'abc') and not is_ascii(b'\xff')
    with pytest.raises(ValueError):
        is_ascii(123)


@pytest.mark.parametrize('nbytes, ndigits, expected', [
    (128991, 0, '126K'),
    (100001221, 0, '95M'),
    (0, 2, '0.00B'),
    (1024, 0, '1K'),
    (1023, 0, '1023B'),
    (1536, 1, '1.5K'),
    (1024 ** 3, 0, '1G'),
    (1024 ** 8, 0, '1Y'),
    (-2048, 0, '-2K'),
])
def test_bytes2human(nbytes, ndigits, expected):
    assert bytes2human(nbytes, ndigits) == expected


def test_html2text_strips_tags_and_decodes_entities():
    assert html2text('<a href="#">Test &amp;<em>(&#956;&#x03CE;)</em></a>') == 'Test &(μώ)'
    assert html2text('<p>one</p><p>two</p>') == 'onetwo'
    assert html2text('no markup') == 'no markup'


def test_gzip_roundtrip():
    payload = b'hello boltons ' * 200
    compressed = gzip_bytes(payload)
    assert len(compressed) < len(payload)
    assert gunzip_bytes(compressed) == payload
    assert gunzip_bytes(gzip_bytes(b'')) == b''


def test_iter_splitlines_handles_mixed_line_endings():
    assert list(iter_splitlines('\nhi\nbye\n')) == ['', 'hi', 'bye', '']
    assert list(iter_splitlines('\r\nhi\rbye\r\n')) == ['', 'hi', 'bye', '']
    assert list(iter_splitlines('')) == []
    assert list(iter_splitlines('single')) == ['single']


def test_indent_skips_empty_lines_by_default():
    assert indent('a\n\nb', '  ') == '  a\n\n  b'
    assert indent('a\n\nb', '> ', key=lambda line: True) == '> a\n> \n> b'


def test_is_uuid_checks_version():
    assert is_uuid('e682ccca-5a4c-4ef2-9711-73f9ad1e15ea')
    assert not is_uuid('0221f0d9-d4b9-11e5-a478-10ddb1c2feb9')
    assert is_uuid('0221f0d9-d4b9-11e5-a478-10ddb1c2feb9', version=1)
    assert is_uuid(uuid.uuid4())
    assert not is_uuid('not-a-uuid')
    assert not is_uuid(None)


def test_args2sh_quotes_unsafe_arguments():
    assert args2sh(['aa', '[bb]', "cc'cc", 'dd"dd']) == '''aa '[bb]' 'cc'"'"'cc' 'dd"dd' '''.strip()
    assert args2sh(['']) == "''"
    assert args2sh(['a', 'b'], sep=',') == 'a,b'


def test_args2cmd_escapes_double_quotes_and_spaces():
    assert args2cmd(['aa', '[bb]', "cc'cc", 'dd"dd']) == '''aa [bb] cc'cc dd\\"dd'''
    assert args2cmd(['has space']) == '"has space"'


def test_escape_shell_args_style_selection():
    assert escape_shell_args(['a b'], style='sh') == "'a b'"
    assert escape_shell_args(['a b'], style='cmd') == '"a b"'
    with pytest.raises(ValueError):
        escape_shell_args(['a'], style='powershell')


def test_parse_int_list_and_format_int_list_are_inverses():
    assert parse_int_list('1,3,5-8,10-11,15') == [1, 3, 5, 6, 7, 8, 10, 11, 15]
    assert format_int_list([1, 3, 5, 6, 7, 8, 10, 11, 15]) == '1,3,5-8,10-11,15'
    assert parse_int_list(format_int_list([4, 5, 6, 9])) == [4, 5, 6, 9]


def test_int_list_custom_delimiters_and_dedupe():
    assert parse_int_list('1;3..5', delim=';', range_delim='..') == [1, 3, 4, 5]
    assert format_int_list([1, 2, 3, 7], delim_space=True) == '1-3, 7'
    assert format_int_list([2, 2, 3]) == '2-3'
    assert parse_int_list('') == []


def test_complement_int_list():
    assert complement_int_list('1,3,5-8,10-11,15') == '0,2,4,9,12-14'
    assert complement_int_list('1,3,5-8,10-11,15', range_start=3) == '4,9,12-14'
    assert complement_int_list('1,3,5-8,10-11,15', range_end=20) == '0,2,4,9,12-14,16-19'
    assert complement_int_list('') == ''


def test_int_ranges_from_int_list():
    assert int_ranges_from_int_list('1,3,5-8,10-11,15') == (
        (1, 1), (3, 3), (5, 8), (10, 11), (15, 15))
    assert int_ranges_from_int_list('') == ()


def test_multi_replace_single_pass_semantics():
    text = 'The foo bar cat ate a bat'
    assert multi_replace(text, {'foo': 'zoo', 'cat': 'hat', 'bat': 'kraken'}) == (
        'The zoo bar hat ate a kraken')
    swap = MultiReplace([('a', 'b'), ('b', 'a')])
    assert swap.sub('aabb') == 'bbaa'


def test_multi_replace_regex_mode():
    mr = MultiReplace({r'\d+': '#', r'\s+': '_'}, regex=True)
    assert mr.sub('a 12  b 3') == 'a_#_b_#'
    assert MultiReplace({}).sub('unchanged') == 'unchanged'


def test_unwrap_text_joins_wrapped_paragraphs():
    text = "Short \n lines  \nwrapped\nsmall.\n\nAnother\nparagraph."
    assert unwrap_text(text) == 'Short lines wrapped small.\n\nAnother paragraph.'
    assert unwrap_text(text, ending=None) == [
        'Short lines wrapped small.', 'Another paragraph.']


def test_removeprefix():
    assert removeprefix('prefix-body', 'prefix-') == 'body'
    assert removeprefix('body', 'prefix-') == 'body'
    assert removeprefix('abc', '') == 'abc'


@pytest.mark.parametrize('items, kwargs, expected', [
    ([], {}, ''),
    (['a'], {}, 'a'),
    (['a', 'b'], {}, 'a and b'),
    (['a', 'b', 'c'], {}, 'a, b, and c'),
    (['a', 'b', 'c'], {'oxford': False}, 'a, b and c'),
    (['a', 'b', 'c'], {'conjunction': 'or'}, 'a, b, or c'),
])
def test_human_readable_list(items, kwargs, expected):
    assert human_readable_list(items, **kwargs) == expected


def test_ellipsize_truncates_at_word_boundaries():
    assert ellipsize('Hello, World!', 16) == 'Hello, World!'
    assert ellipsize('Beautiful is better than ugly. Explicit is better.', 31) == (
        'Beautiful is better than ugly…')
    assert ellipsize('antidisestablishmentarianism', 10) == 'antidises…'


def test_ellipsize_never_exceeds_max_len_and_validates():
    text = 'the quick brown fox jumps over the lazy dog ' * 5
    for max_len in (5, 12, 40, 100):
        assert len(ellipsize(text, max_len)) <= max_len
    assert ellipsize('one two three four', 12, ellipsis='...').endswith('...')
    with pytest.raises(ValueError):
        ellipsize('hello world', 1)
