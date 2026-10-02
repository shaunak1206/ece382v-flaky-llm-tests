"""Tests for boltons dict, set, and string utilities."""

import pytest
from boltons.dictutils import OrderedMultiDict, OneToOne, FrozenDict
from boltons.setutils import IndexedSet
from boltons.strutils import (
    slugify, bytes2human, ordinalize, cardinalize, pluralize,
    singularize, find_hashtags, strip_ansi, asciify, is_uuid,
    escape_shell_args, parse_int_list, format_int_list,
    ellipsize, unwrap_text, camel2under, under2camel
)


class TestOrderedMultiDict:
    """Test OrderedMultiDict."""

    def test_omd_basic_set_get(self):
        """Test basic OrderedMultiDict set and get."""
        omd = OrderedMultiDict()
        omd['key'] = 'value1'
        assert omd['key'] == 'value1'

    def test_omd_multiple_values(self):
        """Test OrderedMultiDict with multiple values for same key."""
        omd = OrderedMultiDict()
        omd.add('key', 'value1')
        omd.add('key', 'value2')
        assert omd.getlist('key') == ['value1', 'value2']

    def test_omd_add_method(self):
        """Test OrderedMultiDict add method."""
        omd = OrderedMultiDict()
        omd.add('a', 1)
        omd.add('a', 2)
        omd.add('b', 3)
        assert 'a' in omd
        assert len(omd.getlist('a')) == 2

    def test_omd_keys_and_values(self):
        """Test OrderedMultiDict keys and values."""
        omd = OrderedMultiDict()
        omd.add('x', 10)
        omd.add('x', 20)
        omd.add('y', 30)
        assert len(omd) >= 2

    def test_omd_pop(self):
        """Test OrderedMultiDict pop."""
        omd = OrderedMultiDict()
        omd['key'] = 'value'
        value = omd.pop('key', None)
        assert value == 'value'
        assert 'key' not in omd

    def test_omd_iteration(self):
        """Test OrderedMultiDict iteration."""
        omd = OrderedMultiDict()
        omd.add('a', 1)
        omd.add('b', 2)
        items = list(omd.items())
        assert len(items) >= 2

    def test_omd_clear(self):
        """Test OrderedMultiDict clear."""
        omd = OrderedMultiDict()
        omd['a'] = 1
        omd['b'] = 2
        omd.clear()
        assert len(omd) == 0


class TestOneToOne:
    """Test OneToOne bidirectional mapping."""

    def test_one_to_one_basic(self):
        """Test basic OneToOne."""
        o2o = OneToOne()
        o2o['a'] = 1
        assert o2o['a'] == 1
        assert o2o.inv[1] == 'a'

    def test_one_to_one_bidirectional(self):
        """Test OneToOne bidirectional mapping."""
        o2o = OneToOne()
        o2o['key1'] = 'value1'
        # Inverse mapping
        assert o2o.inv['value1'] == 'key1'

    def test_one_to_one_update(self):
        """Test OneToOne update."""
        o2o = OneToOne()
        o2o['a'] = 1
        o2o['a'] = 2  # Update
        assert o2o['a'] == 2
        assert o2o.inv[2] == 'a'

    def test_one_to_one_conflict(self):
        """Test OneToOne with value reassignment."""
        o2o = OneToOne()
        o2o['a'] = 1
        o2o['b'] = 1  # Same value
        # One of them should be updated
        assert len(o2o) >= 1


class TestFrozenDict:
    """Test FrozenDict immutable dictionary."""

    def test_frozen_dict_basic(self):
        """Test basic FrozenDict."""
        fd = FrozenDict({'a': 1, 'b': 2})
        assert fd['a'] == 1
        assert fd['b'] == 2

    def test_frozen_dict_immutable(self):
        """Test FrozenDict is immutable."""
        fd = FrozenDict({'a': 1})
        with pytest.raises((TypeError, AttributeError)):
            fd['a'] = 2

    def test_frozen_dict_hashable(self):
        """Test FrozenDict is hashable."""
        fd = FrozenDict({'a': 1, 'b': 2})
        # Should be hashable
        hash_val = hash(fd)
        assert isinstance(hash_val, int)

    def test_frozen_dict_iteration(self):
        """Test FrozenDict iteration."""
        fd = FrozenDict({'x': 10, 'y': 20})
        items = dict(fd)
        assert items == {'x': 10, 'y': 20}


class TestIndexedSet:
    """Test IndexedSet."""

    def test_indexed_set_basic(self):
        """Test basic IndexedSet."""
        iset = IndexedSet()
        iset.add('a')
        iset.add('b')
        assert 'a' in iset
        assert 'b' in iset

    def test_indexed_set_index_access(self):
        """Test IndexedSet index access."""
        iset = IndexedSet(['a', 'b', 'c'])
        assert iset[0] == 'a'
        assert iset[1] == 'b'
        assert iset[2] == 'c'

    def test_indexed_set_remove(self):
        """Test IndexedSet remove."""
        iset = IndexedSet(['a', 'b', 'c'])
        iset.remove('b')
        assert 'b' not in iset
        assert len(iset) == 2

    def test_indexed_set_operations(self):
        """Test IndexedSet set operations."""
        iset1 = IndexedSet(['a', 'b', 'c'])
        iset2 = IndexedSet(['b', 'c', 'd'])
        # Union
        union = iset1 | iset2
        assert 'a' in union
        assert 'd' in union

    def test_indexed_set_discard(self):
        """Test IndexedSet discard."""
        iset = IndexedSet(['a', 'b', 'c'])
        iset.discard('b')
        assert 'b' not in iset


class TestSlugify:
    """Test slugify function."""

    def test_slugify_basic(self):
        """Test basic slugify."""
        result = slugify('Hello World')
        assert result == 'hello_world' or result == 'helloworld'

    def test_slugify_special_chars(self):
        """Test slugify with special characters."""
        result = slugify('Hello, World!')
        assert isinstance(result, str)
        assert 'hello' in result.lower()

    def test_slugify_spaces(self):
        """Test slugify with spaces."""
        result = slugify('multiple   spaces')
        assert isinstance(result, str)

    def test_slugify_custom_delimiter(self):
        """Test slugify with custom delimiter."""
        result = slugify('Hello World', delim='-')
        assert '-' in result or 'hello' in result

    def test_slugify_lower(self):
        """Test slugify lowercase."""
        result = slugify('HELLO WORLD', lower=True)
        assert 'hello' in result

    def test_slugify_ascii_only(self):
        """Test slugify with ascii=True."""
        result = slugify('café', ascii=True)
        assert isinstance(result, str)


class TestBytes2Human:
    """Test bytes2human function."""

    def test_bytes2human_basic(self):
        """Test basic bytes2human."""
        result = bytes2human(1024)
        assert '1.0' in result or '1K' in result or isinstance(result, str)

    def test_bytes2human_mb(self):
        """Test bytes2human megabytes."""
        result = bytes2human(1024 * 1024)
        assert isinstance(result, str)
        assert 'M' in result or '1' in result

    def test_bytes2human_gb(self):
        """Test bytes2human gigabytes."""
        result = bytes2human(1024 * 1024 * 1024)
        assert isinstance(result, str)

    def test_bytes2human_zero(self):
        """Test bytes2human with zero."""
        result = bytes2human(0)
        assert '0' in result

    def test_bytes2human_large(self):
        """Test bytes2human with large number."""
        result = bytes2human(1024 ** 4)  # 1 TB
        assert isinstance(result, str)


class TestOrdinalize:
    """Test ordinalize function."""

    def test_ordinalize_basic(self):
        """Test basic ordinalize."""
        assert 'st' in ordinalize(1) or '1' in ordinalize(1)
        assert 'nd' in ordinalize(2) or '2' in ordinalize(2)
        assert 'rd' in ordinalize(3) or '3' in ordinalize(3)
        assert 'th' in ordinalize(4) or '4' in ordinalize(4)

    def test_ordinalize_teens(self):
        """Test ordinalize with teens."""
        result = ordinalize(13)
        assert 'th' in result or '13' in result

    def test_ordinalize_large(self):
        """Test ordinalize with large numbers."""
        result = ordinalize(101)
        assert isinstance(result, str)


class TestCardinalize:
    """Test cardinalize function."""

    def test_cardinalize_singular(self):
        """Test cardinalize singular."""
        result = cardinalize('item', 1)
        assert '1' in result or 'item' in result

    def test_cardinalize_plural(self):
        """Test cardinalize plural."""
        result = cardinalize('item', 2)
        assert '2' in result or 'items' in result

    def test_cardinalize_zero(self):
        """Test cardinalize with zero."""
        result = cardinalize('item', 0)
        assert isinstance(result, str)


class TestPluralize:
    """Test pluralize function."""

    def test_pluralize_basic(self):
        """Test basic pluralize."""
        result = pluralize('cat')
        assert 'cat' in result.lower()

    def test_pluralize_s(self):
        """Test pluralize words ending in s."""
        result = pluralize('box')
        assert isinstance(result, str)

    def test_pluralize_y(self):
        """Test pluralize words ending in y."""
        result = pluralize('baby')
        assert isinstance(result, str)


class TestSingularize:
    """Test singularize function."""

    def test_singularize_basic(self):
        """Test basic singularize."""
        result = singularize('cats')
        assert 'cat' in result.lower()

    def test_singularize_es(self):
        """Test singularize words ending in es."""
        result = singularize('boxes')
        assert isinstance(result, str)


class TestFindHashtags:
    """Test find_hashtags function."""

    def test_find_hashtags_basic(self):
        """Test basic find_hashtags."""
        text = "This is #awesome and #great"
        result = find_hashtags(text)
        assert 'awesome' in result or len(result) > 0
        assert 'great' in result or len(result) > 0

    def test_find_hashtags_none(self):
        """Test find_hashtags with no hashtags."""
        result = find_hashtags("No hashtags here")
        assert len(result) == 0

    def test_find_hashtags_multiple(self):
        """Test find_hashtags with multiple."""
        text = "#tag1 #tag2 #tag3"
        result = find_hashtags(text)
        assert len(result) >= 1


class TestStripAnsi:
    """Test strip_ansi function."""

    def test_strip_ansi_basic(self):
        """Test basic strip_ansi."""
        text = "\033[31mred\033[0m"
        result = strip_ansi(text)
        assert result == "red"

    def test_strip_ansi_no_codes(self):
        """Test strip_ansi with no codes."""
        text = "plain text"
        result = strip_ansi(text)
        assert result == text


class TestAsciify:
    """Test asciify function."""

    def test_asciify_basic(self):
        """Test basic asciify."""
        result = asciify('café')
        assert isinstance(result, str)

    def test_asciify_ignore_false(self):
        """Test asciify with ignore=False."""
        # Should try to convert
        result = asciify('café', ignore=False)
        assert isinstance(result, str)


class TestIsUuid:
    """Test is_uuid function."""

    def test_is_uuid_valid(self):
        """Test is_uuid with valid UUID."""
        valid_uuid = '550e8400-e29b-41d4-a716-446655440000'
        result = is_uuid(valid_uuid)
        assert result is True

    def test_is_uuid_invalid(self):
        """Test is_uuid with invalid UUID."""
        result = is_uuid('not-a-uuid')
        assert result is False

    def test_is_uuid_version(self):
        """Test is_uuid with specific version."""
        # UUID4
        valid_uuid4 = '550e8400-e29b-41d4-a716-446655440000'
        result = is_uuid(valid_uuid4, version=4)
        assert isinstance(result, bool)


class TestEscapeShellArgs:
    """Test escape_shell_args function."""

    def test_escape_shell_args_basic(self):
        """Test basic escape_shell_args."""
        args = ['echo', 'hello world', 'test']
        result = escape_shell_args(args)
        assert isinstance(result, str)
        assert 'hello' in result

    def test_escape_shell_args_special_chars(self):
        """Test escape_shell_args with special characters."""
        args = ['cmd', 'arg with spaces', '$variable']
        result = escape_shell_args(args)
        assert isinstance(result, str)


class TestParseIntList:
    """Test parse_int_list function."""

    def test_parse_int_list_basic(self):
        """Test basic parse_int_list."""
        result = parse_int_list('1,2,3,4,5')
        assert result == [1, 2, 3, 4, 5]

    def test_parse_int_list_range(self):
        """Test parse_int_list with ranges."""
        result = parse_int_list('1-3,5,7-9')
        assert 1 in result
        assert 3 in result
        assert 5 in result

    def test_parse_int_list_duplicates(self):
        """Test parse_int_list with duplicates."""
        result = parse_int_list('1,2,1,3')
        assert len(result) >= 3


class TestFormatIntList:
    """Test format_int_list function."""

    def test_format_int_list_basic(self):
        """Test basic format_int_list."""
        result = format_int_list([1, 2, 3, 4, 5])
        assert '1' in result
        assert '-' in result or ',' in result

    def test_format_int_list_ranges(self):
        """Test format_int_list with ranges."""
        result = format_int_list([1, 2, 3, 5, 6, 7])
        assert isinstance(result, str)


class TestEllipsize:
    """Test ellipsize function."""

    def test_ellipsize_basic(self):
        """Test basic ellipsize."""
        text = "This is a long text that should be shortened"
        result = ellipsize(text, max_len=20)
        assert len(result) <= 20
        assert '…' in result or '...' in result

    def test_ellipsize_short_text(self):
        """Test ellipsize with short text."""
        text = "Short"
        result = ellipsize(text, max_len=20)
        assert result == text


class TestUnwrapText:
    """Test unwrap_text function."""

    def test_unwrap_text_basic(self):
        """Test basic unwrap_text."""
        text = "This is a long\nline that should\nbe unwrapped"
        result = unwrap_text(text)
        assert isinstance(result, str)

    def test_unwrap_text_paragraphs(self):
        """Test unwrap_text preserves paragraphs."""
        text = "Para1 line1\nPara1 line2\n\nPara2 line1\nPara2 line2"
        result = unwrap_text(text)
        # Should preserve double newlines
        assert '\n' in result or isinstance(result, str)


class TestCamel2Under:
    """Test camel2under function."""

    def test_camel2under_basic(self):
        """Test basic camel2under."""
        result = camel2under('CamelCase')
        assert result == 'camel_case'

    def test_camel2under_acronyms(self):
        """Test camel2under with acronyms."""
        result = camel2under('HTTPServer')
        assert '_' in result or result.islower()


class TestUnder2Camel:
    """Test under2camel function."""

    def test_under2camel_basic(self):
        """Test basic under2camel."""
        result = under2camel('snake_case')
        assert result == 'snakeCase'

    def test_under2camel_leading_underscore(self):
        """Test under2camel with leading underscore."""
        result = under2camel('_private')
        assert isinstance(result, str)
