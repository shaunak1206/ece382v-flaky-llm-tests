"""Unit tests for boltons.iterutils."""
import pytest

from boltons.iterutils import (
    GUIDerator, PathAccessError, SequentialGUIDerator, backoff, bucketize,
    chunk_ranges, chunked, chunked_iter, first, flatten, frange, get_path,
    is_collection, is_iterable, is_scalar, one, pairwise, partition, redundant,
    remap, research, same, soft_sorted, split, split_iter, strip, lstrip,
    rstrip, unique, untyped_sorted, windowed, xfrange, default_enter,
    default_exit)


def test_is_iterable_scalar_collection_classification():
    assert is_iterable([1, 2]) and is_iterable('abc') and not is_iterable(5)
    assert is_scalar(5) and is_scalar('abc') and is_scalar(None)
    assert not is_scalar([1])
    assert is_collection([1]) and is_collection(range(3))
    assert not is_collection('abc') and not is_collection(b'abc')


def test_split_on_none_groups_separators():
    assert split(['hi', 'hello', None, None, 'sup', None, 'soap', None]) == [
        ['hi', 'hello'], ['sup'], ['soap']]


def test_split_with_explicit_separator_keeps_empty_groups():
    assert split([1, 0, 0, 2], sep=[0]) == [[1], [], [2]]
    assert split([1, 2, 3, 4, 5], sep=lambda x: x % 2 == 0) == [[1], [3], [5]]


def test_split_maxsplit_limits_number_of_splits():
    assert split(list('a-b-c-d'), sep='-', maxsplit=1) == [['a'], list('b-c-d')]
    assert split([1, 2, 3], sep=2, maxsplit=0) == [[1, 2, 3]]


def test_split_iter_requires_iterable():
    with pytest.raises(TypeError):
        list(split_iter(5))


def test_strip_helpers_remove_edge_values():
    assert lstrip([0, 0, 1, 2, 0], 0) == [1, 2, 0]
    assert rstrip([0, 1, 2, 0, 0], 0) == [0, 1, 2]
    assert strip([None, 1, None, 2, None], None) == [1, None, 2]


def test_chunked_returns_even_chunks_with_short_tail():
    assert chunked(range(10), 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_chunked_fill_and_count():
    assert chunked(range(10), 3, fill=None)[-1] == [9, None, None]
    assert chunked(range(10), 3, count=2) == [[0, 1, 2], [3, 4, 5]]


def test_chunked_on_strings_returns_string_chunks():
    assert chunked('abcdefg', 3) == ['abc', 'def', 'g']


@pytest.mark.parametrize('size', [0, -2])
def test_chunked_rejects_bad_size(size):
    with pytest.raises(ValueError):
        chunked([1, 2, 3], size)


def test_chunked_iter_rejects_unknown_keyword_and_non_iterable():
    with pytest.raises(ValueError):
        list(chunked_iter([1, 2], 1, bogus=1))
    with pytest.raises(TypeError):
        list(chunked_iter(7, 2))


def test_chunk_ranges_with_offset_overlap_and_align():
    assert list(chunk_ranges(10, 5, input_offset=10)) == [(10, 15), (15, 20)]
    assert list(chunk_ranges(10, 5, input_offset=10, overlap_size=1)) == [
        (10, 15), (14, 19), (18, 20)]
    assert list(chunk_ranges(15, 5, input_offset=4, align=True)) == [
        (4, 5), (5, 10), (10, 15), (15, 19)]


def test_windowed_sliding_window():
    assert windowed(range(5), 3) == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    assert windowed(range(2), 5) == []


def test_windowed_with_fill_pads_the_tail():
    assert windowed(range(4), 3, fill=None) == [
        (0, 1, 2), (1, 2, 3), (2, 3, None), (3, None, None)]


def test_pairwise_with_and_without_end_sentinel():
    assert pairwise(range(4)) == [(0, 1), (1, 2), (2, 3)]
    assert pairwise([]) == []
    assert pairwise('abc', end='END') == [('a', 'b'), ('b', 'c'), ('c', 'END')]


def test_frange_and_xfrange_match_expected_floats():
    assert frange(5) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert frange(100.5, 101.5, 0.25) == [100.5, 100.75, 101.0, 101.25]
    assert frange(5, 0, step=-1.25) == [5.0, 3.75, 2.5, 1.25]
    assert tuple(xfrange(1, 3, step=0.75)) == (1.0, 1.75, 2.5)


def test_frange_zero_step_raises():
    with pytest.raises(ValueError):
        frange(5, step=0)


def test_backoff_is_geometric_and_capped():
    assert backoff(1, 10) == [1.0, 2.0, 4.0, 8.0, 10.0]
    assert backoff(1.0, 10.0, count=8)[-3:] == [10.0, 10.0, 10.0]
    assert backoff(0.25, 100.0, factor=10) == [0.25, 2.5, 25.0, 100.0]


def test_backoff_validation_and_jitter_bounds():
    with pytest.raises(ValueError):
        backoff(5, 1)
    with pytest.raises(ValueError):
        backoff(1, 10, count='repeat')
    values = backoff(1, 10, jitter=1.0)
    assert all(0 <= v <= base for v, base in zip(values, [1, 2, 4, 8, 10]))


def test_bucketize_default_and_custom_key():
    assert bucketize(range(5)) == {False: [0], True: [1, 2, 3, 4]}
    assert bucketize(range(10), lambda x: x % 3) == {
        0: [0, 3, 6, 9], 1: [1, 4, 7], 2: [2, 5, 8]}


def test_bucketize_attribute_name_and_key_list():
    assert bucketize([1 + 1j, 2 + 2j, 1, 2], key='real') == {
        1.0: [1 + 1j, 1], 2.0: [2 + 2j, 2]}
    assert bucketize([1, 2, 365, 4, 98], key=[0, 1, 2, 0, 2]) == {
        0: [1, 4], 1: [2], 2: [365, 98]}


def test_bucketize_value_transform_and_key_filter():
    assert bucketize(range(5), value_transform=lambda x: x * x) == {
        False: [0], True: [1, 4, 9, 16]}
    result = bucketize(range(10), key=lambda x: x % 3,
                       key_filter=lambda k: k != 1)
    assert result == {0: [0, 3, 6, 9], 2: [2, 5, 8]}


def test_bucketize_error_handling():
    with pytest.raises(TypeError):
        bucketize(5)
    with pytest.raises(ValueError):
        bucketize([1, 2, 3], key=[0, 1])
    with pytest.raises(TypeError):
        bucketize([1], key=object())


def test_partition_two_and_many_way():
    nonempty, empty = partition(['', '', 'hi', '', 'bye'])
    assert nonempty == ['hi', 'bye']
    assert empty == ['', '', '']
    pos, neg, zero = partition(range(-2, 3), lambda i: i > 0, lambda i: i < 0)
    assert (pos, neg, zero) == ([1, 2], [-2, -1], [0])


def test_unique_preserves_first_seen_order():
    assert unique([3, 1, 3, 2, 1]) == [3, 1, 2]
    assert ''.join(unique('11010110001010010101010')) == '10'


def test_unique_with_key_function_and_attribute():
    assert unique(['hi', 'hello', 'ok', 'bye', 'yes'], key=len) == [
        'hi', 'hello', 'bye']
    assert unique([1 + 0j, 1 + 2j, 2 + 0j], key='real') == [1 + 0j, 2 + 0j]
    with pytest.raises(TypeError):
        unique([1], key=5)


def test_redundant_returns_first_duplicate_of_each_value():
    assert redundant([1, 2, 3, 4]) == []
    assert redundant([1, 2, 3, 2, 3, 3, 4]) == [2, 3]


def test_redundant_groups_and_key():
    assert redundant([1, 2, 3, 2, 3, 3, 4], groups=True) == [[2, 2], [3, 3, 3]]
    assert redundant(['hi', 'Hi', 'HI', 'hello'], key=str.lower) == ['Hi']
    assert redundant(['hi', 'Hi', 'HI', 'hello'], groups=True,
                     key=str.lower) == [['hi', 'Hi', 'HI']]


def test_first_returns_first_truthy_or_default():
    assert first([0, False, None, [], (), 42]) == 42
    assert first([0, None]) is None
    assert first([0, None], default='fallback') == 'fallback'
    assert first([1, 1, 3, 4, 5], key=lambda x: x % 2 == 0) == 4


def test_one_requires_exactly_one_match():
    assert one((0, 0, 'a')) == 'a'
    assert one((True, False, True)) is None
    assert one((True, True), default=False) is False
    assert one((10, 20, 30, 42), key=lambda i: i > 40) == 42


def test_flatten_and_same():
    assert flatten([[1, 2], [[3], [4, 5]], 'ab']) == [1, 2, 3, 4, 5, 'ab']
    assert same([]) and same([1]) and same(['a'] * 3)
    assert not same(range(4))
    assert same([2, 2], ref=2)
    assert not same([[], []], ref='x')


def test_remap_drops_none_values_recursively():
    reviews = {'Star Trek': {'TNG': 10, 'DS9': 8.5, 'ENT': None},
               'Babylon 5': 6, 'Dr. Who': None}
    result = remap(reviews, lambda p, k, v: v is not None)
    assert result == {'Star Trek': {'TNG': 10, 'DS9': 8.5}, 'Babylon 5': 6}


def test_remap_transforms_values_and_preserves_container_types():
    data = {'a': [1, 2, (3, 4)], 'b': {5, 6}}
    result = remap(data, visit=lambda p, k, v: (k, v * 10) if isinstance(v, int) else (k, v))
    assert result == {'a': [10, 20, (30, 40)], 'b': {50, 60}}
    assert isinstance(result['a'][2], tuple)
    assert isinstance(result['b'], set)


def test_remap_visit_receives_path_of_parent_keys():
    seen = []

    def visit(path, key, value):
        if isinstance(value, int):
            seen.append((path, key))
        return key, value

    remap({'x': {'y': [7]}}, visit=visit)
    assert seen == [(('x', 'y'), 0)]


def test_remap_custom_enter_and_exit_sort_lists():
    def exit_sorted(path, key, old_parent, new_parent, new_items):
        ret = default_exit(path, key, old_parent, new_parent, new_items)
        if isinstance(ret, list):
            ret.sort()
        return ret

    result = remap({'k': [3, 1, 2], 'j': [[9, 8]]}, exit=exit_sorted)
    assert result == {'k': [1, 2, 3], 'j': [[8, 9]]}


def test_remap_handles_self_referential_structures():
    loop = [1]
    loop.append(loop)
    result = remap(loop)
    assert result[0] == 1
    assert result[1] is result


def test_remap_rejects_non_callables_and_bad_root():
    with pytest.raises(TypeError):
        remap({}, visit='nope')
    with pytest.raises(TypeError):
        remap(5)


def test_remap_reraise_visit_false_keeps_items_on_error():
    def bad_visit(path, key, value):
        raise RuntimeError('boom')

    assert remap({'a': 1}, visit=bad_visit, reraise_visit=False) == {'a': 1}
    with pytest.raises(RuntimeError):
        remap({'a': 1}, visit=bad_visit)


def test_get_path_nested_lookup_and_dotted_string():
    root = {'a': {'b': {'c': [[1], [2], [3]]}}}
    assert get_path(root, ('a', 'b', 'c', 2, 0)) == 3
    assert get_path(root, 'a.b.c.1.0') == 2


def test_get_path_errors_and_default():
    root = {'a': [1, 2]}
    with pytest.raises(PathAccessError):
        get_path(root, ('a', 5))
    with pytest.raises(KeyError):
        get_path(root, ('missing',))
    assert get_path(root, ('a', 5), default='dflt') == 'dflt'


def test_research_finds_matching_leaves_with_paths():
    root = {'a': {'b': 1, 'c': (2, 'd', 3)}, 'e': None}
    found = research(root, query=lambda p, k, v: isinstance(v, int))
    assert sorted(found) == [(('a', 'b'), 1), (('a', 'c', 0), 2), (('a', 'c', 2), 3)]


def test_soft_sorted_pins_first_and_last_elements():
    assert soft_sorted(['two', 'b', 'one', 'a'], first=['one', 'two']) == [
        'one', 'two', 'a', 'b']
    assert soft_sorted(range(7), first=[6, 15], last=[2, 4], reverse=True) == [
        6, 5, 3, 1, 0, 2, 4]


def test_untyped_sorted_handles_mixed_types():
    result = untyped_sorted(['abc', 2.0, 1, 2, 'def'])
    assert result[:3] == [1, 2.0, 2] or result[:3] == [1, 2, 2.0]
    assert sorted(x for x in result if isinstance(x, str)) == ['abc', 'def']
    assert len(result) == 5


def test_guiderator_yields_unique_hex_ids_of_requested_size():
    gen = GUIDerator(size=30)
    ids = [next(gen) for _ in range(50)]
    assert len(set(ids)) == 50
    assert all(len(i) == 30 and int(i, 16) >= 0 for i in ids)
    with pytest.raises(ValueError):
        GUIDerator(size=5)


def test_sequential_guiderator_is_monotonic():
    gen = SequentialGUIDerator()
    ids = [next(gen) for _ in range(10)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 10
