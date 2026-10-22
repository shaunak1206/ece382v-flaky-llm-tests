import pytest

from boltons.iterutils import (
    GUIDerator,
    PathAccessError,
    SequentialGUIDerator,
    backoff,
    backoff_iter,
    bucketize,
    chunk_ranges,
    chunked,
    chunked_iter,
    first,
    flatten,
    flatten_iter,
    frange,
    get_path,
    is_collection,
    is_iterable,
    is_scalar,
    lstrip,
    lstrip_iter,
    one,
    pairwise,
    partition,
    redundant,
    remap,
    research,
    rstrip,
    rstrip_iter,
    same,
    soft_sorted,
    split,
    split_iter,
    strip,
    strip_iter,
    unique,
    unique_iter,
    untyped_sorted,
    windowed,
    xfrange,
)


def test_is_iterable_is_scalar_is_collection():
    assert is_iterable([])
    assert not is_iterable(object())
    assert is_scalar(object())
    assert is_scalar('hello')
    assert not is_scalar(range(10))
    assert is_collection(range(10))
    assert not is_collection('hello')
    assert not is_collection(object())


def test_split_and_split_iter():
    src = ['hi', 'hello', None, None, 'sup', None, 'soap', None]
    assert split(src) == [['hi', 'hello'], ['sup'], ['soap']]
    assert list(split_iter(src)) == [['hi', 'hello'], ['sup'], ['soap']]
    assert split([1, 0, 2, 0, 3], 0) == [[1], [2], [3]]
    assert split([1, 0, 2, 0, 3], 0, maxsplit=1) == [[1], [2, 0, 3]]
    assert split([1, 2, 3, 4, 5], lambda x: x % 2 == 0) == [[1], [3], [5]]


def test_strip_family_returns_lists_and_iter_variants_agree():
    assert lstrip(['Foo', 'Bar', 'Bam'], 'Foo') == ['Bar', 'Bam']
    assert rstrip(['Foo', 'Bar', 'Bam'], 'Bam') == ['Foo', 'Bar']
    assert strip(['Fu', 'Foo', 'Bar', 'Bam', 'Fu'], 'Fu') == ['Foo', 'Bar', 'Bam']
    assert list(lstrip_iter(['Foo', 'Bar', 'Bam'], 'Foo')) == ['Bar', 'Bam']
    assert list(rstrip_iter(['Foo', 'Bar', 'Bam'], 'Bam')) == ['Foo', 'Bar']
    assert list(strip_iter(['Fu', 'Foo', 'Bar', 'Bam', 'Fu'], 'Fu')) == ['Foo', 'Bar', 'Bam']
    assert strip([None, 1, 2, None]) == [1, 2]


def test_chunked_and_chunked_iter():
    assert chunked(range(10), 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert chunked(range(10), 3, fill=None) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]
    assert chunked(range(10), 3, count=2) == [[0, 1, 2], [3, 4, 5]]
    assert list(chunked_iter(range(10), 3)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert list(chunked_iter(range(10), 3, fill=None))[-1] == [9, None, None]
    assert chunked([], 3) == []


def test_chunk_ranges_with_offset_and_overlap():
    assert list(chunk_ranges(input_offset=10, input_size=10, chunk_size=5)) == [(10, 15), (15, 20)]
    assert list(chunk_ranges(input_offset=10, input_size=10, chunk_size=5, overlap_size=1)) == [
        (10, 15),
        (14, 19),
        (18, 20),
    ]


def test_pairwise_with_and_without_end():
    assert pairwise(range(5)) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert pairwise([]) == []
    assert pairwise(range(3), end=None) == [(0, 1), (1, 2), (2, None)]


def test_windowed():
    assert windowed(range(5), 3) == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]
    assert windowed(range(3), 5) == []
    assert windowed(range(4), 3, fill=None) == [
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, None),
        (3, None, None),
    ]


def test_frange_and_xfrange():
    assert frange(5) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert frange(6, step=1.25) == [0.0, 1.25, 2.5, 3.75, 5.0]
    assert frange(100.5, 101.5, 0.25) == [100.5, 100.75, 101.0, 101.25]
    assert frange(5, 0) == []
    assert frange(5, 0, step=-1.25) == [5.0, 3.75, 2.5, 1.25]
    assert tuple(xfrange(1, 3, step=0.75)) == (1.0, 1.75, 2.5)


def test_backoff_sequences():
    assert backoff(1, 10) == [1.0, 2.0, 4.0, 8.0, 10.0]
    assert list(backoff_iter(1.0, 10.0, count=5)) == [1.0, 2.0, 4.0, 8.0, 10.0]
    assert list(backoff_iter(1.0, 10.0, count=8)) == [1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0]
    assert list(backoff_iter(0.25, 100.0, factor=10)) == [0.25, 2.5, 25.0, 100.0]


def test_bucketize_with_callable_attribute_and_list_keys():
    assert bucketize(range(5)) == {False: [0], True: [1, 2, 3, 4]}
    assert bucketize(range(5), lambda x: x % 2 == 1) == {False: [0, 2, 4], True: [1, 3]}
    assert bucketize([1 + 1j, 2 + 2j, 1, 2], key='real') == {1.0: [1 + 1j, 1], 2.0: [2 + 2j, 2]}
    assert bucketize([1, 2, 365, 4, 98], key=[0, 1, 2, 0, 2]) == {0: [1, 4], 1: [2], 2: [365, 98]}


def test_partition_single_and_multiple_predicates():
    nonempty, empty = partition(['', '', 'hi', '', 'bye'])
    assert nonempty == ['hi', 'bye']
    assert empty == ['', '', '']
    low, mid, high = partition(range(10), lambda x: x < 3, lambda x: x < 6)
    assert low == [0, 1, 2]
    assert mid == [3, 4, 5]
    assert high == [6, 7, 8, 9]


def test_unique_and_redundant():
    assert ''.join(unique('11010110001010010101010')) == '10'
    assert list(unique_iter([1, 2, 3] * 10)) == [1, 2, 3]
    assert list(unique_iter(['hi', 'hello', 'ok', 'bye', 'yes'], key=len)) == ['hi', 'hello', 'bye']
    assert redundant([1, 2, 3, 4]) == []
    assert redundant([1, 2, 3, 2, 3, 3, 4]) == [2, 3]
    assert redundant([1, 2, 3, 2, 3, 3, 4], groups=True) == [[2, 2], [3, 3, 3]]
    assert redundant(['hi', 'Hi', 'HI', 'hello'], key=str.lower) == ['Hi']


def test_one_and_first():
    assert one((True, False, False)) is True
    assert one((True, False, True)) is None
    assert one((0, 0, 'a')) == 'a'
    assert one((0, False, None)) is None
    assert one((True, True), default=False) is False
    assert first([0, False, None, [], (), 42]) == 42
    assert first([0, False, None, [], ()]) is None
    assert first([0, False, None, [], ()], default='ohai') == 'ohai'
    assert first([1, 1, 3, 4, 5], key=lambda x: x % 2 == 0) == 4


def test_flatten_and_flatten_iter():
    nested = [[1, 2], [[3], [4, 5]]]
    assert flatten(nested) == [1, 2, 3, 4, 5]
    assert list(flatten_iter(nested)) == [1, 2, 3, 4, 5]
    assert flatten(['ab', ['c']]) == ['ab', 'c']
    assert flatten([]) == []


def test_same():
    assert same([])
    assert same([1])
    assert same(['a', 'a', 'a'])
    assert not same(range(20))
    assert same([[], []])
    assert not same([[], []], ref='test')
    assert same([5, 5], ref=5)


def test_remap_drops_values_and_transforms_values():
    data = {'a': 1, 'b': None, 'c': {'d': None, 'e': 2}}
    assert remap(data, visit=lambda p, k, v: v is not None) == {'a': 1, 'c': {'e': 2}}
    assert data['b'] is None

    def double_ints(path, key, value):
        if isinstance(value, int):
            return key, value * 2
        return True

    assert remap({'a': 1, 'b': [1, 2]}, visit=double_ints) == {'a': 2, 'b': [2, 4]}


def test_get_path_and_errors():
    root = {'a': {'b': {'c': [[1], [2], [3]]}}}
    assert get_path(root, ('a', 'b', 'c', 2, 0)) == 3
    assert get_path({'a': {'b': 1}}, 'a.b') == 1
    assert get_path(root, ('a', 'missing'), default=None) is None
    assert get_path(root, ('a', 'missing'), default=0) == 0
    with pytest.raises(PathAccessError):
        get_path(root, ('a', 'b', 'c', 2, 1))
    with pytest.raises(KeyError):
        get_path(root, ('nope',))


def test_research_finds_matching_values_with_paths():
    root = {'a': {'b': 1, 'c': (2, 'd', 3)}, 'e': None}
    res = research(root, query=lambda p, k, v: isinstance(v, int))
    assert sorted(res) == [(('a', 'b'), 1), (('a', 'c', 0), 2), (('a', 'c', 2), 3)]


def test_soft_sorted_and_untyped_sorted():
    assert soft_sorted(['two', 'b', 'one', 'a'], first=['one', 'two']) == ['one', 'two', 'a', 'b']
    assert soft_sorted(range(7), first=[6, 15], last=[2, 4], reverse=True) == [6, 5, 3, 1, 0, 2, 4]
    assert untyped_sorted(['abc', 2.0, 1, 2, 'def']) == [1, 2.0, 2, 'abc', 'def']


def test_guiderators_produce_unique_hex_strings():
    gen = GUIDerator()
    values = [next(gen) for _ in range(100)]
    assert len(set(values)) == 100
    assert all(len(v) == 24 for v in values)
    assert all(int(v, 16) >= 0 for v in values)
    assert len(next(GUIDerator(size=32))) == 32

    seq = SequentialGUIDerator()
    seq_values = [next(seq) for _ in range(50)]
    assert len(set(seq_values)) == 50
    assert seq_values == sorted(seq_values)
