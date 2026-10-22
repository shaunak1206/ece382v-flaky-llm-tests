import itertools

import pytest

from boltons.iterutils import (
    PathAccessError,
    backoff,
    backoff_iter,
    bucketize,
    chunk_ranges,
    chunked,
    chunked_iter,
    first,
    flatten,
    frange,
    get_path,
    is_collection,
    is_iterable,
    is_scalar,
    one,
    pairwise,
    partition,
    redundant,
    remap,
    research,
    same,
    split,
    split_iter,
    strip,
    rstrip,
    lstrip,
    unique,
    windowed,
)


def test_type_predicates():
    assert is_iterable([])
    assert is_iterable('abc')
    assert not is_iterable(object())
    assert not is_iterable(5)

    assert is_scalar('hello')
    assert is_scalar(b'bytes')
    assert is_scalar(3)
    assert not is_scalar(range(3))

    assert is_collection(range(10))
    assert is_collection({'a': 1})
    assert not is_collection('hello')
    assert not is_collection(object())


def test_split():
    src = ['hi', 'hello', None, None, 'sup', None, 'soap', None]
    assert split(src) == [['hi', 'hello'], ['sup'], ['soap']]
    assert (list(split_iter(['hi', 'hello', None, None, 'sup', None], sep=[None]))
            == [['hi', 'hello'], [], ['sup'], []])
    assert split([1, 0, 2, 0, 3], 0, maxsplit=1) == [[1], [2, 0, 3]]
    assert split([1, 0, 2], 0, maxsplit=0) == [[1, 0, 2]]
    assert (list(split_iter(['hi', 'hello', None, '', 'sup', False], lambda x: not x))
            == [['hi', 'hello'], [], ['sup'], []])
    with pytest.raises(TypeError):
        split(5)


def test_strip_family():
    assert lstrip(['Foo', 'Bar', 'Bam'], 'Foo') == ['Bar', 'Bam']
    assert rstrip(['Foo', 'Bar', 'Bam'], 'Bam') == ['Foo', 'Bar']
    assert strip(['Fu', 'Foo', 'Bar', 'Bam', 'Fu'], 'Fu') == ['Foo', 'Bar', 'Bam']
    assert strip([0, 0, 1, 2, 0, 0], 0) == [1, 2]
    # matching values in the middle must be preserved
    assert rstrip(['a', 'x', 'b', 'x', 'x'], 'x') == ['a', 'x', 'b']
    assert strip([None, 1, None, 2, None]) == [1, None, 2]
    assert strip([]) == []


def test_chunked():
    assert chunked(range(10), 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert chunked(range(10), 3, fill=None) == [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]
    assert chunked(range(10), 3, count=2) == [[0, 1, 2], [3, 4, 5]]
    assert chunked('abcdefg', 3) == ['abc', 'def', 'g']
    assert chunked('abcde', 2, fill='x') == ['ab', 'cd', 'ex']
    assert chunked(b'abcde', 2) == [b'ab', b'cd', b'e']
    assert chunked([], 3) == []
    assert list(chunked_iter([1, 2, 3], 2)) == [[1, 2], [3]]

    with pytest.raises(ValueError):
        chunked([1, 2], 0)
    with pytest.raises(ValueError):
        chunked([1, 2], 2, bogus=True)
    with pytest.raises(TypeError):
        chunked(5, 2)


def test_chunk_ranges():
    assert list(chunk_ranges(input_offset=10, input_size=10, chunk_size=5)) == [
        (10, 15), (15, 20)]
    assert list(chunk_ranges(input_offset=10, input_size=10, chunk_size=5,
                             overlap_size=1)) == [(10, 15), (14, 19), (18, 20)]
    assert list(chunk_ranges(input_offset=10, input_size=10, chunk_size=5,
                             overlap_size=2)) == [(10, 15), (13, 18), (16, 20)]
    assert list(chunk_ranges(input_offset=4, input_size=15, chunk_size=5,
                             align=False)) == [(4, 9), (9, 14), (14, 19)]
    assert list(chunk_ranges(input_offset=4, input_size=15, chunk_size=5,
                             align=True)) == [(4, 5), (5, 10), (10, 15), (15, 19)]
    with pytest.raises(ValueError):
        list(chunk_ranges(10, 0))


def test_pairwise_and_windowed():
    assert pairwise(range(5)) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert pairwise([]) == []
    assert pairwise(range(3), end=None) == [(0, 1), (1, 2), (2, None)]
    assert windowed(range(7), 3) == [
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]
    assert windowed(range(3), 5) == []
    assert windowed(range(4), 3, fill=None) == [
        (0, 1, 2), (1, 2, 3), (2, 3, None), (3, None, None)]


def test_frange():
    assert frange(5) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert frange(6, step=1.25) == [0.0, 1.25, 2.5, 3.75, 5.0]
    assert frange(100.5, 101.5, 0.25) == [100.5, 100.75, 101.0, 101.25]
    assert frange(5, 0) == []
    assert frange(5, 0, step=-1.25) == [5.0, 3.75, 2.5, 1.25]
    with pytest.raises(ValueError):
        frange(5, step=0)


def test_backoff():
    assert backoff(1, 10) == [1.0, 2.0, 4.0, 8.0, 10.0]
    assert list(backoff_iter(1.0, 10.0, count=5)) == [1.0, 2.0, 4.0, 8.0, 10.0]
    assert list(backoff_iter(1.0, 10.0, count=8)) == [
        1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0]
    assert list(backoff_iter(0.25, 100.0, factor=10)) == [0.25, 2.5, 25.0, 100.0]
    repeated = list(itertools.islice(backoff_iter(1, 4, count='repeat'), 6))
    assert repeated == [1.0, 2.0, 4.0, 4.0, 4.0, 4.0]

    base = backoff(1, 10)
    jittered = backoff(1, 10, jitter=True)
    assert len(jittered) == len(base)
    for jit, ref in zip(jittered, base):
        assert 0 <= jit <= ref


def test_backoff_invalid_arguments():
    with pytest.raises(ValueError):
        backoff(1, 10, count='repeat')
    with pytest.raises(ValueError):
        backoff(1, 10, factor=0.5)
    with pytest.raises(ValueError):
        backoff(10, 1)
    with pytest.raises(ValueError):
        backoff(-1, 10)
    with pytest.raises(ValueError):
        backoff(1, 10, jitter=2)


def test_bucketize_and_partition():
    assert bucketize(range(5)) == {False: [0], True: [1, 2, 3, 4]}
    assert bucketize(range(10), lambda x: x % 3) == {
        0: [0, 3, 6, 9], 1: [1, 4, 7], 2: [2, 5, 8]}
    assert bucketize([1 + 1j, 2 + 2j, 1, 2], key='real') == {
        1.0: [1 + 1j, 1], 2.0: [2 + 2j, 2]}
    assert bucketize([1, 2, 365, 4, 98], key=[0, 1, 2, 0, 2]) == {
        0: [1, 4], 1: [2], 2: [365, 98]}
    assert bucketize(range(5), value_transform=lambda x: x * x) == {
        False: [0], True: [1, 4, 9, 16]}
    assert bucketize(range(10), key=lambda x: x % 3,
                     key_filter=lambda k: k % 3 != 1) == {
        0: [0, 3, 6, 9], 2: [2, 5, 8]}
    with pytest.raises(TypeError):
        bucketize(5)
    with pytest.raises(ValueError):
        bucketize([1, 2, 3], key=[0, 1])

    nonempty, empty = partition(['', '', 'hi', '', 'bye'])
    assert nonempty == ['hi', 'bye']
    assert empty == ['', '', '']
    assert partition(range(-1, 2), lambda i: i > 0, lambda i: i < 0) == (
        [1], [-1], [0])
    assert partition([1 + 0j, 2j, 3 + 0j], 'imag') == ([2j], [1 + 0j, 3 + 0j])
    with pytest.raises(TypeError):
        partition(5)


def test_unique_and_redundant():
    assert ''.join(unique('11010110001010010101010')) == '10'
    assert unique([1, 2, 3] * 10) == [1, 2, 3]
    assert unique(['hi', 'hello', 'ok', 'bye', 'yes'], key=len) == [
        'hi', 'hello', 'bye']
    with pytest.raises(TypeError):
        unique(5)

    assert redundant([1, 2, 3, 4]) == []
    assert redundant([1, 2, 3, 2, 3, 3, 4]) == [2, 3]
    assert redundant([1, 2, 3, 2, 3, 3, 4], groups=True) == [[2, 2], [3, 3, 3]]
    assert redundant(['hi', 'Hi', 'HI', 'hello'], key=str.lower) == ['Hi']
    assert redundant(['hi', 'Hi', 'HI', 'hello'], groups=True,
                     key=str.lower) == [['hi', 'Hi', 'HI']]


def test_one_and_first():
    assert one((True, False, False)) is True
    assert one((True, False, True)) is None
    assert one((0, 0, 'a')) == 'a'
    assert one((0, False, None)) is None
    assert one((True, True), default=False) is False
    assert one((10, 20, 30, 42), key=lambda i: i > 40) == 42

    assert first([0, False, None, [], (), 42]) == 42
    assert first([0, False, None, [], ()]) is None
    assert first([0, False, None, [], ()], default='ohai') == 'ohai'
    assert first([1, 1, 3, 4, 5], key=lambda x: x % 2 == 0) == 4


def test_flatten_and_same():
    assert flatten([[1, 2], [[3], [4, 5]]]) == [1, 2, 3, 4, 5]
    assert flatten(['ab', ['cd', (b'e',)]]) == ['ab', 'cd', b'e']
    assert flatten([]) == []

    assert same([])
    assert same([1])
    assert same(['a', 'a', 'a'])
    assert not same(range(20))
    assert same([[], []])
    assert not same([[], []], ref='test')
    assert same([5, 5], ref=5)


def test_remap():
    reviews = {'Star Trek': {'TNG': 10, 'DS9': 8.5, 'ENT': None},
               'Babylon 5': 6, 'Dr. Who': None}
    assert remap(reviews, lambda p, k, v: v is not None) == {
        'Star Trek': {'TNG': 10, 'DS9': 8.5}, 'Babylon 5': 6}

    data = {'a': [1, 2, {'b': (3, 4)}], 'c': {5, 6}}
    copy = remap(data)
    assert copy == data
    assert copy is not data
    assert copy['a'] is not data['a']
    assert isinstance(copy['a'][2]['b'], tuple)

    doubled = remap({'a': 1, 'b': [2, 3]},
                    visit=lambda p, k, v: (k, v * 2) if isinstance(v, int) else (k, v))
    assert doubled == {'a': 2, 'b': [4, 6]}

    seen = []
    remap({'a': {'b': 1}}, visit=lambda p, k, v: seen.append((p, k)) or True)
    assert set(seen) == {(('a',), 'b'), ((), 'a')}

    root = {'a': {'b': 1, 'c': (2, 'd', 3)}, 'e': None}
    found = research(root, query=lambda p, k, v: isinstance(v, int))
    assert sorted(found) == [(('a', 'b'), 1), (('a', 'c', 0), 2), (('a', 'c', 2), 3)]

    with pytest.raises(TypeError):
        remap({}, visit=5)
    with pytest.raises(TypeError):
        remap(5)


def test_get_path():
    root = {'a': {'b': {'c': [[1], [2], [3]]}}}
    assert get_path(root, ('a', 'b', 'c', 2, 0)) == 3
    assert get_path(root, 'a.b.c.1.0') == 2
    assert get_path(root, ('a', 'x'), default=None) is None
    assert get_path(root, ('a', 'b', 'c', 9), default='gone') == 'gone'

    with pytest.raises(PathAccessError) as excinfo:
        get_path(root, ('a', 'x'))
    assert isinstance(excinfo.value, KeyError)
    assert 'could not access' in str(excinfo.value)

    with pytest.raises(IndexError):
        get_path(root, ('a', 'b', 'c', 9))
    with pytest.raises(PathAccessError):
        get_path(root, ('a', 'b', 'c', 0, 0, 0))
