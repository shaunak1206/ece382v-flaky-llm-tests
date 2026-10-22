import datetime
from decimal import Decimal
from fractions import Fraction
from itertools import accumulate, count

import pytest

from more_itertools.more import (
    SequenceView,
    always_iterable,
    consecutive_groups,
    count_cycle,
    difference,
    distribute,
    divide,
    exactly_n,
    islice_extended,
    locate,
    map_reduce,
    mark_ends,
    numeric_range,
    padded,
    replace,
    rlocate,
    run_length,
    seekable,
    side_effect,
    sliced,
    sort_together,
    split_after,
    split_at,
    split_before,
    split_into,
    split_when,
    stagger,
    strip,
    lstrip,
    rstrip,
    unzip,
    zip_offset,
)
from more_itertools.recipes import consume


def test_side_effect_calls_function_and_hooks():
    seen = []
    assert list(side_effect(seen.append, [1, 2, 3])) == [1, 2, 3]
    assert seen == [1, 2, 3]

    sums = []
    func = lambda chunk: sums.append(sum(chunk))
    assert list(side_effect(func, [0, 1, 2, 3, 4, 5], 2)) == [0, 1, 2, 3, 4, 5]
    assert sums == [1, 5, 9]

    log = []
    consume(
        side_effect(
            lambda x: log.append(('item', x)),
            'ab',
            before=lambda: log.append('before'),
            after=lambda: log.append('after'),
        )
    )
    assert log == ['before', ('item', 'a'), ('item', 'b'), 'after']

    log = []

    def boom(x):
        raise RuntimeError

    with pytest.raises(RuntimeError):
        list(side_effect(boom, [1], after=lambda: log.append('after')))
    assert log == ['after']


def test_sliced_splits_sequences():
    assert list(sliced((1, 2, 3, 4, 5, 6), 3)) == [(1, 2, 3), (4, 5, 6)]
    assert list(sliced((1, 2, 3, 4, 5, 6, 7, 8), 3)) == [(1, 2, 3), (4, 5, 6), (7, 8)]
    assert list(sliced('abcde', 2)) == ['ab', 'cd', 'e']
    assert list(sliced([], 3)) == []
    assert list(sliced([1, 2, 3, 4], 2, strict=True)) == [[1, 2], [3, 4]]
    with pytest.raises(ValueError):
        list(sliced([1, 2, 3], 2, strict=True))
    with pytest.raises(ValueError):
        list(sliced([1, 2, 3], -1))


def test_split_at():
    assert list(split_at('abcdcba', lambda x: x == 'b')) == [['a'], ['c', 'd', 'c'], ['a']]
    assert list(split_at(range(10), lambda n: n % 2 == 1)) == [[0], [2], [4], [6], [8], []]
    assert list(split_at(range(10), lambda n: n % 2 == 1, maxsplit=2)) == [
        [0],
        [2],
        [4, 5, 6, 7, 8, 9],
    ]
    assert list(split_at('abcdcba', lambda x: x == 'b', keep_separator=True)) == [
        ['a'],
        ['b'],
        ['c', 'd', 'c'],
        ['b'],
        ['a'],
    ]
    assert list(split_at([], lambda x: True)) == [[]]
    assert list(split_at([1, 2, 3], lambda x: x == 2, maxsplit=0)) == [[1, 2, 3]]


def test_split_before_and_split_after():
    assert list(split_before('OneTwo', lambda s: s.isupper())) == [
        ['O', 'n', 'e'],
        ['T', 'w', 'o'],
    ]
    assert list(split_before(range(10), lambda n: n % 3 == 0)) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9],
    ]
    assert list(split_before(range(10), lambda n: n % 3 == 0, maxsplit=2)) == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8, 9],
    ]
    assert list(split_after('one1two2', lambda s: s.isdigit())) == [
        ['o', 'n', 'e', '1'],
        ['t', 'w', 'o', '2'],
    ]
    assert list(split_after(range(10), lambda n: n % 3 == 0)) == [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert list(split_after(range(10), lambda n: n % 3 == 0, maxsplit=2)) == [
        [0],
        [1, 2, 3],
        [4, 5, 6, 7, 8, 9],
    ]
    assert list(split_before([], bool)) == []
    assert list(split_after([], bool)) == []
    assert list(split_before([1, 2, 3], lambda x: x == 2, maxsplit=0)) == [[1, 2, 3]]


def test_split_when_and_split_into():
    data = [1, 2, 3, 3, 2, 5, 2, 4, 2]
    assert list(split_when(data, lambda x, y: x > y)) == [[1, 2, 3, 3], [2, 5], [2, 4], [2]]
    assert list(split_when(data, lambda x, y: x > y, maxsplit=2)) == [
        [1, 2, 3, 3],
        [2, 5],
        [2, 4, 2],
    ]
    assert list(split_when([], lambda x, y: True)) == []
    assert list(split_when([7], lambda x, y: True)) == [[7]]

    assert list(split_into([1, 2, 3, 4, 5, 6], [1, 2, 3])) == [[1], [2, 3], [4, 5, 6]]
    assert list(split_into([1, 2, 3, 4, 5, 6], [2, 3])) == [[1, 2], [3, 4, 5]]
    assert list(split_into([1, 2, 3, 4], [1, 2, 3, 4])) == [[1], [2, 3], [4], []]
    assert list(split_into([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], [2, 3, None])) == [
        [1, 2],
        [3, 4, 5],
        [6, 7, 8, 9, 0],
    ]


def test_padded():
    assert list(padded([1, 2, 3], '?', 5)) == [1, 2, 3, '?', '?']
    assert list(padded([1, 2, 3, 4], n=3, next_multiple=True)) == [1, 2, 3, 4, None, None]
    assert list(padded([1, 2, 3], n=3, next_multiple=True)) == [1, 2, 3]
    assert list(padded([1, 2, 3], '?', 2)) == [1, 2, 3]
    assert list(padded([], 0, 3)) == [0, 0, 0]
    infinite = padded([1, 2, 3], '?')
    assert [next(infinite) for _ in range(6)] == [1, 2, 3, '?', '?', '?']
    with pytest.raises(ValueError):
        padded([1], n=0)


def test_distribute_and_divide():
    group_1, group_2 = distribute(2, [1, 2, 3, 4, 5, 6])
    assert list(group_1) == [1, 3, 5]
    assert list(group_2) == [2, 4, 6]
    assert [list(c) for c in distribute(3, [1, 2, 3, 4, 5, 6, 7])] == [
        [1, 4, 7],
        [2, 5],
        [3, 6],
    ]
    assert [list(c) for c in distribute(5, [1, 2, 3])] == [[1], [2], [3], [], []]

    part_1, part_2 = divide(2, [1, 2, 3, 4, 5, 6])
    assert list(part_1) == [1, 2, 3]
    assert list(part_2) == [4, 5, 6]
    assert [list(c) for c in divide(3, [1, 2, 3, 4, 5, 6, 7])] == [
        [1, 2, 3],
        [4, 5],
        [6, 7],
    ]
    assert [list(c) for c in divide(5, iter([1, 2, 3]))] == [[1], [2], [3], [], []]

    with pytest.raises(ValueError):
        distribute(0, [1, 2])
    with pytest.raises(ValueError):
        divide(0, [1, 2])


def test_stagger_zip_offset_and_always_iterable():
    assert list(stagger([0, 1, 2, 3])) == [(None, 0, 1), (0, 1, 2), (1, 2, 3)]
    assert list(stagger(range(8), offsets=(0, 2, 4))) == [
        (0, 2, 4),
        (1, 3, 5),
        (2, 4, 6),
        (3, 5, 7),
    ]
    assert list(stagger([0, 1, 2, 3], longest=True)) == [
        (None, 0, 1),
        (0, 1, 2),
        (1, 2, 3),
        (2, 3, None),
        (3, None, None),
    ]
    assert list(zip_offset('0123', 'abcdef', offsets=(0, 1))) == [
        ('0', 'b'),
        ('1', 'c'),
        ('2', 'd'),
        ('3', 'e'),
    ]
    assert list(zip_offset('0123', 'abcdef', offsets=(0, 1), longest=True)) == [
        ('0', 'b'),
        ('1', 'c'),
        ('2', 'd'),
        ('3', 'e'),
        (None, 'f'),
    ]
    with pytest.raises(ValueError):
        zip_offset('ab', offsets=(0, 1))

    assert list(always_iterable((1, 2, 3))) == [1, 2, 3]
    assert list(always_iterable(1)) == [1]
    assert list(always_iterable(None)) == []
    assert list(always_iterable('foo')) == ['foo']
    assert list(always_iterable(b'foo')) == [b'foo']
    assert list(always_iterable({'a': 1})) == ['a']
    assert list(always_iterable({'a': 1}, base_type=dict)) == [{'a': 1}]
    assert list(always_iterable('foo', base_type=None)) == ['f', 'o', 'o']


def test_sort_together_and_unzip():
    assert sort_together([(4, 3, 2, 1), ('a', 'b', 'c', 'd')]) == [
        (1, 2, 3, 4),
        ('d', 'c', 'b', 'a'),
    ]
    assert sort_together([(3, 1, 2), (0, 1, 0), ('c', 'b', 'a')], key_list=(1, 2)) == [
        (2, 3, 1),
        (0, 0, 1),
        ('a', 'c', 'b'),
    ]
    names, lengths, widths = ('a', 'b', 'c'), (1, 2, 3), (5, 2, 1)
    assert sort_together(
        [names, lengths, widths], key_list=(1, 2), key=lambda length, width: length * width
    ) == [('c', 'b', 'a'), (3, 2, 1), (1, 2, 5)]
    assert sort_together([(1, 2, 3), ('c', 'b', 'a')], reverse=True) == [
        (3, 2, 1),
        ('a', 'b', 'c'),
    ]
    with pytest.raises(ValueError):
        sort_together([(1, 2), (3,)], strict=True)

    letters, numbers = unzip([('a', 1), ('b', 2), ('c', 3), ('d', 4)])
    assert list(letters) == ['a', 'b', 'c', 'd']
    assert list(numbers) == [1, 2, 3, 4]
    assert unzip([]) == ()
    xs, ys, zs = unzip(iter(zip('abc', range(3), 'xyz')))
    assert (list(xs), list(ys), list(zs)) == (['a', 'b', 'c'], [0, 1, 2], ['x', 'y', 'z'])


def test_numeric_range_construction_and_iteration():
    assert list(numeric_range(3.5)) == [0.0, 1.0, 2.0, 3.0]
    assert list(numeric_range(0, 10, 3)) == [0, 3, 6, 9]
    assert list(numeric_range(5, 0, -2)) == [5, 3, 1]
    assert list(numeric_range(3, -1, -1.0)) == [3.0, 2.0, 1.0, 0.0]
    assert list(numeric_range(0.0, 1.0, 0.25)) == [0.0, 0.25, 0.5, 0.75]
    assert list(numeric_range(5, 5)) == []
    assert list(numeric_range(Decimal('2.1'), Decimal('5.1'))) == [
        Decimal('2.1'),
        Decimal('3.1'),
        Decimal('4.1'),
    ]
    assert list(numeric_range(Fraction(1, 2), Fraction(5, 2), Fraction(1, 2))) == [
        Fraction(1, 2),
        Fraction(1, 1),
        Fraction(3, 2),
        Fraction(2, 1),
    ]
    start = datetime.datetime(2019, 1, 1)
    stop = datetime.datetime(2019, 1, 3)
    step = datetime.timedelta(days=1)
    assert list(numeric_range(start, stop, step)) == [
        datetime.datetime(2019, 1, 1),
        datetime.datetime(2019, 1, 2),
    ]
    with pytest.raises(ValueError):
        numeric_range(0, 10, 0)
    with pytest.raises(TypeError):
        numeric_range()
    with pytest.raises(TypeError):
        numeric_range(1, 2, 3, 4)


def test_numeric_range_sequence_operations():
    nr = numeric_range(0, 10, 3)
    assert len(nr) == 4
    assert nr[0] == 0
    assert nr[-1] == 9
    assert list(nr[1:3]) == [3, 6]
    assert 6 in nr
    assert 7 not in nr
    assert nr.index(6) == 2
    assert nr.count(3) == 1
    assert nr.count(4) == 0
    assert list(reversed(nr)) == [9, 6, 3, 0]
    with pytest.raises(ValueError):
        nr.index(7)
    with pytest.raises(IndexError):
        nr[4]
    with pytest.raises(TypeError):
        nr['a']

    assert bool(numeric_range(3))
    assert not bool(numeric_range(0))
    assert numeric_range(0, 10, 3) == numeric_range(0, 12, 3)
    assert numeric_range(0, 10, 3) != numeric_range(0, 10, 2)
    assert numeric_range(0, 0) == numeric_range(5, 5)
    assert numeric_range(0, 10, 3) != range(0, 10, 3)
    assert hash(numeric_range(0, 10, 3)) == hash(numeric_range(0, 12, 3))
    assert repr(numeric_range(0, 10, 3)) == 'numeric_range(0, 10, 3)'
    assert repr(numeric_range(0, 5)) == 'numeric_range(0, 5)'


def test_count_cycle_and_mark_ends():
    assert list(count_cycle('AB', 3)) == [
        (0, 'A'),
        (0, 'B'),
        (1, 'A'),
        (1, 'B'),
        (2, 'A'),
        (2, 'B'),
    ]
    assert list(count_cycle('AB', 0)) == []
    assert list(count_cycle([])) == []
    inf = count_cycle('AB')
    assert [next(inf) for _ in range(5)] == [(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B'), (2, 'A')]

    assert list(mark_ends('ABC')) == [(True, False, 'A'), (False, False, 'B'), (False, True, 'C')]
    assert list(mark_ends([])) == []
    assert list(mark_ends([1])) == [(True, True, 1)]


def test_locate_and_rlocate():
    assert list(locate([0, 1, 1, 0, 1, 0, 0])) == [1, 2, 4]
    assert list(locate(['a', 'b', 'c', 'b'], lambda x: x == 'b')) == [1, 3]
    iterable = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    pred = lambda *args: args == (1, 2, 3)
    assert list(locate(iterable, pred=pred, window_size=3)) == [1, 5, 9]
    with pytest.raises(ValueError):
        list(locate([1, 2], window_size=0))

    assert list(rlocate([0, 1, 1, 0, 1, 0, 0])) == [4, 2, 1]
    assert list(rlocate(iter('abcb'), lambda x: x == 'b')) == [3, 1]
    assert list(rlocate(iterable, pred=pred, window_size=3)) == [9, 5, 1]
    assert list(rlocate([])) == []


def test_strip_functions():
    iterable = (None, False, None, 1, 2, None, 3, False, None)
    pred = lambda x: x in {None, False, ''}
    assert list(lstrip(iterable, pred)) == [1, 2, None, 3, False, None]
    assert list(rstrip(iterable, pred)) == [None, False, None, 1, 2, None, 3]
    assert list(strip(iterable, pred)) == [1, 2, None, 3]
    assert list(strip([1, 2, 3], pred)) == [1, 2, 3]
    assert list(strip([None, False], pred)) == []
    assert list(strip([], pred)) == []


def test_islice_extended_negative_indices_and_steps():
    assert list(islice_extended(iter('abcdefgh'), -4, -1)) == ['e', 'f', 'g']
    assert list(islice_extended(count(), 110, 99, -2)) == [110, 108, 106, 104, 102, 100]
    assert list(islice_extended(map(str, count()))[10:20:2]) == ['10', '12', '14', '16', '18']
    assert list(islice_extended(range(10), 2, 8, 2)) == [2, 4, 6]
    assert list(islice_extended(range(10), -3, None)) == [7, 8, 9]
    assert list(islice_extended(range(10), None, -2)) == list(range(8))
    assert list(islice_extended(range(5), None, None, -1)) == [4, 3, 2, 1, 0]
    assert list(islice_extended(range(10), 1, None, 3)) == [1, 4, 7]
    assert list(islice_extended(range(3))) == [0, 1, 2]
    with pytest.raises(TypeError):
        islice_extended(range(3))[1]
    with pytest.raises(ValueError):
        list(islice_extended(range(3), 0, 3, 0))


def test_seekable_seek_peek_and_sequence_view():
    s = seekable(iter(range(5)))
    assert [next(s) for _ in range(3)] == [0, 1, 2]
    s.seek(0)
    assert list(s) == [0, 1, 2, 3, 4]
    s.seek(2)
    assert list(s) == [2, 3, 4]
    s.seek(10)
    assert list(s) == []
    s.seek(0)
    assert next(s) == 0

    p = seekable('1234')
    assert p
    assert p.peek() == '1'
    assert list(p) == ['1', '2', '3', '4']
    assert not p
    assert p.peek(default='empty') == 'empty'
    with pytest.raises(StopIteration):
        p.peek()

    it = seekable((str(n) for n in range(20)))
    assert next(it) == '0' and next(it) == '1' and next(it) == '2'
    it.relative_seek(2)
    assert next(it) == '5'
    it.relative_seek(-3)
    assert next(it) == '3'
    assert it[0] == '0'

    bounded = seekable((str(n) for n in count()), maxlen=2)
    assert [next(bounded) for _ in range(4)] == ['0', '1', '2', '3']
    assert list(bounded.elements()) == ['2', '3']
    bounded.seek(0)
    assert [next(bounded) for _ in range(4)] == ['2', '3', '4', '5']

    backing = ['0', '1', '2']
    view = SequenceView(backing)
    backing.append('3')
    assert len(view) == 4
    assert view[1] == '1'
    assert view[1:-1] == ['1', '2']
    assert repr(view) == "SequenceView(['0', '1', '2', '3'])"
    with pytest.raises(TypeError):
        SequenceView(iter([1, 2]))


def test_run_length_and_exactly_n():
    assert list(run_length.encode('abbcccdddd')) == [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
    assert list(run_length.decode([('a', 1), ('b', 2), ('c', 3), ('d', 4)])) == list(
        'abbcccdddd'
    )
    assert list(run_length.encode('')) == []
    text = 'aaabccddddde'
    assert ''.join(run_length.decode(run_length.encode(text))) == text

    assert exactly_n([True, True, False], 2) is True
    assert exactly_n([True, True, False], 1) is False
    assert exactly_n([0, 1, 2, 3, 4, 5], 3, lambda x: x < 3) is True
    assert exactly_n([], 0) is True
    assert exactly_n([1], 0) is False
    assert exactly_n([1, 2], -1) is False
    assert exactly_n([1], 2) is False
    assert exactly_n(iter([True] * 100), 3) is False


def test_map_reduce():
    keyfunc = lambda x: x.upper()
    result = map_reduce('abbccc', keyfunc)
    assert dict(result) == {'A': ['a'], 'B': ['b', 'b'], 'C': ['c', 'c', 'c']}
    assert dict(map_reduce('abbccc', keyfunc, lambda x: 1)) == {
        'A': [1],
        'B': [1, 1],
        'C': [1, 1, 1],
    }
    assert dict(map_reduce('abbccc', keyfunc, lambda x: 1, sum)) == {'A': 1, 'B': 2, 'C': 3}

    items = [x for x in range(30) if 10 <= x <= 20]
    parity = lambda x: x % 2
    assert dict(map_reduce(items, parity, reducefunc=sum)) == {0: 90, 1: 75}

    assert dict(map_reduce([], parity)) == {}
    with pytest.raises(KeyError):
        result['Z']


def test_replace():
    pred = lambda x: x == 0
    assert list(replace([1, 1, 0, 1, 1, 0, 1, 1], pred, (2, 3))) == [1, 1, 2, 3, 1, 1, 2, 3, 1, 1]
    assert list(replace([1, 1, 0, 1, 1, 0, 1, 1, 0], pred, [None], count=2)) == [
        1, 1, None, 1, 1, None, 1, 1, 0,
    ]
    window_pred = lambda *args: args == (0, 1, 2)
    assert list(replace([0, 1, 2, 5, 0, 1, 2, 5], window_pred, [3, 4], window_size=3)) == [
        3, 4, 5, 3, 4, 5,
    ]
    assert list(replace([1, 2, 3], pred, [9])) == [1, 2, 3]
    assert list(replace([], pred, [9])) == []
    assert list(replace([0, 0], pred, [])) == []
    with pytest.raises(ValueError):
        list(replace([1, 2], pred, [3], window_size=0))


def test_difference_and_consecutive_groups():
    assert list(difference(accumulate([0, 1, 2, 3, 4]))) == [0, 1, 2, 3, 4]
    assert list(difference([1, 2, 6, 24, 120], lambda x, y: x // y)) == [1, 2, 3, 4, 5]
    assert list(difference([10, 11, 13, 16], initial=10)) == [1, 2, 3]
    assert list(difference([])) == []

    groups = [list(g) for g in consecutive_groups([1, 10, 11, 12, 20, 30, 31, 32, 33, 40])]
    assert groups == [[1], [10, 11, 12], [20], [30, 31, 32, 33], [40]]
    letters = [list(g) for g in consecutive_groups('abcdfgilmnop', ord)]
    assert letters == [
        ['a', 'b', 'c', 'd'],
        ['f', 'g'],
        ['i'],
        ['l', 'm', 'n', 'o', 'p'],
    ]
    assert [list(g) for g in consecutive_groups([])] == []
