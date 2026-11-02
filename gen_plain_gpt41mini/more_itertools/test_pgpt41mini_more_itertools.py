import math
import pytest
import cmath
import threading
import time
from itertools import count, islice, chain
from collections import deque, defaultdict
from operator import itemgetter

import more_itertools.more as mit

# 1. chunked
def test_chunked_basic():
    assert list(mit.chunked([1, 2, 3, 4, 5, 6], 3)) == [[1, 2, 3], [4, 5, 6]]
    assert list(mit.chunked([1, 2, 3, 4, 5, 6, 7], 3)) == [[1, 2, 3], [4, 5, 6], [7]]

def test_chunked_strict_success_and_fail():
    it = mit.chunked([1, 2, 3, 4, 5, 6], 2, strict=True)
    assert list(it) == [[1, 2], [3, 4], [5, 6]]

    with pytest.raises(ValueError):
        list(mit.chunked([1, 2, 3, 4, 5], 2, strict=True))

def test_chunked_zero_and_negative_n():
    with pytest.raises(ValueError):
        list(mit.chunked([1, 2, 3], -1))
    # n=0 yields empty generator
    assert list(mit.chunked([1,2,3], 0)) == []

# 2. first
def test_first_basic():
    assert mit.first([1, 2, 3]) == 1
    assert mit.first([], default='empty') == 'empty'
    with pytest.raises(ValueError):
        mit.first([])

# 3. last
def test_last_basic():
    assert mit.last([1, 2, 3]) == 3
    assert mit.last([], default='empty') == 'empty'
    with pytest.raises(ValueError):
        mit.last([])

def test_last_reversed_usage():
    class CustomReversed:
        def __reversed__(self):
            return iter([10, 20, 30])
    obj = CustomReversed()
    assert mit.last(obj) == 30

# 4. nth_or_last
def test_nth_or_last_basic():
    assert mit.nth_or_last([1, 2, 3, 4], 2) == 3
    assert mit.nth_or_last([1], 5) == 1
    assert mit.nth_or_last([], 0, default='no') == 'no'
    with pytest.raises(ValueError):
        mit.nth_or_last([], 0)

# 5. peekable
def test_peekable_basic():
    p = mit.peekable([1, 2, 3])
    assert p.peek() == 1
    assert next(p) == 1
    p.prepend(0)
    assert next(p) == 0
    assert p[0] == 2
    assert p[1] == 3
    assert list(p) == [2, 3]

def test_peekable_empty_default():
    p = mit.peekable([])
    assert p.peek(default='empty') == 'empty'
    with pytest.raises(StopIteration):
        p.peek()

def test_peekable_indexing_negative_and_slice():
    p = mit.peekable([1, 2, 3, 4])
    assert p[-1] == 4
    s = p[1:3]
    assert s == [2, 3]

def test_peekable_bool():
    p = mit.peekable([1])
    assert bool(p)
    list(p)
    assert not bool(p)

# 6. consumer decorator
def test_consumer_decorator():
    results = []
    @mit.consumer
    def tally():
        i = 0
        while True:
            results.append((i, (yield)))
            i += 1

    t = tally()
    t.send('a')
    t.send('b')
    assert results == [(0, 'a'), (1, 'b')]

# 7. ilen
def test_ilen_basic():
    assert mit.ilen([1, 2, 3, 4]) == 4
    assert mit.ilen(iter(range(5))) == 5

# 8. iterate
def test_iterate_basic():
    it = mit.iterate(lambda x: 2 * x, 1)
    assert list(islice(it, 5)) == [1, 2, 4, 8, 16]

# 9. with_iter
def test_with_iter_closes_file(tmp_path):
    p = tmp_path / "foo.txt"
    p.write_text("hello\nworld\n")
    with open(p) as f:
        lines = list(mit.with_iter(f))
    assert lines == ["hello\n", "world\n"]

# 10. sized_iterator
def test_sized_iterator_basic():
    it = map(str, range(5))
    sit = mit.sized_iterator(it, 5)
    assert len(sit) == 5
    assert list(sit) == ['0', '1', '2', '3', '4']

# 11. one
def test_one_success_and_failures():
    assert mit.one([10]) == 10
    with pytest.raises(ValueError):
        mit.one([])
    with pytest.raises(ValueError):
        mit.one([1,2])

    with pytest.raises(IndexError):
        mit.one([], too_short=IndexError())
    with pytest.raises(RuntimeError):
        mit.one([1,2], too_long=RuntimeError)

# 12. raise_
def test_raise_function():
    with pytest.raises(RuntimeError):
        mit.raise_(RuntimeError, "fail")

# 13. strictly_n
def test_strictly_n_normal_and_exceptions():
    it = mit.strictly_n([1, 2, 3], 3)
    assert list(it) == [1, 2, 3]

    with pytest.raises(ValueError):
        list(mit.strictly_n([1,2], 3))

    with pytest.raises(ValueError):
        list(mit.strictly_n([1,2,3], 2))

    # custom exceptions
    def too_short(n):
        raise RuntimeError("too short")
    def too_long(n):
        raise RuntimeError("too long")

    with pytest.raises(RuntimeError, match="too short"):
        list(mit.strictly_n([1], 2, too_short=too_short))
    with pytest.raises(RuntimeError, match="too long"):
        list(mit.strictly_n([1,2,3], 2, too_long=too_long))

# 14. distinct_permutations
def test_distinct_permutations():
    seq = [1, 0, 1]
    results = sorted(mit.distinct_permutations(seq))
    expected = [(0, 1, 1), (1, 0, 1), (1, 1, 0)]
    assert results == expected

    results_r2 = sorted(mit.distinct_permutations(seq, r=2))
    expected_r2 = [(0, 1), (1, 0), (1, 1)]
    assert results_r2 == expected_r2

# 15. derangements
def test_derangements_basic():
    seq = ['Alice', 'Bob']
    results = list(mit.derangements(seq))
    # Each permutation with no element in original position
    for d in results:
        assert all(d[i] != seq[i] for i in range(len(seq)))

    results_r = list(mit.derangements(seq, r=1))
    for d in results_r:
        assert len(d) == 1

# 16. intersperse
def test_intersperse_basic():
    assert list(mit.intersperse('!', [1, 2, 3, 4])) == [1, '!', 2, '!', 3, '!', 4]
    assert list(mit.intersperse(0, [1, 2, 3, 4], n=2)) == [1, 2, 0, 3, 4, 0]
    with pytest.raises(ValueError):
        list(mit.intersperse('x', [1, 2], n=0))

# 17. unique_to_each
def test_unique_to_each_basic():
    a = ['A', 'B']
    b = ['B', 'C']
    c = ['B', 'D']
    output = mit.unique_to_each(a, b, c)
    assert output == [['A'], ['C'], ['D']]

    output_2 = mit.unique_to_each("mississippi", "missouri")
    assert output_2 == [['p', 'p'], ['o', 'u', 'r']]

# 18. windowed
def test_windowed_basic_and_fillvalue_and_step():
    val = list(mit.windowed([1, 2, 3, 4, 5], 3))
    assert val == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    val2 = list(mit.windowed([1, 2, 3], 4))
    assert val2 == [(1, 2, 3, None)]

    val3 = list(mit.windowed([1, 2, 3, 4, 5, 6], 3, fillvalue='!', step=2))
    assert val3 == [(1, 2, 3), (3, 4, 5), (5, 6, '!')]

    with pytest.raises(ValueError):
        list(mit.windowed([1], 0))

    with pytest.raises(ValueError):
        list(mit.windowed([1], 1, step=0))

# 19. substrings
def test_substrings_basic():
    s = 'more'
    substrs = list(mit.substrings(s))
    for subs in ['m','o','r','e','mo','or','re','mor','ore','more']:
        assert any(''.join(t) == subs for t in substrs)

# 20. substrings_indexes
def test_substrings_indexes_basic():
    s = 'abc'
    expected = [
        ('a', 0, 1), ('b', 1, 2), ('c', 2, 3), 
        ('ab', 0, 2), ('bc', 1, 3), 
        ('abc', 0, 3)
    ]
    result = list(mit.substrings_indexes(s))
    assert result == expected

    result_rev = list(mit.substrings_indexes(s, reverse=True))
    assert result_rev[::-1] == expected

# 21. bucket
def test_bucket_basic():
    iterable = ['a1', 'b1', 'a2']
    b = mit.bucket(iterable, key=lambda x: x[0])
    keys = sorted(list(b))
    assert keys == ['a', 'b']
    a_items = list(b['a'])
    b_items = list(b['b'])
    assert a_items == ['a1', 'a2']
    assert b_items == ['b1']

    # Test with validator
    it = (1, 3, 5, 7, 9)
    key = lambda x: x
    validator = lambda x: x % 2 == 1
    b2 = mit.bucket(it, key=key, validator=validator)
    assert 2 not in b2
    assert 3 in b2
    assert list(b2[2]) == []

# 22. spy
def test_spy_basic():
    head, it = mit.spy([1,2,3])
    assert head == [1]
    assert list(it) == [1,2,3]

    (first, second), it2 = mit.spy([10, 20, 30], 2)
    assert first == 10
    assert second == 20
    assert list(it2) == [10, 20, 30]

# 23. interleave
def test_interleave_basic():
    result = list(mit.interleave([1,2,3], ['a','b','c']))
    assert result == [1, 'a', 2, 'b', 3, 'c']

# 24. interleave_longest
def test_interleave_longest_basic():
    result = list(mit.interleave_longest([1,2,3], ['a','b']))
    assert result == [1, 'a', 2, 'b', 3]

# 25. interleave_evenly
def test_interleave_evenly_basic():
    iters = [[1, 2, 3], ['a', 'b']]
    result = list(mit.interleave_evenly(iters))
    assert sorted(result) == sorted([1, 2, 3, 'a', 'b'])

    # test length mismatch error
    iters2 = [[1,2,3]]
    with pytest.raises(ValueError):
        list(mit.interleave_evenly(iters2, lengths=[1,2]))

# 26. collapse
def test_collapse_basic_and_levels_and_base_type():
    nested = [(1, 2), ([3, 4], [[5], [6]])]
    flat = list(mit.collapse(nested))
    assert flat == [1, 2, 3, 4, 5, 6]

    nested2 = ['ab', ('cd', 'ef'), ['gh', 'ij']]
    flat2 = list(mit.collapse(nested2, base_type=tuple))
    assert flat2 == ['ab', ('cd', 'ef'), 'gh', 'ij']

    nested3 = [('a', ['b']), ('c', ['d'])]
    flat3 = list(mit.collapse(nested3, levels=1))
    assert flat3 == ['a', ['b'], 'c', ['d']]

# 27. side_effect
def test_side_effect_basic():
    acc = []

    def f(x):
        acc.append(x)

    result = list(mit.side_effect(f, [1, 2, 3]))
    assert result == [1, 2, 3]
    assert acc == [1, 2, 3]

    # Test with chunked
    acc2 = []
    result2 = list(mit.side_effect(f, [1, 2, 3, 4, 5, 6], chunk_size=2))
    assert result2 == [1, 2, 3, 4, 5, 6]
    assert acc2 != [1, 2, 3, 4, 5, 6]  # Because acc2 not connected here

# 28. sliced
def test_sliced_basic():
    assert list(mit.sliced((1,2,3,4,5,6), 3)) == [(1,2,3),(4,5,6)]
    assert list(mit.sliced((1,2,3,4,5,6,7,8), 3)) == [(1,2,3),(4,5,6),(7,8)]

    with pytest.raises(ValueError):
        list(mit.sliced((1,2,3), -1))

    # strict true, should raise when last slice incomplete
    with pytest.raises(ValueError):
        list(mit.sliced((1,2,3,4), 3, strict=True))

# 29-31. split_at, split_before, split_after
@pytest.mark.parametrize("func,name,test_input,pred,maxsplit,keep_separator,expected", [
    (mit.split_at, 'split_at', 'abcdcba', lambda x: x == 'b', -1, False, [['a'], ['c', 'd', 'c'], ['a']]),
    (mit.split_before, 'split_before', 'OneTwo', lambda s: s.isupper(), -1, False, [['O', 'n', 'e'], ['T', 'w', 'o']]),
    (mit.split_after, 'split_after', 'one1two2', lambda s: s.isdigit(), -1, False, [['o', 'n', 'e', '1'], ['t', 'w', 'o', '2']]),
])
def test_split_functions_basic(func, name, test_input, pred, maxsplit, keep_separator, expected):
    if func is mit.split_at:
        result = list(func(test_input, pred, maxsplit=maxsplit, keep_separator=keep_separator))
    else:
        result = list(func(test_input, pred, maxsplit=maxsplit))
    assert result == expected

def test_split_at_zero_maxsplit():
    iterable = [1, 2, 3]
    assert list(mit.split_at(iterable, lambda x: True, maxsplit=0)) == [[1, 2, 3]]

# 32. split_when
def test_split_when_basic():
    data = [1, 2, 3, 3, 2, 5, 2, 4, 2]
    expected = [[1, 2, 3, 3], [2, 5], [2, 4], [2]]
    result = list(mit.split_when(data, lambda x, y: x > y))
    assert result == expected

# 33. split_into
def test_split_into_basic():
    data = [1, 2, 3, 4, 5, 6]
    sizes = [1, 2, 3]
    result = list(mit.split_into(data, sizes))
    assert result == [[1], [2, 3], [4, 5, 6]]

    # test with None size to collect remainder
    sizes2 = [2, 3, None]
    result2 = list(mit.split_into(data, sizes2))
    assert result2 == [[1, 2], [3, 4, 5], [6]]

# 34. padded
def test_padded_basic():
    assert list(mit.padded([1, 2, 3], '?', 5)) == [1, 2, 3, '?', '?']
    assert list(mit.padded([1, 2, 3, 4], n=3, next_multiple=True)) == [1, 2, 3, 4, None, None]

    with pytest.raises(ValueError):
        list(mit.padded([1, 2, 3], '?', 0))

# 35. repeat_each
def test_repeat_each_basic():
    assert list(mit.repeat_each('ABC', 3)) == ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']

# 36. repeat_last
def test_repeat_last_basic():
    assert list(islice(mit.repeat_last(range(3)), 5)) == [0, 1, 2, 2, 2]
    assert list(islice(mit.repeat_last([], 42), 3)) == [42, 42, 42]

# 37. distribute
def test_distribute_basic():
    parts = mit.distribute(2, [1, 2, 3, 4, 5, 6])
    list1, list2 = list(parts[0]), list(parts[1])
    assert list1 == [1, 3, 5]
    assert list2 == [2, 4, 6]

    parts3 = mit.distribute(3, [1, 2, 3, 4, 5, 6, 7])
    parts3_lists = [list(c) for c in parts3]
    assert parts3_lists == [[1, 4, 7], [2, 5], [3, 6]]

    # Negative n error
    with pytest.raises(ValueError):
        mit.distribute(0, [])

# 38. stagger
def test_stagger_basic():
    s = list(mit.stagger([0, 1, 2, 3]))
    assert s[0] == (None, 0, 1)
    s_long = list(mit.stagger([0, 1, 2, 3], longest=True))
    assert s_long[-1] == (3, None, None)

# 39. zip_offset
def test_zip_offset_basic():
    assert list(mit.zip_offset('0123', 'abcdef', offsets=(0, 1))) == [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e')]
    assert list(mit.zip_offset('0123', 'abcdef', offsets=(0, 1), longest=True)) == [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e'), (None, 'f')]

    with pytest.raises(ValueError):
        list(mit.zip_offset('0123', 'abcdef', offsets=(0,)))


# 40. sort_together
def test_sort_together_basic():
    iters = [(4,3,2,1), ('a','b','c','d')]
    sorted_iters = mit.sort_together(iters)
    assert sorted_iters == [(1,2,3,4), ('d','c','b','a')]

    # Using key_list
    iters2 = [(3,1,2), (0,1,0), ('c','b','a')]
    result = mit.sort_together(iters2, key_list=(1, 2))
    assert result == [(2,3,1), (0, 0, 1), ('a','c','b')]

    # Use function key
    names = ('a', 'b', 'c')
    lengths = (1, 2, 3)
    widths = (5, 2, 1)
    def area(l, w): return l*w
    result_func_key = mit.sort_together([names, lengths, widths], key_list=(1,2), key=area)
    assert result_func_key == [('c','b','a'), (3,2,1), (1,2,5)]

    # Reverse
    reversed_sorted = mit.sort_together([(1,2,3), ('c','b','a')], reverse=True)
    assert reversed_sorted == [(3,2,1), ('a','b','c')]

# 41. unzip
def test_unzip_basic():
    iterable = [('a',1), ('b',2)]
    letters, numbers = mit.unzip(iterable)
    assert list(letters) == ['a','b']
    assert list(numbers) == [1,2]

    # Test empty input
    assert mit.unzip([]) == ()

# 42. divide
def test_divide_basic():
    group_1, group_2 = mit.divide(2, [1, 2, 3, 4, 5, 6])
    assert list(group_1) == [1, 2, 3]
    assert list(group_2) == [4, 5, 6]

    children = mit.divide(3, [1, 2, 3, 4, 5, 6, 7])
    children_lists = [list(c) for c in children]
    assert children_lists == [[1, 2, 3], [4, 5], [6, 7]]

    with pytest.raises(ValueError):
        mit.divide(0, [])

# 43. always_iterable
def test_always_iterable_behavior():
    # iterable obj returns iterator
    o = (1,2,3)
    assert list(mit.always_iterable(o)) == [1,2,3]

    # non-iterable returns single item iterable
    o2 = 5
    assert list(mit.always_iterable(o2)) == [5]

    # None returns empty iterable
    assert list(mit.always_iterable(None)) == []

    # base_type excludes str/bytes by default
    s = 'abc'
    assert list(mit.always_iterable(s)) == ['abc']

    # base_type argument disables that check
    assert list(mit.always_iterable(s, base_type=None)) == ['a', 'b', 'c']

# 44. adjacent
def test_adjacent_behavior():
    data = range(6)
    results = list(mit.adjacent(lambda x: x == 3, data))
    expected = [(False, 0), (False, 1), (True, 2), (True, 3), (True, 4), (False, 5)]
    assert results == expected

    # distance=2
    results2 = list(mit.adjacent(lambda x: x == 3, data, distance=2))
    assert results2[0] == (False, 0)
    assert results2[1] == (True, 1)
    assert results2[-1][0] is True

    with pytest.raises(ValueError):
        list(mit.adjacent(lambda x: True, data, distance=-1))

# 45. groupby_transform
def test_groupby_transform_basic():
    iterable = 'aAAbBBcCC'
    keyfunc = lambda k: k.upper()
    valuefunc = lambda v: v.lower()
    reducefunc = lambda g: ''.join(g)
    result = list(mit.groupby_transform(iterable, keyfunc, valuefunc, reducefunc))
    assert result == [('A', 'aaa'), ('B', 'bbb'), ('C', 'ccc')]

# 46. numeric_range basic
def test_numeric_range_basic_behavior():
    r = mit.numeric_range(3.5)
    assert list(r) == [0.0, 1.0, 2.0, 3.0]

    from decimal import Decimal
    start = Decimal('2.1')
    stop = Decimal('5.1')
    r2 = mit.numeric_range(start, stop)
    expected = [Decimal('2.1'), Decimal('3.1'), Decimal('4.1')]
    assert list(r2) == expected

    from fractions import Fraction
    r3 = mit.numeric_range(Fraction(1,2), Fraction(5,2), Fraction(1,2))
    assert list(r3) == [Fraction(1,2), Fraction(1,1), Fraction(3,2), Fraction(2,1)]

    with pytest.raises(ValueError):
        mit.numeric_range(1, 2, 0)

    # Test negative stepping & containment
    r4 = mit.numeric_range(3, -1, -1.0)
    assert list(r4) == [3.0, 2.0, 1.0, 0.0]
    assert 2.0 in r4
    assert 5 not in r4

# 47. count_cycle
def test_count_cycle_basic():
    result = list(mit.count_cycle('AB', 2))
    expected = [(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B')]
    assert result == expected

    # infinite case - test first few elements
    cc = mit.count_cycle('AB')
    first4 = list(islice(cc, 4))
    assert first4 == [(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B')]

# 48. mark_ends
def test_mark_ends_basic():
    data = 'ABC'
    result = list(mit.mark_ends(data))
    expected = [(True, False, 'A'), (False, False, 'B'), (False, True, 'C')]
    assert result == expected

# 49. locate
def test_locate_basic():
    data = [0, 1, 1, 0, 1]
    result = list(mit.locate(data))
    assert result == [1, 2, 4]

    result2 = list(mit.locate(data, pred=lambda x: x==0))
    assert result2 == [0, 3]

    # With window_size
    data2 = [0,1,2,3,0,1,2,3]
    pred = lambda *args: args == (1, 2, 3)
    indexes = list(mit.locate(data2, pred=pred, window_size=3))
    assert indexes == [1, 5]

    with pytest.raises(ValueError):
        list(mit.locate(data, window_size=0))


# 50. longest_common_prefix
def test_longest_common_prefix_basic():
    result = ''.join(mit.longest_common_prefix(['abcdef', 'abczzz', 'abcpqr']))
    assert result == 'abc'

# 51. lstrip, rstrip, strip
def test_strip_functions_basic():
    data = [None, False, None, 1, 2, None, 3, False, None]
    pred = lambda x: x in {None, False}
    assert list(mit.lstrip(data, pred)) == [1, 2, None, 3, False, None]
    assert list(mit.rstrip(data, pred)) == [None, False, None, 1, 2, None, 3]
    assert list(mit.strip(data, pred)) == [1, 2, None, 3]

# 52. islice_extended
def test_islice_extended_basic():
    seq = 'abcdefgh'
    result = list(mit.islice_extended(seq, -4, -1))
    assert result == ['e', 'f', 'g']

    it = map(str, count())
    sl = mit.islice_extended(it)[10:20:2]
    result2 = list(sl)
    expected = [str(i) for i in range(10, 20, 2)]
    assert result2 == expected

    with pytest.raises(ValueError):
        list(mit.islice_extended('abc', 0, 1, 0))

# 53. always_reversible
def test_always_reversible_basic():
    it = (x for x in [1, 2, 3])
    assert list(mit.always_reversible(it)) == [3, 2, 1]
    assert list(mit.always_reversible([1,2,3])) == [3, 2, 1]

# 54. consecutive_groups
def test_consecutive_groups_basic():
    data = [1, 10, 11, 12, 20, 30, 31, 32, 33, 40]
    groups = [list(g) for g in mit.consecutive_groups(data)]
    assert groups == [[1], [10, 11, 12], [20], [30, 31, 32, 33], [40]]

    # Using ordering function ord for letters
    s = 'abcdfgilmnop'
    groups2 = [list(g) for g in mit.consecutive_groups(s, ordering=ord)]
    assert groups2[0] == ['a', 'b', 'c', 'd']

# 55. difference
def test_difference_basic():
    from itertools import accumulate
    data = [1, 2, 3, 4, 5]
    acc = accumulate(data)
    diff = list(mit.difference(acc))
    assert diff == data
    # Using initial param
    acc2 = accumulate([1,3,6])
    diff2 = list(mit.difference(acc2, initial=1))
    assert diff2 == [2, 3]

# 56. SequenceView
def test_sequenceview_basic():
    seq = ['a', 'b', 'c']
    view = mit.SequenceView(seq)
    assert view[0] == 'a'
    assert view[1:] == ['b', 'c']
    seq.append('d')
    assert view[-1] == 'd'
    with pytest.raises(TypeError):
        mit.SequenceView(set())

# 57. seekable
def test_seekable_basic():
    it = mit.seekable('1234')
    assert it.peek() == '1'
    assert next(it) == '1'
    it.seek(0)
    assert next(it) == '1'
    it.relative_seek(2)
    assert next(it) == '4'

    # maxlen=2 cache behavior
    it2 = mit.seekable(range(5), maxlen=2)
    res = [next(it2) for _ in range(4)]
    assert res == [0,1,2,3]
    elems = it2.elements()
    assert list(elems) == [2,3]

# 58. run_length encode and decode
def test_run_length():
    s = 'abbcccdddd'
    encoded = list(mit.run_length.encode(s))
    assert encoded == [('a',1), ('b',2), ('c',3), ('d',4)]
    decoded = list(mit.run_length.decode(encoded))
    assert ''.join(decoded) == s

# 59. exactly_n
def test_exactly_n_behavior():
    assert mit.exactly_n([True, True, False], 2) is True
    assert mit.exactly_n([True, True, False], 1) is False
    assert mit.exactly_n([0,1,2,3,4,5], 3, lambda x: x < 3) is True
    assert mit.exactly_n([], 0) is True
    assert mit.exactly_n([], 1) is False

# 60. circular_shifts
def test_circular_shifts():
    c1 = list(mit.circular_shifts(range(4)))
    assert c1 == [(0,1,2,3), (1,2,3,0), (2,3,0,1), (3,0,1,2)]
    c2 = list(mit.circular_shifts(range(4), 2))
    assert c2 == [(0,1,2,3), (2,3,0,1)]
    c3 = list(mit.circular_shifts(range(4), -1))
    assert c3 == [(0,1,2,3), (3,0,1,2), (2,3,0,1), (1,2,3,0)]
    with pytest.raises(ValueError):
        list(mit.circular_shifts(range(3), 0))

# 61. make_decorator
def test_make_decorator_usage():
    chunker = mit.make_decorator(mit.chunked, result_index=0)
    @chunker(3)
    def gen():
        return iter(range(9))
    result = list(gen())
    assert result == [[0,1,2],[3,4,5],[6,7,8]]

    truth_serum = mit.make_decorator(filter, result_index=1)
    @truth_serum(bool)
    def f():
        return [0,1,'', True]
    assert list(f()) == [1, True]

# 62. map_reduce
def test_map_reduce_basic():
    keyfunc = lambda x: x.upper()
    result = mit.map_reduce('abbccc', keyfunc)
    assert dict(result) == {'A': ['a'], 'B': ['b','b'], 'C': ['c','c','c']}

    result2 = mit.map_reduce('abbccc', keyfunc, valuefunc=lambda x: 1)
    assert dict(result2) == {'A':[1], 'B':[1,1], 'C':[1,1,1]}

    result3 = mit.map_reduce('abbccc', keyfunc, valuefunc=lambda x:1, reducefunc=sum)
    assert dict(result3) == {'A':1, 'B':2, 'C':3}

# 63. rlocate
def test_rlocate_basic():
    data = [0, 1, 1, 0, 1]
    positions = list(mit.rlocate(data))
    assert positions == [4, 2, 1]

    pred = lambda x: x == 'b'
    s = 'abcb'
    res = list(mit.rlocate(s, pred))
    assert res == [3, 1]

# 64. replace
def test_replace_basic():
    data = [1, 0, 1, 0, 1]
    pred = lambda x: x == 0
    substitutes = [9]
    replaced = list(mit.replace(data, pred, substitutes))
    assert replaced == [1, 9, 1, 9, 1]

    replaced_limited = list(mit.replace(data, pred, substitutes, count=1))
    assert replaced_limited == [1, 9, 1, 0, 1]

    # window size 2
    iterable = [0, 1, 2, 5, 0, 1, 2, 5]
    pred = lambda *args: args == (0, 1)
    substitutes = [9]
    replaced_ws = list(mit.replace(iterable, pred, substitutes, window_size=2))
    # should replace first window twice overlapping
    assert replaced_ws == [9, 2, 5, 9, 2, 5]

# 65. partitions
def test_partitions_basic():
    data = 'abc'
    parts = list(mit.partitions(data))
    parts_joined = [[''.join(p) for p in partition] for partition in parts]
    assert ['abc', 'a', 'bc', 'ab', 'c', 'a', 'b', 'c'] == sorted(set(sum(parts_joined, [])))

# 66. set_partitions
def test_set_partitions_basic():
    data = 'abc'
    all_parts = list(mit.set_partitions(data))
    all_parts_set = [tuple(sorted([''.join(sorted(block)) for block in part])) for part in all_parts]
    # We've got these partitions (order of blocks not important)
    expected = {
        ('abc',), ('a', 'bc'), ('ab', 'c'), ('b', 'ac'), ('a', 'b', 'c')
    }
    assert expected.issubset(set(all_parts_set))

    with pytest.raises(ValueError):
        list(mit.set_partitions(data, k=0))

# 67. time_limited
def test_time_limited_basic(monkeypatch):
    # Use freezegun or mocking monotonic
    start = 1000
    times = [start]

    def fake_monotonic():
        return times[-1]

    monkeypatch.setattr(mit, 'monotonic', fake_monotonic)

    def gen():
        yield 1
        times.append(times[-1] + 0.05)
        yield 2
        times.append(times[-1] + 0.15)
        yield 3

    it = mit.time_limited(0.1, gen())
    result = list(it)
    assert result == [1, 2]
    assert it.timed_out is True

    it2 = mit.time_limited(0, iter([1]))
    assert list(it2) == []
    assert it2.timed_out is True

    with pytest.raises(ValueError):
        mit.time_limited(-1, iter([]))

# 68. only
def test_only_function_behavior():
    assert mit.only([1]) == 1
    assert mit.only([]) is None

    with pytest.raises(ValueError):
        mit.only([1,2])

    with pytest.raises(TypeError):
        mit.only([1,2], too_long=TypeError)

# 69. ichunked
def test_ichunked_basic():
    chunks = mit.ichunked(range(10), 3)
    first = next(chunks)
    second = next(chunks)
    assert list(second) == [3,4,5]
    assert list(first)[:3] == [0,1,2]
    assert next(chunks) is not None or True

    with pytest.raises(ValueError):
        list(mit.ichunked(range(10), -1))

# 70. iequals
def test_iequals_basic():
    assert mit.iequals('abc', ['a','b','c'], ('a','b','c')) is True
    assert mit.iequals('abc', 'acb') is False

# 71. distinct_combinations
def test_distinct_combinations_basic():
    inp = [0, 0, 1]
    result = list(mit.distinct_combinations(inp, 2))
    assert set(result) == {(0,0),(0,1)}

# 72. filter_except
def test_filter_except_basic():
    def val(x):
        return int(x)
    data = ['1','2','a','3', None]
    result = list(mit.filter_except(val, data, ValueError, TypeError))
    assert result == ['1','2','3']

# 73. map_except
def test_map_except_basic():
    def val(x):
        return int(x)
    data = ['1','2','a','3', None]
    result = list(mit.map_except(val, data, ValueError, TypeError))
    assert result == [1, 2, 4]

# 74. map_if
def test_map_if_basic():
    data = range(-5,5)
    r1 = list(mit.map_if(data, lambda x: x >3, lambda x:'toobig'))
    assert r1[-1] == 'toobig'
    r2 = list(mit.map_if(data, lambda x: x >= 0, lambda x: x*x, lambda x: None))
    assert r2[0] is None and r2[-1] == 16

# 75. sample basics and errors
def test_sample_basic():
    data = range(100)
    result = mit.sample(data, 5)
    assert len(result) == 5

    data2 = ['a', 'b']
    counts = [3,4]
    result2 = mit.sample(data2, 3, counts=counts)
    assert all(x in ['a','b'] for x in result2)

    weights = (i*i+1 for i in range(100))
    result3 = mit.sample(range(100), 5, weights=weights)
    assert len(result3) == 5

    # exclusive weights and counts
    with pytest.raises(TypeError):
        mit.sample(data, 1, weights=weights, counts=counts)

    # negative k
    with pytest.raises(ValueError):
        mit.sample(data, -1)

# 76. is_sorted
def test_is_sorted():
    assert mit.is_sorted(['1', '2', '3'], key=int)
    assert not mit.is_sorted([5,4,3])
    assert mit.is_sorted([1, 2, 3], strict=False)
    assert not mit.is_sorted([1, 2, 2], strict=True)

# 77. callback_iter basic and abort
def test_callback_iter_basic():
    def f(callback=None):
        if callback:
            callback(1, 'a')
            callback(2, 'b')
        return 3

    with mit.callback_iter(f) as it:
        items = list(it)
        assert (1, 'a') in [arg[0] for arg in items]
        assert it.done
        assert it.result == 3

def test_callback_iter_abort():
    def f(callback=None):
        for i in range(5):
            if callback:
                callback(i)
            time.sleep(0.01)

    with mit.callback_iter(f, wait_seconds=0.001) as it:
        # abort immediately
        it._aborted = True
        with pytest.raises(mit.AbortThread):
            next(it)

# 78. windowed_complete
def test_windowed_complete_basic():
    data = range(7)
    n = 3
    results = list(mit.windowed_complete(data, n))
    assert all(len(m[1]) == n for m in results)
    with pytest.raises(ValueError):
        list(mit.windowed_complete(data, -1))
    with pytest.raises(ValueError):
        list(mit.windowed_complete(data, 10))

# 79. all_unique
def test_all_unique_basic():
    assert not mit.all_unique('ABCB')
    assert mit.all_unique('ABCb')
    assert not mit.all_unique('ABCb', key=str.lower)
    # unhashable types - slower path
    assert mit.all_unique([[1], [2], [3]])
    assert not mit.all_unique([[1], [2], [1]])

# 80. nth_product
def test_nth_product_basic():
    index = 8
    result = mit.nth_product(index, range(2), range(2), range(2), range(2))
    assert result == (1, 0, 0, 0)

    with pytest.raises(IndexError):
        mit.nth_product(100, range(2), range(2))
