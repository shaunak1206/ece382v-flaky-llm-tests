import pytest
from more_itertools import (
    chunked, first, last, windowed, peekable, spy,
    bucket, partition, distinct_permutations, powerset,
    unique_everseen, unique_justseen, roundrobin,
    interleave_longest, grouper, nth, consume, one, only,
    always_iterable, split_at, split_before, collapse,
    run_length, map_reduce, ilen, first_true
)
from more_itertools.recipes import (
    take, tail, random_product, random_permutation,
    random_combination, random_combination_with_replacement
)


class TestChunked:
    def test_chunked_basic(self):
        result = list(chunked([1, 2, 3, 4, 5, 6], 3))
        assert result == [[1, 2, 3], [4, 5, 6]]

    def test_chunked_with_remainder(self):
        result = list(chunked([1, 2, 3, 4, 5, 6, 7, 8], 3))
        assert result == [[1, 2, 3], [4, 5, 6], [7, 8]]

    def test_chunked_strict_mode(self):
        with pytest.raises(ValueError):
            list(chunked([1, 2, 3, 4, 5], 2, strict=True))

    def test_chunked_exact_division(self):
        result = list(chunked([1, 2, 3, 4], 2, strict=True))
        assert result == [[1, 2], [3, 4]]


class TestFirst:
    def test_first_basic(self):
        assert first([0, 1, 2, 3]) == 0

    def test_first_with_default(self):
        assert first([], 'some default') == 'some default'

    def test_first_empty_no_default(self):
        with pytest.raises(ValueError):
            first([])

    def test_first_generator(self):
        gen = (x for x in range(5))
        assert first(gen) == 0


class TestLast:
    def test_last_basic(self):
        assert last([0, 1, 2, 3]) == 3

    def test_last_with_default(self):
        assert last([], 'some default') == 'some default'

    def test_last_empty_no_default(self):
        with pytest.raises(ValueError):
            last([])

    def test_last_iterator(self):
        result = last(iter([1, 2, 3]))
        assert result == 3


class TestWindowed:
    def test_windowed_basic(self):
        result = list(windowed([1, 2, 3, 4, 5], 3))
        assert result == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    def test_windowed_step(self):
        result = list(windowed([1, 2, 3, 4, 5, 6], 2, step=2))
        assert result == [(1, 2), (3, 4), (5, 6)]

    def test_windowed_fillvalue(self):
        result = list(windowed([1, 2, 3], 2, fillvalue=0))
        assert result == [(1, 2), (2, 3), (3, 0)]


class TestPeekable:
    def test_peekable_basic(self):
        p = peekable(['a', 'b'])
        assert p.peek() == 'a'
        assert next(p) == 'a'

    def test_peekable_with_default(self):
        p = peekable([])
        assert p.peek('hi') == 'hi'

    def test_peekable_prepend(self):
        p = peekable([1, 2, 3])
        p.prepend(10, 11, 12)
        assert next(p) == 10
        assert p.peek() == 11
        assert list(p) == [11, 12, 1, 2, 3]

    def test_peekable_indexing(self):
        p = peekable(['a', 'b', 'c', 'd'])
        assert p[0] == 'a'
        assert p[1] == 'b'
        assert next(p) == 'a'

    def test_peekable_truth(self):
        p = peekable(['a', 'b'])
        assert bool(p) is True
        list(p)
        assert bool(p) is False


class TestSpy:
    def test_spy_basic(self):
        iterator = iter([1, 2, 3, 4, 5])
        first_items, new_iter = spy(iterator, n=2)
        assert first_items == [1, 2]
        assert list(new_iter) == [1, 2, 3, 4, 5]

    def test_spy_single(self):
        first_items, new_iter = spy(iter([1, 2, 3]))
        assert first_items == [1]
        assert list(new_iter) == [1, 2, 3]

    def test_spy_empty(self):
        first_items, new_iter = spy(iter([]))
        assert first_items == []
        assert list(new_iter) == []


class TestBucket:
    def test_bucket_basic(self):
        data = [1, 2, 3, 4, 5, 6]
        buckets = bucket(data, key=lambda x: x % 2)
        odds = list(buckets[1])
        evens = list(buckets[0])
        assert odds == [1, 3, 5]
        assert evens == [2, 4, 6]

    def test_bucket_by_type(self):
        data = [1, 'a', 2, 'b', 3]
        buckets = bucket(data, key=type)
        ints = list(buckets[int])
        strs = list(buckets[str])
        assert ints == [1, 2, 3]
        assert strs == ['a', 'b']


class TestPartition:
    def test_partition_basic(self):
        def is_odd(x):
            return x % 2 == 1
        odds, evens = partition(is_odd, [1, 2, 3, 4, 5, 6])
        assert list(odds) == [1, 3, 5]
        assert list(evens) == [2, 4, 6]


class TestDistinctPermutations:
    def test_distinct_permutations_basic(self):
        result = list(distinct_permutations([1, 1, 2]))
        assert len(result) == 3
        assert (1, 1, 2) in result

    def test_distinct_permutations_no_duplicates(self):
        result = list(distinct_permutations([1, 2, 3]))
        assert len(result) == 6

    def test_distinct_permutations_all_same(self):
        result = list(distinct_permutations([1, 1, 1]))
        assert len(result) == 1


class TestPowerset:
    def test_powerset_basic(self):
        result = list(powerset([1, 2, 3]))
        assert () in result
        assert (1,) in result
        assert (1, 2, 3) in result
        assert len(result) == 8

    def test_powerset_empty(self):
        result = list(powerset([]))
        assert result == [()]

    def test_powerset_single(self):
        result = list(powerset([1]))
        assert result == [(), (1,)]


class TestUniqueEverseen:
    def test_unique_everseen_basic(self):
        result = list(unique_everseen('AAAABBBCCD'))
        assert result == ['A', 'B', 'C', 'D']

    def test_unique_everseen_with_key(self):
        result = list(unique_everseen('ABBcCDD', key=str.lower))
        assert result == ['A', 'B', 'c', 'D']

    def test_unique_everseen_numbers(self):
        result = list(unique_everseen([1, 1, 2, 2, 3, 1]))
        assert result == [1, 2, 3, 1]


class TestUniqueJustseen:
    def test_unique_justseen_basic(self):
        result = list(unique_justseen('AAAABBBCCD'))
        assert result == ['A', 'B', 'C', 'D']

    def test_unique_justseen_no_consecutive_duplicates(self):
        result = list(unique_justseen([1, 1, 2, 2, 3, 1, 1]))
        assert result == [1, 2, 3, 1]


class TestRoundrobin:
    def test_roundrobin_basic(self):
        result = list(roundrobin('ABC', 'D', 'EF'))
        assert result == ['A', 'D', 'E', 'B', 'F', 'C']

    def test_roundrobin_different_lengths(self):
        result = list(roundrobin([1, 2], [3, 4, 5]))
        assert result == [1, 3, 2, 4, 5]


class TestInterleavelongest:
    def test_interleave_longest_basic(self):
        result = list(interleave_longest([1, 2], [3, 4, 5]))
        assert result == [1, 3, 2, 4, 5]

    def test_interleave_longest_with_fillvalue(self):
        result = list(interleave_longest([1, 2], [3, 4, 5], fillvalue=None))
        assert result == [1, 3, 2, 4, None, 5]


class TestGrouper:
    def test_grouper_basic(self):
        result = list(grouper([1, 2, 3, 4, 5, 6], 3))
        assert result == [(1, 2, 3), (4, 5, 6)]

    def test_grouper_incomplete_fill(self):
        result = list(grouper([1, 2, 3, 4, 5], 3, incomplete='fill', fillvalue=None))
        assert result == [(1, 2, 3), (4, 5, None)]

    def test_grouper_incomplete_strict(self):
        with pytest.raises(ValueError):
            list(grouper([1, 2, 3, 4, 5], 3, incomplete='strict'))


class TestNth:
    def test_nth_basic(self):
        result = nth(3, iter([0, 1, 2, 3, 4, 5]))
        assert result == 3

    def test_nth_with_default(self):
        result = nth(10, iter([0, 1, 2]), default='x')
        assert result == 'x'


class TestConsume:
    def test_consume_partial(self):
        iterator = iter([1, 2, 3, 4, 5])
        consume(iterator, 2)
        assert list(iterator) == [3, 4, 5]

    def test_consume_all(self):
        iterator = iter([1, 2, 3])
        consume(iterator)
        assert list(iterator) == []


class TestOne:
    def test_one_basic(self):
        result = one([1])
        assert result == 1

    def test_one_empty(self):
        with pytest.raises(ValueError):
            one([])

    def test_one_multiple(self):
        with pytest.raises(ValueError):
            one([1, 2])


class TestOnly:
    def test_only_single_true(self):
        result = only([1])
        assert result == 1

    def test_only_multiple(self):
        result = only([1, 2, 3], too_long='too many')
        assert result == 'too many'


class TestAlwaysIterable:
    def test_always_iterable_list(self):
        result = list(always_iterable([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_always_iterable_string(self):
        result = list(always_iterable('abc'))
        assert result == ['abc']

    def test_always_iterable_single(self):
        result = list(always_iterable(1))
        assert result == [1]

    def test_always_iterable_none(self):
        result = list(always_iterable(None))
        assert result == []


class TestSplitAt:
    def test_split_at_basic(self):
        result = list(split_at(lambda x: x == 3, [1, 2, 3, 4, 5]))
        assert result == [[1, 2], [4, 5]]

    def test_split_at_keep_separator(self):
        result = list(split_at(lambda x: x == 3, [1, 2, 3, 4, 5], keep_separator=True))
        assert result == [[1, 2, 3], [4, 5]]


class TestCollapse:
    def test_collapse_nested_lists(self):
        result = list(collapse([[1, 2], [3, 4, [5, 6]]]))
        assert result == [1, 2, 3, 4, 5, 6]

    def test_collapse_with_levels(self):
        result = list(collapse([[1, [2]], [3, [4]]], levels=1))
        assert result == [1, [2], 3, [4]]


class TestRunLength:
    def test_run_length_basic(self):
        result = list(run_length('AAABBBCDD'))
        assert result == [('A', 3), ('B', 3), ('C', 1), ('D', 2)]

    def test_run_length_numbers(self):
        result = list(run_length([1, 1, 2, 2, 2, 3]))
        assert result == [(1, 2), (2, 3), (3, 1)]


class TestMapReduce:
    def test_map_reduce_basic(self):
        def key_fn(x):
            return x % 2
        def value_fn(x):
            return 1
        def reduce_fn(acc, val):
            return acc + val
        result = map_reduce(key_fn, value_fn, reduce_fn, [1, 2, 3, 4, 5])
        assert result == {1: 3, 0: 2}


class TestIlen:
    def test_ilen_basic(self):
        result = ilen(x for x in range(100))
        assert result == 100

    def test_ilen_empty(self):
        result = ilen([])
        assert result == 0


class TestFirstTrue:
    def test_first_true_basic(self):
        result = first_true([0, 1, 2, 3])
        assert result == 1

    def test_first_true_with_default(self):
        result = first_true([0, 0, 0], default='none')
        assert result == 'none'

    def test_first_true_with_predicate(self):
        result = first_true([1, 2, 3, 4], pred=lambda x: x > 2)
        assert result == 3


class TestRecipesFunctions:
    def test_take_from_recipes(self):
        result = list(take(3, range(10)))
        assert result == [0, 1, 2]

    def test_tail_from_recipes(self):
        result = list(tail(3, range(10)))
        assert result == [7, 8, 9]
