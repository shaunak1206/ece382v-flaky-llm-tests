import pytest
from operator import add, mul
from toolz.itertoolz import (
    remove, accumulate, groupby, frequencies, unique, isdistinct,
    take, drop, nth, first, second, last, concat, get, pluck,
    tail, interleave, partition, partition_all, sliding_window,
    take_nth, count, cons, interpose, mapcat, isiterable,
    iterate, topk, peek, peekn, random_sample
)


class TestRemove:
    def test_remove_basic(self):
        iseven = lambda x: x % 2 == 0
        result = list(remove(iseven, [1, 2, 3, 4]))
        assert result == [1, 3]

    def test_remove_empty(self):
        iseven = lambda x: x % 2 == 0
        result = list(remove(iseven, []))
        assert result == []

    def test_remove_all_match(self):
        iseven = lambda x: x % 2 == 0
        result = list(remove(iseven, [2, 4, 6, 8]))
        assert result == []


class TestAccumulate:
    def test_accumulate_add(self):
        result = list(accumulate(add, [1, 2, 3, 4, 5]))
        assert result == [1, 3, 6, 10, 15]

    def test_accumulate_mul(self):
        result = list(accumulate(mul, [1, 2, 3, 4, 5]))
        assert result == [1, 2, 6, 24, 120]

    def test_accumulate_with_initial(self):
        result = list(accumulate(add, [1, 2, 3], -1))
        assert result == [-1, 0, 2, 5]

    def test_accumulate_empty_with_initial(self):
        result = list(accumulate(add, [], 1))
        assert result == [1]

    def test_accumulate_empty_no_initial(self):
        result = list(accumulate(add, []))
        assert result == []


class TestGroupby:
    def test_groupby_by_length(self):
        names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
        result = groupby(len, names)
        assert result[3] == ['Bob', 'Dan']
        assert result[5] == ['Alice', 'Edith', 'Frank']
        assert result[7] == ['Charlie']

    def test_groupby_by_function(self):
        iseven = lambda x: x % 2 == 0
        result = groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])
        assert set(result[False]) == {1, 3, 5, 7}
        assert set(result[True]) == {2, 4, 6, 8}

    def test_groupby_by_key_non_callable(self):
        data = [{'name': 'Alice', 'gender': 'F'},
                {'name': 'Bob', 'gender': 'M'},
                {'name': 'Charlie', 'gender': 'M'}]
        result = groupby('gender', data)
        assert result['F'][0]['name'] == 'Alice'
        assert len(result['M']) == 2


class TestFrequencies:
    def test_frequencies_basic(self):
        result = frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])
        assert result == {'cat': 3, 'ox': 1, 'pig': 2}

    def test_frequencies_empty(self):
        result = frequencies([])
        assert result == {}

    def test_frequencies_single(self):
        result = frequencies(['a'])
        assert result == {'a': 1}

    def test_frequencies_numbers(self):
        result = frequencies([1, 1, 1, 2, 2, 3])
        assert result == {1: 3, 2: 2, 3: 1}


class TestUnique:
    def test_unique_basic(self):
        result = tuple(unique((1, 2, 3)))
        assert result == (1, 2, 3)

    def test_unique_with_duplicates(self):
        result = tuple(unique((1, 2, 1, 3)))
        assert result == (1, 2, 3)

    def test_unique_by_key(self):
        result = tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len))
        assert result == ('cat', 'mouse')

    def test_unique_empty(self):
        result = tuple(unique([]))
        assert result == ()


class TestIsdistinct:
    def test_isdistinct_true(self):
        assert isdistinct([1, 2, 3]) is True

    def test_isdistinct_false(self):
        assert isdistinct([1, 2, 1]) is False

    def test_isdistinct_string_hello(self):
        assert isdistinct("Hello") is False

    def test_isdistinct_string_world(self):
        assert isdistinct("World") is True

    def test_isdistinct_empty(self):
        assert isdistinct([]) is True


class TestTake:
    def test_take_basic(self):
        result = list(take(2, [10, 20, 30, 40, 50]))
        assert result == [10, 20]

    def test_take_zero(self):
        result = list(take(0, [10, 20, 30]))
        assert result == []

    def test_take_more_than_available(self):
        result = list(take(10, [1, 2, 3]))
        assert result == [1, 2, 3]


class TestDrop:
    def test_drop_basic(self):
        result = list(drop(2, [10, 20, 30, 40, 50]))
        assert result == [30, 40, 50]

    def test_drop_zero(self):
        result = list(drop(0, [10, 20, 30]))
        assert result == [10, 20, 30]

    def test_drop_more_than_available(self):
        result = list(drop(10, [1, 2, 3]))
        assert result == []


class TestNth:
    def test_nth_basic(self):
        result = nth(1, 'ABC')
        assert result == 'B'

    def test_nth_list(self):
        result = nth(2, [10, 20, 30, 40])
        assert result == 30

    def test_nth_iterator(self):
        result = nth(1, iter([10, 20, 30, 40]))
        assert result == 20

    def test_nth_zero(self):
        result = nth(0, 'ABC')
        assert result == 'A'


class TestFirst:
    def test_first_list(self):
        assert first([10, 20, 30]) == 10

    def test_first_string(self):
        assert first('ABC') == 'A'

    def test_first_iterator(self):
        assert first(iter([1, 2, 3])) == 1


class TestSecond:
    def test_second_list(self):
        assert second([10, 20, 30]) == 20

    def test_second_string(self):
        assert second('ABC') == 'B'

    def test_second_iterator(self):
        assert second(iter([1, 2, 3])) == 2


class TestLast:
    def test_last_list(self):
        assert last([10, 20, 30]) == 30

    def test_last_string(self):
        assert last('ABC') == 'C'

    def test_last_tuple(self):
        assert last((1, 2, 3)) == 3


class TestTail:
    def test_tail_basic(self):
        result = tail(2, [10, 20, 30, 40, 50])
        assert result == [40, 50]

    def test_tail_iterator(self):
        result = tail(2, iter([10, 20, 30, 40, 50]))
        assert result == (40, 50)

    def test_tail_zero(self):
        result = tail(0, [10, 20, 30])
        assert result == []


class TestConcat:
    def test_concat_basic(self):
        result = list(concat([[], [1], [2, 3]]))
        assert result == [1, 2, 3]

    def test_concat_strings(self):
        result = list(concat(['ab', 'cd', 'ef']))
        assert result == ['a', 'b', 'c', 'd', 'e', 'f']

    def test_concat_empty(self):
        result = list(concat([]))
        assert result == []


class TestGet:
    def test_get_single_index(self):
        assert get(1, 'ABC') == 'B'

    def test_get_dict(self):
        phonebook = {'Alice': '555-1234', 'Bob': '555-5678'}
        assert get('Alice', phonebook) == '555-1234'

    def test_get_multiple_indices(self):
        result = get([1, 2], 'ABC')
        assert result == ('B', 'C')

    def test_get_with_default(self):
        result = get('missing', {'a': 1}, None)
        assert result is None


class TestPluck:
    def test_pluck_basic(self):
        data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
        result = list(pluck('name', data))
        assert result == ['Cheese', 'Pies']

    def test_pluck_multiple(self):
        data = [[1, 2, 3], [4, 5, 7]]
        result = list(pluck([0, 1], data))
        assert result == [(1, 2), (4, 5)]

    def test_pluck_with_default(self):
        data = [{'id': 1}, {'id': 2, 'name': 'x'}]
        result = list(pluck('name', data, default='N/A'))
        assert result == ['N/A', 'x']


class TestInterleave:
    def test_interleave_basic(self):
        result = list(interleave([[1, 2], [3, 4]]))
        assert result == [1, 3, 2, 4]

    def test_interleave_strings(self):
        result = ''.join(interleave(('ABC', 'XY')))
        assert result == 'AXBYC'

    def test_interleave_different_lengths(self):
        result = list(interleave([[1, 2, 3], [4, 5]]))
        assert result == [1, 4, 2, 5, 3]


class TestPartition:
    def test_partition_basic(self):
        result = list(partition(2, [1, 2, 3, 4]))
        assert result == [(1, 2), (3, 4)]

    def test_partition_with_remainder(self):
        result = list(partition(2, [1, 2, 3, 4, 5]))
        assert result == [(1, 2), (3, 4)]

    def test_partition_with_pad(self):
        result = list(partition(2, [1, 2, 3, 4, 5], pad=None))
        assert result == [(1, 2), (3, 4), (5, None)]


class TestPartitionAll:
    def test_partition_all_basic(self):
        result = list(partition_all(2, [1, 2, 3, 4]))
        assert result == [(1, 2), (3, 4)]

    def test_partition_all_with_remainder(self):
        result = list(partition_all(2, [1, 2, 3, 4, 5]))
        assert result == [(1, 2), (3, 4), (5,)]

    def test_partition_all_empty(self):
        result = list(partition_all(2, []))
        assert result == []


class TestSlidingWindow:
    def test_sliding_window_basic(self):
        result = list(sliding_window(2, [1, 2, 3, 4]))
        assert result == [(1, 2), (2, 3), (3, 4)]

    def test_sliding_window_three(self):
        result = list(sliding_window(3, [1, 2, 3, 4, 5]))
        assert result == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    def test_sliding_window_short_seq(self):
        result = list(sliding_window(2, [1]))
        assert result == []


class TestTakeNth:
    def test_take_nth_basic(self):
        result = list(take_nth(2, [10, 20, 30, 40, 50]))
        assert result == [10, 30, 50]

    def test_take_nth_one(self):
        result = list(take_nth(1, [10, 20, 30, 40]))
        assert result == [10, 20, 30, 40]


class TestCount:
    def test_count_list(self):
        assert count([1, 2, 3, 4, 5]) == 5

    def test_count_generator(self):
        gen = (x for x in range(10))
        assert count(gen) == 10

    def test_count_empty(self):
        assert count([]) == 0


class TestCons:
    def test_cons_basic(self):
        result = list(cons(1, [2, 3]))
        assert result == [1, 2, 3]

    def test_cons_single(self):
        result = list(cons('x', []))
        assert result == ['x']


class TestInterpose:
    def test_interpose_basic(self):
        result = list(interpose("a", [1, 2, 3]))
        assert result == [1, 'a', 2, 'a', 3]

    def test_interpose_strings(self):
        result = list(interpose(' ', ['a', 'b', 'c']))
        assert result == ['a', ' ', 'b', ' ', 'c']


class TestMapcat:
    def test_mapcat_basic(self):
        result = list(mapcat(lambda s: [c.upper() for c in s],
                            [["a", "b"], ["c", "d", "e"]]))
        assert result == ['A', 'B', 'C', 'D', 'E']

    def test_mapcat_flatten_lists(self):
        result = list(mapcat(lambda x: [x, x], [1, 2, 3]))
        assert result == [1, 1, 2, 2, 3, 3]


class TestIsiterable:
    def test_isiterable_list(self):
        assert isiterable([1, 2, 3]) is True

    def test_isiterable_string(self):
        assert isiterable('abc') is True

    def test_isiterable_number(self):
        assert isiterable(5) is False

    def test_isiterable_dict(self):
        assert isiterable({'a': 1}) is True


class TestIterate:
    def test_iterate_basic(self):
        inc = lambda x: x + 1
        counter = iterate(inc, 0)
        result = [next(counter) for _ in range(5)]
        assert result == [0, 1, 2, 3, 4]

    def test_iterate_multiply(self):
        double = lambda x: x * 2
        powers = iterate(double, 1)
        result = [next(powers) for _ in range(4)]
        assert result == [1, 2, 4, 8]


class TestTopk:
    def test_topk_basic(self):
        result = topk(2, [1, 100, 10, 1000])
        assert result == (1000, 100)

    def test_topk_with_key(self):
        result = topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len)
        assert result == ('Charlie', 'Alice')

    def test_topk_all_equal(self):
        result = topk(3, [1, 1, 1, 1])
        assert result == (1, 1, 1)


class TestPeek:
    def test_peek_basic(self):
        seq = [0, 1, 2, 3, 4]
        first_val, seq = peek(seq)
        assert first_val == 0
        assert list(seq) == [0, 1, 2, 3, 4]

    def test_peek_empty(self):
        with pytest.raises(StopIteration):
            peek([])


class TestPeekn:
    def test_peekn_basic(self):
        seq = [0, 1, 2, 3, 4]
        first_two, seq = peekn(2, seq)
        assert first_two == (0, 1)
        assert list(seq) == [0, 1, 2, 3, 4]

    def test_peekn_more_than_available(self):
        seq = [0, 1]
        first_three, seq = peekn(3, seq)
        assert first_three == (0, 1)
        assert list(seq) == [0, 1]


class TestRandomSample:
    def test_random_sample_deterministic(self):
        seq = list(range(100))
        result1 = list(random_sample(0.1, seq, random_state=2016))
        result2 = list(random_sample(0.1, seq, random_state=2016))
        assert result1 == result2

    def test_random_sample_probability(self):
        seq = list(range(1000))
        result = random_sample(0.5, seq, random_state=42)
        count = sum(1 for _ in result)
        # Should be roughly 500, allow variance
        assert 400 < count < 600

    def test_random_sample_zero_probability(self):
        result = list(random_sample(0, [1, 2, 3], random_state=42))
        assert result == []
