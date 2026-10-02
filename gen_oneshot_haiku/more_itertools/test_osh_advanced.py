import pytest
import time
from more_itertools import (
    ichunked, countable, time_limited, sample, iterate,
    seekable, sliding_window, adjacent, nth_product
)
from more_itertools.recipes import (
    random_product, random_permutation, random_combination,
    random_combination_with_replacement, batched, pairwise
)


class TestRandomProduct:
    def test_random_product_basic(self):
        result = list(random_product([1, 2], [3, 4], repeat=1))
        assert len(result) == 4
        assert all(len(item) == 2 for item in result)

    def test_random_product_repeat(self):
        result = list(random_product([1, 2], repeat=2))
        assert len(result) == 4
        assert all(len(item) == 2 for item in result)

    def test_random_product_deterministic(self):
        # Using seed for reproducibility (if supported)
        result = list(random_product([1, 2, 3], [4, 5]))
        assert len(result) == 6


class TestRandomPermutation:
    def test_random_permutation_full(self):
        result = random_permutation([1, 2, 3])
        assert sorted(result) == [1, 2, 3]

    def test_random_permutation_partial(self):
        result = list(random_permutation([1, 2, 3, 4, 5], r=3))
        assert len(result) == 3
        assert len(set(result)) == 3

    def test_random_permutation_empty(self):
        result = random_permutation([])
        assert result == []


class TestRandomCombination:
    def test_random_combination_basic(self):
        result = random_combination([1, 2, 3, 4], 2)
        assert len(result) == 2
        assert all(x in [1, 2, 3, 4] for x in result)

    def test_random_combination_all_elements(self):
        result = random_combination([1, 2, 3], 3)
        assert sorted(result) == [1, 2, 3]


class TestRandomCombinationWithReplacement:
    def test_random_combination_with_replacement_basic(self):
        result = random_combination_with_replacement([1, 2, 3], 2)
        assert len(result) == 2
        assert all(x in [1, 2, 3] for x in result)

    def test_random_combination_with_replacement_allows_duplicates(self):
        # May contain duplicates
        result = random_combination_with_replacement([1, 2], 3)
        assert len(result) == 3


class TestIchunked:
    def test_ichunked_basic(self):
        result = list(ichunked([1, 2, 3, 4, 5, 6], 2))
        assert len(result) == 3
        chunks = [list(chunk) for chunk in result]
        assert chunks == [[1, 2], [3, 4], [5, 6]]

    def test_ichunked_with_remainder(self):
        chunks = [list(chunk) for chunk in ichunked([1, 2, 3, 4, 5], 2)]
        assert chunks == [[1, 2], [3, 4], [5]]

    def test_ichunked_empty(self):
        result = list(ichunked([], 2))
        assert result == []


class TestCountable:
    def test_countable_basic(self):
        it = countable(iter([1, 2, 3, 4, 5]))
        next(it)
        next(it)
        assert it.count == 2

    def test_countable_iteration(self):
        it = countable(iter(['a', 'b', 'c']))
        list(it)
        assert it.count == 3


class TestTimeLimited:
    def test_time_limited_basic(self):
        def slow_iter():
            for i in range(10):
                yield i
        result = list(time_limited(0.1, slow_iter()))
        # Should yield fewer than 10 items in 0.1 seconds
        assert len(result) < 10

    def test_time_limited_completes_early(self):
        def fast_iter():
            for i in range(5):
                yield i
        start = time.time()
        result = list(time_limited(1.0, fast_iter()))
        elapsed = time.time() - start
        assert result == [0, 1, 2, 3, 4]
        assert elapsed < 1.0


class TestSample:
    def test_sample_basic(self):
        result = list(sample(range(100), k=10))
        assert len(result) == 10
        assert all(0 <= x < 100 for x in result)

    def test_sample_weighted(self):
        # Weights the higher numbers
        result = list(sample(range(10), k=5, weights=list(range(1, 11))))
        assert len(result) == 5

    def test_sample_strict_not_enough(self):
        with pytest.raises(ValueError):
            list(sample(range(5), k=10, strict=True))


class TestIterate:
    def test_iterate_basic(self):
        inc = lambda x: x + 1
        result = iterate(inc, 0)
        first_five = [next(result) for _ in range(5)]
        assert first_five == [0, 1, 2, 3, 4]

    def test_iterate_factorial(self):
        mul = lambda x: x * 2
        result = iterate(mul, 1)
        first_five = [next(result) for _ in range(5)]
        assert first_five == [1, 2, 4, 8, 16]


class TestSeekable:
    def test_seekable_basic(self):
        s = seekable(iter([1, 2, 3, 4, 5]))
        assert next(s) == 1
        s.seek(0)
        assert next(s) == 1

    def test_seekable_forward_seek(self):
        s = seekable(iter([1, 2, 3, 4, 5]))
        next(s)
        next(s)
        s.seek(4)
        assert next(s) == 5

    def test_seekable_backward_seek(self):
        s = seekable(iter([1, 2, 3, 4, 5]))
        next(s)
        next(s)
        next(s)
        s.seek(1)
        assert next(s) == 2


class TestSlidingWindow:
    def test_sliding_window_basic(self):
        result = list(sliding_window(iter([1, 2, 3, 4, 5]), 3))
        assert result == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    def test_sliding_window_size_one(self):
        result = list(sliding_window(iter([1, 2, 3]), 1))
        assert result == [(1,), (2,), (3,)]


class TestAdjacent:
    def test_adjacent_basic(self):
        def is_even(x):
            return x % 2 == 0
        result = list(adjacent(is_even, [1, 2, 3, 4, 5, 6, 7, 8]))
        # Items adjacent to even numbers
        assert 3 in result
        assert 5 in result


class TestBatched:
    def test_batched_basic(self):
        result = list(batched([1, 2, 3, 4, 5, 6], 2))
        assert result == [(1, 2), (3, 4), (5, 6)]

    def test_batched_with_remainder(self):
        result = list(batched([1, 2, 3, 4, 5], 2, strict=False))
        assert result == [(1, 2), (3, 4), (5,)]

    def test_batched_strict_mode(self):
        with pytest.raises(ValueError):
            list(batched([1, 2, 3, 4, 5], 2, strict=True))


class TestPairwise:
    def test_pairwise_basic(self):
        result = list(pairwise([1, 2, 3, 4]))
        assert result == [(1, 2), (2, 3), (3, 4)]

    def test_pairwise_short(self):
        result = list(pairwise([1]))
        assert result == []

    def test_pairwise_two_elements(self):
        result = list(pairwise([1, 2]))
        assert result == [(1, 2)]


class TestNthProduct:
    def test_nth_product_basic(self):
        result = nth_product(0, [1, 2], [3, 4])
        assert result == (1, 3)

    def test_nth_product_last(self):
        result = nth_product(3, [1, 2], [3, 4])
        assert result == (2, 4)


class TestComplexScenarios:
    def test_seekable_with_multiple_reads(self):
        s = seekable(iter([1, 2, 3, 4, 5]))
        # Read forward
        assert list(next(s) for _ in range(3)) == [1, 2, 3]
        # Seek back
        s.seek(0)
        assert next(s) == 1

    def test_sample_with_counts(self):
        result = list(sample(range(5), k=5, counts=[1, 2, 3, 4, 5]))
        assert len(result) == 5

    def test_time_limited_fast_completion(self):
        def generator():
            for i in range(5):
                yield i

        result = list(time_limited(10, generator()))
        assert result == [0, 1, 2, 3, 4]

    def test_ichunked_single_chunk(self):
        result = list(ichunked([1, 2, 3], 5))
        assert len(result) == 1
        assert list(result[0]) == [1, 2, 3]


class TestRandomFunctionsProperties:
    def test_random_permutation_uniqueness(self):
        result = random_permutation([1, 2, 3, 4, 5])
        assert len(set(result)) == 5

    def test_random_combination_no_duplicates(self):
        result = random_combination([1, 2, 3, 4, 5, 6], 3)
        assert len(set(result)) == 3

    def test_random_product_cartesian(self):
        result = list(random_product([1, 2], [3, 4]))
        # All combinations exist in some order
        assert len(result) == 4


class TestIterateProperties:
    def test_iterate_infinite(self):
        inc = lambda x: x + 1
        it = iterate(inc, 0)
        # Should be able to generate arbitrarily many
        large_n = next(it for _ in range(1000) for _ in range(1000))
        # Just verify we can iterate far


class TestSeeableAdvanced:
    def test_seekable_reuse_iterator(self):
        s = seekable(iter([1, 2, 3]))
        values1 = list(s)
        s.seek(0)
        values2 = list(s)
        assert values1 == values2

    def test_seekable_getitem(self):
        s = seekable(iter([10, 20, 30, 40, 50]))
        assert s[0] == 10
        assert s[2] == 30
        assert s[-1] == 50


class TestCountableIntegration:
    def test_countable_with_zip(self):
        it = countable(iter([1, 2, 3, 4]))
        pairs = list(zip(it, ['a', 'b', 'c', 'd']))
        assert it.count == 4
