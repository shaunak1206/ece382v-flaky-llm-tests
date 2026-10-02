import itertools
import random
import time

import pytest

from more_itertools import (
    random_combination,
    random_combination_with_replacement,
    random_permutation,
    random_product,
    sample,
    time_limited,
)


# ------------------------------------------------------------------- sample


def test_sample_returns_k_distinct_items_from_population():
    population = list(range(100))
    result = sample(population, 5)
    assert isinstance(result, list)
    assert len(result) == 5
    assert len(set(result)) == 5
    assert set(result) <= set(population)


def test_sample_edge_cases_for_k():
    assert sample(range(10), 0) == []
    assert sorted(sample(range(3), 5)) == [0, 1, 2]
    assert sorted(sample(range(4), 4)) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        sample(range(3), 5, strict=True)
    with pytest.raises(ValueError):
        sample(range(3), -1)


def test_sample_accepts_non_indexable_inputs():
    assert len(sample({1, 2, 3, 4, 5, 6}, 3)) == 3
    assert set(sample({'a': 1, 'b': 2, 'c': 3}, 2)) <= {'a', 'b', 'c'}
    gen = (x * x for x in range(50))
    result = sample(gen, 4)
    assert len(result) == 4
    assert set(result) <= {x * x for x in range(50)}


def test_sample_is_reproducible_with_seed():
    random.seed(1234)
    first_run = sample(range(1000), 10)
    random.seed(1234)
    second_run = sample(range(1000), 10)
    assert first_run == second_run


def test_sample_eventually_selects_every_element():
    seen = set()
    for _ in range(300):
        seen.update(sample(range(5), 1))
    assert seen == {0, 1, 2, 3, 4}


def test_sample_with_counts():
    result = sample(['a', 'b'], k=3, counts=[3, 4])
    assert len(result) == 3
    assert set(result) <= {'a', 'b'}
    assert result.count('a') <= 3
    everything = sample(['a', 'b'], k=7, counts=[3, 4])
    assert sorted(everything) == ['a', 'a', 'a', 'b', 'b', 'b', 'b']


def test_sample_with_weights_prefers_heavy_items():
    population = [0, 1, 2, 3]
    weights = [1, 1, 1, 1000000]
    picks = [sample(population, 1, weights=weights)[0] for _ in range(100)]
    assert picks.count(3) > 90
    result = sample(population, 2, weights=weights)
    assert len(result) == 2
    assert len(set(result)) == 2


def test_sample_weights_and_counts_are_mutually_exclusive():
    with pytest.raises(TypeError):
        sample([1, 2, 3], 2, weights=[1, 1, 1], counts=[1, 1, 1])


# ------------------------------------------------------------ random_product


def test_random_product_draws_one_item_from_each_pool():
    result = random_product('abc', range(4), 'XYZ')
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] in 'abc'
    assert result[1] in range(4)
    assert result[2] in 'XYZ'


def test_random_product_repeat_and_empty():
    result = random_product('abcd', range(4), repeat=2)
    assert len(result) == 4
    assert result[0] in 'abcd' and result[2] in 'abcd'
    assert result[1] in range(4) and result[3] in range(4)
    assert random_product() == ()


def test_random_product_is_reproducible_with_seed():
    random.seed(99)
    a = random_product('abcdef', range(10), repeat=3)
    random.seed(99)
    b = random_product('abcdef', range(10), repeat=3)
    assert a == b


# --------------------------------------------------------- random_permutation


def test_random_permutation_full_length():
    result = random_permutation(range(5))
    assert isinstance(result, tuple)
    assert sorted(result) == [0, 1, 2, 3, 4]


def test_random_permutation_with_r():
    result = random_permutation('abcdef', 3)
    assert len(result) == 3
    assert len(set(result)) == 3
    assert set(result) <= set('abcdef')
    assert random_permutation([]) == ()


def test_random_permutation_changes_order_sometimes():
    seen = {random_permutation(range(4)) for _ in range(200)}
    assert len(seen) > 1


# --------------------------------------------------------- random_combination


def test_random_combination_keeps_input_order():
    result = random_combination(range(10), 3)
    assert len(result) == 3
    assert list(result) == sorted(result)
    assert len(set(result)) == 3


def test_random_combination_edge_cases():
    assert random_combination(range(5), 0) == ()
    assert random_combination(range(5), 5) == (0, 1, 2, 3, 4)
    with pytest.raises(ValueError):
        random_combination(range(3), 5)


def test_random_combination_with_replacement():
    result = random_combination_with_replacement(range(3), 5)
    assert len(result) == 5
    assert list(result) == sorted(result)
    assert set(result) <= {0, 1, 2}
    assert random_combination_with_replacement('abc', 0) == ()
    assert random_combination_with_replacement('a', 3) == ('a', 'a', 'a')


# ---------------------------------------------------------------- time_limited


def test_time_limited_yields_everything_when_fast_enough():
    it = time_limited(10, range(5))
    assert list(it) == [0, 1, 2, 3, 4]
    assert it.timed_out is False


def test_time_limited_stops_after_limit():
    def generator():
        yield 1
        yield 2
        time.sleep(0.2)
        yield 3

    iterable = time_limited(0.1, generator())
    assert list(iterable) == [1, 2]
    assert iterable.timed_out is True


def test_time_limited_zero_and_negative_limits():
    it = time_limited(0, [1, 2, 3])
    assert list(it) == []
    assert it.timed_out is True
    with pytest.raises(ValueError):
        time_limited(-1, [1, 2, 3])


def test_time_limited_on_infinite_iterator():
    it = time_limited(0.05, itertools.count())
    items = list(it)
    assert it.timed_out is True
    assert len(items) > 0
    assert items[:5] == [0, 1, 2, 3, 4]


def test_time_limited_with_fake_clock(monkeypatch):
    ticks = iter([0, 1, 2, 3, 4, 5])
    monkeypatch.setattr('more_itertools.more.monotonic', lambda: next(ticks))
    it = time_limited(2.5, [10, 20, 30, 40])
    assert list(it) == [10, 20]
    assert it.timed_out is True
