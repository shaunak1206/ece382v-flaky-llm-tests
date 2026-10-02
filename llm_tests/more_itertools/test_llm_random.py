import random
from collections import Counter
from itertools import combinations, combinations_with_replacement, permutations, product

import pytest

from more_itertools import (
    random_product,
    random_permutation,
    random_combination,
    random_combination_with_replacement,
    sample,
)


def test_random_product_shape_and_membership():
    result = random_product("abc", range(4), "XYZ")
    assert len(result) == 3
    assert result[0] in "abc"
    assert result[1] in range(4)
    assert result[2] in "XYZ"


def test_random_product_is_in_itertools_product():
    pools = ("abc", [1, 2])
    allowed = set(product(*pools))
    for _ in range(25):
        assert random_product(*pools) in allowed


def test_random_product_seeded_reproducible():
    random.seed(1234)
    a = random_product(range(100), range(100))
    random.seed(1234)
    b = random_product(range(100), range(100))
    assert a == b


def test_random_permutation_full_length():
    result = random_permutation(range(6))
    assert sorted(result) == list(range(6))


def test_random_permutation_r():
    result = random_permutation("abcdef", 3)
    assert len(result) == 3
    assert len(set(result)) == 3
    assert result in set(permutations("abcdef", 3))


def test_random_permutation_r_too_large():
    with pytest.raises(ValueError):
        random_permutation("abc", 5)


def test_random_combination_is_ordered_subsequence():
    data = list(range(10))
    for _ in range(20):
        result = random_combination(data, 4)
        assert result in set(combinations(data, 4))


def test_random_combination_with_replacement_valid():
    allowed = set(combinations_with_replacement("abc", 4))
    for _ in range(20):
        assert random_combination_with_replacement("abc", 4) in allowed


def test_sample_size_and_membership():
    population = list(range(100))
    result = sample(population, 10)
    assert len(result) == 10
    assert len(set(result)) == 10
    assert set(result) <= set(population)


def test_sample_strict_raises_when_too_short():
    with pytest.raises(ValueError):
        sample([1, 2, 3], 10, strict=True)


def test_sample_with_counts():
    result = sample(["a", "b"], k=7, counts=[3, 4])
    assert Counter(result) == Counter({"a": 3, "b": 4})


def test_sample_weighted_zero_weight_never_chosen():
    for _ in range(20):
        result = sample("abc", 2, weights=[1, 1, 0])
        assert set(result) == {"a", "b"}


def test_sample_seeded_reproducible():
    random.seed(2024)
    a = sample(range(1000), 10)
    random.seed(2024)
    b = sample(range(1000), 10)
    assert a == b


