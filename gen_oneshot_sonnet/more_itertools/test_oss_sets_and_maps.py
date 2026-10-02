from collections import defaultdict
from itertools import count, permutations, product
from math import factorial

import pytest

from more_itertools import (
    bucket,
    distinct_permutations,
    map_reduce,
    nth_product,
    powerset,
    unique_everseen,
    unique_justseen,
)


def test_powerset_of_list_and_edge_cases():
    assert list(powerset([1, 2, 3])) == [
        (), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    assert list(powerset([])) == [()]
    assert len(list(powerset(range(5)))) == 2 ** 5
    assert list(powerset([1, 1, 0])) == [
        (), (1,), (1,), (0,), (1, 1), (1, 0), (1, 0), (1, 1, 0)]


def test_powerset_as_set_of_frozensets():
    subsets = {frozenset(s) for s in powerset('abc')}
    assert len(subsets) == 8
    assert frozenset() in subsets
    assert frozenset('abc') in subsets
    assert frozenset('ac') in subsets


def test_distinct_permutations_basic():
    assert sorted(distinct_permutations([1, 0, 1])) == [
        (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    assert list(distinct_permutations([2, 1, 2])) == [
        (1, 2, 2), (2, 1, 2), (2, 2, 1)]
    assert sorted(distinct_permutations([1, 0, 1], r=2)) == [
        (0, 1), (1, 0), (1, 1)]
    assert sorted(distinct_permutations(range(3), r=2)) == [
        (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]


def test_distinct_permutations_match_set_of_permutations():
    items = 'aabc'
    expected = set(permutations(items))
    result = list(distinct_permutations(items))
    assert set(result) == expected
    assert len(result) == len(expected)
    counted = [1, 1, 2, 2, 2, 3]
    formula = factorial(6) // (factorial(2) * factorial(3) * factorial(1))
    assert len(list(distinct_permutations(counted))) == formula


def test_distinct_permutations_edge_cases():
    assert list(distinct_permutations([])) == [()]
    assert list(distinct_permutations([1, 2], r=0)) == [()]
    assert list(distinct_permutations([1, 2], r=5)) == []


def test_unique_everseen_basic_and_key():
    assert list(unique_everseen('AAAABBBCCDAABBB')) == ['A', 'B', 'C', 'D']
    assert list(unique_everseen('ABBCcAD', str.casefold)) == [
        'A', 'B', 'C', 'D']
    assert list(unique_everseen([[1], [2], [1]], key=tuple)) == [[1], [2]]
    with pytest.raises(TypeError):
        list(unique_everseen([[1], [2]]))


def test_unique_everseen_agrees_with_set():
    data = [5, 3, 5, 1, 3, 2, 1]
    result = list(unique_everseen(data))
    assert set(result) == set(data)
    assert result == [5, 3, 1, 2]
    assert list(unique_justseen([5, 5, 3, 3, 5])) == [5, 3, 5]
    assert list(unique_justseen('AAAABBBCCDAABBB')) == [
        'A', 'B', 'C', 'D', 'A', 'B']
    assert list(unique_justseen('ABBCcAD', str.lower)) == [
        'A', 'B', 'C', 'A', 'D']
    assert list(unique_justseen([])) == []


def test_map_reduce_groups_transforms_and_reduces():
    result = map_reduce('abbccc', lambda x: x.upper())
    assert dict(result) == {'A': ['a'], 'B': ['b', 'b'], 'C': ['c', 'c', 'c']}
    keyfunc = lambda x: x.upper()  # noqa: E731
    assert map_reduce('abbccc', keyfunc, lambda x: 1) == {
        'A': [1], 'B': [1, 1], 'C': [1, 1, 1]}
    assert map_reduce('abbccc', keyfunc, lambda x: 1, sum) == {
        'A': 1, 'B': 2, 'C': 3}


def test_map_reduce_odd_even_summaries():
    items = [x for x in range(30) if 10 <= x <= 20]
    categories = map_reduce(items, keyfunc=lambda x: x % 2)
    assert categories == {
        0: [10, 12, 14, 16, 18, 20], 1: [11, 13, 15, 17, 19]}
    summaries = map_reduce(items, keyfunc=lambda x: x % 2, reducefunc=sum)
    assert summaries == {0: 90, 1: 75}


def test_map_reduce_result_behaves_like_plain_dict():
    result = map_reduce([1, 2, 3], lambda x: x % 2)
    assert isinstance(result, defaultdict)
    assert result.default_factory is None
    with pytest.raises(KeyError):
        result[99]
    assert map_reduce([], lambda x: x) == {}


def test_bucket_keys_and_children():
    iterable = ['a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'b3']
    s = bucket(iterable, key=lambda x: x[0])
    assert sorted(list(s)) == ['a', 'b', 'c']
    a_iterable = s['a']
    assert next(a_iterable) == 'a1'
    assert next(a_iterable) == 'a2'
    assert list(s['b']) == ['b1', 'b2', 'b3']
    assert list(s['c']) == ['c1', 'c2']


def test_bucket_children_without_listing_keys_first():
    s = bucket(range(10), key=lambda x: x % 3)
    assert list(s[2]) == [2, 5, 8]
    assert list(s[0]) == [0, 3, 6, 9]
    assert list(s[1]) == [1, 4, 7]
    t = bucket([1, 2, 3, 4], key=lambda x: x % 2)
    assert 1 in t
    assert 0 in t
    assert 5 not in t
    assert list(t[5]) == []


def test_bucket_validator_avoids_exhausting_infinite_iterables():
    key = lambda x: x % 10  # noqa: E731
    validator = lambda x: x in {1, 3, 5, 7, 9}  # noqa: E731
    s = bucket(count(1, 2), key=key, validator=validator)
    assert 2 not in s
    assert list(s[2]) == []
    assert 3 in s


def test_nth_product_basic_and_matches_itertools():
    assert nth_product(8, range(2), range(2), range(2), range(2)) == (
        1, 0, 0, 0)
    assert nth_product(8, range(2), repeat=4) == (1, 0, 0, 0)
    pools = ('ab', range(3), 'xy')
    for index, value in enumerate(product(*pools)):
        assert nth_product(index, *pools) == value


def test_nth_product_negative_index_and_out_of_range():
    pools = (range(3), range(4))
    expected = list(product(*pools))
    assert nth_product(-1, *pools) == expected[-1]
    with pytest.raises(IndexError):
        nth_product(12, *pools)
    with pytest.raises(IndexError):
        nth_product(-13, *pools)
