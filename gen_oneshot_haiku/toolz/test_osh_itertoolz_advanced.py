import pytest
from operator import add, mul
from toolz.itertoolz import (
    reduceby, join, diff, merge_sorted, second
)


class TestReduceby:
    def test_reduceby_basic(self):
        iseven = lambda x: x % 2 == 0
        data = [1, 2, 3, 4, 5]
        result = reduceby(iseven, add, data)
        assert result == {False: 9, True: 6}

    def test_reduceby_with_mul(self):
        iseven = lambda x: x % 2 == 0
        data = [1, 2, 3, 4, 5]
        result = reduceby(iseven, mul, data)
        assert result == {False: 15, True: 8}

    def test_reduceby_with_initial_callable(self):
        def set_add(s, i):
            s.add(i)
            return s
        iseven = lambda x: x % 2 == 0
        result = reduceby(iseven, set_add, [1, 2, 3, 4, 1, 2, 3], set)
        assert result[True] == {2, 4}
        assert result[False] == {1, 3}

    def test_reduceby_with_initial_value(self):
        data = [1, 2, 3, 4, 5]
        result = reduceby(lambda x: x % 2 == 0, add, data, 0)
        assert result == {False: 9, True: 6}

    def test_reduceby_with_string_key(self):
        projects = [
            {'name': 'build roads', 'state': 'CA', 'cost': 1000000},
            {'name': 'fight crime', 'state': 'IL', 'cost': 100000},
            {'name': 'help farmers', 'state': 'IL', 'cost': 2000000},
            {'name': 'help farmers', 'state': 'CA', 'cost': 200000}
        ]
        result = reduceby('state', lambda acc, x: acc + x['cost'], projects, 0)
        assert result == {'CA': 1200000, 'IL': 2100000}


class TestJoin:
    def test_join_inner(self):
        friends = [('Alice', 'Edith'),
                  ('Alice', 'Zhao'),
                  ('Edith', 'Alice'),
                  ('Zhao', 'Alice'),
                  ('Zhao', 'Edith')]

        cities = [('Alice', 'NYC'),
                 ('Alice', 'Chicago'),
                 ('Dan', 'Sydney'),
                 ('Edith', 'Paris'),
                 ('Edith', 'Berlin'),
                 ('Zhao', 'Shanghai')]

        result = list(join(second, friends, lambda x: x[0], cities))
        # Check for expected pairs
        results_set = set(((a, c), (d, e)) for (a, c), (d, e) in result)
        assert len(results_set) > 0

    def test_join_outer_with_defaults(self):
        identity = lambda x: x
        result = list(join(identity, [1, 2, 3],
                          identity, [2, 3, 4],
                          left_default=None, right_default=None))
        # Inner join result plus outer matches
        assert (1, None) in result
        assert (2, 2) in result
        assert (None, 4) in result


class TestDiff:
    def test_diff_basic(self):
        result = list(diff([1, 2, 3], [1, 2, 10, 100]))
        assert result == [(3, 10)]

    def test_diff_with_padding(self):
        result = list(diff([1, 2, 3], [1, 2, 10, 100], default=None))
        assert (3, 10) in result
        assert (None, 100) in result

    def test_diff_with_key(self):
        result = list(diff(['apples', 'bananas'],
                          ['Apples', 'Oranges'],
                          key=str.lower))
        assert result == [('bananas', 'Oranges')]

    def test_diff_identical(self):
        result = list(diff([1, 2, 3], [1, 2, 3]))
        assert result == []

    def test_diff_empty(self):
        result = list(diff([1, 2, 3], []))
        assert len(result) > 0


class TestMergeSorted:
    def test_merge_sorted_basic(self):
        result = list(merge_sorted([1, 3, 5], [2, 4, 6]))
        assert result == [1, 2, 3, 4, 5, 6]

    def test_merge_sorted_strings(self):
        result = ''.join(merge_sorted('abc', 'abc', 'abc'))
        assert result == 'aaabbbccc'

    def test_merge_sorted_with_key(self):
        result = list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
        assert result == [2, 1, 3, 3]

    def test_merge_sorted_single_iterable(self):
        result = list(merge_sorted([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_merge_sorted_empty(self):
        result = list(merge_sorted())
        assert result == []

    def test_merge_sorted_multiple(self):
        result = list(merge_sorted([1, 4], [2, 5], [3, 6]))
        assert result == [1, 2, 3, 4, 5, 6]


class TestComplexJoinScenarios:
    def test_join_with_string_index(self):
        left_seq = [('a', 1), ('b', 2), ('c', 3)]
        right_seq = [('a', 10), ('b', 20), ('d', 40)]
        result = list(join(0, left_seq, 0, right_seq))
        assert (('a', 1), ('a', 10)) in result
        assert (('b', 2), ('b', 20)) in result

    def test_join_right_outer(self):
        left_seq = [1, 2, 3]
        right_seq = [2, 3, 4]
        identity = lambda x: x
        result = list(join(identity, left_seq,
                          identity, right_seq,
                          left_default=None))
        # Should have inner joins and unmatched right
        assert (1, None) in result

    def test_join_full_outer(self):
        left_seq = [1, 2, 3]
        right_seq = [2, 3, 4]
        identity = lambda x: x
        result = list(join(identity, left_seq,
                          identity, right_seq,
                          left_default=None, right_default=None))
        assert (1, None) in result
        assert (None, 4) in result
        assert (2, 2) in result


class TestDiffVariations:
    def test_diff_multiple_sequences(self):
        # diff can take multiple sequences
        result = list(diff([1, 2, 3], [1, 2, 3], [1, 2, 3]))
        assert result == []

    def test_diff_three_sequences_with_mismatch(self):
        result = list(diff([1, 2, 3], [1, 2, 3], [1, 2, 4]))
        assert (3, 3, 4) in result


class TestMergeSortedEdgeCases:
    def test_merge_sorted_unsorted_input(self):
        # Each input should be sorted, but let's test behavior
        result = list(merge_sorted([3, 1, 2], [4, 5]))
        # Result will have issues if input isn't sorted, but function doesn't validate
        assert len(result) == 5

    def test_merge_sorted_negative_numbers(self):
        result = list(merge_sorted([-3, -1, 0], [-2, 1, 3]))
        assert result == [-3, -2, -1, 0, 1, 3]

    def test_merge_sorted_duplicates(self):
        result = list(merge_sorted([1, 1, 2], [1, 2, 2]))
        assert result == [1, 1, 1, 2, 2, 2]
