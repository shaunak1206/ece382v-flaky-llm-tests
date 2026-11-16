import random

import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga


class FakeCondition(list):
    """List-like condition with the API used by the GA strategies."""

    def __init__(self, items, wildcard='#', matches=True):
        super().__init__(items)
        self.wildcard = wildcard
        self.matches = matches

    def generalize(self, idx):
        self[idx] = self.wildcard

    def does_match(self, p):
        return self.matches


class FakeCfg:
    def __init__(self, length=5, wildcard='#'):
        self.classifier_length = length
        self.classifier_wildcard = wildcard


class FakeCl:
    def __init__(self, cond='abc', num=1, tga=0, q=0.5, tav=0.0,
                 marked=False, matches=True, length=None):
        self.condition = FakeCondition(list(cond), matches=matches)
        self.cfg = FakeCfg(length if length is not None else len(cond))
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self._marked = marked

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return isinstance(other, FakeCl) and \
            list(self.condition) == list(other.condition)

    def __hash__(self):
        return id(self)


class FakeSet(list):
    def expand(self):
        result = []
        for cl in self:
            result.extend([cl] * cl.num)
        return result

    def safe_remove(self, cl):
        for i, x in enumerate(self):
            if x is cl:
                del self[i]
                return


def test_should_apply_none_action_set():
    assert ga.should_apply(None, 100, 10) is False


def test_should_apply_empty_action_set():
    assert ga.should_apply([], 100, 10) is False


def test_should_apply_zero_numerosity():
    assert ga.should_apply([FakeCl(num=0, tga=0)], 100, 10) is False


def test_should_apply_true_when_delay_exceeds_threshold():
    action_set = [FakeCl(num=1, tga=0), FakeCl(num=3, tga=20)]
    # average tga = (0 + 60) / 4 = 15; 30 - 15 = 15 > 10
    assert ga.should_apply(action_set, 30, 10) is True


def test_should_apply_false_when_delay_equal_or_lower():
    action_set = [FakeCl(num=2, tga=10)]
    assert ga.should_apply(action_set, 20, 10) is False
    assert ga.should_apply(action_set, 15, 10) is False


def test_set_timestamps_updates_all():
    action_set = [FakeCl(tga=1), FakeCl(tga=5), FakeCl(tga=9)]
    ga.set_timestamps(action_set, 42)
    assert all(cl.tga == 42 for cl in action_set)


def test_roulette_wheel_selection_picks_only_fit_classifier():
    random.seed(1)
    a, b, c = FakeCl('aaa'), FakeCl('bbb'), FakeCl('ccc')
    a.q, b.q, c.q = 0.0, 1.0, 0.0
    p1, p2 = ga.roulette_wheel_selection([a, b, c], lambda cl: cl.q ** 3)
    assert p1 is b
    assert p2 is b


def test_roulette_wheel_selection_returns_population_members():
    random.seed(3)
    pop = [FakeCl('a'), FakeCl('b'), FakeCl('c')]
    for _ in range(20):
        p1, p2 = ga.roulette_wheel_selection(pop, lambda cl: 1.0)
        assert any(p1 is x for x in pop)
        assert any(p2 is x for x in pop)


def test_weighted_random_choice_ignores_zero_weight():
    random.seed(5)
    for _ in range(20):
        assert ga._weighted_random_choice({'a': 0, 'b': 5}) == 'b'


def test_generalizing_mutation_mu_one_generalizes_all():
    cl = FakeCl('abc')
    ga.generalizing_mutation(cl, 1.0)
    assert list(cl.condition) == ['#', '#', '#']


def test_generalizing_mutation_mu_zero_changes_nothing():
    cl = FakeCl('abc')
    ga.generalizing_mutation(cl, 0.0)
    assert list(cl.condition) == ['a', 'b', 'c']


def test_generalizing_mutation_keeps_existing_wildcards():
    cl = FakeCl('a#c')
    ga.generalizing_mutation(cl, 1.0)
    assert list(cl.condition) == ['#', '#', '#']


def test_two_point_crossover_swaps_contiguous_segment():
    np.random.seed(0)
    for _ in range(30):
        parent = FakeCl('aaaaa')
        donor = FakeCl('bbbbb')
        ga.two_point_crossover(parent, donor)
        p, d = list(parent.condition), list(donor.condition)
        assert len(p) == len(d) == 5
        swapped = [i for i in range(5) if p[i] == 'b']
        assert swapped, "at least one position must be swapped"
        assert swapped == list(range(swapped[0], swapped[-1] + 1))
        # donor is the exact complement of parent
        for i in range(5):
            assert {p[i], d[i]} == {'a', 'b'}
            assert (p[i] == 'b') == (d[i] == 'a')


def test_add_classifier_new_is_appended_to_all_sets():
    population, match_set, action_set = [], [], []
    cl = FakeCl('abc', matches=True)
    ga.add_classifier(cl, None, population, match_set, action_set,
                      do_subsumption=False, theta_exp=20)
    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_not_matching_not_added_to_match_set():
    population, match_set, action_set = [], [], []
    cl = FakeCl('abc', matches=False)
    ga.add_classifier(cl, None, population, match_set, action_set,
                      do_subsumption=False, theta_exp=20)
    assert population == [cl]
    assert match_set == []


def test_add_classifier_similar_increases_numerosity():
    old = FakeCl('abc', num=2)
    new = FakeCl('abc')
    population, match_set, action_set = [old], [old], [old]
    ga.add_classifier(new, None, population, match_set, action_set,
                      do_subsumption=False, theta_exp=20)
    assert old.num == 3
    assert len(population) == 1
    assert len(action_set) == 1


def test_add_classifier_marked_old_numerosity_unchanged():
    old = FakeCl('abc', num=2, marked=True)
    new = FakeCl('abc')
    population, match_set, action_set = [old], [], [old]
    ga.add_classifier(new, None, population, match_set, action_set,
                      do_subsumption=False, theta_exp=20)
    assert old.num == 2
    assert population == [old]


def test_add_classifier_uses_subsumer(monkeypatch):
    subsumer = FakeCl('a#c', num=1)
    new = FakeCl('abc')
    monkeypatch.setattr(ga, 'find_subsumers',
                        lambda cl, pop, theta: [subsumer])
    population, action_set = [subsumer], [subsumer]
    ga.add_classifier(new, None, population, None, action_set,
                      do_subsumption=True, theta_exp=20)
    assert subsumer.num == 2
    assert new not in [x for x in population if x is new]


def test_delete_classifiers_nothing_when_space_available():
    random.seed(0)
    cls = [FakeCl('a'), FakeCl('b')]
    population, action_set = FakeSet(cls), FakeSet(cls)
    ga.delete_classifiers(population, None, action_set, insize=1, theta_as=3)
    assert len(population) == 2
    assert len(action_set) == 2


def test_delete_classifiers_removes_single_numerosity_classifier():
    random.seed(0)
    cls = [FakeCl('a'), FakeCl('b'), FakeCl('c')]
    population, action_set = FakeSet(cls), FakeSet(cls)
    match_set = FakeSet(cls)
    ga.delete_classifiers(population, match_set, action_set,
                          insize=1, theta_as=3)
    assert len(action_set) == 2
    assert len(population) == 2
    assert len(match_set) == 2
    assert sum(cl.num for cl in action_set) + 1 <= 3


def test_delete_classifiers_decrements_numerosity():
    random.seed(0)
    cl = FakeCl('a', num=3)
    population, action_set = FakeSet([cl]), FakeSet([cl])
    ga.delete_classifiers(population, None, action_set, insize=1, theta_as=3)
    assert cl.num == 2
    assert len(population) == 1


def test_is_preferred_to_delete_much_lower_quality():
    cl_del = FakeCl(q=0.9)
    cl = FakeCl(q=0.5)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_much_higher_quality():
    cl_del = FakeCl(q=0.2)
    cl = FakeCl(q=0.8)
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_similar_quality_uses_tav_or_mark():
    cl_del = FakeCl(q=0.5, tav=1.0)
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.55, tav=5.0)) is True
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.55, tav=0.5)) is False
    marked = FakeCl(q=0.5, tav=0.0, marked=True)
    assert ga._is_preferred_to_delete(cl_del, marked) is True


def test_find_similar_returns_first_equal_or_none():
    first = FakeCl('abc')
    second = FakeCl('abc')
    other = FakeCl('xyz')
    assert ga._find_similar(FakeCl('abc'), [other, first, second]) is first
    assert ga._find_similar(FakeCl('qqq'), [other, first]) is None
    assert ga._find_similar(FakeCl('abc'), []) is None
