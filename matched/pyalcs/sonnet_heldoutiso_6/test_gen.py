import random

import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga


class FakeCondition(list):
    def __init__(self, items, matches=True):
        super().__init__(items)
        self.matches = matches
        self.generalized = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = '#'

    def does_match(self, p):
        return self.matches


class FakeCfg:
    def __init__(self, length=4, wildcard='#'):
        self.classifier_length = length
        self.classifier_wildcard = wildcard


class FakeCl:
    def __init__(self, ident=0, num=1, tga=0, q=0.5, tav=0.0,
                 marked=False, condition=None, cfg=None):
        self.ident = ident
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self.marked = marked
        self.condition = condition if condition is not None \
            else FakeCondition([])
        self.cfg = cfg if cfg is not None else FakeCfg()

    def is_marked(self):
        return self.marked

    def __eq__(self, other):
        return isinstance(other, FakeCl) and self.ident == other.ident

    def __hash__(self):
        return hash(self.ident)


class FakeList(list):
    def expand(self):
        result = []
        for cl in self:
            result.extend([cl] * cl.num)
        return result

    def safe_remove(self, cl):
        if cl in self:
            self.remove(cl)


# ---------------- should_apply ----------------

def test_should_apply_none_action_set():
    assert ga.should_apply(None, 100, 10) is False


def test_should_apply_empty_action_set():
    assert ga.should_apply([], 100, 10) is False


def test_should_apply_zero_numerosity():
    assert ga.should_apply([FakeCl(num=0, tga=0)], 100, 10) is False


def test_should_apply_above_threshold():
    action_set = [FakeCl(ident=1, num=1, tga=0), FakeCl(ident=2, num=3, tga=0)]
    assert ga.should_apply(action_set, 11, 10) is True


def test_should_apply_equal_or_below_threshold():
    action_set = [FakeCl(num=1, tga=0)]
    assert ga.should_apply(action_set, 10, 10) is False
    assert ga.should_apply(action_set, 5, 10) is False


def test_should_apply_uses_numerosity_weighted_average():
    # average tga = (0*1 + 100*9) / 10 = 90 -> 100 - 90 = 10, not > 10
    action_set = [FakeCl(ident=1, num=1, tga=0), FakeCl(ident=2, num=9, tga=100)]
    assert ga.should_apply(action_set, 100, 10) is False
    assert ga.should_apply(action_set, 101, 10) is True


# ---------------- set_timestamps ----------------

def test_set_timestamps_updates_all():
    action_set = [FakeCl(ident=i, tga=0) for i in range(3)]
    ga.set_timestamps(action_set, 42)
    assert all(cl.tga == 42 for cl in action_set)


# ---------------- selection ----------------

def test_weighted_random_choice_picks_by_pick_position(monkeypatch):
    choices = {'a': 1, 'b': 1, 'c': 1}
    monkeypatch.setattr(random, 'uniform', lambda lo, hi: 0.5)
    assert ga._weighted_random_choice(choices) == 'a'
    monkeypatch.setattr(random, 'uniform', lambda lo, hi: 1.5)
    assert ga._weighted_random_choice(choices) == 'b'
    monkeypatch.setattr(random, 'uniform', lambda lo, hi: 2.5)
    assert ga._weighted_random_choice(choices) == 'c'


def test_roulette_wheel_single_member_returns_it_twice():
    cl = FakeCl(ident=1)
    p1, p2 = ga.roulette_wheel_selection([cl], lambda c: 1.0)
    assert p1 is cl and p2 is cl


def test_roulette_wheel_ignores_zero_fitness():
    good = FakeCl(ident=1, q=1.0)
    bad = FakeCl(ident=2, q=0.0)
    for _ in range(30):
        p1, p2 = ga.roulette_wheel_selection([bad, good], lambda c: c.q ** 3)
        assert p1 is good
        assert p2 is good


# ---------------- mutation ----------------

def test_generalizing_mutation_mu_one_generalizes_non_wildcards():
    cl = FakeCl(condition=FakeCondition(['0', '#', '1', '#']))
    ga.generalizing_mutation(cl, 1.0)
    assert cl.condition.generalized == [0, 2]
    assert list(cl.condition) == ['#', '#', '#', '#']


def test_generalizing_mutation_mu_zero_changes_nothing():
    cl = FakeCl(condition=FakeCondition(['0', '1', '0', '1']))
    ga.generalizing_mutation(cl, 0.0)
    assert cl.condition.generalized == []
    assert list(cl.condition) == ['0', '1', '0', '1']


# ---------------- crossover ----------------

def test_two_point_crossover_swaps_selected_segment(monkeypatch):
    monkeypatch.setattr(np.random, 'choice',
                        lambda *args, **kwargs: np.array([3, 1]))
    parent = FakeCl(ident=1, condition=FakeCondition(['a', 'b', 'c', 'd']))
    donor = FakeCl(ident=2, condition=FakeCondition(['1', '2', '3', '4']))
    ga.two_point_crossover(parent, donor)
    assert list(parent.condition) == ['a', '2', '3', 'd']
    assert list(donor.condition) == ['1', 'b', 'c', '4']


def test_two_point_crossover_preserves_gene_pairs():
    random.seed(1)
    np.random.seed(1)
    for _ in range(20):
        parent = FakeCl(ident=1, condition=FakeCondition(list('abcd')))
        donor = FakeCl(ident=2, condition=FakeCondition(list('1234')))
        ga.two_point_crossover(parent, donor)
        assert len(parent.condition) == 4
        assert len(donor.condition) == 4
        for i in range(4):
            assert {parent.condition[i], donor.condition[i]} == \
                {'abcd'[i], '1234'[i]}
        assert parent.condition != list('abcd') or \
            donor.condition != list('1234') or True


# ---------------- add_classifier ----------------

def test_add_classifier_new_is_added_everywhere_when_matching():
    new = FakeCl(ident=1, condition=FakeCondition(['0'], matches=True))
    population, match_set, action_set = FakeList(), FakeList(), FakeList()
    ga.add_classifier(new, object(), population, match_set, action_set,
                      False, 20)
    assert population == [new]
    assert action_set == [new]
    assert match_set == [new]


def test_add_classifier_new_not_matching_skips_match_set():
    new = FakeCl(ident=1, condition=FakeCondition(['0'], matches=False))
    population, match_set, action_set = FakeList(), FakeList(), FakeList()
    ga.add_classifier(new, object(), population, match_set, action_set,
                      False, 20)
    assert population == [new]
    assert action_set == [new]
    assert match_set == []


def test_add_classifier_none_match_set_is_ok():
    new = FakeCl(ident=1)
    population, action_set = FakeList(), FakeList()
    ga.add_classifier(new, object(), population, None, action_set, False, 20)
    assert population == [new]


def test_add_classifier_similar_increments_numerosity():
    old = FakeCl(ident=1, num=2)
    new = FakeCl(ident=1, num=1)
    population, action_set = FakeList([old]), FakeList([old])
    ga.add_classifier(new, object(), population, None, action_set, False, 20)
    assert old.num == 3
    assert len(population) == 1


def test_add_classifier_similar_marked_not_incremented():
    old = FakeCl(ident=1, num=2, marked=True)
    new = FakeCl(ident=1, num=1)
    population, action_set = FakeList([old]), FakeList([old])
    ga.add_classifier(new, object(), population, None, action_set, False, 20)
    assert old.num == 2
    assert len(population) == 1


# ---------------- finding old classifiers ----------------

def test_find_similar_returns_first_equal_or_none():
    a, b = FakeCl(ident=1), FakeCl(ident=2)
    b_dup = FakeCl(ident=2)
    assert ga._find_similar(FakeCl(ident=2), [a, b, b_dup]) is b
    assert ga._find_similar(FakeCl(ident=3), [a, b]) is None


def test_find_old_classifier_uses_first_subsumer(monkeypatch):
    subsumer = FakeCl(ident=99)
    monkeypatch.setattr(ga, 'find_subsumers',
                        lambda cl, pop, theta: [subsumer, FakeCl(ident=100)])
    result = ga._find_old_classifier([FakeCl(ident=1)], FakeCl(ident=1),
                                     True, 20)
    assert result is subsumer


def test_find_old_classifier_falls_back_to_similar(monkeypatch):
    monkeypatch.setattr(ga, 'find_subsumers', lambda cl, pop, theta: [])
    similar = FakeCl(ident=1)
    result = ga._find_old_classifier([similar], FakeCl(ident=1), True, 20)
    assert result is similar


# ---------------- deletion ----------------

def test_is_preferred_to_delete_much_lower_quality():
    assert ga._is_preferred_to_delete(FakeCl(q=0.9), FakeCl(q=0.5)) is True


def test_is_preferred_to_delete_much_higher_quality():
    assert ga._is_preferred_to_delete(FakeCl(q=0.3), FakeCl(q=0.9)) is False


def test_is_preferred_to_delete_similar_quality_by_tav():
    assert ga._is_preferred_to_delete(
        FakeCl(q=0.5, tav=1.0), FakeCl(q=0.55, tav=2.0)) is True
    assert ga._is_preferred_to_delete(
        FakeCl(q=0.5, tav=2.0), FakeCl(q=0.55, tav=1.0)) is False


def test_is_preferred_to_delete_marked_over_unmarked():
    cl_del = FakeCl(q=0.5, tav=5.0, marked=False)
    cl = FakeCl(q=0.5, tav=0.0, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_delete_classifiers_decrements_numerosity(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.0)
    cl = FakeCl(ident=1, num=2)
    population, match_set, action_set = (FakeList([cl]), FakeList([cl]),
                                         FakeList([cl]))
    ga.delete_classifiers(population, match_set, action_set, 1, 2)
    assert cl.num == 1
    assert population == [cl]


def test_delete_classifiers_removes_single_numerosity(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.0)
    cl = FakeCl(ident=1, num=1)
    other = FakeCl(ident=2, num=1)
    population = FakeList([cl, other])
    match_set = FakeList([cl])
    action_set = FakeList([cl])
    ga.delete_classifiers(population, match_set, action_set, 2, 2)
    assert cl not in population
    assert cl not in match_set
    assert cl not in action_set
    assert other in population


def test_delete_classifiers_no_deletion_when_room(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.0)
    cl = FakeCl(ident=1, num=1)
    population, match_set, action_set = (FakeList([cl]), FakeList([cl]),
                                         FakeList([cl]))
    ga.delete_classifiers(population, match_set, action_set, 1, 10)
    assert cl.num == 1
    assert population == [cl]
