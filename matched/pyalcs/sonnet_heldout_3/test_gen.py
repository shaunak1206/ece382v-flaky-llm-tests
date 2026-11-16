import random
from types import SimpleNamespace

import numpy as np
import pytest

import lcs.strategies.genetic_algorithms as ga


class FakeCl:
    def __init__(self, key=0, num=1, tga=0, q=0.5, tav=0.0, marked=False,
                 condition=None, cfg=None):
        self.key = key
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self._marked = marked
        self.condition = condition
        self.cfg = cfg

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return isinstance(other, FakeCl) and self.key == other.key

    def __hash__(self):
        return hash(self.key)


class FakeCondition(list):
    def __init__(self, items, match=True):
        super().__init__(items)
        self.match = match

    def generalize(self, idx):
        self[idx] = '#'

    def does_match(self, p):
        return self.match


class FakeSet(list):
    def expand(self):
        result = []
        for cl in self:
            result.extend([cl] * cl.num)
        return result

    def safe_remove(self, cl):
        self.remove(cl)


# ---------------- should_apply ----------------

def test_should_apply_none_action_set():
    assert ga.should_apply(None, 100, 10) is False


def test_should_apply_empty_action_set():
    assert ga.should_apply([], 100, 10) is False


def test_should_apply_zero_numerosity():
    action_set = [FakeCl(num=0, tga=0)]
    assert ga.should_apply(action_set, 100, 10) is False


def test_should_apply_weighted_average_threshold():
    action_set = [FakeCl(key=1, num=1, tga=0), FakeCl(key=2, num=3, tga=10)]
    # weighted average tga = 7.5; time - avg = 12.5
    assert ga.should_apply(action_set, 20, 12) is True
    assert ga.should_apply(action_set, 20, 13) is False


def test_should_apply_equal_to_threshold_is_false():
    action_set = [FakeCl(num=1, tga=10)]
    assert ga.should_apply(action_set, 20, 10) is False


# ---------------- set_timestamps ----------------

def test_set_timestamps_updates_all():
    action_set = [FakeCl(key=i, tga=0) for i in range(4)]
    ga.set_timestamps(action_set, 42)
    assert all(cl.tga == 42 for cl in action_set)


# ---------------- selection ----------------

def test_weighted_random_choice_picks_by_pick_value(monkeypatch):
    a, b = FakeCl(key='a'), FakeCl(key='b')
    choices = {a: 1, b: 1}

    monkeypatch.setattr(random, 'uniform', lambda lo, hi: 0.5)
    assert ga._weighted_random_choice(choices) is a

    monkeypatch.setattr(random, 'uniform', lambda lo, hi: 1.5)
    assert ga._weighted_random_choice(choices) is b


def test_roulette_wheel_selection_returns_two_from_population():
    pop = [FakeCl(key=i) for i in range(5)]
    p1, p2 = ga.roulette_wheel_selection(pop, lambda cl: 1.0)
    assert p1 in pop
    assert p2 in pop


def test_roulette_wheel_selection_ignores_zero_fitness():
    good, bad = FakeCl(key='good', q=1.0), FakeCl(key='bad', q=0.0)
    for _ in range(20):
        p1, p2 = ga.roulette_wheel_selection([bad, good], lambda cl: cl.q)
        assert p1 is good
        assert p2 is good


# ---------------- mutation ----------------

def test_generalizing_mutation_mu_one_generalizes_all():
    cfg = SimpleNamespace(classifier_wildcard='#')
    cl = FakeCl(condition=FakeCondition(['1', '0', '#', '1']), cfg=cfg)
    ga.generalizing_mutation(cl, 1.0)
    assert list(cl.condition) == ['#', '#', '#', '#']


def test_generalizing_mutation_mu_zero_changes_nothing():
    cfg = SimpleNamespace(classifier_wildcard='#')
    cl = FakeCl(condition=FakeCondition(['1', '0', '#', '1']), cfg=cfg)
    ga.generalizing_mutation(cl, 0.0)
    assert list(cl.condition) == ['1', '0', '#', '1']


# ---------------- crossover ----------------

def test_two_point_crossover_swaps_segment(monkeypatch):
    cfg = SimpleNamespace(classifier_length=5)
    parent = FakeCl(condition=['a'] * 5, cfg=cfg)
    donor = FakeCl(condition=['b'] * 5, cfg=cfg)
    monkeypatch.setattr(np.random, 'choice', lambda *a, **k: [3, 1])

    ga.two_point_crossover(parent, donor)

    assert parent.condition == ['a', 'b', 'b', 'a', 'a']
    assert donor.condition == ['b', 'a', 'a', 'b', 'b']


def test_two_point_crossover_preserves_alleles_random():
    cfg = SimpleNamespace(classifier_length=6)
    for _ in range(20):
        parent = FakeCl(condition=list('aaaaaa'), cfg=cfg)
        donor = FakeCl(condition=list('bbbbbb'), cfg=cfg)
        ga.two_point_crossover(parent, donor)
        combined = sorted(parent.condition + donor.condition)
        assert combined == sorted(list('aaaaaa') + list('bbbbbb'))
        # positions are complementary
        for x, y in zip(parent.condition, donor.condition):
            assert {x, y} == {'a', 'b'}


# ---------------- add_classifier ----------------

def test_add_classifier_new_is_added_everywhere():
    population, match_set, action_set = [], [], []
    cl = FakeCl(key=1, condition=FakeCondition([], match=True))
    ga.add_classifier(cl, None, population, match_set, action_set, False, 20)
    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_not_matching_not_in_match_set():
    population, match_set, action_set = [], [], []
    cl = FakeCl(key=1, condition=FakeCondition([], match=False))
    ga.add_classifier(cl, None, population, match_set, action_set, False, 20)
    assert population == [cl]
    assert match_set == []


def test_add_classifier_similar_increments_numerosity():
    old = FakeCl(key=1, num=1)
    new = FakeCl(key=1, num=1, condition=FakeCondition([]))
    population, match_set, action_set = [old], [old], [old]
    ga.add_classifier(new, None, population, match_set, action_set, False, 20)
    assert old.num == 2
    assert population == [old]
    assert action_set == [old]


def test_add_classifier_similar_marked_not_incremented():
    old = FakeCl(key=1, num=1, marked=True)
    new = FakeCl(key=1, condition=FakeCondition([]))
    action_set = [old]
    ga.add_classifier(new, None, [old], None, action_set, False, 20)
    assert old.num == 1


# ---------------- finding old classifiers ----------------

def test_find_similar():
    assert ga._find_similar(2, [1, 2, 3]) == 2
    assert ga._find_similar(5, [1, 2, 3]) is None


def test_find_old_classifier_uses_subsumer(monkeypatch):
    subsumer = FakeCl(key='s')
    monkeypatch.setattr(ga, 'find_subsumers',
                        lambda cl, pop, theta: [subsumer])
    new = FakeCl(key='n')
    assert ga._find_old_classifier([FakeCl(key='n')], new, True, 20) is subsumer


def test_find_old_classifier_falls_back_to_similar(monkeypatch):
    monkeypatch.setattr(ga, 'find_subsumers', lambda cl, pop, theta: [])
    similar = FakeCl(key='n')
    new = FakeCl(key='n')
    assert ga._find_old_classifier([similar], new, True, 20) is similar


# ---------------- deletion ----------------

def test_delete_classifiers_decrements_numerosity(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.1)
    cl = FakeCl(key=1, num=2)
    population, action_set = FakeSet([cl]), FakeSet([cl])
    ga.delete_classifiers(population, FakeSet(), action_set, 1, 2)
    assert cl.num == 1
    assert len(population) == 1


def test_delete_classifiers_removes_single_numerosity(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.1)
    cl = FakeCl(key=1, num=1)
    population, match_set, action_set = FakeSet([cl]), FakeSet([cl]), FakeSet([cl])
    ga.delete_classifiers(population, match_set, action_set, 1, 1)
    assert list(population) == []
    assert list(match_set) == []
    assert list(action_set) == []


def test_delete_classifiers_no_deletion_when_room(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.1)
    cl = FakeCl(key=1, num=2)
    population, action_set = FakeSet([cl]), FakeSet([cl])
    ga.delete_classifiers(population, FakeSet(), action_set, 1, 10)
    assert cl.num == 2
    assert list(population) == [cl]


# ---------------- deletion preference ----------------

def test_is_preferred_to_delete_much_lower_quality():
    cl_del = FakeCl(q=0.9)
    cl = FakeCl(q=0.5)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_much_higher_quality():
    cl_del = FakeCl(q=0.5)
    cl = FakeCl(q=0.9)
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_similar_quality_uses_tav():
    cl_del = FakeCl(q=0.5, tav=1.0)
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.55, tav=2.0)) is True
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.55, tav=0.5)) is False


def test_is_preferred_to_delete_marked_vs_unmarked():
    cl_del = FakeCl(q=0.5, marked=False)
    cl = FakeCl(q=0.5, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True
