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
    def __init__(self, length=5, wildcard='#'):
        self.classifier_length = length
        self.classifier_wildcard = wildcard


class FakeCl:
    def __init__(self, key=None, num=1, tga=0, q=0.5, tav=0.0,
                 marked=False, condition=None, cfg=None):
        self.key = key
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self.marked = marked
        self.condition = condition
        self.cfg = cfg

    def is_marked(self):
        return self.marked

    def __eq__(self, other):
        return isinstance(other, FakeCl) and self.key == other.key

    def __hash__(self):
        return id(self)


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
    assert ga.should_apply(None, 100, 25) is False


def test_should_apply_empty_action_set():
    assert ga.should_apply([], 100, 25) is False


def test_should_apply_zero_numerosity():
    assert ga.should_apply([FakeCl(num=0, tga=0)], 100, 25) is False


def test_should_apply_true_when_delay_exceeds_threshold():
    action_set = [FakeCl(num=1, tga=10), FakeCl(num=3, tga=10)]
    assert ga.should_apply(action_set, 100, 25) is True


def test_should_apply_false_when_delay_not_exceeding_threshold():
    action_set = [FakeCl(num=2, tga=90)]
    assert ga.should_apply(action_set, 100, 25) is False
    # exactly equal to threshold -> not strictly greater
    assert ga.should_apply([FakeCl(num=1, tga=75)], 100, 25) is False


def test_should_apply_uses_numerosity_weighted_average():
    # weighted average tga = (0*1 + 100*9) / 10 = 90
    action_set = [FakeCl(num=1, tga=0), FakeCl(num=9, tga=100)]
    assert ga.should_apply(action_set, 100, 5) is True
    assert ga.should_apply(action_set, 100, 15) is False


# ---------------- set_timestamps ----------------

def test_set_timestamps_updates_all():
    action_set = [FakeCl(tga=1), FakeCl(tga=2), FakeCl(tga=3)]
    ga.set_timestamps(action_set, 42)
    assert [cl.tga for cl in action_set] == [42, 42, 42]


def test_set_timestamps_empty_is_noop():
    ga.set_timestamps([], 10)


# ---------------- selection ----------------

def test_weighted_random_choice_picks_by_pick_value(monkeypatch):
    choices = {'a': 1, 'b': 2, 'c': 3}
    monkeypatch.setattr(ga.random, 'uniform', lambda lo, hi: 0.5)
    assert ga._weighted_random_choice(choices) == 'a'
    monkeypatch.setattr(ga.random, 'uniform', lambda lo, hi: 1.5)
    assert ga._weighted_random_choice(choices) == 'b'
    monkeypatch.setattr(ga.random, 'uniform', lambda lo, hi: 5.0)
    assert ga._weighted_random_choice(choices) == 'c'


def test_weighted_random_choice_never_picks_zero_weight():
    random.seed(1)
    choices = {'a': 0, 'b': 1}
    for _ in range(100):
        assert ga._weighted_random_choice(choices) == 'b'


def test_roulette_wheel_selection_returns_two_from_population():
    random.seed(3)
    population = [FakeCl(key=i, q=i + 1) for i in range(4)]
    p1, p2 = ga.roulette_wheel_selection(population, lambda cl: cl.q ** 3)
    assert p1 in population
    assert p2 in population


def test_roulette_wheel_selection_single_fit_classifier():
    good = FakeCl(key='good', q=1.0)
    bad = FakeCl(key='bad', q=0.0)
    p1, p2 = ga.roulette_wheel_selection([good, bad], lambda cl: cl.q)
    assert p1 is good
    assert p2 is good


# ---------------- mutation ----------------

def test_generalizing_mutation_always_with_mu_one():
    cl = FakeCl(condition=FakeCondition(['0', '1', '0']), cfg=FakeCfg(3))
    ga.generalizing_mutation(cl, 1.0)
    assert list(cl.condition) == ['#', '#', '#']
    assert cl.condition.generalized == [0, 1, 2]


def test_generalizing_mutation_never_with_mu_zero():
    cl = FakeCl(condition=FakeCondition(['0', '1', '0']), cfg=FakeCfg(3))
    ga.generalizing_mutation(cl, 0.0)
    assert list(cl.condition) == ['0', '1', '0']
    assert cl.condition.generalized == []


def test_generalizing_mutation_skips_wildcards():
    cl = FakeCl(condition=FakeCondition(['#', '1', '#']), cfg=FakeCfg(3))
    ga.generalizing_mutation(cl, 1.0)
    assert cl.condition.generalized == [1]


# ---------------- crossover ----------------

def test_two_point_crossover_swaps_segment(monkeypatch):
    monkeypatch.setattr(ga.np.random, 'choice',
                        lambda *a, **k: np.array([3, 1]))
    cfg = FakeCfg(5)
    parent = FakeCl(condition=['a', 'a', 'a', 'a', 'a'], cfg=cfg)
    donor = FakeCl(condition=['b', 'b', 'b', 'b', 'b'], cfg=cfg)
    ga.two_point_crossover(parent, donor)
    assert parent.condition == ['a', 'b', 'b', 'a', 'a']
    assert donor.condition == ['b', 'a', 'a', 'b', 'b']


def test_two_point_crossover_preserves_gene_pairs():
    np.random.seed(7)
    cfg = FakeCfg(6)
    for _ in range(20):
        parent = FakeCl(condition=list('abcdef'), cfg=cfg)
        donor = FakeCl(condition=list('ABCDEF'), cfg=cfg)
        ga.two_point_crossover(parent, donor)
        for i in range(6):
            assert {parent.condition[i], donor.condition[i]} == \
                {'abcdef'[i], 'ABCDEF'[i]}
        assert parent.condition != list('abcdef')


# ---------------- similar / old classifier ----------------

def test_find_similar_returns_first_equal():
    a1 = FakeCl(key='x')
    a2 = FakeCl(key='x')
    b = FakeCl(key='y')
    assert ga._find_similar(FakeCl(key='x'), [b, a1, a2]) is a1


def test_find_similar_none_when_absent():
    assert ga._find_similar(FakeCl(key='z'), [FakeCl(key='x')]) is None


def test_find_old_classifier_prefers_subsumer(monkeypatch):
    subsumer = FakeCl(key='sub')
    similar = FakeCl(key='new')
    monkeypatch.setattr(ga, 'find_subsumers',
                        lambda cl, pop, theta: [subsumer])
    result = ga._find_old_classifier([similar], FakeCl(key='new'), True, 20)
    assert result is subsumer


def test_find_old_classifier_falls_back_to_similar(monkeypatch):
    similar = FakeCl(key='new')
    monkeypatch.setattr(ga, 'find_subsumers', lambda cl, pop, theta: [])
    result = ga._find_old_classifier([similar], FakeCl(key='new'), True, 20)
    assert result is similar


def test_find_old_classifier_without_subsumption_skips_subsumers(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not be called")

    monkeypatch.setattr(ga, 'find_subsumers', boom)
    assert ga._find_old_classifier([FakeCl(key='a')], FakeCl(key='b'),
                                   False, 20) is None


# ---------------- add_classifier ----------------

def test_add_classifier_adds_new_when_no_old(monkeypatch):
    monkeypatch.setattr(ga, '_find_old_classifier', lambda *a: None)
    cl = FakeCl(key='n', condition=FakeCondition(['1'], matches=True))
    population, match_set, action_set = [], [], []
    ga.add_classifier(cl, object(), population, match_set, action_set,
                      False, 20)
    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_not_in_match_set_if_not_matching(monkeypatch):
    monkeypatch.setattr(ga, '_find_old_classifier', lambda *a: None)
    cl = FakeCl(key='n', condition=FakeCondition(['1'], matches=False))
    population, match_set, action_set = [], [], []
    ga.add_classifier(cl, object(), population, match_set, action_set,
                      False, 20)
    assert population == [cl]
    assert match_set == []


def test_add_classifier_increments_numerosity_of_old(monkeypatch):
    old = FakeCl(key='o', num=2)
    monkeypatch.setattr(ga, '_find_old_classifier', lambda *a: old)
    cl = FakeCl(key='o', condition=FakeCondition(['1']))
    population, match_set, action_set = [old], [old], [old]
    ga.add_classifier(cl, object(), population, match_set, action_set,
                      False, 20)
    assert old.num == 3
    assert population == [old]


def test_add_classifier_marked_old_not_incremented(monkeypatch):
    old = FakeCl(key='o', num=2, marked=True)
    monkeypatch.setattr(ga, '_find_old_classifier', lambda *a: old)
    cl = FakeCl(key='o', condition=FakeCondition(['1']))
    ga.add_classifier(cl, object(), [old], [old], [old], False, 20)
    assert old.num == 2


# ---------------- delete_classifiers ----------------

def test_delete_classifiers_noop_when_room_available():
    cl = FakeCl(key='a', num=2)
    population = FakeList([cl])
    action_set = FakeList([cl])
    ga.delete_classifiers(population, FakeList(), action_set, 1, 10)
    assert cl.num == 2
    assert list(population) == [cl]


def test_delete_classifiers_decrements_numerosity(monkeypatch):
    monkeypatch.setattr(ga.random, 'random', lambda: 0.0)
    cl = FakeCl(key='a', num=3)
    population = FakeList([cl])
    action_set = FakeList([cl])
    ga.delete_classifiers(population, FakeList(), action_set, 1, 3)
    assert cl.num == 2
    assert list(population) == [cl]


def test_delete_classifiers_removes_single_numerosity(monkeypatch):
    monkeypatch.setattr(ga.random, 'random', lambda: 0.0)
    cl = FakeCl(key='a', num=1)
    other = FakeCl(key='b', num=1, q=0.9)
    population = FakeList([cl])
    match_set = FakeList([cl])
    action_set = FakeList([cl])
    ga.delete_classifiers(population, match_set, action_set, 1, 1)
    assert cl not in population
    assert cl not in match_set
    assert cl not in action_set


# ---------------- _is_preferred_to_delete ----------------

def test_is_preferred_to_delete_much_lower_quality():
    cl_del = FakeCl(q=0.9)
    cl = FakeCl(q=0.5)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_much_higher_quality():
    cl_del = FakeCl(q=0.5)
    cl = FakeCl(q=0.9)
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_similar_quality_marked():
    cl_del = FakeCl(q=0.5, marked=False, tav=10)
    cl = FakeCl(q=0.55, marked=True, tav=1)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_similar_quality_by_tav():
    cl_del = FakeCl(q=0.5, tav=1)
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.5, tav=5)) is True
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.5, tav=0.5)) is False
