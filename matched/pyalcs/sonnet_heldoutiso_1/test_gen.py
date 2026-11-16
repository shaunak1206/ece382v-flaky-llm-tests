import random

import pytest

from lcs.strategies import genetic_algorithms as ga


class FakeCfg:
    classifier_wildcard = '#'

    def __init__(self, length=6):
        self.classifier_length = length


class FakeCondition(list):
    def __init__(self, items, matches=True):
        super().__init__(items)
        self.matches = matches

    def generalize(self, idx):
        self[idx] = '#'

    def does_match(self, p):
        return self.matches


class FakeCl:
    def __init__(self, cond="", num=1, tga=0, q=0.5, tav=0.0, marked=False,
                 key=None, matches=True, length=None):
        self.cfg = FakeCfg(length if length is not None else len(cond))
        self.condition = FakeCondition(list(cond), matches)
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self.marked = marked
        self.key = key

    def is_marked(self):
        return self.marked

    def __eq__(self, other):
        if self.key is None:
            return self is other
        return isinstance(other, FakeCl) and self.key == other.key

    def __hash__(self):
        return id(self)


class FakeSet(list):
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


def test_should_apply_threshold_is_strict():
    action_set = [FakeCl(num=1, tga=0)]
    assert ga.should_apply(action_set, 25, 25) is False
    assert ga.should_apply(action_set, 26, 25) is True


def test_should_apply_uses_numerosity_weighted_average():
    # weighted average tga = (0*1 + 90*9) / 10 = 81
    action_set = [FakeCl(num=1, tga=0), FakeCl(num=9, tga=90)]
    assert ga.should_apply(action_set, 100, 20) is False
    assert ga.should_apply(action_set, 100, 18) is True


# ---------------- set_timestamps ----------------

def test_set_timestamps_updates_all():
    action_set = [FakeCl(tga=1), FakeCl(tga=5), FakeCl(tga=9)]
    ga.set_timestamps(action_set, 42)
    assert all(cl.tga == 42 for cl in action_set)


# ---------------- selection ----------------

def test_weighted_random_choice_respects_pick(monkeypatch):
    choices = {'a': 1, 'b': 2}
    monkeypatch.setattr(random, 'uniform', lambda lo, hi: 0.5)
    assert ga._weighted_random_choice(choices) == 'a'
    monkeypatch.setattr(random, 'uniform', lambda lo, hi: 1.5)
    assert ga._weighted_random_choice(choices) == 'b'


def test_weighted_random_choice_upper_bound_is_sum(monkeypatch):
    seen = {}

    def fake_uniform(lo, hi):
        seen['range'] = (lo, hi)
        return 0.0

    monkeypatch.setattr(random, 'uniform', fake_uniform)
    ga._weighted_random_choice({'a': 1, 'b': 2})
    assert seen['range'] == (0, 3)


def test_roulette_wheel_selection_picks_only_weighted_classifier():
    cl1, cl2, cl3 = FakeCl(q=0.0), FakeCl(q=1.0), FakeCl(q=0.0)
    parents = ga.roulette_wheel_selection([cl1, cl2, cl3], lambda c: c.q ** 3)
    assert len(parents) == 2
    assert parents[0] is cl2
    assert parents[1] is cl2


def test_roulette_wheel_selection_returns_population_members():
    pop = [FakeCl(q=0.3), FakeCl(q=0.6), FakeCl(q=0.9)]
    random.seed(1)
    for _ in range(20):
        p1, p2 = ga.roulette_wheel_selection(pop, lambda c: c.q)
        assert p1 in pop
        assert p2 in pop


# ---------------- mutation ----------------

def test_generalizing_mutation_always():
    cl = FakeCl("0101")
    ga.generalizing_mutation(cl, 1.0)
    assert list(cl.condition) == ['#'] * 4


def test_generalizing_mutation_never():
    cl = FakeCl("0101")
    ga.generalizing_mutation(cl, 0.0)
    assert list(cl.condition) == list("0101")


def test_generalizing_mutation_keeps_wildcards():
    cl = FakeCl("0#1#")
    ga.generalizing_mutation(cl, 1.0)
    assert list(cl.condition) == ['#'] * 4


# ---------------- crossover ----------------

def test_two_point_crossover_swaps_contiguous_segment():
    import numpy as np
    for seed in range(20):
        np.random.seed(seed)
        parent = FakeCl("aaaaaa")
        donor = FakeCl("bbbbbb")
        ga.two_point_crossover(parent, donor)

        p, d = list(parent.condition), list(donor.condition)
        # genes are only exchanged, never lost
        for x, y in zip(p, d):
            assert {x, y} == {'a', 'b'}
        swapped = [i for i, x in enumerate(p) if x == 'b']
        assert len(swapped) >= 1
        assert swapped == list(range(swapped[0], swapped[-1] + 1))
        assert [i for i, x in enumerate(d) if x == 'a'] == swapped


# ---------------- add_classifier ----------------

def test_add_classifier_new_classifier_is_added():
    cl = FakeCl("01", matches=True)
    population, match_set, action_set = FakeSet(), FakeSet(), FakeSet()
    ga.add_classifier(cl, object(), population, match_set, action_set,
                      False, 20)
    assert list(population) == [cl]
    assert list(action_set) == [cl]
    assert list(match_set) == [cl]


def test_add_classifier_not_matching_not_added_to_match_set():
    cl = FakeCl("01", matches=False)
    population, match_set, action_set = FakeSet(), FakeSet(), FakeSet()
    ga.add_classifier(cl, object(), population, match_set, action_set,
                      False, 20)
    assert list(population) == [cl]
    assert list(action_set) == [cl]
    assert list(match_set) == []


def test_add_classifier_without_match_set():
    cl = FakeCl("01")
    population, action_set = FakeSet(), FakeSet()
    ga.add_classifier(cl, object(), population, None, action_set, False, 20)
    assert list(population) == [cl]
    assert list(action_set) == [cl]


def test_add_classifier_similar_increases_numerosity():
    existing = FakeCl("01", num=2, key="x")
    new = FakeCl("01", num=1, key="x")
    population, match_set, action_set = (FakeSet([existing]), FakeSet(),
                                         FakeSet([existing]))
    ga.add_classifier(new, object(), population, match_set, action_set,
                      False, 20)
    assert existing.num == 3
    assert list(population) == [existing]
    assert list(action_set) == [existing]


def test_add_classifier_similar_marked_not_incremented():
    existing = FakeCl("01", num=2, key="x", marked=True)
    new = FakeCl("01", key="x")
    action_set = FakeSet([existing])
    population = FakeSet([existing])
    ga.add_classifier(new, object(), population, FakeSet(), action_set,
                      False, 20)
    assert existing.num == 2
    assert len(population) == 1


def test_add_classifier_uses_subsumer(monkeypatch):
    subsumer = FakeCl("0#", num=4, key="sub")
    new = FakeCl("01", key="new")
    monkeypatch.setattr(ga, "find_subsumers",
                        lambda cl, pop, theta: [subsumer])
    population = FakeSet([subsumer])
    action_set = FakeSet([subsumer])
    ga.add_classifier(new, object(), population, FakeSet(), action_set,
                      True, 20)
    assert subsumer.num == 5
    assert new not in population


# ---------------- _find_similar ----------------

def test_find_similar_returns_first_match_or_none():
    a = FakeCl(key="k")
    b = FakeCl(key="k")
    other = FakeCl(key="z")
    assert ga._find_similar(FakeCl(key="k"), [other, a, b]) is a
    assert ga._find_similar(FakeCl(key="nope"), [a, b]) is None


# ---------------- _is_preferred_to_delete ----------------

def test_is_preferred_to_delete_much_lower_quality():
    cl_del = FakeCl(q=0.9)
    cl = FakeCl(q=0.5)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_much_higher_quality():
    cl_del = FakeCl(q=0.5, tav=1)
    cl = FakeCl(q=0.9, tav=100)
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_similar_quality_uses_tav():
    cl_del = FakeCl(q=0.5, tav=10)
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.55, tav=20)) is True
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.55, tav=5)) is False


def test_is_preferred_to_delete_marked_is_preferred():
    cl_del = FakeCl(q=0.5, tav=100, marked=False)
    cl = FakeCl(q=0.5, tav=0, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


# ---------------- delete_classifiers ----------------

def test_delete_classifiers_no_deletion_when_room():
    cl = FakeCl(num=2)
    population, match_set, action_set = (FakeSet([cl]), FakeSet([cl]),
                                         FakeSet([cl]))
    ga.delete_classifiers(population, match_set, action_set, 1, 10)
    assert cl.num == 2
    assert list(population) == [cl]


def test_delete_classifiers_decrements_numerosity():
    random.seed(0)
    cl = FakeCl(num=3)
    population, match_set, action_set = (FakeSet([cl]), FakeSet([cl]),
                                         FakeSet([cl]))
    ga.delete_classifiers(population, match_set, action_set, 1, 3)
    assert cl.num == 2
    assert list(population) == [cl]


def test_delete_classifiers_removes_single_numerosity_classifier():
    random.seed(0)
    cl = FakeCl(num=1)
    population, match_set, action_set = (FakeSet([cl]), FakeSet([cl]),
                                         FakeSet([cl]))
    ga.delete_classifiers(population, match_set, action_set, 1, 1)
    assert len(population) == 0
    assert len(match_set) == 0
    assert len(action_set) == 0
