import random

import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga


class FakeCl:
    def __init__(self, tga=0, num=1, q=0.5, tav=0.0, marked=False, key=None):
        self.tga = tga
        self.num = num
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


class FakeList(list):
    def expand(self):
        return list(self)

    def safe_remove(self, item):
        for i, el in enumerate(self):
            if el is item:
                del self[i]
                return


class FakeCondition(list):
    def __init__(self, items, wildcard="#", match=True):
        super().__init__(items)
        self.wildcard = wildcard
        self.match = match
        self.generalized = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = self.wildcard

    def does_match(self, p):
        return self.match


class FakeCfg:
    def __init__(self, length=5, wildcard="#"):
        self.classifier_length = length
        self.classifier_wildcard = wildcard


class FakeCondCl(FakeCl):
    def __init__(self, cond, length=None, **kw):
        super().__init__(**kw)
        self.condition = cond
        self.cfg = FakeCfg(length if length is not None else len(cond))


# ---------------- should_apply ----------------

def test_should_apply_none_action_set():
    assert ga.should_apply(None, 100, 10) is False


def test_should_apply_empty_action_set():
    assert ga.should_apply([], 100, 10) is False


def test_should_apply_zero_numerosity():
    assert ga.should_apply([FakeCl(tga=0, num=0)], 100, 10) is False


def test_should_apply_true_when_delay_exceeds_threshold():
    action_set = [FakeCl(tga=0, num=1), FakeCl(tga=10, num=1)]
    # average tga = 5, delay = 95
    assert ga.should_apply(action_set, 100, 50) is True


def test_should_apply_false_when_delay_equals_threshold():
    action_set = [FakeCl(tga=90, num=1)]
    assert ga.should_apply(action_set, 100, 10) is False


def test_should_apply_uses_numerosity_weights():
    # weighted avg tga = (0*1 + 100*9) / 10 = 90 -> delay 10
    action_set = [FakeCl(tga=0, num=1), FakeCl(tga=100, num=9)]
    assert ga.should_apply(action_set, 100, 20) is False
    assert ga.should_apply(action_set, 100, 5) is True


# ---------------- set_timestamps ----------------

def test_set_timestamps_updates_all():
    action_set = [FakeCl(tga=1), FakeCl(tga=2), FakeCl(tga=3)]
    ga.set_timestamps(action_set, 42)
    assert all(cl.tga == 42 for cl in action_set)


def test_set_timestamps_empty_does_nothing():
    ga.set_timestamps([], 5)


# ---------------- selection ----------------

def test_roulette_wheel_selects_only_nonzero_fitness():
    a, b, c = FakeCl(q=0.0), FakeCl(q=1.0), FakeCl(q=0.0)
    for _ in range(20):
        p1, p2 = ga.roulette_wheel_selection([a, b, c], lambda cl: cl.q)
        assert p1 is b
        assert p2 is b


def test_weighted_random_choice_deterministic(monkeypatch):
    a, b = object(), object()
    choices = {a: 1, b: 2}
    monkeypatch.setattr(random, "uniform", lambda lo, hi: 0.5)
    assert ga._weighted_random_choice(choices) is a
    monkeypatch.setattr(random, "uniform", lambda lo, hi: 1.5)
    assert ga._weighted_random_choice(choices) is b


# ---------------- mutation ----------------

def test_generalizing_mutation_mu_one_generalizes_all_specific():
    cond = FakeCondition(["0", "#", "1", "0"])
    cl = FakeCondCl(cond)
    ga.generalizing_mutation(cl, 1.0)
    assert list(cond) == ["#", "#", "#", "#"]
    assert cond.generalized == [0, 2, 3]


def test_generalizing_mutation_mu_zero_changes_nothing():
    cond = FakeCondition(["0", "1", "#"])
    cl = FakeCondCl(cond)
    ga.generalizing_mutation(cl, 0.0)
    assert list(cond) == ["0", "1", "#"]
    assert cond.generalized == []


# ---------------- crossover ----------------

def test_two_point_crossover_swaps_segment(monkeypatch):
    parent = FakeCondCl(["a", "a", "a", "a", "a"])
    donor = FakeCondCl(["b", "b", "b", "b", "b"])
    monkeypatch.setattr(np.random, "choice",
                        lambda *args, **kwargs: np.array([3, 1]))
    ga.two_point_crossover(parent, donor)
    assert parent.condition == ["a", "b", "b", "a", "a"]
    assert donor.condition == ["b", "a", "a", "b", "b"]


def test_two_point_crossover_preserves_genes():
    for _ in range(30):
        parent = FakeCondCl(list("01#10"))
        donor = FakeCondCl(list("1#0#1"))
        before = sorted(parent.condition + donor.condition)
        ga.two_point_crossover(parent, donor)
        after = sorted(parent.condition + donor.condition)
        assert before == after
        assert len(parent.condition) == 5
        assert len(donor.condition) == 5


# ---------------- add_classifier ----------------

def test_add_classifier_new_is_appended_everywhere():
    cl = FakeCondCl(FakeCondition(["0"], match=True), key="new")
    population, match_set, action_set = [], [], []
    ga.add_classifier(cl, None, population, match_set, action_set, False, 20)
    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_not_matching_not_added_to_match_set():
    cl = FakeCondCl(FakeCondition(["0"], match=False), key="new")
    population, match_set, action_set = [], [], []
    ga.add_classifier(cl, None, population, match_set, action_set, False, 20)
    assert population == [cl]
    assert action_set == [cl]
    assert match_set == []


def test_add_classifier_similar_increments_numerosity():
    existing = FakeCondCl(FakeCondition(["0"]), key="same", num=2)
    new = FakeCondCl(FakeCondition(["0"]), key="same", num=1)
    population, match_set, action_set = [existing], [existing], [existing]
    ga.add_classifier(new, None, population, match_set, action_set, False, 20)
    assert existing.num == 3
    assert population == [existing]
    assert action_set == [existing]


def test_add_classifier_similar_marked_not_incremented():
    existing = FakeCondCl(FakeCondition(["0"]), key="same", num=2, marked=True)
    new = FakeCondCl(FakeCondition(["0"]), key="same")
    ga.add_classifier(new, None, [existing], [existing], [existing], False, 20)
    assert existing.num == 2


# ---------------- find similar ----------------

def test_find_similar_returns_first_equal_or_none():
    c1 = FakeCl(key="x")
    c2 = FakeCl(key="y")
    c3 = FakeCl(key="y")
    assert ga._find_similar(FakeCl(key="y"), [c1, c2, c3]) is c2
    assert ga._find_similar(FakeCl(key="z"), [c1, c2, c3]) is None


# ---------------- deletion ----------------

def test_delete_classifiers_noop_when_room_available():
    cl = FakeCl(num=2)
    action_set = FakeList([cl])
    population = FakeList([cl])
    ga.delete_classifiers(population, None, action_set, 1, 10)
    assert cl.num == 2
    assert list(population) == [cl]


def test_delete_classifiers_decrements_numerosity(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    cl = FakeCl(num=3)
    action_set = FakeList([cl])
    population = FakeList([cl])
    ga.delete_classifiers(population, None, action_set, 1, 3)
    assert cl.num == 2
    assert list(population) == [cl]


def test_delete_classifiers_removes_single_numerosity(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    cl = FakeCl(num=1)
    action_set = FakeList([cl])
    population = FakeList([cl])
    match_set = FakeList([cl])
    ga.delete_classifiers(population, match_set, action_set, 1, 1)
    assert len(population) == 0
    assert len(match_set) == 0
    assert len(action_set) == 0


def test_delete_classifiers_prefers_low_quality(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    good = FakeCl(num=1, q=0.9)
    bad = FakeCl(num=1, q=0.1)
    action_set = FakeList([good, bad])
    population = FakeList([good, bad])
    ga.delete_classifiers(population, None, action_set, 1, 2)
    assert list(population) == [good]
    assert list(action_set) == [good]


# ---------------- preference ----------------

def test_is_preferred_to_delete_lower_quality():
    assert ga._is_preferred_to_delete(FakeCl(q=0.9), FakeCl(q=0.5)) is True


def test_is_preferred_to_delete_higher_quality():
    assert ga._is_preferred_to_delete(FakeCl(q=0.2), FakeCl(q=0.8)) is False


def test_is_preferred_to_delete_similar_quality_uses_tav():
    cl_del = FakeCl(q=0.5, tav=1.0)
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.5, tav=2.0)) is True
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.5, tav=0.5)) is False


def test_is_preferred_to_delete_marked_candidate():
    cl_del = FakeCl(q=0.5, marked=False, tav=10.0)
    cl = FakeCl(q=0.55, marked=True, tav=0.0)
    assert ga._is_preferred_to_delete(cl_del, cl) is True
