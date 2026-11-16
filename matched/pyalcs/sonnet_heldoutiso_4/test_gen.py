import random

import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga


class FakeCondition(list):
    def __init__(self, items, wildcard="#"):
        super().__init__(items)
        self.wildcard = wildcard
        self.generalized = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = self.wildcard


class FakeCfg:
    def __init__(self, length=6, wildcard="#"):
        self.classifier_length = length
        self.classifier_wildcard = wildcard


class FakeCl:
    def __init__(self, tga=0, num=1, q=0.5, tav=0.0, marked=False,
                 condition=None, cfg=None, key=None):
        self.tga = tga
        self.num = num
        self.q = q
        self.tav = tav
        self.marked = marked
        self.condition = condition
        self.cfg = cfg
        self.key = key

    def is_marked(self):
        return self.marked

    def __eq__(self, other):
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
            for i, el in enumerate(self):
                if el is cl:
                    del self[i]
                    return


# ---------------- should_apply ----------------

def test_should_apply_none_action_set():
    assert ga.should_apply(None, 100, 25) is False


def test_should_apply_empty_action_set():
    assert ga.should_apply([], 100, 25) is False


def test_should_apply_zero_numerosity():
    assert ga.should_apply([FakeCl(tga=0, num=0)], 100, 25) is False


def test_should_apply_true_when_delay_exceeds_threshold():
    action_set = [FakeCl(tga=10, num=1), FakeCl(tga=30, num=3)]
    # avg tga = (10 + 90) / 4 = 25 ; 100 - 25 = 75 > 50
    assert ga.should_apply(action_set, 100, 50) is True


def test_should_apply_false_when_delay_not_exceeding_threshold():
    action_set = [FakeCl(tga=90, num=2)]
    assert ga.should_apply(action_set, 100, 10) is False
    # exactly equal is not greater
    assert ga.should_apply(action_set, 100, 10) is False


# ---------------- set_timestamps ----------------

def test_set_timestamps_updates_all():
    action_set = [FakeCl(tga=1), FakeCl(tga=5), FakeCl(tga=7)]
    ga.set_timestamps(action_set, 42)
    assert all(cl.tga == 42 for cl in action_set)


# ---------------- selection ----------------

def test_weighted_random_choice_single_option():
    a = object()
    assert ga._weighted_random_choice({a: 3.0}) is a


def test_weighted_random_choice_ignores_zero_weights():
    a, b = object(), object()
    random.seed(1)
    for _ in range(50):
        assert ga._weighted_random_choice({a: 0, b: 2.0}) is b


def test_weighted_random_choice_all_zero_returns_none():
    assert ga._weighted_random_choice({object(): 0}) is None


def test_roulette_wheel_selection_picks_only_fit_classifier():
    good = FakeCl(q=1.0)
    bad1 = FakeCl(q=0.0)
    bad2 = FakeCl(q=0.0)
    random.seed(0)
    for _ in range(20):
        p1, p2 = ga.roulette_wheel_selection([bad1, good, bad2],
                                             lambda cl: cl.q ** 3)
        assert p1 is good
        assert p2 is good


def test_roulette_wheel_selection_returns_population_members():
    pop = [FakeCl(q=0.2), FakeCl(q=0.5), FakeCl(q=0.9)]
    random.seed(3)
    p1, p2 = ga.roulette_wheel_selection(pop, lambda cl: cl.q)
    assert any(p1 is c for c in pop)
    assert any(p2 is c for c in pop)


# ---------------- mutation ----------------

def test_generalizing_mutation_with_probability_one():
    cl = FakeCl(condition=FakeCondition(['1', '#', '0', '1']),
                cfg=FakeCfg(4))
    ga.generalizing_mutation(cl, 1.0)
    assert list(cl.condition) == ['#', '#', '#', '#']
    # the wildcard at index 1 should not have been generalized again
    assert cl.condition.generalized == [0, 2, 3]


def test_generalizing_mutation_with_probability_zero():
    cl = FakeCl(condition=FakeCondition(['1', '0', '1']), cfg=FakeCfg(3))
    ga.generalizing_mutation(cl, 0.0)
    assert list(cl.condition) == ['1', '0', '1']
    assert cl.condition.generalized == []


# ---------------- crossover ----------------

def test_two_point_crossover_swaps_contiguous_segment():
    cfg = FakeCfg(6)
    parent = FakeCl(condition=['a'] * 6, cfg=cfg)
    donor = FakeCl(condition=['b'] * 6, cfg=cfg)
    for seed in range(20):
        np.random.seed(seed)
        parent.condition = ['a'] * 6
        donor.condition = ['b'] * 6
        ga.two_point_crossover(parent, donor)

        swapped = [i for i, c in enumerate(parent.condition) if c == 'b']
        assert len(swapped) >= 1
        assert swapped == list(range(swapped[0], swapped[-1] + 1))
        # donor mirrors parent
        for i in range(6):
            if i in swapped:
                assert donor.condition[i] == 'a'
            else:
                assert parent.condition[i] == 'a'
                assert donor.condition[i] == 'b'


# ---------------- add_classifier ----------------

class _Cond:
    def __init__(self, matches):
        self.matches = matches

    def does_match(self, p):
        return self.matches


def test_add_classifier_new_classifier_added_everywhere(monkeypatch):
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *a, **k: None)
    cl = FakeCl(condition=_Cond(True))
    population, match_set, action_set = [], [], []
    ga.add_classifier(cl, object(), population, match_set, action_set,
                      False, 20)
    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_not_matching_skips_match_set(monkeypatch):
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *a, **k: None)
    cl = FakeCl(condition=_Cond(False))
    population, match_set, action_set = [], [], []
    ga.add_classifier(cl, object(), population, match_set, action_set,
                      False, 20)
    assert population == [cl]
    assert action_set == [cl]
    assert match_set == []


def test_add_classifier_none_match_set(monkeypatch):
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *a, **k: None)
    cl = FakeCl(condition=_Cond(True))
    population, action_set = [], []
    ga.add_classifier(cl, object(), population, None, action_set, False, 20)
    assert population == [cl]
    assert action_set == [cl]


def test_add_classifier_existing_increments_numerosity(monkeypatch):
    old = FakeCl(num=2, marked=False)
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *a, **k: old)
    cl = FakeCl(condition=_Cond(True))
    population, match_set, action_set = [], [], []
    ga.add_classifier(cl, object(), population, match_set, action_set,
                      True, 20)
    assert old.num == 3
    assert population == [] and action_set == [] and match_set == []


def test_add_classifier_existing_marked_not_incremented(monkeypatch):
    old = FakeCl(num=2, marked=True)
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *a, **k: old)
    cl = FakeCl(condition=_Cond(True))
    ga.add_classifier(cl, object(), [], [], [], True, 20)
    assert old.num == 2


# ---------------- delete_classifiers ----------------

def test_delete_classifiers_no_deletion_needed():
    cl = FakeCl(num=2)
    population = FakeSet([cl])
    action_set = FakeSet([cl])
    ga.delete_classifiers(population, FakeSet(), action_set, 1, 10)
    assert cl.num == 2
    assert list(population) == [cl]


def test_delete_classifiers_decrements_numerosity():
    random.seed(0)
    cl = FakeCl(num=3)
    population = FakeSet([cl])
    action_set = FakeSet([cl])
    ga.delete_classifiers(population, FakeSet(), action_set, 1, 3)
    assert cl.num == 2
    assert list(population) == [cl]


def test_delete_classifiers_removes_single_numerosity_classifier():
    random.seed(0)
    cl = FakeCl(num=1)
    other = FakeCl(num=1)
    population = FakeSet([cl])
    match_set = FakeSet([cl])
    action_set = FakeSet([cl])
    ga.delete_classifiers(population, match_set, action_set, 1, 1)
    assert cl not in [c for c in population if c is cl]
    assert len(population) == 0
    assert len(match_set) == 0
    assert len(action_set) == 0
    assert other.num == 1


# ---------------- _is_preferred_to_delete ----------------

def test_is_preferred_to_delete_lower_quality():
    cl_del = FakeCl(q=0.8)
    cl = FakeCl(q=0.3)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_higher_quality():
    cl_del = FakeCl(q=0.3)
    cl = FakeCl(q=0.8)
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_similar_quality_marked():
    cl_del = FakeCl(q=0.5, marked=False, tav=100)
    cl = FakeCl(q=0.55, marked=True, tav=1)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_similar_quality_uses_tav():
    cl_del = FakeCl(q=0.5, tav=1.0)
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.5, tav=5.0)) is True
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.5, tav=0.5)) is False


# ---------------- finding old classifiers ----------------

def test_find_similar_returns_first_equal():
    target = FakeCl(key="x")
    first = FakeCl(key="x")
    second = FakeCl(key="x")
    population = [FakeCl(key="y"), first, second]
    assert ga._find_similar(target, population) is first


def test_find_similar_none_when_missing():
    assert ga._find_similar(FakeCl(key="x"), [FakeCl(key="y")]) is None
    assert ga._find_similar(FakeCl(key="x"), []) is None


def test_find_old_classifier_without_subsumption_uses_similarity():
    similar = FakeCl(key="x")
    result = ga._find_old_classifier([FakeCl(key="z"), similar],
                                     FakeCl(key="x"), False, 20)
    assert result is similar


def test_find_old_classifier_with_subsumer(monkeypatch):
    subsumer = FakeCl(key="s")
    other = FakeCl(key="o")
    monkeypatch.setattr(ga, "find_subsumers",
                        lambda cl, pop, theta: [subsumer, other])
    result = ga._find_old_classifier([FakeCl(key="x")], FakeCl(key="x"),
                                     True, 20)
    assert result is subsumer


def test_find_old_classifier_falls_back_when_no_subsumers(monkeypatch):
    monkeypatch.setattr(ga, "find_subsumers", lambda cl, pop, theta: [])
    similar = FakeCl(key="x")
    result = ga._find_old_classifier([similar], FakeCl(key="x"), True, 20)
    assert result is similar
