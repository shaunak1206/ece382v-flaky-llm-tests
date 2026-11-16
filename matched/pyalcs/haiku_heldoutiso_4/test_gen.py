import random

import numpy as np
import pytest

import lcs.strategies.genetic_algorithms as ga

WILDCARD = "#"


class FakeCfg:
    def __init__(self, length):
        self.classifier_wildcard = WILDCARD
        self.classifier_length = length


class FakeCondition:
    def __init__(self, values, matches=True):
        self._values = list(values)
        self._matches = matches
        self.generalized = []

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, key, value):
        self._values[key] = value

    def generalize(self, idx):
        self.generalized.append(idx)
        self._values[idx] = WILDCARD

    def does_match(self, p):
        return self._matches

    def key(self):
        return tuple(self._values)


class FakeClassifier:
    def __init__(self, values=("0",), action="A", num=1, tga=0,
                 q=0.5, tav=0.0, marked=False, matches=True):
        self.condition = FakeCondition(values, matches=matches)
        self.cfg = FakeCfg(len(values))
        self.action = action
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self.marked = marked

    def is_marked(self):
        return self.marked

    def __eq__(self, other):
        if not isinstance(other, FakeClassifier):
            return NotImplemented
        return (self.condition.key(), self.action) == (
            other.condition.key(), other.action)

    def __hash__(self):
        return hash((self.condition.key(), self.action))


class FakeList(list):
    def expand(self):
        return list(self)

    def safe_remove(self, item):
        for i, x in enumerate(self):
            if x is item:
                del self[i]
                return


def make_cl(**kwargs):
    return FakeClassifier(**kwargs)


# ---------- should_apply ----------

def test_should_apply_returns_false_for_none():
    assert ga.should_apply(None, time=100, theta_ga=5) is False


def test_should_apply_returns_false_for_empty_set():
    assert ga.should_apply(FakeList(), time=100, theta_ga=5) is False


def test_should_apply_true_when_delay_exceeds_threshold():
    action_set = FakeList([make_cl(tga=0, num=1)])
    assert ga.should_apply(action_set, time=10, theta_ga=5) is True


def test_should_apply_false_when_delay_within_threshold():
    action_set = FakeList([make_cl(tga=0, num=1)])
    assert ga.should_apply(action_set, time=3, theta_ga=5) is False


def test_should_apply_uses_numerosity_weighted_average():
    # weighted average tga = (0*3 + 10*1) / 4 = 2.5
    action_set = FakeList([make_cl(tga=0, num=3), make_cl(tga=10, num=1)])
    assert ga.should_apply(action_set, time=8, theta_ga=5) is True
    assert ga.should_apply(action_set, time=7, theta_ga=5) is False


# ---------- set_timestamps ----------

def test_set_timestamps_updates_all_classifiers():
    cls = [make_cl(values=("0",), tga=1), make_cl(values=("1",), tga=2)]
    ga.set_timestamps(FakeList(cls), epoch=42)
    assert all(cl.tga == 42 for cl in cls)


# ---------- generalizing_mutation ----------

def test_generalizing_mutation_generalizes_non_wildcard_positions(monkeypatch):
    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    cl = make_cl(values=("0", WILDCARD, "1"))
    ga.generalizing_mutation(cl, mu=1.0)
    assert cl.condition.generalized == [0, 2]


def test_generalizing_mutation_with_zero_probability_changes_nothing(monkeypatch):
    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    cl = make_cl(values=("0", "1"))
    ga.generalizing_mutation(cl, mu=0.0)
    assert cl.condition.generalized == []
    assert cl.condition.key() == ("0", "1")


# ---------- two_point_crossover ----------

def test_two_point_crossover_swaps_segment_between_cut_points(monkeypatch):
    monkeypatch.setattr(np.random, "choice", lambda pop, size, replace=False: [3, 1])
    parent = make_cl(values=("a", "b", "c", "d"))
    donor = make_cl(values=("w", "x", "y", "z"))

    ga.two_point_crossover(parent, donor)

    assert parent.condition.key() == ("a", "x", "y", "d")
    assert donor.condition.key() == ("w", "b", "c", "z")


def test_two_point_crossover_samples_from_full_range_without_replacement(monkeypatch):
    calls = []

    def fake_choice(population, size, replace=False):
        calls.append((list(population), size, replace))
        return [1, 3]

    monkeypatch.setattr(np.random, "choice", fake_choice)
    parent = make_cl(values=("a", "b", "c", "d"))
    donor = make_cl(values=("w", "x", "y", "z"))

    ga.two_point_crossover(parent, donor)

    assert calls == [(list(range(0, 5)), 2, False)]


# ---------- roulette / weighted choice ----------

def test_weighted_random_choice_picks_first_when_pick_low(monkeypatch):
    monkeypatch.setattr(ga.random, "uniform", lambda a, b: 0.5)
    first, second = "first", "second"
    choices = {first: 1.0, second: 3.0}
    assert ga._weighted_random_choice(choices) == first


def test_weighted_random_choice_picks_second_when_pick_high(monkeypatch):
    monkeypatch.setattr(ga.random, "uniform", lambda a, b: 2.0)
    first, second = "first", "second"
    choices = {first: 1.0, second: 3.0}
    assert ga._weighted_random_choice(choices) == second


def test_roulette_wheel_selection_uses_fitness_and_returns_two_parents(monkeypatch):
    a = make_cl(values=("0",), q=0.2)
    b = make_cl(values=("1",), q=0.8)
    recorded = []
    results = iter([a, b])

    def fake_choice(choices):
        recorded.append(dict(choices))
        return next(results)

    monkeypatch.setattr(ga, "_weighted_random_choice", fake_choice)

    parents = ga.roulette_wheel_selection([a, b], fitnessfunc=lambda cl: cl.q ** 3)

    assert parents == (a, b)
    assert len(recorded) == 2
    assert recorded[0][a] == pytest.approx(0.2 ** 3)
    assert recorded[0][b] == pytest.approx(0.8 ** 3)


# ---------- add_classifier ----------

def test_add_classifier_appends_new_when_no_similar_exists():
    population, match_set, action_set = FakeList(), FakeList(), FakeList()
    cl = make_cl(values=("0", "1"), matches=True)

    ga.add_classifier(cl, object(), population, match_set, action_set,
                      do_subsumption=False, theta_exp=10)

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_skips_match_set_when_not_matching():
    population, match_set, action_set = FakeList(), FakeList(), FakeList()
    cl = make_cl(values=("0", "1"), matches=False)

    ga.add_classifier(cl, object(), population, match_set, action_set,
                      do_subsumption=False, theta_exp=10)

    assert population == [cl]
    assert match_set == []


def test_add_classifier_handles_none_match_set():
    population, action_set = FakeList(), FakeList()
    cl = make_cl(values=("0", "1"))

    ga.add_classifier(cl, object(), population, None, action_set,
                      do_subsumption=False, theta_exp=10)

    assert population == [cl]
    assert action_set == [cl]


def test_add_classifier_increments_numerosity_of_similar_classifier():
    existing = make_cl(values=("0", "1"), num=2)
    population = FakeList([existing])
    action_set = FakeList([existing])
    match_set = FakeList()
    new_cl = make_cl(values=("0", "1"))

    ga.add_classifier(new_cl, object(), population, match_set, action_set,
                      do_subsumption=False, theta_exp=10)

    assert existing.num == 3
    assert len(population) == 1
    assert len(action_set) == 1


def test_add_classifier_does_not_increment_marked_similar_classifier():
    existing = make_cl(values=("0", "1"), num=2, marked=True)
    population = FakeList([existing])
    action_set = FakeList([existing])
    new_cl = make_cl(values=("0", "1"))

    ga.add_classifier(new_cl, object(), population, FakeList(), action_set,
                      do_subsumption=False, theta_exp=10)

    assert existing.num == 2


def test_add_classifier_uses_subsumer_when_subsumption_enabled(monkeypatch):
    subsumer = make_cl(values=(WILDCARD, "1"), num=1)
    population = FakeList([subsumer])
    action_set = FakeList([subsumer])
    seen = {}

    def fake_find_subsumers(cl, pop, theta_exp):
        seen["theta_exp"] = theta_exp
        return [subsumer]

    monkeypatch.setattr(ga, "find_subsumers", fake_find_subsumers)
    new_cl = make_cl(values=("0", "1"))

    ga.add_classifier(new_cl, object(), population, FakeList(), action_set,
                      do_subsumption=True, theta_exp=7)

    assert seen["theta_exp"] == 7
    assert subsumer.num == 2
    assert len(population) == 1


# ---------- delete_classifiers ----------

def test_delete_classifiers_does_nothing_within_threshold():
    cl = make_cl(values=("0",), num=1)
    population = FakeList([cl])
    action_set = FakeList([cl])

    ga.delete_classifiers(population, None, action_set, insize=2, theta_as=3)

    assert cl.num == 1
    assert len(population) == 1


def test_delete_classifiers_decrements_numerosity_above_threshold():
    cl = make_cl(values=("0",), num=5)
    population = FakeList([cl])
    action_set = FakeList([cl])

    ga.delete_classifiers(population, None, action_set, insize=0, theta_as=3)

    assert cl.num == 3
    assert len(population) == 1
    assert len(action_set) == 1


def test_delete_classifiers_removes_classifier_from_all_lists():
    a = make_cl(values=("0",), action="A", num=1)
    b = make_cl(values=("1",), action="B", num=1)
    population = FakeList([a, b])
    match_set = FakeList([a, b])
    action_set = FakeList([a, b])

    ga.delete_classifiers(population, match_set, action_set,
                          insize=0, theta_as=1)

    assert len(action_set) == 1
    survivor = action_set[0]
    assert len(population) == 1 and population[0] is survivor
    assert len(match_set) == 1 and match_set[0] is survivor


# ---------- _is_preferred_to_delete / _find_similar ----------

def test_is_preferred_to_delete_prefers_clearly_lower_quality():
    cl_del = make_cl(q=0.5)
    assert ga._is_preferred_to_delete(cl_del, make_cl(q=0.2)) is True
    assert ga._is_preferred_to_delete(cl_del, make_cl(q=0.9)) is False


def test_is_preferred_to_delete_prefers_marked_at_similar_quality():
    cl_del = make_cl(q=0.5, marked=False)
    cl = make_cl(q=0.55, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_compares_experience_at_similar_quality():
    cl_del = make_cl(q=0.5, tav=2.0)
    cl_more_experienced = make_cl(q=0.5, tav=10.0)
    cl_less_experienced = make_cl(q=0.5, tav=1.0)
    assert ga._is_preferred_to_delete(cl_del, cl_more_experienced) is True
    assert ga._is_preferred_to_delete(cl_del, cl_less_experienced) is False


def test_find_similar_returns_equal_classifier_or_none():
    target = make_cl(values=("0", "1"), action="A")
    other = make_cl(values=("1", "1"), action="A")
    population = FakeList([other, target])

    probe = make_cl(values=("0", "1"), action="A")
    assert ga._find_similar(probe, population) is target

    missing = make_cl(values=("#", "#"), action="A")
    assert ga._find_similar(missing, population) is None
