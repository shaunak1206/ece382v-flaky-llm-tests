import random
import types

import numpy as np
import pytest

import lcs.strategies.genetic_algorithms as ga

WILDCARD = "#"
CFG = types.SimpleNamespace(classifier_wildcard=WILDCARD, classifier_length=4)


class FakeCondition(list):
    """List-based condition that records generalize() calls."""

    def __init__(self, iterable=()):
        super().__init__(iterable)
        self.generalized = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = WILDCARD

    def does_match(self, p):
        return all(c == WILDCARD or c == x for c, x in zip(self, p))


class FakeClassifier:
    def __init__(self, condition=("0", "0", "0", "0"), action=0, q=0.5,
                 tav=0, num=1, tga=0, marked=False, cfg=CFG):
        self.condition = FakeCondition(condition)
        self.action = action
        self.q = q
        self.tav = tav
        self.num = num
        self.tga = tga
        self.cfg = cfg
        self._marked = marked

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return (isinstance(other, FakeClassifier)
                and list(self.condition) == list(other.condition)
                and self.action == other.action)

    def __hash__(self):
        return id(self)


class FakeSet(list):
    """List with the extra methods used by delete_classifiers."""

    def expand(self):
        return list(self)

    def safe_remove(self, item):
        for i, x in enumerate(self):
            if x is item:
                del self[i]
                return


def test_should_apply_returns_false_for_none():
    assert ga.should_apply(None, time=10, theta_ga=1) is False


def test_should_apply_returns_false_for_empty_set():
    assert ga.should_apply(FakeSet(), time=10, theta_ga=1) is False


def test_should_apply_true_when_delay_exceeds_threshold():
    action_set = FakeSet([FakeClassifier(tga=0, num=1)])
    assert ga.should_apply(action_set, time=10, theta_ga=5) is True


def test_should_apply_false_when_delay_equals_threshold():
    action_set = FakeSet([FakeClassifier(tga=5, num=1)])
    assert ga.should_apply(action_set, time=10, theta_ga=5) is False


def test_should_apply_uses_numerosity_weighted_average():
    # weighted average tga = (0*1 + 10*3) / 4 = 7.5
    action_set = FakeSet([
        FakeClassifier(tga=0, num=1),
        FakeClassifier(tga=10, num=3),
    ])
    assert ga.should_apply(action_set, time=10, theta_ga=2) is True
    assert ga.should_apply(action_set, time=10, theta_ga=3) is False


def test_set_timestamps_updates_tga_for_all_classifiers():
    classifiers = [FakeClassifier(tga=1), FakeClassifier(tga=2),
                   FakeClassifier(tga=3)]
    ga.set_timestamps(FakeSet(classifiers), epoch=42)
    assert all(cl.tga == 42 for cl in classifiers)


def test_roulette_wheel_selects_only_positive_fitness_classifier():
    a = FakeClassifier(condition=("0", "0", "0", "0"))
    b = FakeClassifier(condition=("1", "1", "1", "1"))
    fitness = {a: 0.0, b: 5.0}
    parent1, parent2 = ga.roulette_wheel_selection(
        FakeSet([a, b]), fitnessfunc=lambda cl: fitness[cl])
    assert parent1 is b
    assert parent2 is b


def test_roulette_wheel_returns_two_parents_from_picks(monkeypatch):
    a = FakeClassifier(condition=("0", "0", "0", "0"))
    b = FakeClassifier(condition=("1", "1", "1", "1"))
    picks = iter([0.5, 1.5])
    calls = []

    def fake_uniform(lo, hi):
        calls.append((lo, hi))
        return next(picks)

    monkeypatch.setattr(random, "uniform", fake_uniform)
    parent1, parent2 = ga.roulette_wheel_selection(
        FakeSet([a, b]), fitnessfunc=lambda cl: 1.0)

    assert parent1 is a
    assert parent2 is b
    assert calls == [(0, 2.0), (0, 2.0)]


def test_weighted_random_choice_picks_by_cumulative_weight(monkeypatch):
    a, b, c = (FakeClassifier(), FakeClassifier(), FakeClassifier())
    choices = {a: 1.0, b: 2.0, c: 3.0}  # total 6.0

    for pick, expected in [(0.5, a), (1.5, b), (3.5, c)]:
        monkeypatch.setattr(random, "uniform", lambda lo, hi, p=pick: p)
        assert ga._weighted_random_choice(choices) is expected


def test_generalizing_mutation_with_zero_probability_changes_nothing():
    cl = FakeClassifier(condition=("0", "1", "#", "0"))
    ga.generalizing_mutation(cl, mu=0.0)
    assert list(cl.condition) == ["0", "1", "#", "0"]
    assert cl.condition.generalized == []


def test_generalizing_mutation_with_full_probability_skips_wildcards():
    cl = FakeClassifier(condition=("0", "#", "1", "0"))
    ga.generalizing_mutation(cl, mu=1.0)
    assert cl.condition.generalized == [0, 2, 3]
    assert list(cl.condition) == [WILDCARD] * 4


def test_two_point_crossover_swaps_middle_segment(monkeypatch):
    monkeypatch.setattr(np.random, "choice",
                        lambda a, size, replace: np.array([3, 1]))
    parent = FakeClassifier(condition=("a", "b", "c", "d"))
    donor = FakeClassifier(condition=("w", "x", "y", "z"))

    ga.two_point_crossover(parent, donor)

    assert list(parent.condition) == ["a", "x", "y", "d"]
    assert list(donor.condition) == ["w", "b", "c", "z"]


def test_two_point_crossover_preserves_all_alleles():
    np.random.seed(1234)
    parent = FakeClassifier(condition=("a", "b", "c", "d"))
    donor = FakeClassifier(condition=("w", "x", "y", "z"))
    original = sorted(list(parent.condition) + list(donor.condition))

    ga.two_point_crossover(parent, donor)

    assert len(parent.condition) == 4
    assert len(donor.condition) == 4
    assert sorted(list(parent.condition) + list(donor.condition)) == original


def test_add_classifier_appends_new_classifier_without_match_set():
    population = FakeSet()
    action_set = FakeSet()
    cl = FakeClassifier(condition=("1", "1", "1", "1"))

    ga.add_classifier(cl, ("0", "0", "0", "0"), population, None,
                      action_set, do_subsumption=False, theta_exp=10)

    assert len(population) == 1 and population[0] is cl
    assert len(action_set) == 1 and action_set[0] is cl


def test_add_classifier_adds_to_match_set_only_when_matching():
    p = ("0", "0", "0", "0")
    population = FakeSet()
    action_set = FakeSet()
    match_set = FakeSet()

    matching = FakeClassifier(condition=("0", WILDCARD, "0", "0"))
    ga.add_classifier(matching, p, population, match_set, action_set,
                      do_subsumption=False, theta_exp=10)

    non_matching = FakeClassifier(condition=("1", "0", "0", "0"))
    ga.add_classifier(non_matching, p, population, match_set, action_set,
                      do_subsumption=False, theta_exp=10)

    assert any(x is matching for x in match_set)
    assert not any(x is non_matching for x in match_set)
    assert len(population) == 2
    assert len(action_set) == 2


def test_add_classifier_increments_numerosity_of_similar_classifier():
    old = FakeClassifier(condition=("0", "1", "0", "1"), action=2, num=1)
    population = FakeSet([old])
    action_set = FakeSet([old])
    new = FakeClassifier(condition=("0", "1", "0", "1"), action=2)

    ga.add_classifier(new, ("0", "1", "0", "1"), population, FakeSet(),
                      action_set, do_subsumption=False, theta_exp=10)

    assert old.num == 2
    assert len(population) == 1
    assert not any(x is new for x in population)


def test_add_classifier_does_not_increment_marked_classifier():
    old = FakeClassifier(condition=("0", "1", "0", "1"), action=2,
                         num=1, marked=True)
    population = FakeSet([old])
    action_set = FakeSet([old])
    new = FakeClassifier(condition=("0", "1", "0", "1"), action=2)

    ga.add_classifier(new, ("0", "1", "0", "1"), population, FakeSet(),
                      action_set, do_subsumption=False, theta_exp=10)

    assert old.num == 1
    assert len(population) == 1


def test_add_classifier_uses_subsumer_when_subsumption_enabled(monkeypatch):
    general = FakeClassifier(condition=(WILDCARD, WILDCARD, "0", "0"),
                             num=1)
    population = FakeSet([general])
    action_set = FakeSet([general])
    new = FakeClassifier(condition=("1", "1", "0", "0"))

    monkeypatch.setattr(ga, "find_subsumers",
                        lambda cl, pop, theta: [general])

    ga.add_classifier(new, ("1", "1", "0", "0"), population, FakeSet(),
                      action_set, do_subsumption=True, theta_exp=10)

    assert general.num == 2
    assert len(population) == 1
    assert not any(x is new for x in population)


def test_delete_classifiers_noop_when_under_threshold():
    a = FakeClassifier(num=1)
    b = FakeClassifier(num=2)
    population = FakeSet([a, b])
    match_set = FakeSet([a, b])
    action_set = FakeSet([a, b])

    ga.delete_classifiers(population, match_set, action_set,
                          insize=2, theta_as=5)

    assert a.num == 1 and b.num == 2
    assert len(population) == 2


def test_delete_classifiers_decrements_numerosity_before_removal(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    cl = FakeClassifier(num=3)
    population = FakeSet([cl])
    match_set = FakeSet([cl])
    action_set = FakeSet([cl])

    ga.delete_classifiers(population, match_set, action_set,
                          insize=0, theta_as=2)

    assert cl.num == 2
    assert len(action_set) == 1
    assert len(population) == 1


def test_delete_classifiers_removes_classifier_with_numerosity_one(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    a = FakeClassifier(condition=("0", "0", "0", "0"), q=0.5, num=1)
    b = FakeClassifier(condition=("1", "1", "1", "1"), q=0.9, num=1)
    population = FakeSet([a, b])
    match_set = FakeSet([a, b])
    action_set = FakeSet([a, b])

    ga.delete_classifiers(population, match_set, action_set,
                          insize=0, theta_as=1)

    assert not any(x is a for x in population)
    assert not any(x is a for x in match_set)
    assert not any(x is a for x in action_set)
    assert any(x is b for x in population)


def test_is_preferred_to_delete_true_for_clearly_worse_classifier():
    cl_del = FakeClassifier(q=0.8)
    cl = FakeClassifier(q=0.5)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_false_for_clearly_better_classifier():
    cl_del = FakeClassifier(q=0.5)
    cl = FakeClassifier(q=0.95)
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_tie_breaks_on_marking_and_tav():
    # Equal quality, cl marked while cl_del is not -> cl preferred for deletion
    cl_del = FakeClassifier(q=0.5, tav=5, marked=False)
    cl_marked = FakeClassifier(q=0.5, tav=5, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl_marked) is True

    # Equal quality, both unmarked: older (higher tav) cl is preferred
    cl_old = FakeClassifier(q=0.5, tav=10, marked=False)
    cl_young = FakeClassifier(q=0.5, tav=2, marked=False)
    assert ga._is_preferred_to_delete(cl_young, cl_old) is True
    assert ga._is_preferred_to_delete(cl_old, cl_young) is False
