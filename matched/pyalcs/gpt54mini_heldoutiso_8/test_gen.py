import random

import numpy as np
import pytest

import lcs.strategies.genetic_algorithms as ga


class DummyCfg:
    def __init__(self, classifier_length=4, classifier_wildcard="#"):
        self.classifier_length = classifier_length
        self.classifier_wildcard = classifier_wildcard


class DummyCondition:
    def __init__(self, values, match=True):
        self.values = list(values)
        self.generalized = []
        self.match = match

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.values[item]
        return self.values[item]

    def __setitem__(self, idx, value):
        self.values[idx] = value

    def generalize(self, idx):
        self.generalized.append(idx)
        self.values[idx] = "*"

    def does_match(self, perception):
        return self.match

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return repr(self.values)


class DummyClassifier:
    def __init__(self, condition, q=0.0, tga=0, tav=0, num=1, cfg=None, marked=False):
        self.condition = condition
        self.q = q
        self.tga = tga
        self.tav = tav
        self.num = num
        self.cfg = cfg or DummyCfg()
        self._marked = marked

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return (
            isinstance(other, DummyClassifier)
            and self.condition.values == other.condition.values
            and self.q == other.q
            and self.tga == other.tga
            and self.tav == other.tav
            and self.num == other.num
        )

    def __hash__(self):
        return id(self)


class DummyPop(list):
    def expand(self):
        for cl in list(self):
            for _ in range(cl.num):
                yield cl

    def safe_remove(self, cl):
        self.remove(cl)


def test_should_apply_returns_false_for_none_action_set():
    assert ga.should_apply(None, time=10, theta_ga=5) is False


def test_should_apply_returns_false_for_empty_numerosity():
    cl = DummyClassifier(DummyCondition([1, 2]), tga=1, num=0)
    assert ga.should_apply([cl], time=10, theta_ga=5) is False


def test_should_apply_true_when_average_delay_exceeds_threshold():
    cl1 = DummyClassifier(DummyCondition([1]), tga=2, num=1)
    cl2 = DummyClassifier(DummyCondition([2]), tga=4, num=3)
    assert ga.should_apply([cl1, cl2], time=20, theta_ga=10) is True


def test_set_timestamps_updates_all_classifiers():
    cl1 = DummyClassifier(DummyCondition([1]), tga=1)
    cl2 = DummyClassifier(DummyCondition([2]), tga=2)
    action_set = [cl1, cl2]

    ga.set_timestamps(action_set, epoch=99)

    assert cl1.tga == 99
    assert cl2.tga == 99


def test_roulette_wheel_selection_uses_weighted_random_choice(monkeypatch):
    population = [object(), object()]
    calls = []

    def fake_choice(choices):
        calls.append(dict(choices))
        return population[0] if len(calls) == 1 else population[1]

    monkeypatch.setattr(ga, "_weighted_random_choice", fake_choice)

    p1, p2 = ga.roulette_wheel_selection(population, lambda cl: 1.5)

    assert p1 is population[0]
    assert p2 is population[1]
    assert len(calls) == 2
    assert set(calls[0].keys()) == set(population)


def test_generalizing_mutation_generalizes_only_when_probability_and_non_wildcard(monkeypatch):
    cl = DummyClassifier(DummyCondition(["a", "#", "c"], match=True), cfg=DummyCfg(classifier_wildcard="#"))
    seq = iter([0.1, 0.9, 0.2])  # mutate idx 0 and 2 only

    monkeypatch.setattr(random, "random", lambda: next(seq))

    ga.generalizing_mutation(cl, mu=0.5)

    assert cl.condition.generalized == [0, 2]
    assert cl.condition.values == ["*", "#", "*"]


def test_two_point_crossover_swaps_condition_segments(monkeypatch):
    parent = DummyClassifier(DummyCondition(["p0", "p1", "p2", "p3"]), cfg=DummyCfg(classifier_length=4))
    donor = DummyClassifier(DummyCondition(["d0", "d1", "d2", "d3"]), cfg=DummyCfg(classifier_length=4))

    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: np.array([1, 3]))

    ga.two_point_crossover(parent, donor)

    assert parent.condition.values == ["p0", "d1", "d2", "p3"]
    assert donor.condition.values == ["d0", "p1", "p2", "d3"]


def test_add_classifier_appends_new_classifier_and_match_set_when_matching(monkeypatch):
    cl = DummyClassifier(DummyCondition([1, 2, 3], match=True))
    p = object()
    population = []
    match_set = []
    action_set = []

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: None)

    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=5)

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_does_not_append_when_existing_classifier_is_found(monkeypatch):
    cl = DummyClassifier(DummyCondition([1, 2, 3]), num=1)
    old_cl = DummyClassifier(DummyCondition([1, 2, 3]), num=2)
    population = [old_cl]
    match_set = []
    action_set = [old_cl]

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old_cl)

    ga.add_classifier(cl, object(), population, match_set, action_set, do_subsumption=False, theta_exp=5)

    assert old_cl.num == 3
    assert population == [old_cl]
    assert action_set == [old_cl]
    assert match_set == []


def test_add_classifier_does_not_increment_marked_old_classifier(monkeypatch):
    cl = DummyClassifier(DummyCondition([1, 2, 3]), num=1)
    old_cl = DummyClassifier(DummyCondition([1, 2, 3]), num=2, marked=True)
    population = [old_cl]
    action_set = [old_cl]

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old_cl)

    ga.add_classifier(cl, object(), population, None, action_set, do_subsumption=False, theta_exp=5)

    assert old_cl.num == 2


def test_delete_classifiers_decrements_numerosity_when_selected(monkeypatch):
    cl1 = DummyClassifier(DummyCondition([1]), num=2, q=0.5, tav=1)
    cl2 = DummyClassifier(DummyCondition([2]), num=1, q=0.1, tav=1)
    population = DummyPop([cl1, cl2])
    match_set = DummyPop([cl1, cl2])
    action_set = DummyPop([cl1, cl2])

    monkeypatch.setattr(random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=0, theta_as=10)

    assert cl1.num == 1
    assert cl2.num == 1
    assert cl1 in population and cl2 in population


def test_delete_classifiers_removes_classifier_from_all_collections(monkeypatch):
    cl1 = DummyClassifier(DummyCondition([1]), num=1, q=0.5, tav=1)
    cl2 = DummyClassifier(DummyCondition([2]), num=1, q=0.1, tav=1)
    population = DummyPop([cl1, cl2])
    match_set = DummyPop([cl1, cl2])
    action_set = DummyPop([cl1, cl2])

    monkeypatch.setattr(random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=0, theta_as=1)

    assert cl1 not in population and cl1 not in match_set and cl1 not in action_set
    assert len(population) == 1
    assert len(match_set) == 1
    assert len(action_set) == 1


def test_is_preferred_to_delete_prefers_lower_quality():
    cl_del = DummyClassifier(DummyCondition([1]), q=0.9, tav=1, marked=False)
    cl = DummyClassifier(DummyCondition([2]), q=0.7, tav=10, marked=False)

    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_prefers_marked_classifier_when_equal_quality():
    cl_del = DummyClassifier(DummyCondition([1]), q=0.5, tav=1, marked=False)
    cl = DummyClassifier(DummyCondition([2]), q=0.5, tav=1, marked=True)

    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_find_old_classifier_uses_first_subsumer_when_available(monkeypatch):
    target = DummyClassifier(DummyCondition([9]))
    subsumer = DummyClassifier(DummyCondition([1]))
    population = [subsumer, target]

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, pop, theta_exp: [subsumer, target])

    assert ga._find_old_classifier(population, target, use_subsumption=True, theta_exp=7) is subsumer


def test_find_old_classifier_falls_back_to_similarity_when_no_subsumer(monkeypatch):
    target = DummyClassifier(DummyCondition([9]))
    similar = DummyClassifier(DummyCondition([9]))
    population = [similar, target]

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, pop, theta_exp: [])
    assert ga._find_old_classifier(population, target, use_subsumption=True, theta_exp=7) is similar


def test_find_similar_returns_first_equal_classifier_or_none():
    target = DummyClassifier(DummyCondition([1, 2, 3]))
    other = DummyClassifier(DummyCondition([9, 9, 9]))
    population = [other, target]

    assert ga._find_similar(target, population) is target
    assert ga._find_similar(other, [target]) is None


def test_weighted_random_choice_returns_key_based_on_pick(monkeypatch):
    a = object()
    b = object()
    choices = {a: 1.0, b: 2.0}

    monkeypatch.setattr(random, "uniform", lambda low, high: 0.5)

    assert ga._weighted_random_choice(choices) is a
