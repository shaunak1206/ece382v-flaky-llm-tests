import random

import numpy as np
import pytest

import lcs.strategies.genetic_algorithms as ga


class DummyCfg:
    def __init__(self, classifier_length=4, classifier_wildcard="#"):
        self.classifier_length = classifier_length
        self.classifier_wildcard = classifier_wildcard


class DummyCondition(list):
    def __init__(self, values, match=True):
        super().__init__(values)
        self._match = match
        self.generalized = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = "G"

    def does_match(self, p):
        return self._match


class DummyClassifier:
    def __init__(self, condition, q=0.0, tga=0, num=1, tav=0, marked=False, cfg=None):
        self.condition = condition
        self.q = q
        self.tga = tga
        self.num = num
        self.tav = tav
        self._marked = marked
        self.cfg = cfg or DummyCfg()

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return (
            isinstance(other, DummyClassifier)
            and list(self.condition) == list(other.condition)
            and self.q == other.q
            and self.tga == other.tga
            and self.num == other.num
            and self.tav == other.tav
            and self._marked == other._marked
        )

    def __hash__(self):
        return id(self)


class DummyContainer(list):
    def expand(self):
        return list(self)

    def safe_remove(self, item):
        self.remove(item)


def test_should_apply_returns_false_for_none_action_set():
    assert ga.should_apply(None, time=10, theta_ga=3) is False


def test_should_apply_returns_false_for_zero_numerosity():
    action_set = [DummyClassifier(["a"], tga=5, num=0)]
    assert ga.should_apply(action_set, time=20, theta_ga=3) is False


def test_should_apply_true_when_average_delay_exceeds_threshold():
    action_set = [
        DummyClassifier(["a"], tga=1, num=2),
        DummyClassifier(["b"], tga=2, num=2),
    ]
    assert ga.should_apply(action_set, time=10, theta_ga=4) is True


def test_should_apply_false_when_delay_equals_threshold():
    action_set = [DummyClassifier(["a"], tga=5, num=2)]
    assert ga.should_apply(action_set, time=8, theta_ga=3) is False


def test_set_timestamps_updates_all_classifiers():
    action_set = [DummyClassifier(["a"], tga=1), DummyClassifier(["b"], tga=2)]
    ga.set_timestamps(action_set, epoch=99)
    assert [cl.tga for cl in action_set] == [99, 99]


def test_generalizing_mutation_generalizes_matching_nonwildcard_positions(monkeypatch):
    cl = DummyClassifier(DummyCondition(["x", "#", "y", "z"]), cfg=DummyCfg(classifier_wildcard="#"))

    seq = iter([0.1, 0.9, 0.2, 0.1])  # mutate idx 0, 2, 3 only
    monkeypatch.setattr(random, "random", lambda: next(seq))

    ga.generalizing_mutation(cl, mu=0.5)

    assert cl.condition.generalized == [0, 2, 3]
    assert cl.condition == ["G", "#", "G", "G"]


def test_generalizing_mutation_skips_wildcard_positions(monkeypatch):
    cl = DummyClassifier(DummyCondition(["#", "#", "a"], match=True), cfg=DummyCfg(classifier_wildcard="#"))
    monkeypatch.setattr(random, "random", lambda: 0.0)

    ga.generalizing_mutation(cl, mu=1.0)

    assert cl.condition.generalized == [2]
    assert cl.condition == ["#", "#", "G"]


def test_two_point_crossover_swaps_selected_slices(monkeypatch):
    parent = DummyClassifier(DummyCondition(["p0", "p1", "p2", "p3"]), cfg=DummyCfg(classifier_length=4))
    donor = DummyClassifier(DummyCondition(["d0", "d1", "d2", "d3"]), cfg=DummyCfg(classifier_length=4))

    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: np.array([1, 3]))

    ga.two_point_crossover(parent, donor)

    assert parent.condition == ["p0", "d1", "d2", "p3"]
    assert donor.condition == ["d0", "p1", "p2", "d3"]


def test_add_classifier_appends_to_population_match_and_action_set_when_new():
    cl = DummyClassifier(DummyCondition(["a"], match=True))
    p = object()
    population = []
    match_set = []
    action_set = []

    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_does_not_append_to_match_set_when_not_matching():
    cl = DummyClassifier(DummyCondition(["a"], match=False))
    p = object()
    population = []
    match_set = []
    action_set = []

    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == []


def test_add_classifier_increments_numerosity_for_similar_existing_classifier(monkeypatch):
    existing = DummyClassifier(DummyCondition(["a"]), num=1, marked=False)
    new_cl = DummyClassifier(DummyCondition(["a"]), num=1, marked=False)
    population = [existing]
    action_set = [existing]
    match_set = []

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: existing)

    ga.add_classifier(new_cl, object(), population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert existing.num == 2
    assert population == [existing]
    assert action_set == [existing]
    assert match_set == []


def test_add_classifier_does_not_increment_marked_existing_classifier(monkeypatch):
    existing = DummyClassifier(DummyCondition(["a"]), num=1, marked=True)
    new_cl = DummyClassifier(DummyCondition(["a"]), num=1, marked=False)
    population = [existing]
    action_set = [existing]
    match_set = []

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: existing)

    ga.add_classifier(new_cl, object(), population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert existing.num == 1


def test_is_preferred_to_delete_prefers_lower_q():
    cl_del = DummyClassifier(DummyCondition(["a"]), q=0.8, tav=1, marked=False)
    cl = DummyClassifier(DummyCondition(["b"]), q=0.6, tav=10, marked=False)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_prefers_marked_classifier_when_q_similar():
    cl_del = DummyClassifier(DummyCondition(["a"]), q=1.0, tav=1, marked=False)
    cl = DummyClassifier(DummyCondition(["b"]), q=1.05, tav=2, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_find_old_classifier_prefers_subsumer_when_available(monkeypatch):
    subsumer = DummyClassifier(DummyCondition(["s"]))
    similar = DummyClassifier(DummyCondition(["x"]))
    target = DummyClassifier(DummyCondition(["t"]))

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, population, theta_exp: [subsumer, similar])
    monkeypatch.setattr(ga, "_find_similar", lambda *args, **kwargs: similar)

    found = ga._find_old_classifier([subsumer, similar], target, use_subsumption=True, theta_exp=5)
    assert found is subsumer


def test_find_old_classifier_falls_back_to_similarity_when_no_subsumers(monkeypatch):
    similar = DummyClassifier(DummyCondition(["x"]))
    target = DummyClassifier(DummyCondition(["t"]))

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, population, theta_exp: [])
    monkeypatch.setattr(ga, "_find_similar", lambda other_cl, population: similar)

    found = ga._find_old_classifier([similar], target, use_subsumption=True, theta_exp=5)
    assert found is similar


def test_find_similar_returns_first_equal_classifier():
    target = DummyClassifier(DummyCondition(["a"]), q=1.0, tga=0, num=1, tav=0, marked=False)
    population = [
        DummyClassifier(DummyCondition(["b"]), q=1.0),
        target,
        DummyClassifier(DummyCondition(["a"]), q=2.0),
    ]

    assert ga._find_similar(target, population) is target


def test_weighted_random_choice_selects_expected_key(monkeypatch):
    choices = {"a": 1, "b": 2, "c": 3}
    monkeypatch.setattr(random, "uniform", lambda a, b: 1.5)

    assert ga._weighted_random_choice(choices) == "b"
