import random
from types import SimpleNamespace

import numpy as np
import pytest

import lcs.strategies.genetic_algorithms as ga


class DummyCondition(list):
    def __init__(self, values, wildcard=None, match=True):
        super().__init__(values)
        self._wildcard = wildcard
        self._match = match
        self.generalized = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = self._wildcard if self._wildcard is not None else "*"

    def does_match(self, p):
        return self._match


class DummyClassifier:
    def __init__(
        self,
        condition=None,
        num=1,
        tga=0,
        q=0.0,
        tav=0,
        marked=False,
        cfg=None,
    ):
        self.condition = condition if condition is not None else DummyCondition([0, 1, 2])
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self._marked = marked
        self.cfg = cfg if cfg is not None else SimpleNamespace(
            classifier_wildcard="*",
            classifier_length=len(self.condition),
        )

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return isinstance(other, DummyClassifier) and list(self.condition) == list(other.condition)


class SafeList(list):
    def expand(self):
        return self

    def safe_remove(self, item):
        self.remove(item)


def test_should_apply_returns_false_for_none_action_set():
    assert ga.should_apply(None, time=10, theta_ga=5) is False


def test_should_apply_returns_false_for_empty_numerosity():
    cl = DummyClassifier(num=0, tga=1)
    assert ga.should_apply([cl], time=100, theta_ga=5) is False


def test_should_apply_true_when_average_delay_exceeds_threshold():
    action_set = [
        DummyClassifier(num=2, tga=10),
        DummyClassifier(num=1, tga=20),
    ]
    # average tga = 40/3, delay > 5
    assert ga.should_apply(action_set, time=20, theta_ga=5) is True


def test_should_apply_false_when_average_delay_not_exceed_threshold():
    action_set = [
        DummyClassifier(num=2, tga=10),
        DummyClassifier(num=1, tga=20),
    ]
    assert ga.should_apply(action_set, time=18, theta_ga=5) is False


def test_set_timestamps_updates_all_classifiers():
    action_set = [DummyClassifier(tga=1), DummyClassifier(tga=2), DummyClassifier(tga=3)]
    ga.set_timestamps(action_set, epoch=42)
    assert [cl.tga for cl in action_set] == [42, 42, 42]


def test_roulette_wheel_selection_uses_weighted_choice_twice(monkeypatch):
    population = [DummyClassifier(q=1), DummyClassifier(q=2), DummyClassifier(q=3)]
    calls = []

    def fake_weighted(choices):
        calls.append(choices)
        return population[len(calls) - 1]

    monkeypatch.setattr(ga, "_weighted_random_choice", fake_weighted)
    p1, p2 = ga.roulette_wheel_selection(population, fitnessfunc=lambda cl: cl.q)

    assert p1 is population[0]
    assert p2 is population[1]
    assert len(calls) == 2
    assert set(calls[0].keys()) == set(population)
    assert calls[0][population[2]] == 3


def test_generalizing_mutation_generalizes_only_selected_positions(monkeypatch):
    cond = DummyCondition(["a", "b", "*", "d"], wildcard="*")
    cl = DummyClassifier(condition=cond, cfg=SimpleNamespace(classifier_wildcard="*", classifier_length=4))

    seq = iter([0.1, 0.9, 0.2, 0.05])  # mutate idx 0 and 3 only with mu=0.5
    monkeypatch.setattr(random, "random", lambda: next(seq))

    ga.generalizing_mutation(cl, mu=0.5)

    assert cond.generalized == [0, 3]
    assert list(cond) == ["*", "b", "*", "*"]


def test_generalizing_mutation_does_not_generalize_wildcards(monkeypatch):
    cond = DummyCondition(["*", "b", "*"], wildcard="*")
    cl = DummyClassifier(condition=cond, cfg=SimpleNamespace(classifier_wildcard="*", classifier_length=3))

    monkeypatch.setattr(random, "random", lambda: 0.0)
    ga.generalizing_mutation(cl, mu=1.0)

    assert cond.generalized == [1]
    assert list(cond) == ["*", "*", "*"]


def test_two_point_crossover_swaps_selected_slice(monkeypatch):
    cfg = SimpleNamespace(classifier_length=5)
    parent = DummyClassifier(condition=DummyCondition([0, 1, 2, 3, 4], wildcard="*", match=True), cfg=cfg)
    donor = DummyClassifier(condition=DummyCondition(["a", "b", "c", "d", "e"], wildcard="*", match=True), cfg=cfg)

    monkeypatch.setattr(np.random, "choice", lambda *args, **kwargs: np.array([1, 4]))
    ga.two_point_crossover(parent, donor)

    assert list(parent.condition) == [0, "b", "c", "d", 4]
    assert list(donor.condition) == ["a", 1, 2, 3, "e"]


def test_add_classifier_appends_new_classifier_and_match_set_when_matching():
    cl = DummyClassifier(condition=DummyCondition([1, 2, 3], match=True))
    population = SafeList()
    action_set = SafeList()
    match_set = SafeList()
    p = object()

    ga.add_classifier(
        cl,
        p=p,
        population=population,
        match_set=match_set,
        action_set=action_set,
        do_subsumption=False,
        theta_exp=1,
    )

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_increments_existing_unmarked_classifier_numerosity(monkeypatch):
    existing = DummyClassifier(condition=DummyCondition([1, 2, 3]), num=2, marked=False)
    new = DummyClassifier(condition=DummyCondition([1, 2, 3]), num=1)
    population = SafeList([existing])
    action_set = SafeList([existing])
    match_set = SafeList()

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: existing)

    ga.add_classifier(
        new,
        p=object(),
        population=population,
        match_set=match_set,
        action_set=action_set,
        do_subsumption=False,
        theta_exp=1,
    )

    assert existing.num == 3
    assert population == [existing]
    assert action_set == [existing]
    assert match_set == []


def test_add_classifier_does_not_increment_marked_existing_classifier(monkeypatch):
    existing = DummyClassifier(condition=DummyCondition([1, 2, 3]), num=2, marked=True)
    new = DummyClassifier(condition=DummyCondition([1, 2, 3]), num=1)
    population = SafeList([existing])
    action_set = SafeList([existing])
    match_set = SafeList()

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: existing)

    ga.add_classifier(
        new,
        p=object(),
        population=population,
        match_set=match_set,
        action_set=action_set,
        do_subsumption=False,
        theta_exp=1,
    )

    assert existing.num == 2


def test_is_preferred_to_delete_prefers_lower_quality():
    cl_del = DummyClassifier(q=1.0, tav=10)
    cl = DummyClassifier(q=0.8, tav=20)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_prefers_marked_or_older_when_quality_similar():
    cl_del = DummyClassifier(q=1.0, tav=10, marked=False)
    cl = DummyClassifier(q=1.05, tav=20, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True

    cl2 = DummyClassifier(q=1.0, tav=50, marked=False)
    assert ga._is_preferred_to_delete(cl_del, cl2) is True


def test_find_old_classifier_prefers_subsumer_when_available(monkeypatch):
    target = DummyClassifier(condition=DummyCondition([9, 9, 9]))
    subsumer = DummyClassifier(condition=DummyCondition([1, 1, 1]))

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, population, theta_exp: [subsumer])
    monkeypatch.setattr(ga, "_find_similar", lambda *args, **kwargs: None)

    result = ga._find_old_classifier([subsumer], target, use_subsumption=True, theta_exp=5)
    assert result is subsumer


def test_find_old_classifier_falls_back_to_similarity_when_no_subsumer(monkeypatch):
    target = DummyClassifier(condition=DummyCondition([1, 2, 3]))
    similar = DummyClassifier(condition=DummyCondition([1, 2, 3]))

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, population, theta_exp: [])
    monkeypatch.setattr(ga, "_find_similar", lambda cl, population: similar)

    result = ga._find_old_classifier([similar], target, use_subsumption=True, theta_exp=5)
    assert result is similar


def test_find_similar_returns_first_equal_classifier():
    target = DummyClassifier(condition=DummyCondition([1, 2, 3]))
    other = DummyClassifier(condition=DummyCondition([1, 2, 3]))
    population = [DummyClassifier(condition=DummyCondition([0, 0, 0])), other]

    assert ga._find_similar(target, population) is other


def test_weighted_random_choice_selects_first_and_later_entries(monkeypatch):
    choices = {"a": 1.0, "b": 2.0, "c": 3.0}

    monkeypatch.setattr(random, "uniform", lambda a, b: 0.5)
    assert ga._weighted_random_choice(choices) == "a"

    monkeypatch.setattr(random, "uniform", lambda a, b: 1.5)
    assert ga._weighted_random_choice(choices) == "b"

    monkeypatch.setattr(random, "uniform", lambda a, b: 5.5)
    assert ga._weighted_random_choice(choices) == "c"


def test_delete_classifiers_reduces_numerosity_when_classifier_over_threshold(monkeypatch):
    cl = DummyClassifier(num=3, q=1.0, tav=1)
    population = SafeList([cl])
    match_set = SafeList([cl])
    action_set = SafeList([cl])

    monkeypatch.setattr(random, "random", lambda: 0.0)  # always consider cl for deletion

    ga.delete_classifiers(
        population=population,
        match_set=match_set,
        action_set=action_set,
        insize=0,
        theta_as=2,
    )

    assert cl.num == 2
    assert population == [cl]
    assert match_set == [cl]
    assert action_set == [cl]


def test_delete_classifiers_removes_classifier_from_all_lists_when_num_is_one(monkeypatch):
    cl = DummyClassifier(num=1, q=1.0, tav=1)
    population = SafeList([cl])
    match_set = SafeList([cl])
    action_set = SafeList([cl])

    monkeypatch.setattr(random, "random", lambda: 0.0)

    ga.delete_classifiers(
        population=population,
        match_set=match_set,
        action_set=action_set,
        insize=0,
        theta_as=0,
    )

    assert population == []
    assert match_set == []
    assert action_set == []
