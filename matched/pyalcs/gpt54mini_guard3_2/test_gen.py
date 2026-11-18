import random

import numpy as np
import pytest

import lcs.strategies.genetic_algorithms as ga


class DummyCfg:
    def __init__(self, classifier_length=4, wildcard="#"):
        self.classifier_length = classifier_length
        self.classifier_wildcard = wildcard


class DummyCondition:
    def __init__(self, values):
        self.values = list(values)
        self.generalized = []

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
        self.values[idx] = "G"

    def does_match(self, perception):
        return True


class DummyClassifier:
    def __init__(self, condition, q=0.0, tga=0, num=1, tav=0, marked=False, cfg=None):
        self.condition = condition
        self.q = q
        self.tga = tga
        self.num = num
        self.tav = tav
        self._marked = marked
        self.cfg = cfg or DummyCfg(classifier_length=len(condition))

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return (
            isinstance(other, DummyClassifier)
            and self.condition.values == other.condition.values
            and self.q == other.q
            and self.tga == other.tga
            and self.num == other.num
            and self.tav == other.tav
            and self._marked == other._marked
        )

    def __hash__(self):
        return id(self)


class DummyList(list):
    def expand(self):
        for item in list(self):
            for _ in range(item.num):
                yield item

    def safe_remove(self, item):
        self.remove(item)


def test_should_apply_none_returns_false():
    assert ga.should_apply(None, time=10, theta_ga=5) is False


def test_should_apply_false_when_average_delay_not_exceeded():
    action_set = [DummyClassifier(DummyCondition([1]), tga=5, num=2), DummyClassifier(DummyCondition([1]), tga=7, num=2)]
    assert ga.should_apply(action_set, time=10, theta_ga=2) is False


def test_should_apply_true_when_average_delay_exceeded():
    action_set = [DummyClassifier(DummyCondition([1]), tga=1, num=1), DummyClassifier(DummyCondition([1]), tga=2, num=1)]
    assert ga.should_apply(action_set, time=10, theta_ga=5) is True


def test_should_apply_false_when_overall_num_zero():
    action_set = [DummyClassifier(DummyCondition([1]), tga=3, num=0)]
    assert ga.should_apply(action_set, time=10, theta_ga=1) is False


def test_set_timestamps_updates_all_classifiers():
    action_set = [DummyClassifier(DummyCondition([1]), tga=0), DummyClassifier(DummyCondition([2]), tga=3)]
    ga.set_timestamps(action_set, epoch=42)
    assert [cl.tga for cl in action_set] == [42, 42]


def test_generalizing_mutation_generalizes_only_non_wildcards(monkeypatch):
    random.seed(0)
    cl = DummyClassifier(DummyCondition(["a", "#", "b", "c"]), cfg=DummyCfg(classifier_length=4, wildcard="#"))

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    ga.generalizing_mutation(cl, mu=1.0)

    assert cl.condition.values == ["G", "#", "G", "G"]
    assert cl.condition.generalized == [0, 2, 3]


def test_generalizing_mutation_can_skip_when_probability_is_low(monkeypatch):
    random.seed(0)
    cl = DummyClassifier(DummyCondition(["a", "b"]), cfg=DummyCfg(classifier_length=2, wildcard="#"))

    monkeypatch.setattr(ga.random, "random", lambda: 0.99)
    ga.generalizing_mutation(cl, mu=0.5)

    assert cl.condition.values == ["a", "b"]
    assert cl.condition.generalized == []


def test_two_point_crossover_swaps_segment(monkeypatch):
    parent = DummyClassifier(DummyCondition([1, 2, 3, 4]), cfg=DummyCfg(classifier_length=4))
    donor = DummyClassifier(DummyCondition([9, 8, 7, 6]), cfg=DummyCfg(classifier_length=4))

    monkeypatch.setattr(ga.np.random, "choice", lambda *args, **kwargs: np.array([1, 3]))
    ga.two_point_crossover(parent, donor)

    assert parent.condition.values == [1, 8, 7, 4]
    assert donor.condition.values == [9, 2, 3, 6]


def test_add_classifier_appends_new_classifier_when_no_old_found(monkeypatch):
    cl = DummyClassifier(DummyCondition([1, 2]), num=1)
    population = DummyList()
    match_set = DummyList()
    action_set = DummyList()
    perception = object()

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: None)
    ga.add_classifier(cl, perception, population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_increments_numerosity_when_old_unmarked(monkeypatch):
    old = DummyClassifier(DummyCondition([1, 2]), num=3, marked=False)
    cl = DummyClassifier(DummyCondition([1, 2]), num=1)
    population = DummyList([old])
    match_set = DummyList([old])
    action_set = DummyList([old])

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old)
    ga.add_classifier(cl, object(), population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert old.num == 4
    assert cl not in population


def test_add_classifier_does_not_increment_marked_old(monkeypatch):
    old = DummyClassifier(DummyCondition([1, 2]), num=3, marked=True)
    cl = DummyClassifier(DummyCondition([1, 2]), num=1)
    population = DummyList([old])
    match_set = DummyList([old])
    action_set = DummyList([old])

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old)
    ga.add_classifier(cl, object(), population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert old.num == 3
    assert cl not in population


def test_find_old_classifier_prefers_subsumers_when_available(monkeypatch):
    target = DummyClassifier(DummyCondition([1]), q=1.0)
    subsumer = DummyClassifier(DummyCondition([1]), q=2.0)

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, population, theta_exp: [subsumer])
    result = ga._find_old_classifier([target], target, use_subsumption=True, theta_exp=5)

    assert result is subsumer


def test_find_old_classifier_falls_back_to_similarity(monkeypatch):
    target = DummyClassifier(DummyCondition([1]), q=1.0)
    similar = DummyClassifier(DummyCondition([1]), q=1.0)

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, population, theta_exp: [])
    result = ga._find_old_classifier([similar], target, use_subsumption=True, theta_exp=5)

    assert result is similar


def test_find_similar_returns_first_equal_classifier():
    target = DummyClassifier(DummyCondition([1, 2]), q=1.0)
    not_equal = DummyClassifier(DummyCondition([9, 9]), q=1.0)
    equal = DummyClassifier(DummyCondition([1, 2]), q=1.0)
    population = [not_equal, equal]

    assert ga._find_similar(target, population) is equal


def test_weighted_random_choice_selects_expected_item(monkeypatch):
    choices = {"a": 1.0, "b": 2.0, "c": 3.0}
    monkeypatch.setattr(ga.random, "uniform", lambda a, b: 1.5)

    assert ga._weighted_random_choice(choices) == "b"


def test_is_preferred_to_delete_prefers_lower_q():
    cl_del = DummyClassifier(DummyCondition([1]), q=1.0, tav=1, marked=False)
    cl = DummyClassifier(DummyCondition([1]), q=0.8, tav=10, marked=False)

    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_considers_marking_and_tav():
    cl_del = DummyClassifier(DummyCondition([1]), q=1.0, tav=1, marked=False)
    cl = DummyClassifier(DummyCondition([1]), q=1.05, tav=10, marked=True)

    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_delete_classifiers_decrements_numerosity_when_selected(monkeypatch):
    random.seed(0)
    cl = DummyClassifier(DummyCondition([1]), q=1.0, num=2, tav=0, marked=False)
    population = DummyList([cl])
    match_set = DummyList([cl])
    action_set = DummyList([cl])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=2)

    assert cl.num == 1
    assert population == [cl]
    assert match_set == [cl]
    assert action_set == [cl]


def test_delete_classifiers_removes_classifier_from_all_lists_when_num_is_one(monkeypatch):
    random.seed(0)
    cl = DummyClassifier(DummyCondition([1]), q=1.0, num=1, tav=0, marked=False)
    population = DummyList([cl])
    match_set = DummyList([cl])
    action_set = DummyList([cl])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=1)

    assert population == []
    assert match_set == []
    assert action_set == []
