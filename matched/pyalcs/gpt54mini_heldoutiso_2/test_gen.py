import random
from types import SimpleNamespace

import pytest

import lcs.strategies.genetic_algorithms as ga


class DummyCondition:
    def __init__(self, values, wildcard=None):
        self.values = list(values)
        self.generalized = []
        self.wildcard = wildcard

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return list(self.values[item])
        return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value

    def generalize(self, idx):
        self.generalized.append(idx)
        self.values[idx] = f"g{idx}"

    def does_match(self, p):
        return True

    def __eq__(self, other):
        return isinstance(other, DummyCondition) and self.values == other.values


class DummyClassifier:
    def __init__(
        self,
        condition,
        q=0.0,
        tga=0,
        num=1,
        tav=0,
        marked=False,
        cfg=None,
    ):
        self.condition = condition
        self.q = q
        self.tga = tga
        self.num = num
        self.tav = tav
        self._marked = marked
        self.cfg = cfg or SimpleNamespace(classifier_wildcard="#", classifier_length=len(condition))

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return (
            isinstance(other, DummyClassifier)
            and self.condition == other.condition
            and self.q == other.q
        )


class DummyList(list):
    def expand(self):
        for item in self:
            for _ in range(item.num):
                yield item

    def safe_remove(self, item):
        if item in self:
            self.remove(item)


def test_should_apply_returns_false_for_none_action_set():
    assert ga.should_apply(None, time=10, theta_ga=5) is False


def test_should_apply_returns_false_for_zero_numerosity():
    cl = DummyClassifier(["a"], tga=5, num=0)
    assert ga.should_apply([cl], time=20, theta_ga=5) is False


def test_should_apply_returns_true_when_average_delay_exceeds_threshold():
    cl1 = DummyClassifier(["a"], tga=1, num=1)
    cl2 = DummyClassifier(["b"], tga=2, num=3)
    # average tga = 1.75; delay at time 10 is 8.25 > 5
    assert ga.should_apply([cl1, cl2], time=10, theta_ga=5) is True


def test_should_apply_returns_false_when_delay_equals_threshold():
    cl = DummyClassifier(["a"], tga=5, num=1)
    assert ga.should_apply([cl], time=10, theta_ga=5) is False


def test_set_timestamps_updates_all_classifiers():
    cl1 = DummyClassifier(["a"], tga=1)
    cl2 = DummyClassifier(["b"], tga=2)
    ga.set_timestamps([cl1, cl2], epoch=99)
    assert cl1.tga == 99
    assert cl2.tga == 99


def test_roulette_wheel_selection_uses_weighted_choice_twice(monkeypatch):
    population = [DummyClassifier(["a"]), DummyClassifier(["b"]), DummyClassifier(["c"])]
    calls = []

    def fake_weighted_choice(choices):
        calls.append(dict(choices))
        return population[len(calls) - 1]

    monkeypatch.setattr(ga, "_weighted_random_choice", fake_weighted_choice)

    p1, p2 = ga.roulette_wheel_selection(population, fitnessfunc=lambda cl: 1.0)
    assert p1 is population[0]
    assert p2 is population[1]
    assert len(calls) == 2
    assert set(calls[0].keys()) == set(population)


def test_generalizing_mutation_generalizes_only_non_wildcards_when_probability_hits(monkeypatch):
    cl = DummyClassifier(["x", "#", "y"], cfg=SimpleNamespace(classifier_wildcard="#"))
    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    ga.generalizing_mutation(cl, mu=1.0)
    assert cl.condition.generalized == [0, 2]
    assert cl.condition.values == ["g0", "#", "g2"]


def test_generalizing_mutation_skips_when_probability_fails(monkeypatch):
    cl = DummyClassifier(["x", "y"], cfg=SimpleNamespace(classifier_wildcard="#"))
    monkeypatch.setattr(ga.random, "random", lambda: 0.999)
    ga.generalizing_mutation(cl, mu=0.1)
    assert cl.condition.generalized == []


def test_two_point_crossover_swaps_slices(monkeypatch):
    parent = DummyClassifier(["a", "b", "c", "d"], cfg=SimpleNamespace(classifier_length=4))
    donor = DummyClassifier(["w", "x", "y", "z"], cfg=SimpleNamespace(classifier_length=4))
    monkeypatch.setattr(ga.np.random, "choice", lambda *args, **kwargs: [1, 3])

    ga.two_point_crossover(parent, donor)

    assert parent.condition.values == ["a", "x", "y", "d"]
    assert donor.condition.values == ["w", "b", "c", "z"]


def test_add_classifier_appends_new_classifier_and_match_set_when_no_old_classifier(monkeypatch):
    cl = DummyClassifier(["a"])
    population = DummyList()
    match_set = DummyList()
    action_set = DummyList()
    p = object()

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: None)

    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_increments_numerosity_of_existing_unmarked_classifier(monkeypatch):
    cl = DummyClassifier(["a"])
    old_cl = DummyClassifier(["a"], num=2, marked=False)
    population = DummyList([old_cl])
    match_set = DummyList([old_cl])
    action_set = DummyList([old_cl])

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old_cl)

    ga.add_classifier(cl, object(), population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert old_cl.num == 3
    assert population == [old_cl]
    assert action_set == [old_cl]
    assert match_set == [old_cl]


def test_add_classifier_does_not_increment_marked_classifier(monkeypatch):
    cl = DummyClassifier(["a"])
    old_cl = DummyClassifier(["a"], num=2, marked=True)
    population = DummyList([old_cl])
    match_set = DummyList([old_cl])
    action_set = DummyList([old_cl])

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old_cl)

    ga.add_classifier(cl, object(), population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert old_cl.num == 2


def test_delete_classifiers_decrements_numerosity_when_selected_classifier_has_num_gt_one(monkeypatch):
    cl = DummyClassifier(["a"], num=2, q=0.0)
    population = DummyList([cl])
    match_set = DummyList([cl])
    action_set = DummyList([cl])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=0, theta_as=1)

    assert cl.num == 1
    assert cl in population
    assert cl in match_set
    assert cl in action_set


def test_delete_classifiers_removes_classifier_from_all_lists_when_num_equals_one(monkeypatch):
    cl = DummyClassifier(["a"], num=1, q=0.0)
    population = DummyList([cl])
    match_set = DummyList([cl])
    action_set = DummyList([cl])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=1)

    assert cl not in population
    assert cl not in match_set
    assert cl not in action_set


def test_is_preferred_to_delete_prefers_lower_q(monkeypatch):
    cl_del = DummyClassifier(["a"], q=1.0, tav=1, marked=False)
    cl = DummyClassifier(["b"], q=0.8, tav=100, marked=False)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_uses_marked_and_tav_when_qs_are_close():
    cl_del = DummyClassifier(["a"], q=1.0, tav=1, marked=False)
    cl = DummyClassifier(["b"], q=1.05, tav=2, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_find_old_classifier_uses_subsumers_first(monkeypatch):
    cl = DummyClassifier(["a"])
    subsumer = DummyClassifier(["s"])

    monkeypatch.setattr(ga, "find_subsumers", lambda *args, **kwargs: [subsumer])
    monkeypatch.setattr(ga, "_find_similar", lambda *args, **kwargs: None)

    result = ga._find_old_classifier([subsumer], cl, use_subsumption=True, theta_exp=10)
    assert result is subsumer


def test_find_old_classifier_falls_back_to_similar_when_no_subsumers(monkeypatch):
    cl = DummyClassifier(["a"])
    similar = DummyClassifier(["a"])

    monkeypatch.setattr(ga, "find_subsumers", lambda *args, **kwargs: [])
    monkeypatch.setattr(ga, "_find_similar", lambda other, population: similar)

    result = ga._find_old_classifier([similar], cl, use_subsumption=True, theta_exp=10)
    assert result is similar


def test_find_similar_returns_first_equal_classifier():
    target = DummyClassifier(["x"], q=1.0)
    other1 = DummyClassifier(["y"], q=1.0)
    other2 = DummyClassifier(["x"], q=1.0)
    population = [other1, other2]

    assert ga._find_similar(target, population) is other2


def test_weighted_random_choice_returns_first_item_when_pick_is_zero(monkeypatch):
    items = {"a": 1, "b": 2, "c": 3}
    monkeypatch.setattr(ga.random, "uniform", lambda a, b: 0.0)

    assert ga._weighted_random_choice(items) == "a"
