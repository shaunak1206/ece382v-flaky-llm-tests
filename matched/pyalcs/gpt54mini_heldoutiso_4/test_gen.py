import random

import pytest

import lcs.strategies.genetic_algorithms as ga


class DummyCfg:
    def __init__(self, classifier_length=4, wildcard="#"):
        self.classifier_length = classifier_length
        self.classifier_wildcard = wildcard


class DummyCondition:
    def __init__(self, values, match=True):
        self.values = list(values)
        self.generalized = []
        self._match = match

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
        self.values[idx] = "W"

    def does_match(self, p):
        return self._match


class DummyClassifier:
    def __init__(self, condition, q=0.5, num=1, tga=0, tav=0, marked=False, cfg=None):
        self.condition = condition
        self.q = q
        self.num = num
        self.tga = tga
        self.tav = tav
        self._marked = marked
        self.cfg = cfg or DummyCfg(classifier_length=len(condition.values))

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return isinstance(other, DummyClassifier) and self.condition.values == other.condition.values

    def __hash__(self):
        return id(self)


class DummyActionSet(list):
    def expand(self):
        return list(self)

    def safe_remove(self, item):
        self.remove(item)


def test_should_apply_returns_false_for_none_action_set():
    assert ga.should_apply(None, time=10, theta_ga=5) is False


def test_should_apply_returns_false_for_empty_numerosity():
    cl = DummyClassifier(DummyCondition([1, 2]), num=0, tga=10)
    assert ga.should_apply([cl], time=20, theta_ga=5) is False


def test_should_apply_returns_true_when_threshold_exceeded():
    cl1 = DummyClassifier(DummyCondition([1, 2]), num=2, tga=1)
    cl2 = DummyClassifier(DummyCondition([3, 4]), num=1, tga=2)
    assert ga.should_apply([cl1, cl2], time=10, theta_ga=5) is True


def test_should_apply_returns_false_when_threshold_not_exceeded():
    cl1 = DummyClassifier(DummyCondition([1, 2]), num=2, tga=7)
    cl2 = DummyClassifier(DummyCondition([3, 4]), num=1, tga=8)
    assert ga.should_apply([cl1, cl2], time=10, theta_ga=5) is False


def test_set_timestamps_updates_all_classifiers():
    action_set = [
        DummyClassifier(DummyCondition([1]), tga=1),
        DummyClassifier(DummyCondition([2]), tga=2),
    ]
    ga.set_timestamps(action_set, epoch=42)
    assert [cl.tga for cl in action_set] == [42, 42]


def test_generalizing_mutation_generalizes_only_when_random_allows(monkeypatch):
    cl = DummyClassifier(DummyCondition(["a", "#", "b", "c"], match=True))
    seq = iter([0.1, 0.9, 0.2, 0.8])
    monkeypatch.setattr(random, "random", lambda: next(seq))

    ga.generalizing_mutation(cl, mu=0.5)

    assert cl.condition.values == ["W", "#", "W", "c"]
    assert cl.condition.generalized == [0, 2]


def test_generalizing_mutation_skips_wildcards(monkeypatch):
    cl = DummyClassifier(DummyCondition(["#", "x", "#"], match=True))
    monkeypatch.setattr(random, "random", lambda: 0.0)

    ga.generalizing_mutation(cl, mu=1.0)

    assert cl.condition.values == ["#", "W", "#"]
    assert cl.condition.generalized == [1]


def test_two_point_crossover_swaps_selected_segment(monkeypatch):
    parent = DummyClassifier(DummyCondition([1, 2, 3, 4]), cfg=DummyCfg(classifier_length=4))
    donor = DummyClassifier(DummyCondition(["a", "b", "c", "d"]), cfg=DummyCfg(classifier_length=4))
    monkeypatch.setattr(ga.np.random, "choice", lambda *args, **kwargs: [1, 3])

    ga.two_point_crossover(parent, donor)

    assert parent.condition.values == [1, "b", "c", 4]
    assert donor.condition.values == ["a", 2, 3, "d"]


def test_add_classifier_appends_new_classifier_and_match_set_when_matching():
    cl = DummyClassifier(DummyCondition([1, 2], match=True))
    population = []
    match_set = []
    action_set = []

    ga.add_classifier(
        cl=cl,
        p=object(),
        population=population,
        match_set=match_set,
        action_set=action_set,
        do_subsumption=False,
        theta_exp=10,
    )

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_increments_numerosity_for_existing_unmarked_classifier(monkeypatch):
    old = DummyClassifier(DummyCondition([1, 2]), num=2, marked=False)
    new = DummyClassifier(DummyCondition([1, 2]), num=1)
    population = [old]
    action_set = [old]

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old)

    ga.add_classifier(
        cl=new,
        p=object(),
        population=population,
        match_set=[],
        action_set=action_set,
        do_subsumption=False,
        theta_exp=10,
    )

    assert old.num == 3
    assert population == [old]
    assert action_set == [old]


def test_add_classifier_does_not_increment_marked_existing_classifier(monkeypatch):
    old = DummyClassifier(DummyCondition([1, 2]), num=2, marked=True)
    new = DummyClassifier(DummyCondition([1, 2]), num=1)
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old)

    ga.add_classifier(
        cl=new,
        p=object(),
        population=[],
        match_set=[],
        action_set=[old],
        do_subsumption=False,
        theta_exp=10,
    )

    assert old.num == 2


def test_find_similar_returns_equal_classifier():
    target = DummyClassifier(DummyCondition([1, 2]))
    population = [DummyClassifier(DummyCondition([0, 0])), target]

    assert ga._find_similar(target, population) is target


def test_find_old_classifier_uses_subsumers_before_similarity(monkeypatch):
    target = DummyClassifier(DummyCondition([1, 2]))
    similar = DummyClassifier(DummyCondition([1, 2]))
    subsumer = DummyClassifier(DummyCondition([9, 9]))

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, pop, theta_exp: [subsumer])
    monkeypatch.setattr(ga, "_find_similar", lambda other_cl, population: similar)

    result = ga._find_old_classifier([similar], target, use_subsumption=True, theta_exp=3)
    assert result is subsumer


def test_find_old_classifier_falls_back_to_similarity_when_no_subsumers(monkeypatch):
    target = DummyClassifier(DummyCondition([1, 2]))
    similar = DummyClassifier(DummyCondition([1, 2]))

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, pop, theta_exp: [])
    monkeypatch.setattr(ga, "_find_similar", lambda other_cl, population: similar)

    result = ga._find_old_classifier([similar], target, use_subsumption=True, theta_exp=3)
    assert result is similar


def test_weighted_random_choice_selects_correct_key(monkeypatch):
    a = DummyClassifier(DummyCondition([1]))
    b = DummyClassifier(DummyCondition([2]))
    choices = {a: 1.0, b: 3.0}
    monkeypatch.setattr(random, "uniform", lambda low, high: 0.5)

    assert ga._weighted_random_choice(choices) is a


def test_roulette_wheel_selection_calls_weighted_random_choice_twice(monkeypatch):
    p1 = DummyClassifier(DummyCondition([1]))
    p2 = DummyClassifier(DummyCondition([2]))
    population = [p1, p2]
    calls = []

    monkeypatch.setattr(ga, "_weighted_random_choice", lambda choices: calls.append(list(choices.keys())[0]) or list(choices.keys())[0])

    parent1, parent2 = ga.roulette_wheel_selection(population, lambda cl: 1.0)

    assert parent1 is p1
    assert parent2 is p1
    assert len(calls) == 2


def test_is_preferred_to_delete_prefers_lower_q():
    cl_del = DummyClassifier(DummyCondition([1]), q=0.8, tav=1, marked=False)
    cl = DummyClassifier(DummyCondition([2]), q=0.6, tav=10, marked=False)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_prefers_marked_when_q_close():
    cl_del = DummyClassifier(DummyCondition([1]), q=0.5, tav=1, marked=False)
    cl = DummyClassifier(DummyCondition([2]), q=0.55, tav=1, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_delete_classifiers_reduces_numerosity_when_possible(monkeypatch):
    cl = DummyClassifier(DummyCondition([1]), num=2, q=0.5, tav=0)
    population = DummyActionSet([cl])
    match_set = DummyActionSet([cl])
    action_set = DummyActionSet([cl])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=2)

    assert cl.num == 1
    assert cl in population and cl in match_set and cl in action_set


def test_delete_classifiers_removes_classifier_when_numerosity_one(monkeypatch):
    cl = DummyClassifier(DummyCondition([1]), num=1, q=0.5, tav=0)
    population = DummyActionSet([cl])
    match_set = DummyActionSet([cl])
    action_set = DummyActionSet([cl])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=1)

    assert cl not in population
    assert cl not in match_set
    assert cl not in action_set
