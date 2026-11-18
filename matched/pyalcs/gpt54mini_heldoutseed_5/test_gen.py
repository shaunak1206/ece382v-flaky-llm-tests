import random

import pytest

from lcs.strategies import genetic_algorithms as ga


class DummyCfg:
    def __init__(self, classifier_length=4, classifier_wildcard="#"):
        self.classifier_length = classifier_length
        self.classifier_wildcard = classifier_wildcard


class DummyCondition:
    def __init__(self, values, match=True):
        self.values = list(values)
        self.match = match
        self.generalized = []

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.values[item]
        return self.values[item]

    def __setitem__(self, idx, value):
        self.values[idx] = value

    def generalize(self, idx):
        self.generalized.append(idx)
        self.values[idx] = "#"

    def does_match(self, p):
        return self.match


class DummyClassifier:
    def __init__(
        self,
        condition,
        q=1.0,
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
        self.cfg = cfg or DummyCfg(len(condition))

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return (
            isinstance(other, DummyClassifier)
            and self.condition.values == other.condition.values
        )

    def __hash__(self):
        return hash(tuple(self.condition.values))


class DummyList(list):
    def expand(self):
        return list(self)

    def safe_remove(self, item):
        if item in self:
            self.remove(item)


def test_should_apply_returns_false_for_none_action_set():
    assert ga.should_apply(None, time=10, theta_ga=3) is False


def test_should_apply_returns_false_for_zero_numerosity():
    action_set = [DummyClassifier([1], tga=5, num=0)]
    assert ga.should_apply(action_set, time=10, theta_ga=3) is False


def test_should_apply_returns_true_when_average_delay_exceeds_threshold():
    action_set = [
        DummyClassifier([1], tga=1, num=2),
        DummyClassifier([2], tga=3, num=2),
    ]
    assert ga.should_apply(action_set, time=10, theta_ga=4) is True


def test_set_timestamps_updates_all_classifiers():
    action_set = [DummyClassifier([1], tga=0), DummyClassifier([2], tga=5)]
    ga.set_timestamps(action_set, epoch=42)
    assert [cl.tga for cl in action_set] == [42, 42]


def test_roulette_wheel_selection_uses_weighted_choice(monkeypatch):
    population = [DummyClassifier([1], q=1.0), DummyClassifier([2], q=5.0)]

    calls = []

    def fake_weighted_random_choice(choices):
        calls.append(list(choices.keys()))
        return list(choices.keys())[0]

    monkeypatch.setattr(ga, "_weighted_random_choice", fake_weighted_random_choice)

    parent1, parent2 = ga.roulette_wheel_selection(population, lambda cl: cl.q)

    assert parent1 is population[0]
    assert parent2 is population[0]
    assert len(calls) == 2
    assert all(set(call) == set(population) for call in calls)


def test_generalizing_mutation_generalizes_selected_positions(monkeypatch):
    cl = DummyClassifier(["a", "b", "#", "d"], cfg=DummyCfg(4, classifier_wildcard="#"))

    rand_values = iter([0.0, 0.9, 0.0])
    monkeypatch.setattr(random, "random", lambda: next(rand_values))

    ga.generalizing_mutation(cl, mu=0.5)

    assert cl.condition.generalized == [0, 3]
    assert cl.condition.values == ["#", "b", "#", "#"]


def test_two_point_crossover_swaps_selected_segment(monkeypatch):
    parent = DummyClassifier(["a", "b", "c", "d"], cfg=DummyCfg(4))
    donor = DummyClassifier(["w", "x", "y", "z"], cfg=DummyCfg(4))

    monkeypatch.setattr(ga.np.random, "choice", lambda *args, **kwargs: [1, 3])

    ga.two_point_crossover(parent, donor)

    assert parent.condition.values == ["a", "x", "y", "d"]
    assert donor.condition.values == ["w", "b", "c", "z"]


def test_add_classifier_appends_new_classifier_and_match_set_when_matching():
    cl = DummyClassifier([1, 2], match=True)
    p = object()
    population = DummyList()
    match_set = DummyList()
    action_set = DummyList()

    ga.add_classifier(cl, p, population, match_set, action_set, False, theta_exp=0)

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_increments_existing_unmarked_classifier_numerosity(monkeypatch):
    old = DummyClassifier([1, 2], num=1, marked=False)
    new = DummyClassifier([1, 2], num=1, marked=False)
    population = DummyList([old])
    action_set = DummyList([old])
    match_set = DummyList()

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old)

    ga.add_classifier(new, object(), population, match_set, action_set, False, theta_exp=0)

    assert old.num == 2
    assert population == [old]
    assert action_set == [old]
    assert match_set == []


def test_add_classifier_does_not_increment_marked_classifier(monkeypatch):
    old = DummyClassifier([1, 2], num=3, marked=True)
    new = DummyClassifier([1, 2], num=1, marked=False)
    population = DummyList([old])
    action_set = DummyList([old])
    match_set = DummyList()

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old)

    ga.add_classifier(new, object(), population, match_set, action_set, False, theta_exp=0)

    assert old.num == 3


def test_find_similar_returns_first_equal_classifier():
    target = DummyClassifier([1, 2, 3])
    other1 = DummyClassifier([0, 0, 0])
    other2 = DummyClassifier([1, 2, 3])
    population = [other1, other2]

    assert ga._find_similar(target, population) is other2


def test_find_old_classifier_prefers_first_subsumer(monkeypatch):
    cl = DummyClassifier([1])
    subsumer = DummyClassifier([9])
    other = DummyClassifier([8])

    monkeypatch.setattr(ga, "find_subsumers", lambda cl_, pop, theta: [subsumer, other])

    assert ga._find_old_classifier([other], cl, True, theta_exp=1) is subsumer


def test_find_old_classifier_falls_back_to_similarity_when_no_subsumers(monkeypatch):
    cl = DummyClassifier([1, 2])
    similar = DummyClassifier([1, 2])
    population = [DummyClassifier([0, 0]), similar]

    monkeypatch.setattr(ga, "find_subsumers", lambda *args, **kwargs: [])

    assert ga._find_old_classifier(population, cl, True, theta_exp=1) is similar


def test_is_preferred_to_delete_prefers_lower_q():
    cl_del = DummyClassifier([1], q=0.8, tav=1, marked=False)
    cl = DummyClassifier([2], q=0.6, tav=1, marked=False)

    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_delete_classifiers_removes_classifier_when_population_over_threshold(monkeypatch):
    cl1 = DummyClassifier([1], q=1.0, num=1)
    cl2 = DummyClassifier([2], q=0.5, num=1)
    population = DummyList([cl1, cl2])
    match_set = DummyList([cl1, cl2])
    action_set = DummyList([cl1, cl2])

    monkeypatch.setattr(random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=1)

    assert len(population) == 1
    assert len(match_set) == 1
    assert len(action_set) == 1


def test_delete_classifiers_decrements_numerosity_instead_of_removing(monkeypatch):
    cl1 = DummyClassifier([1], q=1.0, num=2)
    population = DummyList([cl1])
    match_set = DummyList([cl1])
    action_set = DummyList([cl1])

    monkeypatch.setattr(random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=1)

    assert cl1.num == 1
    assert population == [cl1]
    assert match_set == [cl1]
    assert action_set == [cl1]
