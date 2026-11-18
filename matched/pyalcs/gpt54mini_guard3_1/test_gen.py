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


class DummyCl:
    def __init__(
        self,
        condition=None,
        q=0.0,
        tga=0,
        num=1,
        tav=0,
        marked=False,
        cfg=None,
        name=None,
    ):
        self.cfg = cfg or DummyCfg()
        self.condition = condition if condition is not None else DummyCondition(["a", "b", "c", "d"])
        self.q = q
        self.tga = tga
        self.num = num
        self.tav = tav
        self._marked = marked
        self.name = name

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return isinstance(other, DummyCl) and self.name == other.name

    def __hash__(self):
        return id(self)


class DummyPopulation(list):
    def safe_remove(self, cl):
        self.remove(cl)


class DummyActionSet(DummyPopulation):
    def expand(self):
        return list(self)


def test_should_apply_none_action_set_returns_false():
    assert ga.should_apply(None, time=10, theta_ga=3) is False


def test_should_apply_zero_numerosity_returns_false():
    cl = DummyCl(tga=5, num=0)
    assert ga.should_apply([cl], time=20, theta_ga=1) is False


def test_should_apply_true_when_average_delay_exceeds_threshold():
    cl1 = DummyCl(tga=2, num=1)
    cl2 = DummyCl(tga=4, num=1)
    assert ga.should_apply([cl1, cl2], time=10, theta_ga=2) is True


def test_should_apply_false_when_average_delay_does_not_exceed_threshold():
    cl1 = DummyCl(tga=8, num=1)
    cl2 = DummyCl(tga=10, num=1)
    assert ga.should_apply([cl1, cl2], time=12, theta_ga=3) is False


def test_set_timestamps_updates_all_classifiers():
    action_set = [DummyCl(tga=1), DummyCl(tga=2), DummyCl(tga=3)]
    ga.set_timestamps(action_set, epoch=99)
    assert [cl.tga for cl in action_set] == [99, 99, 99]


def test_generalizing_mutation_no_mutation_when_mu_zero(monkeypatch):
    random.seed(0)
    np.random.seed(0)

    cl = DummyCl(condition=DummyCondition(["x", "y", "z", "w"]))
    ga.generalizing_mutation(cl, mu=0.0)
    assert cl.condition == ["x", "y", "z", "w"]
    assert cl.condition.generalized == []


def test_generalizing_mutation_generalizes_non_wildcards_when_random_below_mu(monkeypatch):
    random.seed(0)
    np.random.seed(0)

    seq = iter([0.1, 0.9, 0.2, 0.05])
    monkeypatch.setattr(ga.random, "random", lambda: next(seq))

    cfg = DummyCfg(classifier_wildcard="#")
    cl = DummyCl(condition=DummyCondition(["a", "#", "c", "d"], match=True), cfg=cfg)
    ga.generalizing_mutation(cl, mu=0.5)

    assert cl.condition.generalized == [0, 2, 3]
    assert cl.condition == ["G", "#", "G", "G"]


def test_two_point_crossover_swaps_selected_segment(monkeypatch):
    random.seed(0)
    np.random.seed(0)

    monkeypatch.setattr(ga.np.random, "choice", lambda *args, **kwargs: np.array([1, 3]))

    cfg = DummyCfg(classifier_length=4)
    parent = DummyCl(condition=DummyCondition(["p0", "p1", "p2", "p3"]), cfg=cfg)
    donor = DummyCl(condition=DummyCondition(["d0", "d1", "d2", "d3"]), cfg=cfg)

    ga.two_point_crossover(parent, donor)

    assert parent.condition == ["p0", "d1", "d2", "p3"]
    assert donor.condition == ["d0", "p1", "p2", "d3"]


def test_add_classifier_appends_new_classifier_and_match_set_when_matching():
    population = DummyPopulation()
    action_set = DummyActionSet()
    match_set = DummyPopulation()
    p = object()
    cl = DummyCl(condition=DummyCondition(["a"], match=True), name="new")

    ga.add_classifier(
        cl=cl,
        p=p,
        population=population,
        match_set=match_set,
        action_set=action_set,
        do_subsumption=False,
        theta_exp=0,
    )

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_increments_numerosity_of_existing_classifier(monkeypatch):
    population = DummyPopulation([DummyCl(name="existing", num=2)])
    action_set = DummyActionSet([population[0]])
    match_set = DummyPopulation()
    p = object()
    new_cl = DummyCl(name="existing", num=1)

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, pop, theta_exp: [])

    ga.add_classifier(
        cl=new_cl,
        p=p,
        population=population,
        match_set=match_set,
        action_set=action_set,
        do_subsumption=False,
        theta_exp=0,
    )

    assert population[0].num == 3
    assert len(population) == 1
    assert len(action_set) == 1
    assert match_set == []


def test_add_classifier_does_not_increment_marked_existing_classifier(monkeypatch):
    existing = DummyCl(name="existing", num=2, marked=True)
    population = DummyPopulation([existing])
    action_set = DummyActionSet([existing])
    match_set = DummyPopulation()
    p = object()
    new_cl = DummyCl(name="existing", num=1)

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, pop, theta_exp: [])

    ga.add_classifier(
        cl=new_cl,
        p=p,
        population=population,
        match_set=match_set,
        action_set=action_set,
        do_subsumption=False,
        theta_exp=0,
    )

    assert existing.num == 2


def test_delete_classifiers_decrements_numerosity_when_selected(monkeypatch):
    random.seed(0)
    np.random.seed(0)

    cl1 = DummyCl(name="a", q=0.0, num=2, tav=1, marked=False)
    cl2 = DummyCl(name="b", q=0.0, num=1, tav=0, marked=False)
    population = DummyPopulation([cl1, cl2])
    match_set = DummyPopulation([cl1, cl2])
    action_set = DummyActionSet([cl1, cl2])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=3)

    assert cl1.num == 1 or cl2.num == 0
    assert sum(cl.num for cl in action_set) == 2


def test_delete_classifiers_removes_singleton_from_all_sets(monkeypatch):
    random.seed(0)
    np.random.seed(0)

    cl1 = DummyCl(name="a", q=0.0, num=1, tav=1, marked=False)
    cl2 = DummyCl(name="b", q=0.0, num=1, tav=0, marked=False)
    population = DummyPopulation([cl1, cl2])
    match_set = DummyPopulation([cl1, cl2])
    action_set = DummyActionSet([cl1, cl2])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=2)

    assert len(population) == 1
    assert len(match_set) == 1
    assert len(action_set) == 1
    assert population[0] == match_set[0] == action_set[0]


def test_is_preferred_to_delete_prefers_lower_quality():
    better = DummyCl(q=0.0, tav=0, marked=False)
    worse = DummyCl(q=-0.2, tav=10, marked=False)
    assert ga._is_preferred_to_delete(better, worse) is True
    assert ga._is_preferred_to_delete(worse, better) is False


def test_is_preferred_to_delete_prefers_marked_when_quality_similar():
    unmarked = DummyCl(q=1.0, tav=0, marked=False)
    marked = DummyCl(q=1.05, tav=0, marked=True)
    assert ga._is_preferred_to_delete(unmarked, marked) is True


def test_is_preferred_to_delete_prefers_higher_tav_when_quality_similar():
    cl_del = DummyCl(q=1.0, tav=1, marked=False)
    cl = DummyCl(q=1.05, tav=5, marked=False)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_find_old_classifier_uses_subsumers_first(monkeypatch):
    subsumer = DummyCl(name="subsumer")
    population = [DummyCl(name="other"), subsumer]
    cl = DummyCl(name="new")

    monkeypatch.setattr(ga, "find_subsumers", lambda cl_, pop, theta_exp: [subsumer])

    result = ga._find_old_classifier(population, cl, use_subsumption=True, theta_exp=1)
    assert result == subsumer


def test_find_old_classifier_falls_back_to_similarity_when_no_subsumers(monkeypatch):
    existing = DummyCl(name="same")
    population = [DummyCl(name="other"), existing]
    cl = DummyCl(name="same")

    monkeypatch.setattr(ga, "find_subsumers", lambda cl_, pop, theta_exp: [])

    result = ga._find_old_classifier(population, cl, use_subsumption=True, theta_exp=1)
    assert result == existing


def test_find_similar_returns_first_equal_classifier():
    target = DummyCl(name="x")
    population = [DummyCl(name="a"), target, DummyCl(name="x")]
    assert ga._find_similar(target, population) == target


def test_weighted_random_choice_selects_first_item_for_low_pick(monkeypatch):
    choices = {"a": 1.0, "b": 2.0, "c": 3.0}
    monkeypatch.setattr(ga.random, "uniform", lambda low, high: 0.0)
    assert ga._weighted_random_choice(choices) == "a"


def test_weighted_random_choice_selects_last_item_for_high_pick(monkeypatch):
    choices = {"a": 1.0, "b": 2.0, "c": 3.0}
    monkeypatch.setattr(ga.random, "uniform", lambda low, high: high - 1e-12)
    assert ga._weighted_random_choice(choices) == "c"
