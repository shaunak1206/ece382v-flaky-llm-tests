import random

import pytest

from lcs.strategies import genetic_algorithms as ga


class FakeConditionList(list):
    def generalize(self, idx):
        self[idx] = "W"

    def does_match(self, perception):
        return self == list(perception)


class FakeCfg:
    def __init__(self, classifier_length=4, wildcard="#"):
        self.classifier_length = classifier_length
        self.classifier_wildcard = wildcard


class FakeClassifier:
    def __init__(
        self,
        condition=None,
        q=0.0,
        tav=0,
        tga=0,
        num=1,
        marked=False,
        cfg=None,
        name=None,
    ):
        self.condition = FakeConditionList(condition or [])
        self.q = q
        self.tav = tav
        self.tga = tga
        self.num = num
        self._marked = marked
        self.cfg = cfg or FakeCfg(classifier_length=len(self.condition))
        self.name = name

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return (
            isinstance(other, FakeClassifier)
            and self.condition == other.condition
            and self.name == other.name
        )

    def __hash__(self):
        return id(self)


class FakePopulation(list):
    def safe_remove(self, item):
        self.remove(item)

    def expand(self):
        for cl in self:
            for _ in range(cl.num):
                yield cl


def test_should_apply_returns_false_for_none_action_set():
    assert ga.should_apply(None, time=10, theta_ga=5) is False


def test_should_apply_returns_false_for_empty_numerosity():
    cl = FakeClassifier(tga=3, num=0)
    assert ga.should_apply([cl], time=10, theta_ga=2) is False


def test_should_apply_true_when_average_delay_exceeds_threshold():
    action_set = [FakeClassifier(tga=1, num=2), FakeClassifier(tga=5, num=1)]
    assert ga.should_apply(action_set, time=10, theta_ga=3) is True


def test_should_apply_false_when_average_delay_not_exceeding_threshold():
    action_set = [FakeClassifier(tga=8, num=2), FakeClassifier(tga=7, num=1)]
    assert ga.should_apply(action_set, time=10, theta_ga=3) is False


def test_set_timestamps_updates_all_classifiers():
    action_set = [FakeClassifier(tga=1), FakeClassifier(tga=2), FakeClassifier(tga=3)]
    ga.set_timestamps(action_set, epoch=42)
    assert [cl.tga for cl in action_set] == [42, 42, 42]


def test_roulette_wheel_selection_uses_weighted_choice(monkeypatch):
    population = [FakeClassifier(name="a"), FakeClassifier(name="b"), FakeClassifier(name="c")]
    seen = []

    def fitness(cl):
        return {"a": 1, "b": 2, "c": 3}[cl.name]

    def fake_weighted(choices):
        seen.append(dict(choices))
        return population[len(seen) - 1]

    monkeypatch.setattr(ga, "_weighted_random_choice", fake_weighted)
    p1, p2 = ga.roulette_wheel_selection(population, fitness)

    assert p1 is population[0]
    assert p2 is population[1]
    assert len(seen) == 2
    assert seen[0][population[0]] == 1
    assert seen[0][population[1]] == 2
    assert seen[0][population[2]] == 3


def test_generalizing_mutation_generalizes_only_non_wildcards(monkeypatch):
    cl = FakeClassifier(condition=["a", "#", "b"], cfg=FakeCfg(classifier_length=3, wildcard="#"))

    seq = iter([0.0, 0.9, 0.0])

    monkeypatch.setattr(random, "random", lambda: next(seq))
    ga.generalizing_mutation(cl, mu=0.5)

    assert cl.condition == ["W", "#", "W"]


def test_two_point_crossover_swaps_slices(monkeypatch):
    parent = FakeClassifier(condition=["a", "b", "c", "d"], cfg=FakeCfg(classifier_length=4))
    donor = FakeClassifier(condition=["A", "B", "C", "D"], cfg=FakeCfg(classifier_length=4))

    monkeypatch.setattr(ga.np.random, "choice", lambda *args, **kwargs: [1, 3])
    ga.two_point_crossover(parent, donor)

    assert parent.condition == ["a", "B", "C", "d"]
    assert donor.condition == ["A", "b", "c", "D"]


def test_add_classifier_appends_new_classifier_and_match_set(monkeypatch):
    cl = FakeClassifier(condition=["x", "y"], name="new")
    p = ["x", "y"]
    population = FakePopulation()
    match_set = FakePopulation()
    action_set = FakePopulation()

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: None)

    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_does_not_add_to_match_set_if_not_matching(monkeypatch):
    cl = FakeClassifier(condition=["x", "y"], name="new")
    p = ["x", "z"]
    population = FakePopulation()
    match_set = FakePopulation()
    action_set = FakePopulation()

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: None)

    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == []


def test_add_classifier_increments_numerosity_for_unmarked_existing_classifier(monkeypatch):
    old = FakeClassifier(condition=["x"], num=2, marked=False, name="old")
    cl = FakeClassifier(condition=["x"], name="old")
    population = FakePopulation([old])
    action_set = FakePopulation([old])
    match_set = FakePopulation([old])

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old)

    ga.add_classifier(cl, ["x"], population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert old.num == 3
    assert population == [old]
    assert action_set == [old]
    assert match_set == [old]


def test_delete_classifiers_decrements_numerosity_when_selected(monkeypatch):
    cl = FakeClassifier(condition=["x"], num=2, q=1.0, tav=1, name="a")
    population = FakePopulation([cl])
    match_set = FakePopulation([cl])
    action_set = FakePopulation([cl])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=0, theta_as=1)

    assert cl.num == 1
    assert population == [cl]
    assert match_set == [cl]
    assert action_set == [cl]


def test_delete_classifiers_removes_classifier_from_all_sets_when_numerosity_one(monkeypatch):
    cl = FakeClassifier(condition=["x"], num=1, q=1.0, tav=1, name="a")
    population = FakePopulation([cl])
    match_set = FakePopulation([cl])
    action_set = FakePopulation([cl])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=1)

    assert population == []
    assert match_set == []
    assert action_set == []


def test_is_preferred_to_delete_prefers_lower_q():
    cl_del = FakeClassifier(q=1.0, tav=1, marked=False)
    cl = FakeClassifier(q=0.8, tav=10, marked=False)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_uses_marking_and_tav_tiebreakers():
    cl_del = FakeClassifier(q=1.0, tav=1, marked=False)
    cl = FakeClassifier(q=1.0, tav=5, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True

    cl2 = FakeClassifier(q=1.0, tav=10, marked=False)
    cl_del2 = FakeClassifier(q=1.0, tav=3, marked=False)
    assert ga._is_preferred_to_delete(cl_del2, cl2) is True


def test_find_old_classifier_prefers_subsumers(monkeypatch):
    cl = FakeClassifier(condition=["x"], name="target")
    subsumer = FakeClassifier(condition=["x"], name="subsumer")

    monkeypatch.setattr(ga, "find_subsumers", lambda *_args, **_kwargs: [subsumer])
    monkeypatch.setattr(ga, "_find_similar", lambda *_args, **_kwargs: None)

    result = ga._find_old_classifier([subsumer], cl, use_subsumption=True, theta_exp=5)
    assert result is subsumer


def test_find_old_classifier_falls_back_to_similar_when_no_subsumers(monkeypatch):
    cl = FakeClassifier(condition=["x"], name="target")
    similar = FakeClassifier(condition=["x"], name="target")

    monkeypatch.setattr(ga, "find_subsumers", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(ga, "_find_similar", lambda *_args, **_kwargs: similar)

    result = ga._find_old_classifier([similar], cl, use_subsumption=True, theta_exp=5)
    assert result is similar


def test_find_similar_returns_first_equal_classifier():
    target = FakeClassifier(condition=["x", "y"], name="same")
    population = [
        FakeClassifier(condition=["a", "b"], name="other"),
        FakeClassifier(condition=["x", "y"], name="same"),
        FakeClassifier(condition=["x", "y"], name="same"),
    ]

    result = ga._find_similar(target, population)
    assert result is population[1]
