import random

import pytest

import lcs.strategies.genetic_algorithms as ga


class DummyCfg:
    def __init__(self, classifier_length=4, classifier_wildcard="#"):
        self.classifier_length = classifier_length
        self.classifier_wildcard = classifier_wildcard


class DummyCondition:
    def __init__(self, values, wildcard="#"):
        self.values = list(values)
        self.wildcard = wildcard
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
        self.values[idx] = self.wildcard

    def does_match(self, perception):
        return True


class DummyClassifier:
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
        self.condition = condition if condition is not None else DummyCondition([0, 1, 2, 3])
        self.q = q
        self.tav = tav
        self.tga = tga
        self.num = num
        self._marked = marked
        self.cfg = cfg if cfg is not None else DummyCfg(classifier_length=len(self.condition))
        self.name = name or "cl"

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return (
            isinstance(other, DummyClassifier)
            and self.condition.values == other.condition.values
            and self.name == other.name
        )

    def __hash__(self):
        return id(self)


class DummyList(list):
    def safe_remove(self, item):
        self.remove(item)

    def expand(self):
        out = []
        for cl in self:
            out.extend([cl] * cl.num)
        return out


def test_should_apply_returns_false_for_none_action_set():
    assert ga.should_apply(None, time=10, theta_ga=3) is False


def test_should_apply_returns_false_for_empty_numerosity():
    action_set = [DummyClassifier(tga=5, num=0)]
    assert ga.should_apply(action_set, time=20, theta_ga=1) is False


def test_should_apply_true_when_average_delay_exceeds_threshold():
    action_set = [DummyClassifier(tga=1, num=1), DummyClassifier(tga=2, num=1)]
    assert ga.should_apply(action_set, time=10, theta_ga=5) is True


def test_should_apply_false_when_average_delay_not_exceed_threshold():
    action_set = [DummyClassifier(tga=7, num=1), DummyClassifier(tga=9, num=1)]
    assert ga.should_apply(action_set, time=10, theta_ga=5) is False


def test_set_timestamps_updates_all_classifiers():
    action_set = [DummyClassifier(tga=1), DummyClassifier(tga=2), DummyClassifier(tga=3)]
    ga.set_timestamps(action_set, epoch=42)
    assert [cl.tga for cl in action_set] == [42, 42, 42]


def test_roulette_wheel_selection_uses_weighted_choice(monkeypatch):
    population = [DummyClassifier(name="a"), DummyClassifier(name="b"), DummyClassifier(name="c")]

    monkeypatch.setattr(ga, "_weighted_random_choice", lambda choices: max(choices, key=choices.get))

    parent1, parent2 = ga.roulette_wheel_selection(population, lambda cl: {"a": 1, "b": 3, "c": 2}[cl.name])
    assert parent1.name == "b"
    assert parent2.name == "b"


def test_generalizing_mutation_generalizes_only_selected_non_wildcard_positions(monkeypatch):
    cond = DummyCondition(["a", "#", "b", "c"], wildcard="#")
    cl = DummyClassifier(condition=cond, cfg=DummyCfg(classifier_length=4, classifier_wildcard="#"))

    seq = iter([0.0, 0.9, 0.0, 0.2])
    monkeypatch.setattr(random, "random", lambda: next(seq))

    ga.generalizing_mutation(cl, mu=0.5)
    assert cond.generalized == [0, 2]
    assert cond.values == ["#", "#", "#", "c"]


def test_generalizing_mutation_skips_wildcards():
    cond = DummyCondition(["a", "#", "b"], wildcard="#")
    cl = DummyClassifier(condition=cond, cfg=DummyCfg(classifier_length=3, classifier_wildcard="#"))

    original_random = random.random
    try:
        random.random = lambda: 0.0
        ga.generalizing_mutation(cl, mu=1.0)
    finally:
        random.random = original_random

    assert cond.generalized == [0, 2]
    assert cond.values == ["#", "#", "#"]


def test_two_point_crossover_swaps_selected_slice(monkeypatch):
    parent = DummyClassifier(condition=DummyCondition(["p0", "p1", "p2", "p3"]), cfg=DummyCfg(classifier_length=4))
    donor = DummyClassifier(condition=DummyCondition(["d0", "d1", "d2", "d3"]), cfg=DummyCfg(classifier_length=4))

    monkeypatch.setattr(ga.np.random, "choice", lambda rng, size, replace=False: [1, 3])

    ga.two_point_crossover(parent, donor)
    assert parent.condition.values == ["p0", "d1", "d2", "p3"]
    assert donor.condition.values == ["d0", "p1", "p2", "d3"]


def test_add_classifier_appends_new_classifier_and_match_set_when_no_old_classifier(monkeypatch):
    cl = DummyClassifier(condition=DummyCondition([1, 2, 3]), name="new")
    population = DummyList()
    match_set = DummyList()
    action_set = DummyList()
    p = object()

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: None)

    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=1)
    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_increments_numerosity_for_unmarked_existing_classifier(monkeypatch):
    old = DummyClassifier(condition=DummyCondition([1, 2, 3]), num=2, marked=False, name="old")
    cl = DummyClassifier(condition=DummyCondition([4, 5, 6]), name="new")
    population = DummyList([old])
    match_set = DummyList()
    action_set = DummyList([old])

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old)

    ga.add_classifier(cl, object(), population, match_set, action_set, do_subsumption=False, theta_exp=1)
    assert old.num == 3
    assert population == [old]
    assert action_set == [old]
    assert match_set == []


def test_add_classifier_does_not_increment_marked_existing_classifier(monkeypatch):
    old = DummyClassifier(condition=DummyCondition([1, 2, 3]), num=2, marked=True, name="old")
    cl = DummyClassifier(condition=DummyCondition([4, 5, 6]), name="new")
    population = DummyList([old])
    match_set = DummyList()
    action_set = DummyList([old])

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old)

    ga.add_classifier(cl, object(), population, match_set, action_set, do_subsumption=False, theta_exp=1)
    assert old.num == 2


def test_find_similar_returns_first_equal_classifier():
    target = DummyClassifier(condition=DummyCondition([1, 2]), name="same")
    population = [DummyClassifier(condition=DummyCondition([9, 9]), name="other"), target]
    assert ga._find_similar(target, population) is target


def test_find_old_classifier_prefers_subsumers_when_available(monkeypatch):
    cl = DummyClassifier()
    subsumer = DummyClassifier(name="subsumer")

    monkeypatch.setattr(ga, "find_subsumers", lambda *args, **kwargs: [subsumer])
    monkeypatch.setattr(ga, "_find_similar", lambda *args, **kwargs: None)

    assert ga._find_old_classifier([subsumer], cl, use_subsumption=True, theta_exp=3) is subsumer


def test_find_old_classifier_falls_back_to_similar_when_no_subsumer(monkeypatch):
    cl = DummyClassifier()
    similar = DummyClassifier(name="similar")

    monkeypatch.setattr(ga, "find_subsumers", lambda *args, **kwargs: [])
    monkeypatch.setattr(ga, "_find_similar", lambda other, population: similar)

    assert ga._find_old_classifier([similar], cl, use_subsumption=True, theta_exp=3) is similar


def test_is_preferred_to_delete_prefers_lower_q():
    cl_del = DummyClassifier(q=1.0, tav=10, marked=False)
    cl = DummyClassifier(q=0.8, tav=5, marked=False)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_prefers_marked_when_q_tied():
    cl_del = DummyClassifier(q=1.0, tav=10, marked=False)
    cl = DummyClassifier(q=1.0, tav=5, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_delete_classifiers_removes_one_classifier_when_over_threshold(monkeypatch):
    cl1 = DummyClassifier(num=1, q=1.0, tav=1, marked=False, name="a")
    cl2 = DummyClassifier(num=1, q=2.0, tav=2, marked=False, name="b")
    population = DummyList([cl1, cl2])
    match_set = DummyList([cl1, cl2])
    action_set = DummyList([cl1, cl2])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=1)

    assert len(population) == 1
    assert len(match_set) == 1
    assert len(action_set) == 1


def test_delete_classifiers_decrements_numerosity_instead_of_removing(monkeypatch):
    cl1 = DummyClassifier(num=2, q=1.0, tav=1, marked=False, name="a")
    action_set = DummyList([cl1])
    population = DummyList([cl1])
    match_set = DummyList([cl1])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=1)

    assert cl1.num == 1
    assert population == [cl1]
    assert match_set == [cl1]
    assert action_set == [cl1]
