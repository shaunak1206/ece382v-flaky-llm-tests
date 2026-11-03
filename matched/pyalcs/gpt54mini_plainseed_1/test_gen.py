import types

import pytest

import lcs.strategies.genetic_algorithms as ga


class FakeCfg:
    def __init__(self, classifier_wildcard="*"):
        self.classifier_wildcard = classifier_wildcard
        self.classifier_length = 4


class FakeCondition:
    def __init__(self, values, match_result=True):
        self._values = list(values)
        self.generalized = []
        self._match_result = match_result

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, item):
        return self._values[item]

    def __setitem__(self, idx, value):
        self._values[idx] = value

    def __len__(self):
        return len(self._values)

    def generalize(self, idx):
        self.generalized.append(idx)
        self._values[idx] = None

    def does_match(self, perception):
        return self._match_result

    def __repr__(self):
        return f"FakeCondition({self._values!r})"


class FakeClassifier:
    def __init__(self, condition, q=0.0, tga=0, num=1, tav=0, marked=False, cfg=None, name=None):
        self.condition = condition
        self.q = q
        self.tga = tga
        self.num = num
        self.tav = tav
        self._marked = marked
        self.cfg = cfg or FakeCfg()
        self.name = name or f"cl-{id(self)}"

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        return isinstance(other, FakeClassifier) and (
            self.condition._values == other.condition._values
            and self.q == other.q
            and self.tga == other.tga
            and self.num == other.num
            and self.tav == other.tav
            and self._marked == other._marked
        )

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return self.name


class ExpandableList(list):
    def expand(self):
        return list(self)

    def safe_remove(self, item):
        self.remove(item)


def test_should_apply_returns_false_for_none_action_set():
    assert ga.should_apply(None, time=10, theta_ga=5) is False


def test_should_apply_returns_false_for_zero_numerosity():
    cl = FakeClassifier(FakeCondition([1, 2]), tga=1, num=0)
    assert ga.should_apply([cl], time=10, theta_ga=1) is False


def test_should_apply_uses_weighted_average_timestamp():
    cl1 = FakeClassifier(FakeCondition([1]), tga=0, num=1)
    cl2 = FakeClassifier(FakeCondition([2]), tga=10, num=3)
    # average tga = 7.5, time - avg = 2.5
    assert ga.should_apply([cl1, cl2], time=10, theta_ga=2) is True
    assert ga.should_apply([cl1, cl2], time=10, theta_ga=3) is False


def test_set_timestamps_updates_all_classifiers():
    cl1 = FakeClassifier(FakeCondition([1]), tga=0)
    cl2 = FakeClassifier(FakeCondition([2]), tga=5)
    ga.set_timestamps([cl1, cl2], epoch=42)
    assert cl1.tga == 42
    assert cl2.tga == 42


def test_roulette_wheel_selection_returns_two_parents(monkeypatch):
    population = [FakeClassifier(FakeCondition([1]), name="a"), FakeClassifier(FakeCondition([2]), name="b")]

    def fake_choice(choices):
        # return the first key deterministically
        return next(iter(choices))

    monkeypatch.setattr(ga, "_weighted_random_choice", fake_choice)
    p1, p2 = ga.roulette_wheel_selection(population, fitnessfunc=lambda cl: 1.0)
    assert p1 is population[0]
    assert p2 is population[0]


def test_generalizing_mutation_generalizes_only_non_wildcards(monkeypatch):
    cl = FakeClassifier(FakeCondition([1, "*", 3], match_result=True), cfg=FakeCfg(classifier_wildcard="*"))
    values = iter([0.0, 0.0, 0.9])  # mutate first two only, third not reached for generalization
    monkeypatch.setattr(ga.random, "random", lambda: next(values))
    ga.generalizing_mutation(cl, mu=0.5)
    assert cl.condition.generalized == [0]
    assert cl.condition._values[0] is None
    assert cl.condition._values[1] == "*"
    assert cl.condition._values[2] == 3


def test_two_point_crossover_swaps_selected_slice(monkeypatch):
    parent = FakeClassifier(FakeCondition([1, 2, 3, 4, 5]), cfg=FakeCfg())
    donor = FakeClassifier(FakeCondition([9, 8, 7, 6, 5]), cfg=FakeCfg())
    monkeypatch.setattr(ga.np.random, "choice", lambda *args, **kwargs: [1, 4])
    ga.two_point_crossover(parent, donor)
    assert parent.condition._values == [1, 8, 7, 6, 5]
    assert donor.condition._values == [9, 2, 3, 4, 5]


def test_add_classifier_appends_to_population_match_and_action_set_when_new():
    cl = FakeClassifier(FakeCondition([1, 2], match_result=True), name="new")
    population = []
    match_set = []
    action_set = []
    p = types.SimpleNamespace()
    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=0)
    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_does_not_append_to_match_set_when_not_matching():
    cl = FakeClassifier(FakeCondition([1, 2], match_result=False), name="new")
    population = []
    match_set = []
    action_set = []
    p = types.SimpleNamespace()
    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=0)
    assert population == [cl]
    assert action_set == [cl]
    assert match_set == []


def test_add_classifier_increases_numerosity_for_existing_classifier(monkeypatch):
    old = FakeClassifier(FakeCondition([1, 2]), num=2, name="old")
    cl = FakeClassifier(FakeCondition([1, 2]), num=1, name="new")
    population = [old]
    match_set = []
    action_set = [old]

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old)
    ga.add_classifier(cl, types.SimpleNamespace(), population, match_set, action_set, do_subsumption=False, theta_exp=0)
    assert old.num == 3
    assert population == [old]
    assert action_set == [old]
    assert match_set == []


def test_add_classifier_does_not_increase_numerosity_when_old_classifier_marked(monkeypatch):
    old = FakeClassifier(FakeCondition([1, 2]), num=2, marked=True, name="old")
    cl = FakeClassifier(FakeCondition([1, 2]), num=1, name="new")
    population = [old]
    match_set = []
    action_set = [old]

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old)
    ga.add_classifier(cl, types.SimpleNamespace(), population, match_set, action_set, do_subsumption=False, theta_exp=0)
    assert old.num == 2


def test_find_old_classifier_prefers_subsumer_when_available(monkeypatch):
    cl = FakeClassifier(FakeCondition([1, 2]), name="new")
    subsumer = FakeClassifier(FakeCondition([0, 0]), name="subsumer")
    similar = FakeClassifier(FakeCondition([1, 2]), name="similar")

    monkeypatch.setattr(ga, "find_subsumers", lambda *args, **kwargs: [subsumer])
    assert ga._find_old_classifier([similar, subsumer], cl, use_subsumption=True, theta_exp=1) is subsumer


def test_find_old_classifier_falls_back_to_similar_when_no_subsumer(monkeypatch):
    cl = FakeClassifier(FakeCondition([1, 2]), name="new")
    similar = FakeClassifier(FakeCondition([1, 2]), name="similar")
    monkeypatch.setattr(ga, "find_subsumers", lambda *args, **kwargs: [])
    assert ga._find_old_classifier([similar], cl, use_subsumption=True, theta_exp=1) is similar


def test_is_preferred_to_delete_prefers_lower_q():
    cl_del = FakeClassifier(FakeCondition([1]), q=0.8, tav=1)
    cl = FakeClassifier(FakeCondition([2]), q=0.6, tav=10)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_tie_uses_marking_and_tav():
    cl_del = FakeClassifier(FakeCondition([1]), q=1.0, tav=1, marked=False)
    cl = FakeClassifier(FakeCondition([2]), q=1.0, tav=5, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_delete_classifiers_removes_item_when_numerosity_is_one(monkeypatch):
    target = FakeClassifier(FakeCondition([1]), q=0.1, num=1, tav=0, name="target")
    other = FakeClassifier(FakeCondition([2]), q=0.9, num=1, tav=0, name="other")
    population = ExpandableList([target, other])
    match_set = ExpandableList([target, other])
    action_set = ExpandableList([target, other])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=1)

    assert target not in population
    assert target not in match_set
    assert target not in action_set


def test_delete_classifiers_decrements_numerosity_when_greater_than_one(monkeypatch):
    cl = FakeClassifier(FakeCondition([1]), q=0.1, num=2, tav=0, name="target")
    population = ExpandableList([cl])
    match_set = ExpandableList([cl])
    action_set = ExpandableList([cl])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=1)

    assert cl.num == 1
    assert cl in population
    assert cl in match_set
    assert cl in action_set
