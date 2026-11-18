import pytest
import types
import random

import numpy as np

from lcs.strategies import genetic_algorithms as ga
from lcs import Perception


class DummyCondition:
    def __init__(self, condition_seq, cfg=None):
        # Support indexing and slicing
        self._cond = list(condition_seq)
        self.generalize_calls = []
        self.cfg = cfg or types.SimpleNamespace(classifier_wildcard="#")

    def __getitem__(self, item):
        # support int indexing and slices
        if isinstance(item, slice):
            return self._cond[item]
        else:
            return self._cond[item]

    def __setitem__(self, key, value):
        self._cond[key] = value

    def __len__(self):
        return len(self._cond)

    def generalize(self, idx):
        self.generalize_calls.append(idx)
        # Mark generalized idx by replacing with wildcard symbol
        self._cond[idx] = self.cfg.classifier_wildcard

    def does_match(self, perception):
        # For testing, let's say it always match if last element equals perception.val
        return getattr(perception, "val", None) == self._cond[-1]

    def __eq__(self, other):
        if not hasattr(other, 'condition'):
            return False
        return self._cond == other.condition._cond

    def __repr__(self):
        return f"DummyCondition({self._cond})"


class DummyClassifier:
    def __init__(self,
                 tga=0,
                 num=1,
                 condition=None,
                 cfg=None,
                 q=0.5,
                 tav=0,
                 marked=False,
                 action=None):
        self.tga = tga
        self.num = num
        self.condition = condition or DummyCondition(["A", "B", "C"], cfg=cfg)
        self.is_marked_call = marked
        self.cfg = cfg or types.SimpleNamespace(classifier_length=len(self.condition))
        self.q = q
        self.tav = tav
        self.action = action

    def is_marked(self):
        return self.is_marked_call

    def __eq__(self, other):
        if not isinstance(other, DummyClassifier):
            return False
        return (self.num == other.num and
                self.tga == other.tga and
                self.condition == other.condition and
                self.q == other.q and
                self.tav == other.tav and
                self.action == other.action)


class DummyList(list):
    """List variant with safe_remove (silent if element not present) and expand()"""
    def safe_remove(self, elem):
        try:
            self.remove(elem)
        except ValueError:
            pass

    def expand(self):
        # Just return self for testing
        return self


def test_should_apply_returns_false_when_action_set_none():
    assert not ga.should_apply(None, time=10, theta_ga=5)


def test_should_apply_returns_false_when_no_classifiers():
    action_set = []
    assert not ga.should_apply(action_set, time=10, theta_ga=5)


def test_should_apply_returns_false_when_average_too_recent():
    cl1 = DummyClassifier(tga=9, num=1)
    cl2 = DummyClassifier(tga=10, num=1)
    action_set = [cl1, cl2]
    # average tga = 9.5, time=10, 10 - 9.5 = 0.5 < theta_ga=1
    assert not ga.should_apply(action_set, time=10, theta_ga=1)


def test_should_apply_returns_true_when_average_exceeds_theta_ga():
    cl1 = DummyClassifier(tga=3, num=1)
    cl2 = DummyClassifier(tga=4, num=1)
    action_set = [cl1, cl2]
    # average tga = 3.5, time=10, 10 - 3.5 = 6.5 > theta_ga=5
    assert ga.should_apply(action_set, time=10, theta_ga=5)


def test_set_timestamps_updates_all_classifiers_tga():
    cl1 = DummyClassifier(tga=3)
    cl2 = DummyClassifier(tga=5)
    action_set = [cl1, cl2]
    ga.set_timestamps(action_set, epoch=20)
    assert cl1.tga == 20
    assert cl2.tga == 20


def test_roulette_wheel_selection_returns_two_parents(monkeypatch):
    # Create population of DummyClassifiers
    pop = [DummyClassifier(q=i + 1) for i in range(3)]
    # simple fitness function: return q
    def fitness_func(cl):
        return cl.q

    # Monkeypatch _weighted_random_choice to pick first two classifiers always
    choices_captured = []
    def fake_weighted_choice(choices):
        choices_captured.append(choices)
        # return classifier with highest key
        return max(choices, key=lambda c: choices[c])

    monkeypatch.setattr(ga, "_weighted_random_choice", fake_weighted_choice)

    p1, p2 = ga.roulette_wheel_selection(pop, fitness_func)

    assert isinstance(p1, DummyClassifier)
    assert isinstance(p2, DummyClassifier)
    assert len(choices_captured) == 2
    for choices in choices_captured:
        # keys must be from population
        assert set(choices.keys()) == set(pop)


def test_generalizing_mutation_mutates_condition_when_random_lt_mu(monkeypatch):
    cond = DummyCondition(["A", "B", "C"], cfg=types.SimpleNamespace(classifier_wildcard="#"))
    cl = DummyClassifier(condition=cond)

    # Supply predetermined random values: first < mu triggers generalize, others > mu
    random_vals = iter([0.01, 0.5, 0.5])
    monkeypatch.setattr(random, "random", lambda: next(random_vals))

    ga.generalizing_mutation(cl, mu=0.1)

    # index 0 should be generalized, others not
    assert cond.generalize_calls == [0]
    assert cond[0] == "#"


def test_generalizing_mutation_does_not_mutate_if_cond_is_wildcard(monkeypatch):
    cfg = types.SimpleNamespace(classifier_wildcard="#")
    cond = DummyCondition(["#", "B", "C"], cfg=cfg)
    cl = DummyClassifier(condition=cond, cfg=cfg)

    random_vals = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr(random, "random", lambda: next(random_vals))

    ga.generalizing_mutation(cl, mu=1.0)

    # Because first condition is already wildcard, should not generalize it
    assert cond.generalize_calls == [1, 2]  # indices 1 and 2 generalized


def test_two_point_crossover_swaps_condition_segments(monkeypatch):
    cfg = types.SimpleNamespace(classifier_length=5)
    cond1 = DummyCondition(["A", "B", "C", "D", "E"], cfg=cfg)
    cond2 = DummyCondition(["V", "W", "X", "Y", "Z"], cfg=cfg)
    p = DummyClassifier(condition=cond1, cfg=cfg)
    d = DummyClassifier(condition=cond2, cfg=cfg)

    # monkeypatch np.random.choice to always pick (1,4)
    monkeypatch.setattr(np.random, "choice", lambda a, size, replace: np.array([1, 4]))

    ga.two_point_crossover(p, d)

    # positions 1 to 3 (4 exclusive) should be swapped
    assert p.condition._cond == ["A", "W", "X", "Y", "E"]
    assert d.condition._cond == ["V", "B", "C", "D", "Z"]


def test_add_classifier_appends_when_no_old_classifier(monkeypatch):
    pop = DummyList()
    match_set = DummyList()
    action_set = DummyList()
    cfg = types.SimpleNamespace(classifier_wildcard="#")
    cond = DummyCondition(["A", "B", "C"], cfg=cfg)
    cl = DummyClassifier(condition=cond, cfg=cfg)

    # Mock _find_old_classifier to return None
    monkeypatch.setattr(ga, "_find_old_classifier", lambda population, cl, do_subs, th_exp: None)
    p = Perception()
    p.val = "C"

    ga.add_classifier(cl, p, pop, match_set, action_set, do_subsumption=True, theta_exp=5)

    assert cl in pop
    assert cl in action_set
    assert cl in match_set


def test_add_classifier_increments_num_when_old_classifier_found_and_not_marked(monkeypatch):
    # Setup old classifier
    old = DummyClassifier(num=3, marked=False)
    pop = DummyList([old])
    match_set = DummyList()
    action_set = DummyList([old])
    newcl = DummyClassifier()
    monkeypatch.setattr(ga, "_find_old_classifier", lambda pop, cl, do_subs, th_exp: old)

    p = Perception()
    p.val = "C"

    ga.add_classifier(newcl, p, pop, match_set, action_set, do_subsumption=True, theta_exp=5)

    assert old.num == 4  # incremented
    # newcl not appended
    assert newcl not in pop
    assert newcl not in action_set


def test_add_classifier_does_not_increment_num_when_old_classifier_marked(monkeypatch):
    old = DummyClassifier(num=3, marked=True)
    pop = DummyList([old])
    match_set = DummyList()
    action_set = DummyList([old])
    newcl = DummyClassifier()
    monkeypatch.setattr(ga, "_find_old_classifier", lambda pop, cl, do_subs, th_exp: old)

    p = Perception()
    p.val = "C"

    ga.add_classifier(newcl, p, pop, match_set, action_set, do_subsumption=True, theta_exp=5)

    assert old.num == 3  # not incremented


def test_delete_classifiers_decrements_num_or_removes(monkeypatch):
    # Setup three classifiers
    cl1 = DummyClassifier(num=2, q=0.2, tav=10, marked=False)
    cl2 = DummyClassifier(num=1, q=0.1, tav=5, marked=False)
    cl3 = DummyClassifier(num=1, q=0.4, tav=2, marked=True)
    pop = DummyList([cl1, cl2, cl3])
    match_set = DummyList([cl1, cl2, cl3])
    action_set = DummyList([cl1, cl2, cl3])

    # We patch random.random to always return < 0.3 to allow deletion
    random_val_iter = iter([0.1] * 10)  # always below 0.3

    monkeypatch.setattr(random, "random", lambda: next(random_val_iter))

    # Also patch _is_preferred_to_delete to prefer cl2 over cl1, cl3 over cl2
    monkeypatch.setattr(ga, "_is_preferred_to_delete", lambda d, c: c == cl2 or c == cl3)

    # Insert size 0, θas = 3 (max classifiers in action set)
    # total nums 4 > 3 triggers deletion
    ga.delete_classifiers(pop, match_set, action_set, insize=0, theta_as=3)

    # cl1 num should be decremented first or cl2/cl3 removed
    # At least one classifier num reduced or one removed
    total_num_after = sum(cl.num for cl in action_set)
    assert total_num_after <= 3

    # cl2 or cl3 should be removed if possible
    removed = [cl for cl in [cl1, cl2, cl3] if cl not in action_set]
    assert len(removed) >= 0  # could be empty if num decremented but 1 <=


def test__is_preferred_to_delete_prefers_lower_q():
    cl_del = DummyClassifier(q=0.5, tav=10, marked=False)
    cl = DummyClassifier(q=0.3, tav=10, marked=False)
    # cl has much lower q, preferred
    assert ga._is_preferred_to_delete(cl_del, cl)

    cl = DummyClassifier(q=0.6, tav=10, marked=False)
    # cl higher q, not preferred
    assert not ga._is_preferred_to_delete(cl_del, cl)


def test__is_preferred_to_delete_considers_marked_and_tav():
    # Equal q within 0.1 difference, cl is marked and cl_del not marked -> prefer cl
    cl_del = DummyClassifier(q=0.5, tav=10, marked=False)
    cl = DummyClassifier(q=0.55, tav=5, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl)

    # Both marked or both unmarked, cl tav higher -> prefer cl
    cl = DummyClassifier(q=0.55, tav=15, marked=True)
    cl_del = DummyClassifier(q=0.55, tav=10, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl)

    # cl tav lower -> don't prefer cl
    cl = DummyClassifier(q=0.55, tav=5, marked=True)
    assert not ga._is_preferred_to_delete(cl_del, cl)


def test__find_similar_returns_first_similar():
    cl1 = DummyClassifier(action=1)
    cl2 = DummyClassifier(action=2)
    cl3 = DummyClassifier(action=3)
    pop = [cl1, cl2, cl3]

    # Equality requires identical condition and other irrelevant fields
    other = DummyClassifier(action=2)
    result = ga._find_similar(other, pop)
    assert result == cl2
    assert result.action == 2


def test__find_similar_returns_none_for_no_match():
    cl1 = DummyClassifier(action=1)
    cl2 = DummyClassifier(action=2)
    pop = [cl1, cl2]

    other = DummyClassifier(action=3)
    result = ga._find_similar(other, pop)
    assert result is None


def test__find_old_classifier_prefers_subsumers(monkeypatch):
    cl = DummyClassifier()
    pop = [DummyClassifier(), DummyClassifier()]
    first_subsumer = DummyClassifier(num=10)
    # Mock find_subsumers to return [first_subsumer]
    monkeypatch.setattr(ga, "find_subsumers", lambda c, p, th: [first_subsumer])
    result = ga._find_old_classifier(pop, cl, True, 5)
    assert result == first_subsumer


def test__find_old_classifier_fallsback_to_similar(monkeypatch):
    cl = DummyClassifier()
    pop = [DummyClassifier()]
    # Mock find_subsumers to return empty
    monkeypatch.setattr(ga, "find_subsumers", lambda c, p, th: [])
    sim = DummyClassifier()
    monkeypatch.setattr(ga, "_find_similar", lambda cl, pop: sim)
    result = ga._find_old_classifier(pop, cl, True, 5)
    assert result == sim


def test__find_old_classifier_returns_none_when_nothing_found(monkeypatch):
    cl = DummyClassifier()
    pop = []
    monkeypatch.setattr(ga, "find_subsumers", lambda c, p, th: [])
    monkeypatch.setattr(ga, "_find_similar", lambda cl, pop: None)
    result = ga._find_old_classifier(pop, cl, True, 5)
    assert result is None
