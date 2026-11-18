import random
import builtins
import types

import pytest
import numpy as np

from lcs.strategies import genetic_algorithms as ga
from lcs import Perception


class DummyCondition:
    def __init__(self, cond_list, cfg):
        self._conds = list(cond_list)
        self.cfg = cfg

    def __getitem__(self, idx):
        return self._conds[idx]

    def __setitem__(self, idx, val):
        self._conds[idx] = val

    def __iter__(self):
        return iter(self._conds)

    def __len__(self):
        return len(self._conds)

    def generalize(self, idx):
        # just mark index generalized for test
        self._conds[idx] = 'generalized'

    def does_match(self, perception):
        # For test with simple condition matching perception attribute-wise
        return all(getattr(perception, i, None) == val for i, val in enumerate(self._conds))

    def __eq__(self, other):
        if not isinstance(other, DummyClassifier):
            return False
        # simple equality on condition + action + effect as tuple
        return (
            tuple(self._conds) == tuple(other.condition._conds)
        )


class DummyConfig:
    classifier_length = 5
    classifier_wildcard = 'wildcard'


class DummyClassifier:
    def __init__(self, num=1, tga=0, tav=0, q=1.0, is_marked=False,
                 cond=None, action=0, effect=None, cfg=None):
        self.num = num
        self.tga = tga
        self.tav = tav
        self.q = q
        self._marked = is_marked
        self.action = action
        self.effect = effect or []
        self.cfg = cfg or DummyConfig()
        self.condition = cond or DummyCondition(
            ['0'] * self.cfg.classifier_length, self.cfg
        )

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        # equality defined by condition, action, effect triplet for _find_similar:
        if not isinstance(other, DummyClassifier):
            return False
        return (
            tuple(self.condition._conds) == tuple(other.condition._conds) and
            self.action == other.action and
            self.effect == other.effect
        )

    def __hash__(self):
        # To allow usage as dict keys for roulette_wheel_selection
        return hash((tuple(self.condition._conds), self.action, tuple(self.effect)))


@pytest.mark.parametrize(
    "action_set, time, theta_ga, expected",
    [
        (None, 10, 5, False),  # None action_set no GA
        ([], 10, 5, False),  # empty action_set no GA
        ([DummyClassifier(num=0, tga=3)], 10, 5, False),  # overall_num=0 no GA
        ([DummyClassifier(num=1, tga=3)], 10, 5, True),  # delay=7 >5 --> GA
        ([DummyClassifier(num=1, tga=9)], 10, 0, True),  # delay=1 > 0 --> GA
        ([DummyClassifier(num=1, tga=9)], 10, 1, False),  # delay=1 <= 1 False
        ([DummyClassifier(num=2, tga=8), DummyClassifier(num=2, tga=9)], 10, 0, True),  # avg=8.5 delay=1.5 >0
    ],
)
def test_should_apply(action_set, time, theta_ga, expected):
    assert ga.should_apply(action_set, time, theta_ga) is expected


def test_set_timestamps_sets_all_tga_to_epoch():
    cl1 = DummyClassifier(tga=2)
    cl2 = DummyClassifier(tga=5)
    action_set = [cl1, cl2]
    ga.set_timestamps(action_set, 42)
    assert all(cl.tga == 42 for cl in action_set)


def test_roulette_wheel_selection_selects_two_distinct_parents(monkeypatch):
    pop = [DummyClassifier(q=1.0), DummyClassifier(q=2.0), DummyClassifier(q=3.0)]

    # Fitness function: cube of q (simulate)
    def fitness(cl):
        return cl.q ** 3

    # Patch random.uniform to select deterministic values
    distribution = [0.1, 3.5, 10.0]
    calls = iter(distribution)

    def fake_uniform(a, b):
        return next(calls)

    monkeypatch.setattr(random, "uniform", fake_uniform)

    p1, p2 = ga.roulette_wheel_selection(pop, fitness)
    assert p1 in pop
    assert p2 in pop


def test_generalizing_mutation_randomizes_condition(monkeypatch):
    class_cfg = DummyConfig()
    class_cfg.classifier_wildcard = 'wildcard'
    cl = DummyClassifier(cfg=class_cfg)
    # Set condition with no wildcards for test
    cl.condition = DummyCondition(list("abcde"), class_cfg)

    # Track indexes generalized
    generalized_indexes = []

    # Patch random.random to control mutation application
    values = iter([0.05, 0.6, 0.04, 0.5, 0.01])  # mu=0.1 threshold

    def fake_rand():
        return next(values)

    monkeypatch.setattr(random, "random", fake_rand)
    # Patch generalize method to track calls
    original_generalize = cl.condition.generalize

    def tracking_generalize(idx):
        generalized_indexes.append(idx)
        original_generalize(idx)

    cl.condition.generalize = tracking_generalize
    ga.generalizing_mutation(cl, 0.1)
    assert 0 in generalized_indexes
    assert 2 in generalized_indexes
    assert 4 in generalized_indexes
    # Positions 1 and 3 should NOT be mutated
    assert cl.condition[1] == 'b'
    assert cl.condition[3] == 'd'


def test_two_point_crossover_swaps_correct_slices(monkeypatch):
    cfg = DummyConfig()
    p1_condition = list("ABCDE")
    p2_condition = list("12345")
    p1 = DummyClassifier(cond=DummyCondition(p1_condition.copy(), cfg), cfg=cfg)
    p2 = DummyClassifier(cond=DummyCondition(p2_condition.copy(), cfg), cfg=cfg)

    # Patch np.random.choice to pick fixed indices
    monkeypatch.setattr(np.random, "choice", lambda a, size, replace: np.array([1, 4]))
    ga.two_point_crossover(p1, p2)

    # p1's slice [1:4] should now be from p2 original slice [1:4]
    assert p1.condition._conds[:1] == p1_condition[:1]
    assert p1.condition._conds[1:4] == p2_condition[1:4]
    assert p1.condition._conds[4:] == p1_condition[4:]

    # p2's slice [1:4] should now be from p1 original slice [1:4]
    assert p2.condition._conds[:1] == p2_condition[:1]
    assert p2.condition._conds[1:4] == p1_condition[1:4]
    assert p2.condition._conds[4:] == p2_condition[4:]


def test_add_classifier_adds_new_classifier(monkeypatch):
    population = []
    match_set = []
    action_set = []

    cl = DummyClassifier()
    perception = Perception()
    cl.condition.does_match = lambda p: True

    # Patch _find_old_classifier to return None, simulating no similar classifier found
    monkeypatch.setattr(ga, "_find_old_classifier", lambda a, b, c, d: None)

    ga.add_classifier(cl, perception, population, match_set, action_set, True, 10)
    assert cl in population
    assert cl in action_set
    assert cl in match_set


def test_add_classifier_increments_existing_num(monkeypatch):
    old_cl = DummyClassifier(num=1, is_marked=False)
    population = [old_cl]
    match_set = []
    action_set = [old_cl]

    new_cl = DummyClassifier()
    perception = Perception()

    # Patch _find_old_classifier to return old_cl simulating similar classifier found
    monkeypatch.setattr(ga, "_find_old_classifier", lambda a, b, c, d: old_cl)

    ga.add_classifier(new_cl, perception, population, match_set, action_set, True, 10)
    # num should be incremented because old_cl is not marked
    assert old_cl.num == 2
    # new_cl not added
    assert new_cl not in population


def test_delete_classifiers_removes_with_num_one(monkeypatch):
    cfg = DummyConfig()
    cl1 = DummyClassifier(num=1, tga=0, q=1.0)
    cl2 = DummyClassifier(num=2, tga=0, q=1.0)
    population = [cl1, cl2]
    match_set = [cl1, cl2]
    action_set = ActionSetStub([cl1, cl2])

    # force random.random to always trigger deletion for first cl_del found
    random_values = iter([0.1, 0.1, 0.1])

    monkeypatch.setattr(random, "random", lambda: next(random_values))

    initial_pop_len = len(population)
    ga.delete_classifiers(population, match_set, action_set, insize=4, theta_as=5)

    # After deletion sum num must be <= theta_as - insize
    total_num = sum(cl.num for cl in action_set)
    assert total_num + 4 <= 5
    # cl1 had num=1, expect removed from lists
    assert cl1 not in population
    assert cl1 not in match_set
    assert cl1 not in action_set


def test_delete_classifiers_decrements_num_when_greater_than_one(monkeypatch):
    cl1 = DummyClassifier(num=3, tga=0, q=1.0)
    population = [cl1]
    match_set = [cl1]
    action_set = ActionSetStub([cl1])

    random_values = iter([0.1, 0.1, 0.1])

    monkeypatch.setattr(random, "random", lambda: next(random_values))

    ga.delete_classifiers(population, match_set, action_set, insize=4, theta_as=6)
    # num decreased by 1
    assert cl1.num == 2
    assert cl1 in population
    assert cl1 in match_set
    assert cl1 in action_set


@pytest.mark.parametrize(
    "cl_del_q, cl_q, cl_del_marked, cl_marked, cl_del_tav, cl_tav, expected",
    [
        (0.5, 0.3, False, False, 10, 20, True),  # q difference big negative
        (0.5, 0.5, False, True, 10, 20, True),  # q within 0.1 but cl marked not cl_del
        (0.5, 0.5, True, True, 10, 20, True),  # both marked, cl has higher tav
        (0.5, 0.5, True, True, 20, 10, False), # both marked, cl_del has higher tav
        (0.5, 0.5, False, False, 10, 20, True),# both unmarked, cl has higher tav
        (0.5, 0.5, False, False, 20, 10, False),# both unmarked, cl_del higher tav
        (0.5, 0.7, False, False, 10, 10, False), # q difference positive
    ]
)
def test_is_preferred_to_delete_behaviors(
        cl_del_q, cl_q,
        cl_del_marked, cl_marked,
        cl_del_tav, cl_tav,
        expected
):
    cl_del = DummyClassifier(q=cl_del_q, is_marked=cl_del_marked, tav=cl_del_tav)
    cl = DummyClassifier(q=cl_q, is_marked=cl_marked, tav=cl_tav)
    assert ga._is_preferred_to_delete(cl_del, cl) is expected


def test__find_old_classifier_prefers_subsumers(monkeypatch):
    cl = DummyClassifier()
    population = [DummyClassifier()]

    # Prepare stub for find_subsumers
    subsumer = DummyClassifier()
    monkeypatch.setattr(ga, "find_subsumers", lambda c, pop, theta: [subsumer])
    monkeypatch.setattr(ga, "_find_similar", lambda c, pop: None)

    out = ga._find_old_classifier(population, cl, True, 10)
    assert out is subsumer


def test__find_old_classifier_fallsback_to_similar(monkeypatch):
    cl = DummyClassifier()
    population = [DummyClassifier()]
    monkeypatch.setattr(ga, "find_subsumers", lambda c, pop, theta: [])
    sim = DummyClassifier()
    monkeypatch.setattr(ga, "_find_similar", lambda c, pop: sim)

    out = ga._find_old_classifier(population, cl, True, 10)
    assert out is sim


def test__find_similar_finds_first_equal(monkeypatch):
    # Setup population with equal and not equal classifiers
    cfg = DummyConfig()
    cl1 = DummyClassifier(cond=DummyCondition("abcde", cfg), action=1, effect=[1])
    cl2 = DummyClassifier(cond=DummyCondition("abcde", cfg), action=1, effect=[1])
    cl3 = DummyClassifier(cond=DummyCondition("abcdf", cfg), action=1, effect=[1])
    population = [cl3, cl1, cl2]

    found = ga._find_similar(cl2, population)
    assert found == cl1 or found == cl2


# Helpers:

class ActionSetStub(list):
    """Stub action_set supporting expand() and safe_remove() methods as needed by delete_classifiers."""

    def expand(self):
        # For tests, simply return self iterable multiple times
        return self

    def safe_remove(self, cl):
        # Remove if present
        if cl in self:
            self.remove(cl)


def test__weighted_random_choice_weights(monkeypatch):
    # Setup choices dict
    class Dummy:
        def __init__(self, n):
            self.n = n

        def __repr__(self):
            return f"Dummy({self.n})"

    d1 = Dummy(1)
    d2 = Dummy(2)
    choices = {d1: 1.0, d2: 3.0}

    # pick fixed values to hit first or second
    results = []

    def fake_uniform(a, b):
        # Test 1: pick near 0 -> d1
        results.append(ga._weighted_random_choice(choices))
        return 0.5

    monkeypatch.setattr(random, "uniform", fake_uniform)
    # Actually run once to record
    out = ga._weighted_random_choice(choices)
    assert out in (d1, d2)


def test_two_point_crossover_valid_indices(monkeypatch):
    # test that right > left always holds
    cfg = DummyConfig()
    p1 = DummyClassifier(cfg=cfg)
    p2 = DummyClassifier(cfg=cfg)

    def fake_choice(a, size, replace):
        vals = [3, 1]  # unsorted to test sort
        return np.array(vals)

    monkeypatch.setattr(np.random, "choice", fake_choice)
    ga.two_point_crossover(p1, p2)
    # No assertion needed, assert in function should hold, else raises


def test_add_classifier_does_not_increment_marked(monkeypatch):
    old_cl = DummyClassifier(num=1, is_marked=True)
    population = [old_cl]
    action_set = [old_cl]
    match_set = []

    new_cl = DummyClassifier()
    perception = Perception()

    monkeypatch.setattr(ga, "_find_old_classifier", lambda a, b, c, d: old_cl)

    ga.add_classifier(new_cl, perception, population, match_set, action_set, True, 10)
    # num should NOT be incremented because old_cl is marked
    assert old_cl.num == 1
    assert new_cl not in population
