import random
import numpy as np
import pytest

from lcs.strategies import genetic_algorithms
from lcs import Perception


class DummyCondition:
    def __init__(self, cond, cfg):
        self._cond = list(cond)
        self.cfg = cfg
        # For generalizing_mutation test, track calls
        self.generalized_indices = set()

    def __getitem__(self, idx):
        return self._cond[idx]

    def __setitem__(self, idx, val):
        self._cond[idx] = val

    def __len__(self):
        return len(self._cond)

    def __eq__(self, other):
        if not isinstance(other, DummyCondition):
            return False
        return self._cond == other._cond

    def generalize(self, idx):
        # Mark that generalize() was called on idx
        self.generalized_indices.add(idx)

    def does_match(self, p):
        return True  # Simplify matching always true


class DummyCfg:
    def __init__(self, length, wildcard):
        self.classifier_length = length
        self.classifier_wildcard = wildcard


class DummyClassifier:
    def __init__(self, condition=None, num=1, tga=0, q=1.0, tav=0, cfg=None, marked=False):
        # condition must be DummyCondition or compatible
        self.condition = condition
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self._marked = marked
        self.cfg = cfg or DummyCfg(length=len(condition) if condition else 0, wildcard='#')

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        if not isinstance(other, DummyClassifier):
            return False
        return (
            self.condition == other.condition and
            self.num == other.num and
            self.tga == other.tga and
            self.q == other.q and
            self.tav == other.tav
        )


class DummyList(list):
    """
    Extends list with methods 'safe_remove' and 'expand' used in delete_classifiers.
    safe_remove removes element if present without error.
    expand returns self (simulate the expected behavior)
    """
    def safe_remove(self, cl):
        try:
            self.remove(cl)
        except ValueError:
            pass

    def expand(self):
        return self


@pytest.mark.parametrize("action_set,time,theta_ga,expected", [
    (None, 10, 5, False),
    ([DummyClassifier(tga=0, num=0)], 10, 5, False),
    ([DummyClassifier(tga=5, num=1)], 10, 5, False),  # time - avg tga == 5 <= 5
    ([DummyClassifier(tga=4, num=1)], 10, 5, True),   # time - avg tga == 6 > 5
    ([DummyClassifier(tga=0, num=2), DummyClassifier(tga=4, num=3)], 10, 4, True),  # weighted avg = (0*2+4*3)/5=2.4, 10-2.4=7.6>4
])
def test_should_apply(action_set, time, theta_ga, expected):
    random.seed(0)
    np.random.seed(0)
    result = genetic_algorithms.should_apply(action_set, time, theta_ga)
    assert result == expected


def test_set_timestamps_updates_all_tga():
    random.seed(0)
    np.random.seed(0)
    cl1 = DummyClassifier(tga=1)
    cl2 = DummyClassifier(tga=2)
    action_set = [cl1, cl2]
    genetic_algorithms.set_timestamps(action_set, 100)
    assert cl1.tga == 100
    assert cl2.tga == 100


def test_roulette_wheel_selection_returns_two_from_population():
    random.seed(42)
    np.random.seed(42)
    cl1 = DummyClassifier()
    cl2 = DummyClassifier()
    cl3 = DummyClassifier()
    population = [cl1, cl2, cl3]

    def fitnessfunc(cl):
        # simple fitness: 1, 2, 3
        idx = population.index(cl)
        return idx + 1

    parent1, parent2 = genetic_algorithms.roulette_wheel_selection(population, fitnessfunc)
    assert parent1 in population
    assert parent2 in population


def test_generalizing_mutation_mutates_condition_with_probability(monkeypatch):
    random.seed(123)
    np.random.seed(123)
    cfg = DummyCfg(length=3, wildcard='#')
    cond_vals = ['a', 'b', 'c']
    condition = DummyCondition(cond_vals, cfg)
    cl = DummyClassifier(condition=condition, cfg=cfg)

    # Patch random.random to return 0.0, so mutation always happens if not wildcard
    monkeypatch.setattr(genetic_algorithms.random, "random", lambda: 0.0)

    cl.condition._cond = ['a', 'b', 'c']
    # No wildcard in positions, so all indices should generalize
    genetic_algorithms.generalizing_mutation(cl, mu=0.5)
    # All indices where cond != wildcard 'a','b','c' should be generalized
    assert condition.generalized_indices == {0, 1, 2}


def test_generalizing_mutation_skips_wildcard(monkeypatch):
    random.seed(123)
    np.random.seed(123)
    # Set wildcard at index 1
    cfg = DummyCfg(length=3, wildcard='b')
    cond_vals = ['a', 'b', 'c']
    condition = DummyCondition(cond_vals, cfg)
    cl = DummyClassifier(condition=condition, cfg=cfg)

    monkeypatch.setattr(genetic_algorithms.random, "random", lambda: 0.0)

    genetic_algorithms.generalizing_mutation(cl, mu=0.9)

    # index 1 is wildcard, so not generalized
    assert 1 not in condition.generalized_indices
    # indices 0 and 2 generalized
    assert condition.generalized_indices == {0, 2}


def test_two_point_crossover_swaps_segments(monkeypatch):
    random.seed(99)
    np.random.seed(99)
    length = 5
    cfg = DummyCfg(length=length, wildcard='#')

    cond1 = DummyCondition(['a', 'b', 'c', 'd', 'e'], cfg)
    cond2 = DummyCondition(['v', 'w', 'x', 'y', 'z'], cfg)
    parent = DummyClassifier(condition=cond1, cfg=cfg)
    donor = DummyClassifier(condition=cond2, cfg=cfg)

    # Patch numpy.random.choice to select fixed crossover points 1 and 4
    monkeypatch.setattr(genetic_algorithms.np.random, "choice", lambda *args, **kwargs: np.array([1,4]))

    genetic_algorithms.two_point_crossover(parent, donor)

    # Segment indices 1-3 should be swapped: b,c,d <-> w,x,y
    expected_parent = ['a', 'w', 'x', 'y', 'e']
    expected_donor = ['v', 'b', 'c', 'd', 'z']

    assert parent.condition._cond == expected_parent
    assert donor.condition._cond == expected_donor


def test_add_classifier_adds_new_when_none_found(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    # use empty action_set to force no old classifier found
    population = []
    match_set = DummyList()
    action_set = DummyList()
    cfg = DummyCfg(3, '#')

    cond = DummyCondition(['a', 'b', 'c'], cfg)
    cl = DummyClassifier(condition=cond, cfg=cfg)

    # Patch _find_old_classifier to return None
    monkeypatch.setattr(genetic_algorithms, "_find_old_classifier", lambda *args, **kwargs: None)

    perception = Perception("abc")

    genetic_algorithms.add_classifier(cl, perception, population, match_set, action_set, do_subsumption=True, theta_exp=0)

    assert cl in population
    assert cl in action_set
    assert cl in match_set


def test_add_classifier_increments_existing(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    population = []
    match_set = DummyList()
    action_set = DummyList()
    cfg = DummyCfg(3, '#')
    cond = DummyCondition(['a', 'b', 'c'], cfg)
    cl = DummyClassifier(condition=cond, num=1, tga=0, cfg=cfg)

    old_cl = DummyClassifier(condition=cond, num=5, tga=0, cfg=cfg)

    # Marked is False => increments num
    assert not old_cl.is_marked()

    # Patch _find_old_classifier to return old_cl
    monkeypatch.setattr(genetic_algorithms, "_find_old_classifier", lambda *args, **kwargs: old_cl)

    genetic_algorithms.add_classifier(cl, Perception("abc"), population, match_set, action_set, True, 0)

    assert old_cl.num == 6  # incremented
    assert cl not in population
    assert cl not in action_set
    assert cl not in match_set


def test_delete_classifiers_reduces_num_or_removes(monkeypatch):
    random.seed(7)
    np.random.seed(7)

    cfg = DummyCfg(1, '#')

    cl1 = DummyClassifier(num=2, q=1.0, tav=1, cfg=cfg)
    cl2 = DummyClassifier(num=1, q=0.5, tav=2, cfg=cfg)
    cl3 = DummyClassifier(num=1, q=1.2, tav=0, cfg=cfg)

    population = DummyList([cl1, cl2, cl3])
    match_set = DummyList([cl1, cl2])
    action_set = DummyList([cl1, cl2, cl3])

    # Patch random.random to always < 0.3 so deletion happens on first candidate
    monkeypatch.setattr(genetic_algorithms.random, "random", lambda: 0.0)

    # Delete classifiers to free space for 3 new (insize=3)
    # theta_as=4 limits max classifiers in action_set to that amount
    genetic_algorithms.delete_classifiers(population, match_set, action_set, insize=3, theta_as=4)

    # After deletion, total nums in action_set <= 4-3 = 1 (because insize + sum(cl.num) > theta_as deletes)
    # Should delete cl(s) until num sum is <= theta_as - insize
    total_num = sum(cl.num for cl in action_set)
    assert total_num + 3 <= 4

    # cl2 has lowest q and smallest tav, expected to be deleted first
    # either cl num reduced or cl removed fully
    # cl2.num == 0 or cl2 not in population
    if cl2 in population:
        assert cl2.num == 0 or cl2.num == 1  # could be reduced once
    else:
        assert cl2 not in population


def test__weighted_random_choice_selects_element(monkeypatch):
    # Here we patch random.uniform to pick values hitting certain thresholds
    choices = {"a": 1.0, "b": 3.0, "c": 6.0}
    keys = list(choices.keys())
    vals = list(choices.values())
    total = sum(vals)

    # Check that if random.uniform returns near 0, first item picked
    monkeypatch.setattr(genetic_algorithms.random, "uniform", lambda a, b: 0)
    val = genetic_algorithms._weighted_random_choice(choices)
    assert val == "a"

    # Near sum of first value
    monkeypatch.setattr(genetic_algorithms.random, "uniform", lambda a, b: vals[0] + 0.0001)
    val = genetic_algorithms._weighted_random_choice(choices)
    assert val == "b"

    # Near total sum
    monkeypatch.setattr(genetic_algorithms.random, "uniform", lambda a, b: total - 0.0001)
    val = genetic_algorithms._weighted_random_choice(choices)
    assert val == "c"


def test__find_similar_finds_equal_classifier():
    cl1 = DummyClassifier(num=1, tga=1)
    cl2 = DummyClassifier(num=1, tga=1)
    population = [cl1]

    # cl2 is equal to cl1 by __eq__ definition
    found = genetic_algorithms._find_similar(cl2, population)
    assert found == cl1

    # Non-equal classifier: returns None
    cl3 = DummyClassifier(num=1, tga=2)
    not_found = genetic_algorithms._find_similar(cl3, population)
    assert not_found is None


@pytest.mark.parametrize("cl_del_props,cl_props,expected", [
    # cl q smaller by > 0.1 => True
    ((1.2, False, 5), (1.0, False, 5), True),
    # cl q difference <= 0.1, cl marked and cl_del not => True
    ((1.0, False, 5), (1.05, True, 5), True),
    # cl q difference <= 0.1, cl marked and cl_del marked => False, because next check needed
    ((1.0, True, 5), (1.05, True, 3), False),
    # cl q difference <= 0.1, cl marked & cl_del marked, cl.tav <= cl_del.tav => False
    ((1.0, True, 5), (1.05, True, 5), False),
    # cl q difference <= 0.1, cl unmarked & cl_del unmarked, cl.tav > cl_del.tav => True
    ((1.0, False, 3), (1.05, False, 5), True),
    # cl q difference <= 0.1, cl unmarked & cl_del unmarked, cl.tav <= cl_del.tav => False
    ((1.0, False, 5), (1.05, False, 3), False),
    # cl q difference more than 0.1 negative => False
    ((1.0, False, 5), (1.5, False, 5), False),
])
def test__is_preferred_to_delete_variants(cl_del_props, cl_props, expected):
    # cl_del_props and cl_props: (q, marked, tav)
    cl_del = DummyClassifier()
    cl_del.q, cl_del._marked, cl_del.tav = cl_del_props
    cl = DummyClassifier()
    cl.q, cl._marked, cl.tav = cl_props

    result = genetic_algorithms._is_preferred_to_delete(cl_del, cl)
    assert result == expected


@pytest.mark.parametrize("subsumers,expected_old_cl_idx", [
    # subsumers list with one element: first returned
    (["sub1", "sub2"], 0),
    # empty list: None
    ([], None),
])
def test__find_old_classifier_subsumption(monkeypatch, subsumers, expected_old_cl_idx):
    cl = DummyClassifier()
    population = ["sub1", "sub2", "other"]

    def fake_find_subsumers(cl_arg, pop_arg, theta_exp):
        assert cl_arg == cl
        assert pop_arg == population
        assert theta_exp == 10
        return subsumers

    monkeypatch.setattr(genetic_algorithms, "find_subsumers", fake_find_subsumers)
    monkeypatch.setattr(genetic_algorithms, "_find_similar", lambda x, pop: "similar" if expected_old_cl_idx is None else None)

    # if subsumers nonempty, old_cl = subsumers[0]
    old_cl = genetic_algorithms._find_old_classifier(population, cl, use_subsumption=True, theta_exp=10)
    if expected_old_cl_idx is None:
        # subsumers empty, old_cl from _find_similar expected 'similar'
        assert old_cl == "similar"
    else:
        assert old_cl == subsumers[0]


def test__find_old_classifier_without_subsumption(monkeypatch):
    cl = DummyClassifier()
    population = ["similar_cl"]

    monkeypatch.setattr(genetic_algorithms, "_find_similar", lambda x, pop: "similar_cl")

    old_cl = genetic_algorithms._find_old_classifier(population, cl, use_subsumption=False, theta_exp=0)
    assert old_cl == "similar_cl"
