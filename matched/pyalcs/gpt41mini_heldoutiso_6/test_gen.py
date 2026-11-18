import math
import random
import pytest
import numpy as np

from lcs.strategies import genetic_algorithms as ga
from lcs import Perception


class DummyCfg:
    # Dummy classifier config for testing
    classifier_length = 5
    classifier_wildcard = '#'


class DummyCondition(list):
    def __init__(self, condition_list):
        super().__init__(condition_list)
        self.cfg = DummyCfg()

    def generalize(self, idx):
        # Mark generalization as setting to wildcard
        self[idx] = self.cfg.classifier_wildcard

    def does_match(self, perception):
        # Simplified match: True if all elements of condition are in perception.data
        # (Assuming perception.data is iterable)
        for i, c in enumerate(self):
            if c != self.cfg.classifier_wildcard and (i >= len(perception.data) or perception.data[i] != c):
                return False
        return True


class DummyClassifier:
    def __init__(self, num=1, tga=0, q=1.0, tav=0, condition=None, is_marked=False):
        self.num = num  # numerosity
        self.tga = tga  # time of last GA
        self.q = q      # quality/fitness value
        self.tav = tav  # average time last updated
        self._is_marked = is_marked
        self.cfg = DummyCfg()
        if condition is None:
            self.condition = DummyCondition(['a', 'b', 'c', 'd', 'e'])
        else:
            if isinstance(condition, DummyCondition):
                self.condition = condition
            else:
                self.condition = DummyCondition(condition)

    def __eq__(self, other):
        if not isinstance(other, DummyClassifier):
            return False
        # equality by condition + other attributes for similarity
        return (list(self.condition) == list(other.condition))

    def is_marked(self):
        return self._is_marked


class DummyList(list):
    # To simulate .expand() returning iterators or list with safe_remove method
    def expand(self):
        # Return self for iteration
        return iter(self)

    def safe_remove(self, item):
        # Remove item if present
        try:
            self.remove(item)
        except ValueError:
            pass


def test_should_apply_none_action_set():
    assert ga.should_apply(None, time=100, theta_ga=10) is False


def test_should_apply_empty_action_set():
    action_set = []
    assert ga.should_apply(action_set, time=50, theta_ga=10) is False


def test_should_apply_zero_numerosity():
    cl1 = DummyClassifier(num=0, tga=10)
    cl2 = DummyClassifier(num=0, tga=20)
    action_set = [cl1, cl2]
    assert ga.should_apply(action_set, time=50, theta_ga=10) is False


def test_should_apply_true_and_false_cases():
    # Case where average delay is > theta_ga: Should return True
    cl1 = DummyClassifier(num=3, tga=20)
    cl2 = DummyClassifier(num=2, tga=15)
    action_set = [cl1, cl2]
    time = 50
    theta_ga = 5  # delay > 5 should trigger True

    assert ga.should_apply(action_set, time=time, theta_ga=theta_ga) is True

    # Case where delay <= theta_ga: Should return False
    theta_ga = 50
    assert ga.should_apply(action_set, time=time, theta_ga=theta_ga) is False


def test_set_timestamps_updates_all():
    cl1 = DummyClassifier(tga=0)
    cl2 = DummyClassifier(tga=5)
    action_set = [cl1, cl2]

    epoch = 100
    ga.set_timestamps(action_set, epoch)

    assert all(cl.tga == epoch for cl in action_set)


def test_roulette_wheel_selection_returns_parents(monkeypatch):
    population = [
        DummyClassifier(q=1.0),
        DummyClassifier(q=2.0),
        DummyClassifier(q=3.0)
    ]

    def fitness_func(cl):
        return cl.q

    # Monkeypatch random.uniform to predictable values covering weight ranges:
    # sum fitness = 6, ranges: 0-1,1-3,3-6
    calls = iter([0.5, 2.5])  # will pick first and second classifier
    monkeypatch.setattr(random, "uniform", lambda a, b: next(calls))

    parent1, parent2 = ga.roulette_wheel_selection(population, fitness_func)
    # parent1 should be population[0], parent2 population[1]
    assert parent1 == population[0]
    assert parent2 == population[1]


def test_generalizing_mutation_all_mutate(monkeypatch):
    # Setup classifier with known condition (no wildcards)
    cl = DummyClassifier(condition=['x', 'y', 'z', 'w', 'v'])
    mu = 1.0  # mutate all

    calls = iter([0, 0, 0, 0, 0])  # always less than mu
    monkeypatch.setattr(random, "random", lambda: next(calls))

    ga.generalizing_mutation(cl, mu)

    # All index replaced by wildcard except any original wildcard (none in our condition)
    assert all(elem == cl.cfg.classifier_wildcard for elem in cl.condition)


def test_generalizing_mutation_no_mutate(monkeypatch):
    cl = DummyClassifier(condition=['a', 'b', 'c', 'd', 'e'])
    mu = 0.0  # mutate none

    monkeypatch.setattr(random, "random", lambda: 1.0)  # always greater than mu

    ga.generalizing_mutation(cl, mu)

    # Condition unchanged
    assert list(cl.condition) == ['a', 'b', 'c', 'd', 'e']


def test_two_point_crossover_flips_segments(monkeypatch):
    # parent and donor have distinct condition arrays
    parent = DummyClassifier(condition=['A', 'B', 'C', 'D', 'E'])
    donor = DummyClassifier(condition=['V', 'W', 'X', 'Y', 'Z'])

    # Force random choice of points 1 and 4
    monkeypatch.setattr(np.random, "choice", lambda a, size, replace: np.array([1, 4]))

    ga.two_point_crossover(parent, donor)

    # Check swapped segment in indices 1:4: parent's replaced with donor's, and donor's replaced with parent's
    assert parent.condition[:1] == ['A']
    assert list(parent.condition[1:4]) == ['W', 'X', 'Y']
    assert parent.condition[4:] == ['E']

    assert donor.condition[:1] == ['V']
    assert list(donor.condition[1:4]) == ['B', 'C', 'D']
    assert donor.condition[4:] == ['Z']


def test_add_classifier_new_adds_to_all_lists():
    population = []
    match_set = DummyList()
    action_set = DummyList()
    cl = DummyClassifier(condition=['a', 'b', 'c', 'd', 'e'])
    p = Perception(['a', 'b', 'c', 'd', 'e'])

    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert cl in population
    assert cl in action_set
    assert cl in match_set


def test_add_classifier_existing_increments_num(monkeypatch):
    # Setup to find subsumer and reject adding new

    old_cl = DummyClassifier(num=2, is_marked=lambda: False)
    old_cl_condition = DummyCondition(['a', 'b', 'c', 'd', 'e'])
    old_cl.condition = old_cl_condition
    new_cl = DummyClassifier(condition=['a', 'b', 'c', 'd', 'e'])

    population = [old_cl]
    action_set = [old_cl]
    match_set = DummyList()  # empty

    def fake_find_subsumers(cl, pop, theta):
        return [old_cl]

    monkeypatch.setattr(ga, "find_subsumers", fake_find_subsumers)

    ga.add_classifier(new_cl, Perception(['a', 'b', 'c', 'd', 'e']), population, match_set, action_set,
                      do_subsumption=True, theta_exp=0)

    assert old_cl.num == 3  # incremented numerosity
    assert new_cl not in population
    assert new_cl not in action_set


def test_delete_classifiers_removes_and_decrements(monkeypatch):
    pop = DummyList()
    ms = DummyList()
    as_ = DummyList()

    cl1 = DummyClassifier(num=3)
    cl2 = DummyClassifier(num=1)
    cl3 = DummyClassifier(num=1)
    pop.extend([cl1, cl2, cl3])
    ms.extend([cl1, cl2, cl3])
    as_.extend([cl1, cl2, cl3])

    # Patch random.random to force choosing cl2 then cl3 for deletion
    # less than 0.3 triggers pick, else keep looking
    picks = iter([0.1, 0.4, 0.2, 0.5])  # triggers first pick cl2 first, then cl3

    monkeypatch.setattr(random, "random", lambda: next(picks))

    # We want to insert 1 classifier, theta_as threshold 4, currently sum(cl.num) = 5 (3+1+1)
    # insize + 5 = 6 > 4 triggers deletion cycle
    ga.delete_classifiers(pop, ms, as_, insize=1, theta_as=4)

    # cl2 num=1 -> removed completely, cl3 num=1 -> removed completely
    # cl1 num=3 remains
    assert cl1 in pop and cl1.num == 3
    assert cl2 not in pop
    assert cl3 not in pop
    assert cl2 not in ms
    assert cl3 not in ms
    assert cl2 not in as_
    assert cl3 not in as_


@pytest.mark.parametrize("cl_del_q,cl_q,cl_del_marked,cl_marked,cl_del_tav,cl_tav,expected", [
    (0.5, 0.3, False, False, 10, 15, True),  # cl q significantly lower
    (0.5, 0.4, False, False, 10, 20, False),  # difference <= 0.1 no other conditions true
    (0.5, 0.55, False, True, 10, 20, True),  # cl marked and cl_del not marked
    (0.5, 0.55, True, True, 10, 12, False),  # both marked no tav higher
    (0.5, 0.55, False, False, 10, 20, True),  # both unmarked, cl tav higher
    (0.5, 0.55, True, False, 10, 20, False),  # cl_del marked, cl not marked => False
])
def test_is_preferred_to_delete_variants(cl_del_q, cl_q,
                                        cl_del_marked, cl_marked,
                                        cl_del_tav, cl_tav,
                                        expected):
    cl_del = DummyClassifier(q=cl_del_q, tav=cl_del_tav, is_marked=lambda: cl_del_marked)
    cl = DummyClassifier(q=cl_q, tav=cl_tav, is_marked=lambda: cl_marked)
    assert ga._is_preferred_to_delete(cl_del, cl) == expected


def test_find_old_classifier_uses_subsumers(monkeypatch):
    cl = DummyClassifier()
    population = [DummyClassifier(), DummyClassifier()]

    expected_subsumer = DummyClassifier()
    monkeypatch.setattr(ga, "find_subsumers", lambda cl_, pop, theta: [expected_subsumer])

    old_cl = ga._find_old_classifier(population, cl, use_subsumption=True, theta_exp=0)
    assert old_cl is expected_subsumer


def test_find_old_classifier_subsumer_empty_falls_back(monkeypatch):
    cl = DummyClassifier()
    population = [DummyClassifier(condition=['a','b','c','d','e']),
                  DummyClassifier(condition=['1','2','3','4','5'])]

    monkeypatch.setattr(ga, "find_subsumers", lambda cl_, pop, theta: [])

    # Similar classifier is population[0] (by __eq__)
    similar_cl = population[0]

    old_cl = ga._find_old_classifier(population, similar_cl, use_subsumption=True, theta_exp=0)
    assert old_cl == similar_cl


def test_find_old_classifier_no_subsumption_uses_similar():
    population = [DummyClassifier(condition=['a','b','c','d','e']),
                  DummyClassifier(condition=['1','2','3','4','5'])]
    cl = DummyClassifier(condition=['a','b','c','d','e'])

    old_cl = ga._find_old_classifier(population, cl, use_subsumption=False, theta_exp=0)
    assert old_cl == population[0]


def test_find_similar_returns_none_when_no_match():
    population = [DummyClassifier(condition=['x','y','z','w','v'])]
    other_cl = DummyClassifier(condition=['a','b','c','d','e'])

    res = ga._find_similar(other_cl, population)
    assert res is None


def test_find_similar_returns_first_match():
    population = [DummyClassifier(condition=['x','y','z','w','v']),
                  DummyClassifier(condition=['a','b','c','d','e']),
                  DummyClassifier(condition=['a','b','c','d','e'])]

    other_cl = DummyClassifier(condition=['a','b','c','d','e'])

    res = ga._find_similar(other_cl, population)
    assert res == population[1]


def test_weighted_random_choice_selection_sum(monkeypatch):
    keys = ['a', 'b', 'c']
    choices = {k: i + 1 for i, k in enumerate(keys)}  # a:1, b:2, c:3, sum=6

    # random.uniform will return values that pick b and c
    calls = iter([1.8, 5.9])

    monkeypatch.setattr(random, "uniform", lambda a, b: next(calls))

    first_pick = ga._weighted_random_choice(choices)
    second_pick = ga._weighted_random_choice(choices)

    assert first_pick == 'b'  # b range is (1..3)
    assert second_pick == 'c'  # c range is (3..6)
