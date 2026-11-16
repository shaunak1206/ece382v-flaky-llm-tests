import random
from types import SimpleNamespace

import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga

WILDCARD = "#"
CFG = SimpleNamespace(classifier_wildcard=WILDCARD, classifier_length=4)


class FakeCondition(list):
    """List-based condition supporting the methods used by genetic_algorithms."""

    def __init__(self, values, matches=True):
        super().__init__(values)
        self.matches = matches
        self.generalized = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = WILDCARD

    def does_match(self, p):
        return self.matches


class FakeClassifier:
    def __init__(self, condition=None, action=0, q=0.5, tav=0,
                 num=1, tga=0, marked=False, cfg=CFG):
        self.condition = condition if condition is not None else FakeCondition([0, 1, 0, 1])
        self.action = action
        self.q = q
        self.tav = tav
        self.num = num
        self.tga = tga
        self.cfg = cfg
        self._marked = marked

    def is_marked(self):
        return self._marked

    @property
    def key(self):
        return (tuple(self.condition), self.action)

    def __eq__(self, other):
        return isinstance(other, FakeClassifier) and self.key == other.key

    def __hash__(self):
        return hash(self.key)


class PopList(list):
    """List with the extra methods used by delete_classifiers."""

    def expand(self):
        return list(self)

    def safe_remove(self, item):
        if item in self:
            self.remove(item)


def make_cl(values, matches=True, **kwargs):
    return FakeClassifier(condition=FakeCondition(values, matches=matches), **kwargs)


# ---------------------------------------------------------------- should_apply

def test_should_apply_returns_false_for_none_action_set():
    assert ga.should_apply(None, time=100, theta_ga=1) is False


def test_should_apply_returns_false_for_empty_action_set():
    assert ga.should_apply([], time=100, theta_ga=1) is False


def test_should_apply_returns_false_when_all_numerosities_zero():
    action_set = [FakeClassifier(num=0, tga=0)]
    assert ga.should_apply(action_set, time=100, theta_ga=1) is False


def test_should_apply_true_when_average_delay_exceeds_threshold():
    action_set = [FakeClassifier(num=1, tga=0)]
    assert ga.should_apply(action_set, time=10, theta_ga=5) is True


def test_should_apply_false_when_delay_equals_threshold():
    action_set = [FakeClassifier(num=1, tga=0)]
    assert ga.should_apply(action_set, time=5, theta_ga=5) is False


def test_should_apply_uses_numerosity_weighted_average():
    # weighted average tga = (0*3 + 10*1) / 4 = 2.5
    action_set = [FakeClassifier(num=3, tga=0), FakeClassifier(num=1, tga=10)]
    assert ga.should_apply(action_set, time=8, theta_ga=5) is True   # 8 - 2.5 = 5.5 > 5
    assert ga.should_apply(action_set, time=7, theta_ga=5) is False  # 7 - 2.5 = 4.5 <= 5


# ---------------------------------------------------------------- set_timestamps

def test_set_timestamps_updates_tga_of_all_classifiers():
    action_set = [FakeClassifier(tga=1), FakeClassifier(tga=2), FakeClassifier(tga=3)]
    ga.set_timestamps(action_set, epoch=42)
    assert [cl.tga for cl in action_set] == [42, 42, 42]


# ---------------------------------------------------------- roulette_wheel_selection

def test_roulette_wheel_selection_single_classifier_returned_as_both_parents():
    only = make_cl([0, 0, 0, 0])
    parent1, parent2 = ga.roulette_wheel_selection([only], fitnessfunc=lambda cl: 1.0)
    assert parent1 is only
    assert parent2 is only


def test_roulette_wheel_selection_never_picks_zero_fitness_classifier():
    zero = make_cl([0, 0, 0, 0])
    winner = make_cl([1, 1, 1, 1])
    fitness = {id(zero): 0.0, id(winner): 1.0}
    for _ in range(50):
        parents = ga.roulette_wheel_selection(
            [zero, winner], fitnessfunc=lambda cl: fitness[id(cl)])
        assert all(p is winner for p in parents)


# ----------------------------------------------------------- generalizing_mutation

def test_generalizing_mutation_with_full_probability_generalizes_non_wildcards():
    cl = make_cl([0, WILDCARD, 1, WILDCARD])
    ga.generalizing_mutation(cl, mu=1.0)
    assert cl.condition.generalized == [0, 2]
    assert list(cl.condition) == [WILDCARD, WILDCARD, WILDCARD, WILDCARD]


def test_generalizing_mutation_with_zero_probability_changes_nothing():
    cl = make_cl([0, 1, 0, 1])
    ga.generalizing_mutation(cl, mu=0.0)
    assert cl.condition.generalized == []
    assert list(cl.condition) == [0, 1, 0, 1]


# ------------------------------------------------------------- two_point_crossover

def test_two_point_crossover_swaps_complementary_segments():
    np.random.seed(1)
    parent = make_cl([0, 0, 0, 0])
    donor = make_cl([1, 1, 1, 1])
    ga.two_point_crossover(parent, donor)
    for i in range(4):
        assert parent.condition[i] + donor.condition[i] == 1


def test_two_point_crossover_swapped_positions_form_contiguous_block():
    np.random.seed(2)
    parent = make_cl([0, 0, 0, 0])
    donor = make_cl([1, 1, 1, 1])
    ga.two_point_crossover(parent, donor)
    ones = [i for i in range(4) if parent.condition[i] == 1]
    assert len(ones) >= 1
    assert ones == list(range(ones[0], ones[-1] + 1))


def test_two_point_crossover_identical_parents_unchanged():
    np.random.seed(3)
    parent = make_cl([0, 1, 0, 1])
    donor = make_cl([0, 1, 0, 1])
    ga.two_point_crossover(parent, donor)
    assert list(parent.condition) == [0, 1, 0, 1]
    assert list(donor.condition) == [0, 1, 0, 1]


# ---------------------------------------------------------------- add_classifier

def test_add_classifier_appends_new_classifier_to_all_lists_when_matching():
    p = (0, 1, 0, 1)
    cl = make_cl([0, 1, 0, 1], matches=True)
    population, match_set, action_set = [], [], []
    ga.add_classifier(cl, p, population, match_set, action_set,
                      do_subsumption=False, theta_exp=0)
    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_skips_match_set_when_not_matching_or_none():
    p = (0, 1, 0, 1)
    cl = make_cl([1, 1, 1, 1], matches=False)
    population, match_set, action_set = [], [], []
    ga.add_classifier(cl, p, population, match_set, action_set,
                      do_subsumption=False, theta_exp=0)
    assert population == [cl]
    assert action_set == [cl]
    assert match_set == []

    cl2 = make_cl([0, 0, 0, 0], matches=True)
    population2, action_set2 = [], []
    ga.add_classifier(cl2, p, population2, None, action_set2,
                      do_subsumption=False, theta_exp=0)
    assert population2 == [cl2]
    assert action_set2 == [cl2]


def test_add_classifier_increments_numerosity_of_unmarked_existing_classifier():
    p = (0, 1, 0, 1)
    existing = make_cl([0, 1, 0, 1], num=2, marked=False)
    new = make_cl([0, 1, 0, 1], num=1)
    population, action_set = [existing], [existing]
    ga.add_classifier(new, p, population, None, action_set,
                      do_subsumption=False, theta_exp=0)
    assert existing.num == 3
    assert population == [existing]
    assert action_set == [existing]


def test_add_classifier_does_not_increment_marked_existing_classifier():
    p = (0, 1, 0, 1)
    existing = make_cl([0, 1, 0, 1], num=2, marked=True)
    new = make_cl([0, 1, 0, 1], num=1)
    population, action_set = [existing], [existing]
    ga.add_classifier(new, p, population, None, action_set,
                      do_subsumption=False, theta_exp=0)
    assert existing.num == 2
    assert len(population) == 1


# -------------------------------------------------------------- delete_classifiers

def test_delete_classifiers_noop_when_under_threshold():
    random.seed(0)
    c1 = make_cl([0, 0, 0, 0])
    c2 = make_cl([1, 1, 1, 1])
    population, match_set, action_set = PopList([c1, c2]), PopList([c1]), PopList([c1, c2])
    ga.delete_classifiers(population, match_set, action_set, insize=0, theta_as=2)
    assert len(action_set) == 2
    assert len(population) == 2
    assert len(match_set) == 1


def test_delete_classifiers_removes_classifiers_until_under_threshold():
    random.seed(0)
    c1 = make_cl([0, 0, 0, 0])
    c2 = make_cl([1, 1, 1, 1])
    c3 = make_cl([0, 1, 0, 1])
    population = PopList([c1, c2, c3])
    match_set = PopList([c1, c2, c3])
    action_set = PopList([c1, c2, c3])
    ga.delete_classifiers(population, match_set, action_set, insize=0, theta_as=2)
    assert len(action_set) == 2
    assert len(population) == 2
    assert len(match_set) == 2
    assert all(cl in action_set for cl in population if cl in action_set)


def test_delete_classifiers_decrements_numerosity_before_removing():
    random.seed(0)
    cl = make_cl([0, 1, 0, 1], num=5)
    population = PopList([cl])
    action_set = PopList([cl])
    ga.delete_classifiers(population, None, action_set, insize=0, theta_as=3)
    assert cl.num == 3
    assert action_set == [cl]
    assert population == [cl]


# ------------------------------------------------------------ _is_preferred_to_delete

def test_is_preferred_to_delete_true_when_candidate_has_clearly_lower_quality():
    cl_del = FakeClassifier(q=0.9, tav=0)
    cl = FakeClassifier(q=0.1, tav=0)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_false_when_candidate_has_clearly_higher_quality():
    cl_del = FakeClassifier(q=0.1, tav=0)
    cl = FakeClassifier(q=0.9, tav=0)
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_within_tolerance_prefers_marked_candidate():
    cl_del = FakeClassifier(q=0.5, tav=0, marked=False)
    cl = FakeClassifier(q=0.5, tav=0, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_within_tolerance_keeps_lower_tav_candidate():
    cl_del = FakeClassifier(q=0.5, tav=10, marked=False)
    cl = FakeClassifier(q=0.5, tav=2, marked=False)
    assert ga._is_preferred_to_delete(cl_del, cl) is False


# ----------------------------------------------------------------- _find_similar

def test_find_similar_returns_classifier_with_same_condition_and_action():
    target = make_cl([0, 1, 0, 1])
    other = make_cl([1, 1, 1, 1])
    same = make_cl([0, 1, 0, 1])
    population = [other, same]
    found = ga._find_similar(target, population)
    assert found is same


def test_find_similar_returns_none_when_no_match():
    target = make_cl([0, 1, 0, 1])
    population = [make_cl([1, 1, 1, 1]), make_cl([0, 0, 0, 0])]
    assert ga._find_similar(target, population) is None
