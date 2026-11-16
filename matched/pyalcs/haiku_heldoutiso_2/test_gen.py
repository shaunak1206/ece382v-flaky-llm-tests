import random
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga


class FakeCondition(list):
    """List-based condition with the generalize/does_match API used by the GA."""

    def __init__(self, values, matches=False):
        super().__init__(values)
        self.matches = matches
        self.generalized = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = '#'

    def does_match(self, perception):
        return self.matches


class FakeClassifier:
    """Minimal classifier stub. Uses identity equality and hashing."""

    def __init__(self, condition=None, q=0.5, tav=0, num=1, tga=0,
                 marked=False, cfg=None):
        self.condition = condition if condition is not None else FakeCondition([])
        self.q = q
        self.tav = tav
        self.num = num
        self.tga = tga
        self.marked = marked
        self.cfg = cfg if cfg is not None else SimpleNamespace(
            classifier_wildcard='#', classifier_length=4)

    def is_marked(self):
        return self.marked


class FakeActionSet(list):
    """List-based population/action set with the methods the GA relies on."""

    def expand(self):
        return list(self)

    def safe_remove(self, cl):
        if cl in self:
            self.remove(cl)


@pytest.mark.parametrize("action_set", [None, [], [FakeClassifier(num=0)]])
def test_should_apply_returns_false_when_no_numerosity(action_set):
    assert ga.should_apply(action_set, time=100, theta_ga=1) is False


def test_should_apply_true_when_average_delay_exceeds_threshold():
    action_set = [FakeClassifier(num=1, tga=0)]
    assert ga.should_apply(action_set, time=10, theta_ga=5) is True


def test_should_apply_false_when_delay_equals_threshold():
    action_set = [FakeClassifier(num=1, tga=0)]
    assert ga.should_apply(action_set, time=5, theta_ga=5) is False


def test_should_apply_uses_numerosity_weighted_average():
    # weighted average = (1*0 + 3*10) / 4 = 7.5, so the delay at time 10 is 2.5
    action_set = [FakeClassifier(num=1, tga=0), FakeClassifier(num=3, tga=10)]
    assert ga.should_apply(action_set, time=10, theta_ga=2) is True
    assert ga.should_apply(action_set, time=10, theta_ga=3) is False


def test_set_timestamps_sets_tga_for_every_classifier():
    classifiers = [FakeClassifier(tga=0), FakeClassifier(tga=3)]
    ga.set_timestamps(classifiers, epoch=42)
    assert [cl.tga for cl in classifiers] == [42, 42]


def test_roulette_wheel_selection_picks_parents_by_cumulative_fitness(monkeypatch):
    a = FakeClassifier(q=1.0)
    b = FakeClassifier(q=3.0)
    draws = iter([0.5, 1.5])
    monkeypatch.setattr(random, "uniform", lambda lo, hi: next(draws))

    parent1, parent2 = ga.roulette_wheel_selection([a, b], lambda cl: cl.q)

    assert parent1 is a
    assert parent2 is b


def test_weighted_random_choice_handles_interval_boundaries(monkeypatch):
    choices = {'a': 1.0, 'b': 3.0}

    monkeypatch.setattr(random, "uniform", lambda lo, hi: 0.0)
    assert ga._weighted_random_choice(choices) == 'a'

    monkeypatch.setattr(random, "uniform", lambda lo, hi: 1.0)
    assert ga._weighted_random_choice(choices) == 'b'


def test_generalizing_mutation_skips_wildcards_when_mu_is_one():
    cl = FakeClassifier(condition=FakeCondition(['0', '#', '1']))

    ga.generalizing_mutation(cl, mu=1.0)

    assert cl.condition.generalized == [0, 2]
    assert list(cl.condition) == ['#', '#', '#']


def test_generalizing_mutation_does_nothing_when_mu_is_zero():
    cl = FakeClassifier(condition=FakeCondition(['0', '1']))

    ga.generalizing_mutation(cl, mu=0.0)

    assert cl.condition.generalized == []
    assert list(cl.condition) == ['0', '1']


def test_generalizing_mutation_follows_random_draws(monkeypatch):
    cl = FakeClassifier(condition=FakeCondition(['0', '1', '1']))

    monkeypatch.setattr(random, "random", lambda: 0.4)
    ga.generalizing_mutation(cl, mu=0.5)
    assert cl.condition.generalized == [0, 1, 2]

    cl2 = FakeClassifier(condition=FakeCondition(['0', '1', '1']))
    monkeypatch.setattr(random, "random", lambda: 0.6)
    ga.generalizing_mutation(cl2, mu=0.5)
    assert cl2.condition.generalized == []


def test_two_point_crossover_swaps_segment_between_cut_points():
    parent = FakeClassifier(condition=FakeCondition(list('abcd')))
    donor = FakeClassifier(condition=FakeCondition(list('wxyz')))

    # Unsorted cut points on purpose: the function must sort them.
    with mock.patch.object(np.random, "choice", return_value=np.array([3, 1])):
        ga.two_point_crossover(parent, donor)

    assert list(parent.condition) == list('axyd')
    assert list(donor.condition) == list('wbcz')


def test_two_point_crossover_draws_cut_points_from_full_range():
    parent = FakeClassifier(condition=FakeCondition(list('abcd')))
    donor = FakeClassifier(condition=FakeCondition(list('wxyz')))

    with mock.patch.object(np.random, "choice",
                           return_value=np.array([0, 4])) as choice:
        ga.two_point_crossover(parent, donor)

    args, kwargs = choice.call_args
    assert list(args[0]) == [0, 1, 2, 3, 4]
    assert args[1] == 2
    assert kwargs == {'replace': False}
    assert list(parent.condition) == list('wxyz')


@pytest.mark.parametrize("matches", [True, False])
def test_add_classifier_adds_new_classifier_and_matches_condition(matches):
    cl = FakeClassifier(condition=FakeCondition(['0'], matches=matches))
    population, action_set, match_set = [], [], []

    with mock.patch.object(ga, "_find_old_classifier", return_value=None):
        ga.add_classifier(cl, p=object(), population=population,
                          match_set=match_set, action_set=action_set,
                          do_subsumption=False, theta_exp=5)

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == ([cl] if matches else [])


@pytest.mark.parametrize("marked, expected_num", [(False, 3), (True, 2)])
def test_add_classifier_increments_numerosity_only_when_old_is_unmarked(
        marked, expected_num):
    old = FakeClassifier(num=2, marked=marked)
    cl = FakeClassifier()
    population, action_set = [old], [old]

    with mock.patch.object(ga, "_find_old_classifier", return_value=old):
        ga.add_classifier(cl, p=None, population=population,
                          match_set=None, action_set=action_set,
                          do_subsumption=True, theta_exp=5)

    assert old.num == expected_num
    assert population == [old]
    assert action_set == [old]


def test_add_classifier_forwards_subsumption_arguments():
    cl = FakeClassifier()
    action_set = []

    with mock.patch.object(ga, "_find_old_classifier",
                           return_value=None) as finder:
        ga.add_classifier(cl, p=None, population=[], match_set=None,
                          action_set=action_set, do_subsumption=True,
                          theta_exp=7)

    finder.assert_called_once_with(action_set, cl, True, 7)


def test_find_similar_returns_first_equal_element_or_none():
    assert ga._find_similar(2, [1, 2, 2, 3]) == 2
    assert ga._find_similar(9, [1, 2]) is None


def test_find_old_classifier_without_subsumption_uses_similarity():
    with mock.patch.object(ga, "find_subsumers") as subsumers:
        result = ga._find_old_classifier([1, 2, 3], 2,
                                         use_subsumption=False, theta_exp=5)

    assert result == 2
    subsumers.assert_not_called()


def test_find_old_classifier_prefers_first_subsumer_returned():
    population = ['general', 'specific']
    with mock.patch.object(ga, "find_subsumers",
                           return_value=['general', 'specific']) as subsumers:
        result = ga._find_old_classifier(population, 'x',
                                         use_subsumption=True, theta_exp=5)

    assert result == 'general'
    subsumers.assert_called_once_with('x', population, 5)


def test_find_old_classifier_falls_back_to_similar_when_no_subsumer():
    with mock.patch.object(ga, "find_subsumers", return_value=[]):
        result = ga._find_old_classifier([1, 2, 3], 3,
                                         use_subsumption=True, theta_exp=5)

    assert result == 3


@pytest.mark.parametrize("del_kwargs, cl_kwargs, expected", [
    ({'q': 0.8}, {'q': 0.5}, True),                       # much worse fitness
    ({'q': 0.5}, {'q': 0.9}, False),                      # much better fitness
    ({'q': 0.5, 'marked': False}, {'q': 0.5, 'marked': True}, True),
    ({'q': 0.5, 'tav': 2}, {'q': 0.5, 'tav': 10}, True),  # older candidate
    ({'q': 0.5, 'tav': 5}, {'q': 0.5, 'tav': 1}, False),  # younger candidate
])
def test_is_preferred_to_delete_compares_fitness_marking_and_age(
        del_kwargs, cl_kwargs, expected):
    cl_del = FakeClassifier(**del_kwargs)
    cl = FakeClassifier(**cl_kwargs)

    assert ga._is_preferred_to_delete(cl_del, cl) is expected


def test_delete_classifiers_removes_least_preferred_when_over_capacity(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    strong = FakeClassifier(q=0.9)
    weak = FakeClassifier(q=0.1)
    population = FakeActionSet([strong, weak])
    action_set = FakeActionSet([strong, weak])

    ga.delete_classifiers(population, None, action_set, insize=1, theta_as=2)

    assert action_set == [strong]
    assert population == [strong]


def test_delete_classifiers_decrements_numerosity_before_removing(monkeypatch):
    monkeypatch.setattr(random, "random", lambda: 0.0)
    cl = FakeClassifier(num=3)
    population = FakeActionSet([cl])
    action_set = FakeActionSet([cl])

    ga.delete_classifiers(population, None, action_set, insize=1, theta_as=3)

    assert cl.num == 2
    assert action_set == [cl]
    assert population == [cl]


def test_delete_classifiers_does_nothing_when_within_capacity(monkeypatch):
    random_mock = mock.Mock(return_value=0.0)
    monkeypatch.setattr(random, "random", random_mock)
    cl = FakeClassifier(num=2)
    population = FakeActionSet([cl])
    action_set = FakeActionSet([cl])

    ga.delete_classifiers(population, None, action_set, insize=1, theta_as=3)

    assert random_mock.call_count == 0
    assert cl.num == 2
    assert action_set == [cl]
    assert population == [cl]
