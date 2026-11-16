import random
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga


class FakeCondition(list):
    """List-like condition that records generalize calls and supports does_match."""

    def __init__(self, items, matches=True):
        super().__init__(items)
        self.matches = matches
        self.generalized = []
        self.match_calls = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = '#'

    def does_match(self, p):
        self.match_calls.append(p)
        return self.matches


class FakeCl:
    """Minimal classifier stand-in with the attributes used by genetic_algorithms."""

    def __init__(self, key='c', q=0.5, tav=0, tga=0, num=1,
                 marked=False, condition=None, cfg=None):
        self.key = key
        self.q = q
        self.tav = tav
        self.tga = tga
        self.num = num
        self.marked = marked
        self.condition = condition if condition is not None else FakeCondition([])
        self.cfg = cfg if cfg is not None else SimpleNamespace(
            classifier_wildcard='#', classifier_length=4)

    def is_marked(self):
        return self.marked

    def __eq__(self, other):
        return isinstance(other, FakeCl) and self.key == other.key

    __hash__ = object.__hash__

    def __repr__(self):
        return 'FakeCl({!r})'.format(self.key)


class FakeList(list):
    """List that mimics the population/action-set containers used by the GA."""

    def expand(self):
        return list(self)

    def safe_remove(self, item):
        if item in self:
            self.remove(item)


def test_should_apply_returns_false_for_none_or_empty_set():
    assert ga.should_apply(None, time=10, theta_ga=1) is False
    assert ga.should_apply([], time=10, theta_ga=1) is False


def test_should_apply_true_when_average_delay_exceeds_threshold():
    action_set = [FakeCl(tga=0, num=1)]
    assert ga.should_apply(action_set, time=10, theta_ga=5) is True
    assert ga.should_apply(action_set, time=4, theta_ga=5) is False


def test_should_apply_uses_numerosity_weighted_average_and_strict_threshold():
    # weighted average = (2*1 + 8*3) / 4 = 6.5
    action_set = [FakeCl(key='a', tga=2, num=1), FakeCl(key='b', tga=8, num=3)]
    assert ga.should_apply(action_set, time=10, theta_ga=3) is True
    assert ga.should_apply(action_set, time=10, theta_ga=4) is False
    # boundary: time - avg == theta is not greater, so False
    assert ga.should_apply([FakeCl(tga=0, num=1)], time=5, theta_ga=5) is False


def test_set_timestamps_updates_tga_of_every_classifier():
    cls = [FakeCl(key='a', tga=1), FakeCl(key='b', tga=2)]
    ga.set_timestamps(cls, epoch=42)
    assert all(cl.tga == 42 for cl in cls)


def test_roulette_wheel_selection_picks_parents_by_fitness_intervals(monkeypatch):
    a = FakeCl(key='a')
    b = FakeCl(key='b')
    fitness = {a: 1.0, b: 3.0}
    picks = iter([0.5, 1.5])  # first pick lands in a's interval, second in b's
    monkeypatch.setattr(random, 'uniform', lambda lo, hi: next(picks))

    parent1, parent2 = ga.roulette_wheel_selection([a, b], lambda cl: fitness[cl])

    assert parent1 is a
    assert parent2 is b


def test_roulette_wheel_selection_selects_high_fitness_at_upper_pick(monkeypatch):
    a = FakeCl(key='a')
    b = FakeCl(key='b')
    fitness = {a: 1.0, b: 3.0}
    monkeypatch.setattr(random, 'uniform', lambda lo, hi: 3.9)

    parent1, parent2 = ga.roulette_wheel_selection([a, b], lambda cl: fitness[cl])

    assert parent1 is b
    assert parent2 is b


def test_generalizing_mutation_generalizes_non_wildcard_positions(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.0)
    cond = FakeCondition(['0', '#', '1'])
    cl = FakeCl(condition=cond)

    ga.generalizing_mutation(cl, mu=0.5)

    assert cond.generalized == [0, 2]
    assert list(cond) == ['#', '#', '#']


def test_generalizing_mutation_does_nothing_when_probability_not_met(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.5)
    cond = FakeCondition(['0', '1'])
    cl = FakeCl(condition=cond)

    ga.generalizing_mutation(cl, mu=0.0)

    assert cond.generalized == []
    assert list(cond) == ['0', '1']


def test_two_point_crossover_swaps_segment_between_parent_and_donor(monkeypatch):
    # Return unsorted indices to also verify sorting of the cut points.
    monkeypatch.setattr(np.random, 'choice',
                        lambda *args, **kwargs: np.array([3, 1]))
    parent = FakeCl(key='p', condition=FakeCondition(['a', 'b', 'c', 'd']))
    donor = FakeCl(key='d', condition=FakeCondition(['w', 'x', 'y', 'z']))

    ga.two_point_crossover(parent, donor)

    assert list(parent.condition) == ['a', 'x', 'y', 'd']
    assert list(donor.condition) == ['w', 'b', 'c', 'z']


def test_add_classifier_appends_new_classifier_to_population_action_and_match_set():
    population, action_set, match_set = FakeList(), FakeList(), FakeList()
    cond = FakeCondition(['0', '1'], matches=True)
    new_cl = FakeCl(key='new', condition=cond)
    perception = object()

    ga.add_classifier(new_cl, perception, population, match_set, action_set,
                      do_subsumption=False, theta_exp=10)

    assert new_cl in population
    assert new_cl in action_set
    assert new_cl in match_set
    assert cond.match_calls == [perception]


def test_add_classifier_does_not_add_to_match_set_when_condition_does_not_match():
    population, action_set, match_set = FakeList(), FakeList(), FakeList()
    cond = FakeCondition(['0'], matches=False)
    new_cl = FakeCl(key='new', condition=cond)

    ga.add_classifier(new_cl, object(), population, match_set, action_set,
                      do_subsumption=False, theta_exp=10)

    assert new_cl in population
    assert new_cl in action_set
    assert new_cl not in match_set


def test_add_classifier_skips_match_set_when_it_is_none():
    population, action_set = FakeList(), FakeList()
    new_cl = FakeCl(key='new', condition=FakeCondition(['0'], matches=True))

    ga.add_classifier(new_cl, object(), population, None, action_set,
                      do_subsumption=False, theta_exp=10)

    assert new_cl in population
    assert new_cl in action_set


def test_add_classifier_increments_numerosity_of_similar_unmarked_classifier():
    existing = FakeCl(key='same', num=2)
    population, action_set = FakeList([existing]), FakeList([existing])
    new_cl = FakeCl(key='same')

    ga.add_classifier(new_cl, object(), population, None, action_set,
                      do_subsumption=False, theta_exp=10)

    assert existing.num == 3
    assert len(population) == 1
    assert len(action_set) == 1


def test_add_classifier_does_not_increment_marked_similar_classifier():
    existing = FakeCl(key='same', num=2, marked=True)
    population, action_set = FakeList([existing]), FakeList([existing])
    new_cl = FakeCl(key='same')

    ga.add_classifier(new_cl, object(), population, None, action_set,
                      do_subsumption=False, theta_exp=10)

    assert existing.num == 2
    assert len(population) == 1


def test_add_classifier_uses_subsumer_when_subsumption_enabled(monkeypatch):
    subsumer = FakeCl(key='general', num=1)
    calls = []

    def fake_find_subsumers(cl, population, theta_exp):
        calls.append(theta_exp)
        return [subsumer]

    monkeypatch.setattr(ga, 'find_subsumers', fake_find_subsumers)
    population, action_set = FakeList([subsumer]), FakeList([subsumer])
    new_cl = FakeCl(key='specific')

    ga.add_classifier(new_cl, object(), population, None, action_set,
                      do_subsumption=True, theta_exp=7)

    assert calls == [7]
    assert subsumer.num == 2
    assert new_cl not in population


def test_find_similar_returns_first_equal_classifier_or_none():
    first = FakeCl(key='x', num=1)
    second = FakeCl(key='x', num=5)
    population = [FakeCl(key='y'), first, second]

    assert ga._find_similar(FakeCl(key='x'), population) is first
    assert ga._find_similar(FakeCl(key='z'), population) is None


def test_find_old_classifier_falls_back_to_similar_when_no_subsumers(monkeypatch):
    similar = FakeCl(key='x')
    monkeypatch.setattr(ga, 'find_subsumers', lambda cl, pop, theta: [])

    result = ga._find_old_classifier([similar], FakeCl(key='x'),
                                     use_subsumption=True, theta_exp=3)

    assert result is similar


def test_delete_classifiers_does_nothing_when_under_capacity():
    cl1 = FakeCl(key='a', num=1)
    population = FakeList([cl1])
    match_set = FakeList([cl1])
    action_set = FakeList([cl1])

    ga.delete_classifiers(population, match_set, action_set,
                          insize=1, theta_as=2)

    assert list(population) == [cl1]
    assert list(match_set) == [cl1]
    assert list(action_set) == [cl1]


def test_delete_classifiers_removes_classifier_from_all_lists_when_num_is_one(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.0)
    cl1 = FakeCl(key='a', q=0.5, tav=0, num=1)
    cl2 = FakeCl(key='b', q=0.5, tav=0, num=1)
    population = FakeList([cl1, cl2])
    match_set = FakeList([cl1, cl2])
    action_set = FakeList([cl1, cl2])

    ga.delete_classifiers(population, match_set, action_set,
                          insize=0, theta_as=1)

    assert cl1 not in population
    assert cl1 not in match_set
    assert cl1 not in action_set
    assert cl2 in action_set


def test_delete_classifiers_decrements_numerosity_when_greater_than_one(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.0)
    cl1 = FakeCl(key='a', q=0.5, tav=0, num=2)
    cl2 = FakeCl(key='b', q=0.5, tav=0, num=1)
    population = FakeList([cl1, cl2])
    match_set = FakeList([cl1, cl2])
    action_set = FakeList([cl1, cl2])

    ga.delete_classifiers(population, match_set, action_set,
                          insize=0, theta_as=2)

    assert cl1.num == 1
    assert cl1 in population
    assert cl1 in action_set


def test_is_preferred_to_delete_true_when_candidate_has_much_lower_quality():
    cl_del = FakeCl(key='del', q=0.5)
    cl = FakeCl(key='cand', q=0.1)

    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_false_when_candidate_has_much_higher_quality():
    cl_del = FakeCl(key='del', q=0.1)
    cl = FakeCl(key='cand', q=0.9)

    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_prefers_marked_candidate_when_quality_is_close():
    cl_del = FakeCl(key='del', q=0.5, marked=False)
    cl = FakeCl(key='cand', q=0.5, marked=True)

    assert ga._is_preferred_to_delete(cl_del, cl) is True
