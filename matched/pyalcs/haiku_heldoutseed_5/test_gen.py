import numpy as np
import pytest

import random

from lcs.strategies import genetic_algorithms as ga


class FakeCfg:
    classifier_wildcard = '#'
    classifier_length = 3


class FakeCondition(list):
    def __init__(self, items, matches=True):
        super().__init__(items)
        self.matches = matches
        self.generalized = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = '#'

    def does_match(self, p):
        return self.matches


class FakeClassifier:
    def __init__(self, condition=None, action=0, q=0.5, tav=0,
                 num=1, tga=0, marked=False, matches=True):
        self.condition = FakeCondition(
            condition if condition is not None else ['0', '1', '0'],
            matches=matches)
        self.action = action
        self.q = q
        self.tav = tav
        self.num = num
        self.tga = tga
        self.cfg = FakeCfg()
        self._marked = marked

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        if not isinstance(other, FakeClassifier):
            return NotImplemented
        return (list(self.condition) == list(other.condition)
                and self.action == other.action)

    def __hash__(self):
        return hash((tuple(self.condition), self.action))


class FakeList(list):
    def expand(self):
        return list(self)

    def safe_remove(self, item):
        self[:] = [x for x in self if x is not item]


def test_should_apply_returns_false_for_none():
    assert ga.should_apply(None, time=100, theta_ga=1) is False


def test_should_apply_returns_false_when_total_numerosity_is_zero():
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
    assert ga.should_apply(action_set, time=8, theta_ga=5) is True
    assert ga.should_apply(action_set, time=7, theta_ga=5) is False


def test_set_timestamps_updates_every_classifier():
    action_set = [FakeClassifier(tga=0), FakeClassifier(tga=3)]
    ga.set_timestamps(action_set, epoch=42)
    assert all(cl.tga == 42 for cl in action_set)


def test_weighted_random_choice_picks_by_cumulative_weight(monkeypatch):
    choices = {'a': 1, 'b': 3}
    monkeypatch.setattr(ga.random, "uniform", lambda lo, hi: 0.5)
    assert ga._weighted_random_choice(choices) == 'a'
    monkeypatch.setattr(ga.random, "uniform", lambda lo, hi: 2.0)
    assert ga._weighted_random_choice(choices) == 'b'


def test_weighted_random_choice_returns_none_for_empty_choices():
    assert ga._weighted_random_choice({}) is None


def test_roulette_wheel_selection_returns_two_parents(monkeypatch):
    a = FakeClassifier(action=0, q=0.25)
    b = FakeClassifier(action=1, q=0.75)
    values = iter([0.5, 3.5])
    monkeypatch.setattr(ga.random, "uniform", lambda lo, hi: next(values))

    parent1, parent2 = ga.roulette_wheel_selection([a, b], lambda cl: cl.q)

    assert parent1 is a
    assert parent2 is b


def test_generalizing_mutation_with_mu_one_generalizes_non_wildcards():
    cl = FakeClassifier(condition=['0', '#', '1'])
    ga.generalizing_mutation(cl, mu=1.0)
    assert cl.condition.generalized == [0, 2]
    assert list(cl.condition) == ['#', '#', '#']


def test_generalizing_mutation_with_mu_zero_changes_nothing():
    cl = FakeClassifier(condition=['0', '1', '0'])
    ga.generalizing_mutation(cl, mu=0.0)
    assert cl.condition.generalized == []
    assert list(cl.condition) == ['0', '1', '0']


def test_two_point_crossover_swaps_segment_between_parents(monkeypatch):
    parent = FakeClassifier(condition=['a', 'b', 'c'])
    donor = FakeClassifier(condition=['x', 'y', 'z'])
    monkeypatch.setattr(np.random, "choice",
                        lambda *args, **kwargs: np.array([1, 3]))

    ga.two_point_crossover(parent, donor)

    assert list(parent.condition) == ['a', 'y', 'z']
    assert list(donor.condition) == ['x', 'b', 'c']


def test_two_point_crossover_preserves_per_position_elements():
    for seed in range(20):
        np.random.seed(seed)
        parent = FakeClassifier(condition=['a', 'b', 'c'])
        donor = FakeClassifier(condition=['x', 'y', 'z'])
        before = [(parent.condition[i], donor.condition[i]) for i in range(3)]

        ga.two_point_crossover(parent, donor)

        after = [(parent.condition[i], donor.condition[i]) for i in range(3)]
        for old, new in zip(before, after):
            assert sorted(old) == sorted(new)


def test_add_classifier_appends_new_when_no_similar_exists():
    population, match_set, action_set = FakeList(), FakeList(), FakeList()
    new_cl = FakeClassifier(condition=['0', '1', '0'], matches=True)

    ga.add_classifier(new_cl, p=None, population=population,
                      match_set=match_set, action_set=action_set,
                      do_subsumption=False, theta_exp=0)

    assert population == [new_cl]
    assert action_set == [new_cl]
    assert match_set == [new_cl]


def test_add_classifier_increments_numerosity_of_similar_unmarked():
    existing = FakeClassifier(condition=['0', '1', '0'], num=2)
    population, match_set = FakeList([existing]), FakeList()
    action_set = FakeList([existing])
    new_cl = FakeClassifier(condition=['0', '1', '0'])

    ga.add_classifier(new_cl, p=None, population=population,
                      match_set=match_set, action_set=action_set,
                      do_subsumption=False, theta_exp=0)

    assert existing.num == 3
    assert len(population) == 1
    assert len(action_set) == 1


def test_add_classifier_does_not_increment_marked_similar():
    existing = FakeClassifier(condition=['0', '1', '0'], num=2, marked=True)
    population, match_set = FakeList([existing]), FakeList()
    action_set = FakeList([existing])
    new_cl = FakeClassifier(condition=['0', '1', '0'])

    ga.add_classifier(new_cl, p=None, population=population,
                      match_set=match_set, action_set=action_set,
                      do_subsumption=False, theta_exp=0)

    assert existing.num == 2
    assert len(population) == 1


def test_add_classifier_uses_subsumer_when_subsumption_enabled(monkeypatch):
    subsumer = FakeClassifier(condition=['#', '#', '#'], num=1)
    monkeypatch.setattr(ga, "find_subsumers",
                        lambda cl, pop, theta: [subsumer])
    population, match_set = FakeList(), FakeList()
    action_set = FakeList([subsumer])
    new_cl = FakeClassifier(condition=['0', '1', '0'])

    ga.add_classifier(new_cl, p=None, population=population,
                      match_set=match_set, action_set=action_set,
                      do_subsumption=True, theta_exp=5)

    assert subsumer.num == 2
    assert population == []
    assert new_cl not in action_set


def test_delete_classifiers_does_nothing_when_within_threshold():
    population = FakeList()
    match_set = FakeList()
    action_set = FakeList([FakeClassifier(num=1), FakeClassifier(num=1)])
    before = list(action_set)

    ga.delete_classifiers(population, match_set, action_set,
                          insize=1, theta_as=5)

    assert action_set == before


def test_delete_classifiers_removes_lower_quality_classifier(monkeypatch):
    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    worse = FakeClassifier(action=0, q=0.1, num=1)
    better = FakeClassifier(action=1, q=0.9, num=1)
    population = FakeList([worse, better])
    match_set = FakeList([worse, better])
    action_set = FakeList([worse, better])

    ga.delete_classifiers(population, match_set, action_set,
                          insize=1, theta_as=2)

    assert worse not in action_set
    assert better in action_set
    assert worse not in population


def test_delete_classifiers_decrements_numerosity_before_removing(monkeypatch):
    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    cl = FakeClassifier(num=3)
    population = FakeList([cl])
    match_set = FakeList([cl])
    action_set = FakeList([cl])

    ga.delete_classifiers(population, match_set, action_set,
                          insize=1, theta_as=2)

    assert cl.num == 1
    assert action_set == [cl]


def test_is_preferred_to_delete_true_when_candidate_has_much_lower_quality():
    cl_del = FakeClassifier(q=0.5)
    cl = FakeClassifier(q=0.3)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_false_when_candidate_has_much_higher_quality():
    cl_del = FakeClassifier(q=0.5)
    cl = FakeClassifier(q=0.9)
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_true_when_similar_q_and_only_candidate_marked():
    cl_del = FakeClassifier(q=0.5, marked=False)
    cl = FakeClassifier(q=0.5, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_compares_experience_when_similar_q_unmarked():
    cl_del = FakeClassifier(q=0.5, tav=10)
    older = FakeClassifier(q=0.5, tav=20)
    newer = FakeClassifier(q=0.5, tav=5)
    assert ga._is_preferred_to_delete(cl_del, older) is True
    assert ga._is_preferred_to_delete(cl_del, newer) is False


def test_find_similar_returns_first_equal_classifier():
    target = FakeClassifier(condition=['0', '1', '0'], action=2)
    other = FakeClassifier(condition=['1', '1', '1'], action=2)
    match = FakeClassifier(condition=['0', '1', '0'], action=2)
    population = [other, match]

    assert ga._find_similar(target, population) is match


def test_find_similar_returns_none_when_no_equal_classifier():
    target = FakeClassifier(condition=['0', '1', '0'], action=2)
    population = [FakeClassifier(condition=['0', '1', '0'], action=3)]

    assert ga._find_similar(target, population) is None
