import random

import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga


class FakeCfg:
    def __init__(self, length=6, wildcard='#'):
        self.classifier_length = length
        self.classifier_wildcard = wildcard


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


class FakeCl:
    def __init__(self, name='a', num=1, tga=0, q=0.5, tav=0.0, marked=False,
                 condition=None, cfg=None):
        self.name = name
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self.marked = marked
        self.condition = condition if condition is not None \
            else FakeCondition(['1', '0', '#'])
        self.cfg = cfg if cfg is not None else FakeCfg(len(self.condition))

    def is_marked(self):
        return self.marked

    def __eq__(self, other):
        return isinstance(other, FakeCl) and self.name == other.name

    def __hash__(self):
        return id(self)


class FakeList(list):
    def expand(self):
        return list(self)

    def safe_remove(self, item):
        if item in self:
            self.remove(item)


def test_should_apply_none_action_set():
    assert ga.should_apply(None, 100, 10) is False


def test_should_apply_empty_action_set():
    assert ga.should_apply([], 100, 10) is False


def test_should_apply_true_when_delay_exceeds_threshold():
    cls = [FakeCl(num=1, tga=0)]
    assert ga.should_apply(cls, 11, 10) is True


def test_should_apply_false_when_delay_equals_threshold():
    cls = [FakeCl(num=1, tga=0)]
    assert ga.should_apply(cls, 10, 10) is False


def test_should_apply_uses_numerosity_weighted_average():
    # weighted avg tga = (0*1 + 30*3) / 4 = 22.5
    cls = [FakeCl(num=1, tga=0), FakeCl(num=3, tga=30)]
    assert ga.should_apply(cls, 32, 10) is False   # 9.5 delay
    assert ga.should_apply(cls, 33, 10) is True    # 10.5 delay


def test_set_timestamps_updates_all():
    cls = [FakeCl(name='a', tga=1), FakeCl(name='b', tga=5)]
    ga.set_timestamps(cls, 42)
    assert [cl.tga for cl in cls] == [42, 42]


def test_weighted_random_choice_respects_pick(monkeypatch):
    choices = {'a': 1.0, 'b': 1.0}
    monkeypatch.setattr(ga.random, 'uniform', lambda lo, hi: 0.5)
    assert ga._weighted_random_choice(choices) == 'a'
    monkeypatch.setattr(ga.random, 'uniform', lambda lo, hi: 1.5)
    assert ga._weighted_random_choice(choices) == 'b'


def test_roulette_wheel_selection_returns_two_population_members():
    random.seed(1)
    pop = [FakeCl(name=str(i), q=0.1 * (i + 1)) for i in range(4)]
    p1, p2 = ga.roulette_wheel_selection(pop, lambda cl: cl.q ** 3)
    assert p1 in pop
    assert p2 in pop


def test_roulette_wheel_selection_ignores_zero_fitness():
    random.seed(2)
    good = FakeCl(name='good', q=1.0)
    bad = FakeCl(name='bad', q=0.0)
    for _ in range(20):
        p1, p2 = ga.roulette_wheel_selection([bad, good], lambda cl: cl.q)
        assert p1 is good
        assert p2 is good


def test_generalizing_mutation_mu_one_generalizes_all_specific():
    cl = FakeCl(condition=FakeCondition(['1', '0', '#', '1']))
    ga.generalizing_mutation(cl, 1.0)
    assert list(cl.condition) == ['#', '#', '#', '#']
    # wildcard position 2 must be skipped
    assert cl.condition.generalized == [0, 1, 3]


def test_generalizing_mutation_mu_zero_changes_nothing():
    cl = FakeCl(condition=FakeCondition(['1', '0', '#', '1']))
    ga.generalizing_mutation(cl, 0.0)
    assert list(cl.condition) == ['1', '0', '#', '1']
    assert cl.condition.generalized == []


def test_two_point_crossover_swaps_segment(monkeypatch):
    monkeypatch.setattr(np.random, 'choice', lambda a, n, replace: [1, 3])
    cfg = FakeCfg(5)
    parent = FakeCl(condition=FakeCondition(list('AAAAA')), cfg=cfg)
    donor = FakeCl(condition=FakeCondition(list('BBBBB')), cfg=cfg)
    ga.two_point_crossover(parent, donor)
    assert ''.join(parent.condition) == 'ABBAA'
    assert ''.join(donor.condition) == 'BAABB'


def test_two_point_crossover_sorts_cut_points(monkeypatch):
    monkeypatch.setattr(np.random, 'choice', lambda a, n, replace: [4, 2])
    cfg = FakeCfg(5)
    parent = FakeCl(condition=FakeCondition(list('AAAAA')), cfg=cfg)
    donor = FakeCl(condition=FakeCondition(list('BBBBB')), cfg=cfg)
    ga.two_point_crossover(parent, donor)
    assert ''.join(parent.condition) == 'AABBA'
    assert ''.join(donor.condition) == 'BBAAB'


def test_two_point_crossover_random_preserves_alleles_per_position():
    np.random.seed(3)
    cfg = FakeCfg(6)
    for _ in range(25):
        parent = FakeCl(condition=FakeCondition(list('AAAAAA')), cfg=cfg)
        donor = FakeCl(condition=FakeCondition(list('BBBBBB')), cfg=cfg)
        ga.two_point_crossover(parent, donor)
        assert len(parent.condition) == 6
        assert len(donor.condition) == 6
        for a, b in zip(parent.condition, donor.condition):
            assert {a, b} == {'A', 'B'}
        assert 'B' in parent.condition


def test_find_similar_returns_first_equal_or_none():
    first = FakeCl(name='x')
    second = FakeCl(name='x')
    other = FakeCl(name='y')
    probe = FakeCl(name='x')
    assert ga._find_similar(probe, [other, first, second]) is first
    assert ga._find_similar(FakeCl(name='z'), [other, first]) is None


def test_find_old_classifier_without_subsumption_uses_similar(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError('find_subsumers must not be called')

    monkeypatch.setattr(ga, 'find_subsumers', boom)
    existing = FakeCl(name='x')
    result = ga._find_old_classifier([existing], FakeCl(name='x'), False, 20)
    assert result is existing


def test_find_old_classifier_prefers_subsumer(monkeypatch):
    subsumer = FakeCl(name='general')
    monkeypatch.setattr(ga, 'find_subsumers',
                        lambda cl, pop, theta: [subsumer])
    similar = FakeCl(name='x')
    result = ga._find_old_classifier([similar], FakeCl(name='x'), True, 20)
    assert result is subsumer


def test_add_classifier_new_is_added_to_all_matching_sets():
    population, match_set, action_set = FakeList(), FakeList(), FakeList()
    new = FakeCl(name='new', condition=FakeCondition(['1'], matches=True))
    ga.add_classifier(new, object(), population, match_set, action_set,
                      False, 20)
    assert population == [new]
    assert action_set == [new]
    assert match_set == [new]


def test_add_classifier_new_not_matching_skips_match_set():
    population, match_set, action_set = FakeList(), FakeList(), FakeList()
    new = FakeCl(name='new', condition=FakeCondition(['1'], matches=False))
    ga.add_classifier(new, object(), population, match_set, action_set,
                      False, 20)
    assert population == [new]
    assert action_set == [new]
    assert match_set == []


def test_add_classifier_existing_increments_numerosity():
    old = FakeCl(name='x', num=2)
    population, match_set, action_set = (
        FakeList([old]), FakeList([old]), FakeList([old]))
    ga.add_classifier(FakeCl(name='x'), object(), population, match_set,
                      action_set, False, 20)
    assert old.num == 3
    assert len(population) == 1
    assert len(action_set) == 1


def test_add_classifier_existing_marked_not_incremented():
    old = FakeCl(name='x', num=2, marked=True)
    population, match_set, action_set = (
        FakeList([old]), FakeList([old]), FakeList([old]))
    ga.add_classifier(FakeCl(name='x'), object(), population, match_set,
                      action_set, False, 20)
    assert old.num == 2
    assert len(population) == 1


def test_is_preferred_to_delete_by_quality():
    cl_del = FakeCl(q=0.8)
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.5)) is True
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.95)) is False


def test_is_preferred_to_delete_marked_and_tav_tiebreaks():
    cl_del = FakeCl(q=0.5, marked=False, tav=1.0)
    assert ga._is_preferred_to_delete(
        cl_del, FakeCl(q=0.55, marked=True, tav=0.0)) is True
    assert ga._is_preferred_to_delete(
        cl_del, FakeCl(q=0.5, marked=False, tav=2.0)) is True
    assert ga._is_preferred_to_delete(
        cl_del, FakeCl(q=0.5, marked=False, tav=0.5)) is False


def test_delete_classifiers_decrements_numerosity(monkeypatch):
    monkeypatch.setattr(ga.random, 'random', lambda: 0.0)
    cl = FakeCl(name='a', num=3)
    population, match_set, action_set = (
        FakeList([cl]), FakeList([cl]), FakeList([cl]))
    ga.delete_classifiers(population, match_set, action_set, 1, 3)
    assert cl.num == 2
    assert population == [cl]


def test_delete_classifiers_removes_single_numerosity_classifier(monkeypatch):
    monkeypatch.setattr(ga.random, 'random', lambda: 0.0)
    cl = FakeCl(name='a', num=1)
    population, match_set, action_set = (
        FakeList([cl]), FakeList([cl]), FakeList([cl]))
    ga.delete_classifiers(population, match_set, action_set, 1, 1)
    assert population == []
    assert match_set == []
    assert action_set == []
