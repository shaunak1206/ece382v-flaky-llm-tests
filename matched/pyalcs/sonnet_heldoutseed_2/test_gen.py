import random

import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga


class FakeCfg:
    def __init__(self, length=5, wildcard='#'):
        self.classifier_length = length
        self.classifier_wildcard = wildcard


class FakeCondition(list):
    def __init__(self, items, matches=True, wildcard='#'):
        super().__init__(items)
        self.matches = matches
        self.wildcard = wildcard

    def generalize(self, idx):
        self[idx] = self.wildcard

    def does_match(self, p):
        return self.matches


class FakeCl:
    def __init__(self, num=1, tga=0, q=0.5, tav=0.0, marked=False,
                 condition=None, cfg=None, key=None):
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self.marked = marked
        self.condition = condition if condition is not None else FakeCondition([])
        self.cfg = cfg if cfg is not None else FakeCfg()
        self.key = key

    def is_marked(self):
        return self.marked

    def __eq__(self, other):
        if self.key is None or getattr(other, 'key', None) is None:
            return self is other
        return self.key == other.key

    def __hash__(self):
        return id(self)


class FakeSet(list):
    def expand(self):
        return list(self)

    def safe_remove(self, cl):
        if cl in self:
            self.remove(cl)


# --- should_apply -----------------------------------------------------------

def test_should_apply_none_returns_false():
    assert ga.should_apply(None, 100, 10) is False


def test_should_apply_empty_set_returns_false():
    assert ga.should_apply([], 100, 10) is False


def test_should_apply_zero_numerosity_returns_false():
    assert ga.should_apply([FakeCl(num=0, tga=0)], 100, 10) is False


def test_should_apply_uses_weighted_average_and_threshold():
    action_set = [FakeCl(num=3, tga=10), FakeCl(num=1, tga=50)]
    # weighted average tga = (30 + 50) / 4 = 20
    assert ga.should_apply(action_set, 31, 10) is True
    assert ga.should_apply(action_set, 30, 10) is False  # equal, not greater
    assert ga.should_apply(action_set, 25, 10) is False


# --- set_timestamps ---------------------------------------------------------

def test_set_timestamps_updates_all_classifiers():
    action_set = [FakeCl(tga=1), FakeCl(tga=5), FakeCl(tga=9)]
    ga.set_timestamps(action_set, 42)
    assert [cl.tga for cl in action_set] == [42, 42, 42]


# --- selection --------------------------------------------------------------

def test_weighted_random_choice_respects_pick(monkeypatch):
    choices = {'a': 1, 'b': 3}
    monkeypatch.setattr(random, 'uniform', lambda lo, hi: 0.5)
    assert ga._weighted_random_choice(choices) == 'a'
    monkeypatch.setattr(random, 'uniform', lambda lo, hi: 2.0)
    assert ga._weighted_random_choice(choices) == 'b'


def test_roulette_wheel_selection_returns_two_members():
    population = [FakeCl(q=0.2), FakeCl(q=0.5), FakeCl(q=0.9)]
    random.seed(1)
    p1, p2 = ga.roulette_wheel_selection(population, lambda cl: cl.q ** 3)
    assert p1 in population
    assert p2 in population


def test_roulette_wheel_selection_ignores_zero_fitness():
    good = FakeCl(q=1.0)
    bad = FakeCl(q=0.0)
    random.seed(3)
    for _ in range(50):
        p1, p2 = ga.roulette_wheel_selection([bad, good], lambda cl: cl.q)
        assert p1 is good
        assert p2 is good


# --- mutation ---------------------------------------------------------------

def test_generalizing_mutation_with_mu_one_generalizes_all_specific():
    cl = FakeCl(condition=FakeCondition(['0', '1', '#', '1']))
    ga.generalizing_mutation(cl, 1.0)
    assert list(cl.condition) == ['#', '#', '#', '#']


def test_generalizing_mutation_with_mu_zero_changes_nothing():
    cl = FakeCl(condition=FakeCondition(['0', '1', '#', '1']))
    ga.generalizing_mutation(cl, 0.0)
    assert list(cl.condition) == ['0', '1', '#', '1']


# --- crossover --------------------------------------------------------------

def test_two_point_crossover_swaps_selected_segment(monkeypatch):
    cfg = FakeCfg(length=5)
    parent = FakeCl(condition=FakeCondition(list('aaaaa')), cfg=cfg)
    donor = FakeCl(condition=FakeCondition(list('bbbbb')), cfg=cfg)
    monkeypatch.setattr(np.random, 'choice', lambda *a, **k: [3, 1])

    ga.two_point_crossover(parent, donor)

    assert list(parent.condition) == list('abbaa')
    assert list(donor.condition) == list('baabb')


def test_two_point_crossover_random_preserves_genes():
    cfg = FakeCfg(length=6)
    np.random.seed(0)
    for _ in range(20):
        parent = FakeCl(condition=FakeCondition(list('aaaaaa')), cfg=cfg)
        donor = FakeCl(condition=FakeCondition(list('bbbbbb')), cfg=cfg)
        ga.two_point_crossover(parent, donor)
        for x, y in zip(parent.condition, donor.condition):
            assert {x, y} == {'a', 'b'}
        assert 'b' in parent.condition


# --- add_classifier ---------------------------------------------------------

def test_add_classifier_new_is_added_to_all_sets():
    new = FakeCl(key='new', condition=FakeCondition(['1'], matches=True))
    population, match_set, action_set = FakeSet(), FakeSet(), FakeSet()
    ga.add_classifier(new, object(), population, match_set, action_set,
                      do_subsumption=False, theta_exp=20)
    assert new in population
    assert new in action_set
    assert new in match_set


def test_add_classifier_not_matching_skips_match_set():
    new = FakeCl(key='new', condition=FakeCondition(['1'], matches=False))
    population, match_set, action_set = FakeSet(), FakeSet(), FakeSet()
    ga.add_classifier(new, object(), population, match_set, action_set,
                      False, 20)
    assert new in population and new in action_set
    assert new not in match_set


def test_add_classifier_with_none_match_set():
    new = FakeCl(key='new')
    population, action_set = FakeSet(), FakeSet()
    ga.add_classifier(new, object(), population, None, action_set, False, 20)
    assert new in population and new in action_set


def test_add_classifier_similar_increases_numerosity():
    old = FakeCl(key='same', num=2)
    new = FakeCl(key='same', num=1)
    population, action_set = FakeSet([old]), FakeSet([old])
    ga.add_classifier(new, object(), population, FakeSet(), action_set,
                      False, 20)
    assert old.num == 3
    assert new not in population
    assert len(population) == 1


def test_add_classifier_similar_marked_does_not_increase_numerosity():
    old = FakeCl(key='same', num=2, marked=True)
    new = FakeCl(key='same')
    action_set = FakeSet([old])
    ga.add_classifier(new, object(), FakeSet([old]), FakeSet(), action_set,
                      False, 20)
    assert old.num == 2


def test_add_classifier_uses_subsumer_when_enabled(monkeypatch):
    subsumer = FakeCl(key='subsumer', num=1)
    new = FakeCl(key='new')
    monkeypatch.setattr(ga, 'find_subsumers',
                        lambda cl, pop, theta: [subsumer])
    population, action_set = FakeSet([subsumer]), FakeSet([subsumer])
    ga.add_classifier(new, object(), population, FakeSet(), action_set,
                      True, 20)
    assert subsumer.num == 2
    assert new not in population


# --- find similar -----------------------------------------------------------

def test_find_similar_returns_first_match_or_none():
    a = FakeCl(key='x')
    b = FakeCl(key='x')
    other = FakeCl(key='y')
    assert ga._find_similar(FakeCl(key='x'), [other, a, b]) is a
    assert ga._find_similar(FakeCl(key='z'), [other, a, b]) is None


# --- deletion ---------------------------------------------------------------

def test_is_preferred_to_delete_lower_quality():
    cl_del = FakeCl(q=0.9)
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.5)) is True
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=1.0)) is False


def test_is_preferred_to_delete_similar_quality_uses_mark_and_tav():
    cl_del = FakeCl(q=0.5, tav=1.0)
    assert ga._is_preferred_to_delete(
        cl_del, FakeCl(q=0.55, marked=True)) is True
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.55, tav=2.0)) is True
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.55, tav=0.5)) is False


def test_delete_classifiers_decrements_numerosity(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.0)
    cl = FakeCl(num=3)
    population, action_set = FakeSet([cl]), FakeSet([cl])
    ga.delete_classifiers(population, FakeSet(), action_set,
                          insize=1, theta_as=3)
    assert cl.num == 2
    assert cl in population and cl in action_set


def test_delete_classifiers_removes_single_numerosity_classifier(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.0)
    cl = FakeCl(num=1)
    other = FakeCl(num=1, q=0.9)
    population = FakeSet([cl, other])
    match_set = FakeSet([cl, other])
    action_set = FakeSet([cl, other])
    ga.delete_classifiers(population, match_set, action_set,
                          insize=1, theta_as=2)
    # exactly one classifier (the worse one) was removed everywhere
    assert cl not in population
    assert cl not in match_set
    assert cl not in action_set
    assert other in population and other in action_set


def test_delete_classifiers_noop_when_room_available():
    cl = FakeCl(num=1)
    population, action_set = FakeSet([cl]), FakeSet([cl])
    ga.delete_classifiers(population, FakeSet(), action_set,
                          insize=1, theta_as=10)
    assert cl.num == 1
    assert cl in population
