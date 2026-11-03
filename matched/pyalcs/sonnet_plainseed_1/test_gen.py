import random
from types import SimpleNamespace

import numpy as np
import pytest

from lcs import Perception
from lcs.strategies import genetic_algorithms as ga


WILDCARD = '#'


class FakeCondition(list):
    def __init__(self, items=(), matches=True, specificity=0, subsumes=True):
        super().__init__(items)
        self.matches = matches
        self.specificity = specificity
        self._subsumes = subsumes

    def generalize(self, idx):
        self[idx] = WILDCARD

    def does_match(self, p):
        return self.matches

    def subsumes(self, other):
        return self._subsumes


class FakeEffect:
    def subsumes(self, other):
        return True


class FakeCl:
    def __init__(self, key='a', num=1, tga=0, q=0.5, tav=0.0, marked=False,
                 exp=0, reliable=False, condition=None, action=0,
                 general=True):
        self.key = key
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self.marked = marked
        self.exp = exp
        self.reliable = reliable
        self.condition = condition if condition is not None \
            else FakeCondition()
        self.action = action
        self.general = general
        self.effect = FakeEffect()
        self.cfg = SimpleNamespace(classifier_wildcard=WILDCARD,
                                   classifier_length=len(self.condition))

    def __eq__(self, other):
        return isinstance(other, FakeCl) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return 'FakeCl(%s)' % self.key

    def is_marked(self):
        return self.marked

    def is_reliable(self):
        return self.reliable

    def is_more_general(self, other):
        return self.general


class FakeSet(list):
    def expand(self):
        result = []
        for cl in self:
            result.extend([cl] * cl.num)
        return result

    def safe_remove(self, cl):
        if cl in self:
            self.remove(cl)


# ---------------------------------------------------------------- should_apply

def test_should_apply_none_action_set():
    assert ga.should_apply(None, 100, 10) is False

def test_should_apply_zero_numerosity():
    action_set = [FakeCl(num=0, tga=0)]
    assert ga.should_apply(action_set, 100, 10) is False


def test_should_apply_true_when_delay_exceeds_threshold():
    action_set = [FakeCl(num=1, tga=0), FakeCl(key='b', num=1, tga=10)]
    # average tga = 5, time - 5 = 45 > 20
    assert ga.should_apply(action_set, 50, 20) is True


def test_should_apply_false_when_delay_equals_threshold():
    action_set = [FakeCl(num=1, tga=10)]
    # 30 - 10 = 20, not strictly greater than 20
    assert ga.should_apply(action_set, 30, 20) is False


def test_should_apply_uses_numerosity_weighted_average():
    heavy = FakeCl(key='h', num=9, tga=100)
    light = FakeCl(key='l', num=1, tga=0)
    # weighted average tga = 90; time 100 -> delay 10
    assert ga.should_apply([heavy, light], 100, 5) is True
    assert ga.should_apply([heavy, light], 100, 10) is False


# -------------------------------------------------------------- set_timestamps

def test_set_timestamps_updates_all_classifiers():
    action_set = [FakeCl(key=str(i), tga=i) for i in range(4)]
    ga.set_timestamps(action_set, 77)
    assert [cl.tga for cl in action_set] == [77] * 4

# ------------------------------------------------------ roulette wheel selection

def test_weighted_random_choice_picks_by_cumulative_weight(monkeypatch):
    choices = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    picks = {0.5: 'a', 1.5: 'b', 3.5: 'c', 5.9: 'c'}
    for pick, expected in picks.items():
        monkeypatch.setattr(random, 'uniform', lambda lo, hi, p=pick: p)
        assert ga._weighted_random_choice(choices) == expected
def test_roulette_wheel_selection_only_positive_fitness_selected():
    random.seed(1234)
    winner = FakeCl(key='win', q=1.0)
    losers = [FakeCl(key='l%d' % i, q=0.0) for i in range(3)]
    for _ in range(20):
        p1, p2 = ga.roulette_wheel_selection(losers + [winner],
                                             lambda cl: cl.q)
        assert p1 is winner
        assert p2 is winner

# ---------------------------------------------------------- generalizing_mutation

def test_generalizing_mutation_mu_one_generalizes_all_specific():
    cl = FakeCl(condition=FakeCondition(['0', '1', '0', '1']))
    ga.generalizing_mutation(cl, 1.0)
    assert list(cl.condition) == [WILDCARD] * 4


def test_generalizing_mutation_mu_zero_changes_nothing():
    cl = FakeCl(condition=FakeCondition(['0', '1', WILDCARD, '1']))
    ga.generalizing_mutation(cl, 0.0)
    assert list(cl.condition) == ['0', '1', WILDCARD, '1']

# ---------------------------------------------------------- two_point_crossover

def test_two_point_crossover_swaps_selected_segment(monkeypatch):
    monkeypatch.setattr(np.random, 'choice',
                        lambda *a, **k: np.array([3, 1]))
    parent = FakeCl(key='p', condition=FakeCondition(list('00000')))
    donor = FakeCl(key='d', condition=FakeCondition(list('11111')))
    ga.two_point_crossover(parent, donor)
    assert list(parent.condition) == list('01100')
    assert list(donor.condition) == list('10011')

def test_two_point_crossover_preserves_genes_with_random_points():
    np.random.seed(7)
    parent = FakeCl(key='p', condition=FakeCondition(list('01010101')))
    donor = FakeCl(key='d', condition=FakeCondition(list('11110000')))
    before = sorted(list(parent.condition) + list(donor.condition))
    ga.two_point_crossover(parent, donor)
    after = sorted(list(parent.condition) + list(donor.condition))
    assert before == after
    assert len(parent.condition) == 8
    assert len(donor.condition) == 8


# --------------------------------------------------------------- add_classifier

def test_add_classifier_new_classifier_appended_everywhere():
    new = FakeCl(key='new', condition=FakeCondition(['0'], matches=True))
    population, match_set, action_set = FakeSet(), FakeSet(), FakeSet()
    ga.add_classifier(new, Perception(['0']), population, match_set,
                      action_set, False, 20)
    assert population == [new]
    assert action_set == [new]
    assert match_set == [new]


def test_add_classifier_not_matching_perception_skips_match_set():
    new = FakeCl(key='new', condition=FakeCondition(['1'], matches=False))
    population, match_set, action_set = FakeSet(), FakeSet(), FakeSet()
    ga.add_classifier(new, Perception(['0']), population, match_set,
                      action_set, False, 20)
    assert population == [new]
    assert action_set == [new]
    assert match_set == []

def test_add_classifier_similar_increments_numerosity():
    old = FakeCl(key='same', num=2)
    new = FakeCl(key='same', num=1)
    population, match_set, action_set = (FakeSet([old]), FakeSet([old]),
                                         FakeSet([old]))
    ga.add_classifier(new, Perception(['0']), population, match_set,
                      action_set, False, 20)
    assert old.num == 3
    assert len(population) == 1
    assert len(action_set) == 1


def test_add_classifier_similar_marked_does_not_increment():
    old = FakeCl(key='same', num=2, marked=True)
    new = FakeCl(key='same')
    population, action_set = FakeSet([old]), FakeSet([old])
    ga.add_classifier(new, Perception(['0']), population, None,
                      action_set, False, 20)
    assert old.num == 2
    assert population == [old]


def test_add_classifier_subsumed_by_general_classifier():
    subsumer = FakeCl(key='general', num=1, exp=50, reliable=True,
                      condition=FakeCondition(['#'], specificity=0))
    new = FakeCl(key='specific', condition=FakeCondition(['0'],
                                                         specificity=1))
    population, action_set = FakeSet([subsumer]), FakeSet([subsumer])
    ga.add_classifier(new, Perception(['0']), population, None,
                      action_set, True, 20)
    assert subsumer.num == 2
    assert population == [subsumer]


# ------------------------------------------------------ _find_old_classifier etc.

def test_find_old_classifier_prefers_most_general_subsumer():
    specific = FakeCl(key='s', exp=50, reliable=True,
                      condition=FakeCondition(['0', '#'], specificity=1))
    general = FakeCl(key='g', exp=50, reliable=True,
                     condition=FakeCondition(['#', '#'], specificity=0))
    new = FakeCl(key='n', condition=FakeCondition(['0', '0'], specificity=2))
    found = ga._find_old_classifier([specific, general], new, True, 20)
    assert found is general


def test_find_old_classifier_falls_back_to_similar_without_subsumption():
    similar = FakeCl(key='x', exp=50, reliable=True)
    other = FakeCl(key='y')
    new = FakeCl(key='x')
    assert ga._find_old_classifier([other, similar], new, False, 20) is similar
# ------------------------------------------------------------- delete_classifiers

def test_delete_classifiers_removes_worst_single_numerosity(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.0)
    good = FakeCl(key='good', num=1, q=0.9)
    bad = FakeCl(key='bad', num=1, q=0.1)
    population = FakeSet([good, bad])
    match_set = FakeSet([good, bad])
    action_set = FakeSet([good, bad])
    ga.delete_classifiers(population, match_set, action_set, 1, 2)
    assert population == [good]
    assert match_set == [good]
    assert action_set == [good]


def test_delete_classifiers_decrements_numerosity_when_above_one(monkeypatch):
    monkeypatch.setattr(random, 'random', lambda: 0.0)
    cl = FakeCl(key='a', num=3)
    population, action_set = FakeSet([cl]), FakeSet([cl])
    ga.delete_classifiers(population, None, action_set, 1, 3)
    assert cl.num == 2
    assert population == [cl]
    assert action_set == [cl]


def test_delete_classifiers_nothing_to_do_when_room_available():
    cl = FakeCl(key='a', num=2)
    population, action_set = FakeSet([cl]), FakeSet([cl])
    ga.delete_classifiers(population, None, action_set, 1, 10)
    assert cl.num == 2
    assert population == [cl]


# ------------------------------------------------------ _is_preferred_to_delete

def test_is_preferred_to_delete_much_lower_quality():
    cl_del = FakeCl(q=0.9)
    cl = FakeCl(q=0.5)
    assert ga._is_preferred_to_delete(cl_del, cl) is True

def test_is_preferred_to_delete_similar_quality_marked_candidate():
    cl_del = FakeCl(q=0.5, marked=False)
    cl = FakeCl(q=0.55, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_similar_quality_uses_tav():
    cl_del = FakeCl(q=0.5, tav=1.0)
    higher_tav = FakeCl(q=0.5, tav=5.0)
    lower_tav = FakeCl(q=0.5, tav=0.5)
    assert ga._is_preferred_to_delete(cl_del, higher_tav) is True
    assert ga._is_preferred_to_delete(cl_del, lower_tav) is False
