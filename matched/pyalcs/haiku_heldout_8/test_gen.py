import random
from types import SimpleNamespace

import numpy as np

from lcs.strategies import genetic_algorithms as ga


class FakeCondition:
    """Minimal stand-in for a classifier condition."""

    def __init__(self, values, matches=True):
        self.values = list(values)
        self.matches = matches
        self.generalized = []

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def generalize(self, idx):
        self.generalized.append(idx)
        self.values[idx] = '#'

    def does_match(self, p):
        return self.matches


class FakeClassifier:
    """Minimal stand-in for a classifier. Equality is based on `key`."""

    def __init__(self, key, num=1, q=0.5, tga=0, tav=0,
                 marked=False, condition=None, cfg=None):
        self.key = key
        self.num = num
        self.q = q
        self.tga = tga
        self.tav = tav
        self.marked = marked
        self.condition = condition
        self.cfg = cfg

    def is_marked(self):
        return self.marked

    def __eq__(self, other):
        if not isinstance(other, FakeClassifier):
            return NotImplemented
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return f"FakeClassifier({self.key!r})"


class FakeList(list):
    """List that supports the extra methods used by delete_classifiers."""

    def expand(self):
        return list(self)

    def safe_remove(self, item):
        if item in self:
            self.remove(item)


def make_cl(key, values=('1', '0'), num=1, q=0.5, tga=0, tav=0,
            marked=False, matches=True):
    cfg = SimpleNamespace(classifier_wildcard='#',
                          classifier_length=len(values))
    return FakeClassifier(key, num=num, q=q, tga=tga, tav=tav,
                          marked=marked,
                          condition=FakeCondition(values, matches),
                          cfg=cfg)


# ---------------------------------------------------------------- should_apply

def test_should_apply_returns_false_for_none_action_set():
    assert ga.should_apply(None, time=100, theta_ga=1) is False


def test_should_apply_returns_false_for_empty_or_zero_numerosity():
    assert ga.should_apply(FakeList(), time=100, theta_ga=1) is False
    zero = FakeList([make_cl('a', num=0, tga=0)])
    assert ga.should_apply(zero, time=100, theta_ga=1) is False


def test_should_apply_true_when_average_delay_exceeds_threshold():
    action_set = FakeList([make_cl('a', num=1, tga=0)])
    assert ga.should_apply(action_set, time=10, theta_ga=5) is True


def test_should_apply_false_when_delay_equals_threshold():
    # Comparison is strict: delay must be greater than theta_ga.
    action_set = FakeList([make_cl('a', num=1, tga=0)])
    assert ga.should_apply(action_set, time=5, theta_ga=5) is False


def test_should_apply_uses_numerosity_weighted_average():
    # Weighted average tga = (2*1 + 8*3) / 4 = 6.5, so delay = 10 - 6.5 = 3.5
    action_set = FakeList([
        make_cl('a', num=1, tga=2),
        make_cl('b', num=3, tga=8),
    ])
    assert ga.should_apply(action_set, time=10, theta_ga=3) is True
    assert ga.should_apply(action_set, time=10, theta_ga=4) is False


# -------------------------------------------------------------- set_timestamps

def test_set_timestamps_updates_all_classifiers():
    cls = [make_cl('a', tga=0), make_cl('b', tga=3), make_cl('c', tga=7)]
    ga.set_timestamps(cls, epoch=42)
    assert all(cl.tga == 42 for cl in cls)


# ------------------------------------------------------------- roulette wheel

def test_weighted_random_choice_selects_by_cumulative_weight(monkeypatch):
    choices = {'a': 1, 'b': 2, 'c': 3}  # total weight 6
    for pick, expected in [(0.5, 'a'), (1.5, 'b'), (1.0, 'b'), (3.5, 'c')]:
        monkeypatch.setattr(random, 'uniform', lambda lo, hi, p=pick: p)
        assert ga._weighted_random_choice(choices) == expected


def test_roulette_wheel_never_selects_zero_fitness_classifier():
    good = make_cl('good')
    bad = make_cl('bad')
    population = [bad, good]
    fitness = {bad: 0.0, good: 1.0}

    for _ in range(50):
        parent1, parent2 = ga.roulette_wheel_selection(
            population, lambda cl: fitness[cl])
        assert parent1 is good
        assert parent2 is good


# --------------------------------------------------------- generalizing mutation

def test_generalizing_mutation_full_probability_generalizes_all_non_wildcards():
    cl = make_cl('a', values=['1', '0', '1'])
    ga.generalizing_mutation(cl, mu=1.0)
    assert cl.condition.generalized == [0, 1, 2]
    assert cl.condition.values == ['#', '#', '#']


def test_generalizing_mutation_zero_probability_changes_nothing():
    cl = make_cl('a', values=['1', '0', '1'])
    ga.generalizing_mutation(cl, mu=0.0)
    assert cl.condition.generalized == []
    assert cl.condition.values == ['1', '0', '1']


def test_generalizing_mutation_skips_wildcards():
    cl = make_cl('a', values=['#', '1', '#', '0'])
    ga.generalizing_mutation(cl, mu=1.0)
    assert cl.condition.generalized == [1, 3]


# -------------------------------------------------------------- two-point crossover

def _crossover_pair(length=6):
    cfg = SimpleNamespace(classifier_wildcard='#', classifier_length=length)
    parent = FakeClassifier(
        'p', condition=FakeCondition([f'p{i}' for i in range(length)]),
        cfg=cfg)
    donor = FakeClassifier(
        'd', condition=FakeCondition([f'd{i}' for i in range(length)]),
        cfg=cfg)
    return parent, donor


def test_two_point_crossover_swaps_one_contiguous_segment():
    np.random.seed(0)
    parent, donor = _crossover_pair(length=6)

    ga.two_point_crossover(parent, donor)

    swapped = [i for i in range(6) if parent.condition[i] == f'd{i}']
    assert swapped, "at least one position must be exchanged"
    assert swapped == list(range(swapped[0], swapped[-1] + 1))

    for i in range(6):
        if i in swapped:
            assert parent.condition[i] == f'd{i}'
            assert donor.condition[i] == f'p{i}'
        else:
            assert parent.condition[i] == f'p{i}'
            assert donor.condition[i] == f'd{i}'


def test_two_point_crossover_preserves_genes_across_seeds():
    for seed in range(20):
        np.random.seed(seed)
        parent, donor = _crossover_pair(length=6)
        ga.two_point_crossover(parent, donor)
        for i in range(6):
            assert sorted([parent.condition[i], donor.condition[i]]) == \
                sorted([f'p{i}', f'd{i}'])


# -------------------------------------------------------------- add_classifier

def test_add_classifier_new_is_appended_to_all_lists_when_matching():
    population, action_set, match_set = FakeList(), FakeList(), FakeList()
    new_cl = make_cl('new', matches=True)

    ga.add_classifier(new_cl, object(), population, match_set, action_set,
                      do_subsumption=False, theta_exp=0)

    assert new_cl in population
    assert new_cl in action_set
    assert new_cl in match_set


def test_add_classifier_new_not_matching_is_not_added_to_match_set():
    population, action_set, match_set = FakeList(), FakeList(), FakeList()
    new_cl = make_cl('new', matches=False)

    ga.add_classifier(new_cl, object(), population, match_set, action_set,
                      do_subsumption=False, theta_exp=0)

    assert new_cl in population
    assert new_cl in action_set
    assert new_cl not in match_set


def test_add_classifier_existing_unmarked_increments_numerosity():
    existing = make_cl('k', num=2)
    population = FakeList([existing])
    action_set = FakeList([existing])
    match_set = FakeList()
    new_cl = make_cl('k')

    ga.add_classifier(new_cl, object(), population, match_set, action_set,
                      do_subsumption=False, theta_exp=0)

    assert existing.num == 3
    assert len(population) == 1
    assert new_cl not in population


def test_add_classifier_existing_marked_does_not_increment_numerosity():
    existing = make_cl('k', num=2, marked=True)
    population = FakeList([existing])
    action_set = FakeList([existing])
    match_set = FakeList()

    ga.add_classifier(make_cl('k'), object(), population, match_set,
                      action_set, do_subsumption=False, theta_exp=0)

    assert existing.num == 2


# ------------------------------------------------------------- similar search

def test_find_similar_returns_first_equal_classifier_or_none():
    first = make_cl('x', num=1)
    second = make_cl('x', num=5)
    population = [make_cl('y'), first, second]

    assert ga._find_similar(make_cl('x'), population) is first
    assert ga._find_similar(make_cl('z'), population) is None


# ------------------------------------------------------------ delete_classifiers

def test_delete_classifiers_noop_when_under_threshold():
    cl = make_cl('a', num=1)
    population = FakeList([cl])
    action_set = FakeList([cl])
    match_set = FakeList([cl])

    random.seed(0)
    ga.delete_classifiers(population, match_set, action_set,
                          insize=1, theta_as=5)

    assert cl.num == 1
    assert cl in population and cl in action_set and cl in match_set


def test_delete_classifiers_decrements_numerosity_of_multi_copy_classifier():
    cl = make_cl('a', num=3)
    population = FakeList([cl])
    action_set = FakeList([cl])
    match_set = FakeList()

    random.seed(1)
    # 0 + 3 > 2 triggers one deletion step, which only decrements num.
    ga.delete_classifiers(population, match_set, action_set,
                          insize=0, theta_as=2)

    assert cl.num == 2
    assert cl in population
    assert cl in action_set


def test_delete_classifiers_removes_single_copy_from_all_lists():
    cl = make_cl('a', num=1)
    other = make_cl('b', num=1)
    population = FakeList([cl, other])
    action_set = FakeList([cl])
    match_set = FakeList([cl])

    random.seed(2)
    ga.delete_classifiers(population, match_set, action_set,
                          insize=1, theta_as=1)

    assert cl not in population
    assert cl not in action_set
    assert cl not in match_set
    assert other in population


# ------------------------------------------------------- _is_preferred_to_delete

def test_is_preferred_to_delete_true_for_clearly_lower_fitness():
    cl_del = make_cl('del', q=0.5)
    cl = make_cl('low', q=0.2)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_false_for_clearly_higher_fitness():
    cl_del = make_cl('del', q=0.5)
    cl = make_cl('high', q=0.9)
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_marked_preferred_when_fitness_close():
    cl_del = make_cl('del', q=0.5, marked=False)
    cl = make_cl('marked', q=0.45, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True
