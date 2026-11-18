import builtins
import random
import types

import numpy as np
import pytest

from lcs import Perception
from lcs.strategies import genetic_algorithms as ga


class DummyCondition:
    def __init__(self, cond_list, cfg):
        # store list or tuple of values
        self._cond = list(cond_list)
        self.cfg = cfg

    def __getitem__(self, idx):
        return self._cond[idx]

    def __setitem__(self, idx, val):
        self._cond[idx] = val

    def __eq__(self, other):
        if not isinstance(other, DummyClassifier):
            return False
        return (self._cond == other.condition._cond)

    def generalize(self, idx):
        self._cond[idx] = self.cfg.classifier_wildcard

    def does_match(self, perception):
        # For testing match_set append logic
        return True

    def __len__(self):
        return len(self._cond)


class DummyClassifier:
    def __init__(self, cond, num=1, tga=0, q=1.0, tav=0, marked=False,
                 cfg=None):
        self.condition = DummyCondition(cond, cfg)
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self._marked = marked
        self.cfg = cfg

    def __eq__(self, other):
        if not isinstance(other, DummyClassifier):
            return False
        # Matches if condition is same and num, q, tav, marked are same for easier tests.
        return (self.condition == other.condition and
                self.num == other.num and
                self.q == other.q and
                self.tav == other.tav and
                self._marked == other._marked)

    def is_marked(self):
        return self._marked

    def __hash__(self):
        # To allow usage as dict keys
        return id(self)


class DummyList(list):
    """
    List subclass with methods called safe_remove (same as remove, ignoring if item not present)
    and expand which returns all items in self (to simulate action_set.expand())
    """

    def safe_remove(self, item):
        try:
            self.remove(item)
        except ValueError:
            pass

    def expand(self):
        return self


class DummyCfg:
    def __init__(self):
        self.classifier_wildcard = "#"
        self.classifier_length = 5


@pytest.fixture
def cfg():
    return DummyCfg()


@pytest.fixture
def population(cfg):
    # create some dummy classifiers
    return [
        DummyClassifier(("A", "B", "C", "D", "E"), num=2, tga=5, q=0.8, tav=10,
                        marked=False, cfg=cfg),
        DummyClassifier(("A", "B", "C", "D", "#"), num=3, tga=3, q=0.5, tav=20,
                        marked=True, cfg=cfg),
    ]


@pytest.fixture
def action_set(cfg):
    cl1 = DummyClassifier(("A", "B", "C", "D", "E"), num=1, tga=7, q=0.6,
                        tav=15, marked=False, cfg=cfg)
    cl2 = DummyClassifier(("A", "B", "C", "D", "F"), num=1, tga=8, q=0.7,
                        tav=12, marked=False, cfg=cfg)
    return DummyList([cl1, cl2])


@pytest.fixture
def match_set():
    return DummyList()


def test_should_apply_none_action_set():
    assert ga.should_apply(None, time=100, theta_ga=5) is False


def test_should_apply_zero_numerosity_returns_false():
    class Cl:
        def __init__(self):
            self.tga = 10
            self.num = 0
    aset = [Cl(), Cl()]
    assert ga.should_apply(azet:=aset, time=20, theta_ga=2) is False


def test_should_apply_returns_true_when_avg_delay_exceeds_theta_ga():
    class Cl:
        def __init__(self, tga, num):
            self.tga = tga
            self.num = num

    aset = [Cl(0, 2), Cl(5, 3)]  # weighted avg = (0*2 + 5*3)/5 = 3.0
    time = 10
    theta_ga = 5

    # time - avg = 7 > theta_ga(5)
    assert ga.should_apply(azet:=aset, time=time, theta_ga=theta_ga) is True


def test_should_apply_returns_false_when_avg_delay_not_exceed():
    class Cl:
        def __init__(self, tga, num):
            self.tga = tga
            self.num = num

    aset = [Cl(9, 1), Cl(10, 3)]  # avg = (9+30)/4 = 9.75
    time = 13
    theta_ga = 5

    # time - avg = 3.25 <= theta_ga
    assert ga.should_apply(azet:=aset, time=time, theta_ga=theta_ga) is False


def test_set_timestamps_updates_tga_in_action_set(action_set):
    epoch = 12345
    ga.set_timestamps(action_set, epoch)

    for cl in action_set:
        assert cl.tga == epoch


def test_roulette_wheel_selection_returns_two_parents(population):
    # Define a simple fitness returning the q cube (as described)
    fitness = lambda cl: cl.q ** 3

    parent1, parent2 = ga.roulette_wheel_selection(population, fitness)
    assert parent1 in population
    assert parent2 in population
    # Parents can be the same (duplicates allowed)


def test_generalizing_mutation_random_calls(monkeypatch, cfg):
    # classifier with condition all specific (no wildcard initially)
    cond = ['a', 'b', 'c', 'd', 'e']
    cl = DummyClassifier(cond, cfg=cfg)
    mu = 1.0  # mutate all

    called_idxs = []

    def fake_random():
        # Always less than mu=1.0 to force generalization
        return 0.0

    def fake_generalize(idx):
        called_idxs.append(idx)
        cl.condition._cond[idx] = cl.cfg.classifier_wildcard

    monkeypatch.setattr(random, "random", fake_random)
    monkeypatch.setattr(cl.condition, "generalize", fake_generalize)

    ga.generalizing_mutation(cl, mu)

    # Should have called generalize once for each element not wildcard initially
    assert called_idxs == [0, 1, 2, 3, 4]
    # All entries replaced by wildcard now
    assert all(el == cl.cfg.classifier_wildcard for el in cl.condition._cond)


def test_generalizing_mutation_does_not_generalize_if_random_high(monkeypatch, cfg):
    # classifier with condition all specific (no wildcard initially)
    cond = ['a', 'b', 'c', 'd', 'e']
    cl = DummyClassifier(cond, cfg=cfg)
    mu = 0.0  # zero chance

    called = False

    def fake_random():
        # Always more than mu=0, so no generalize
        return 1.0

    def fake_generalize(idx):
        nonlocal called
        called = True

    monkeypatch.setattr(random, "random", fake_random)
    monkeypatch.setattr(cl.condition, "generalize", fake_generalize)

    ga.generalizing_mutation(cl, mu)

    assert called is False
    # Condition unchanged
    assert cl.condition._cond == cond


def test_two_point_crossover_swaps_chromosomes(cfg):
    parent = DummyClassifier(("A", "B", "C", "D", "E"), cfg=cfg)
    donor = DummyClassifier(("1", "2", "3", "4", "5"), cfg=cfg)

    length = parent.cfg.classifier_length
    # We monkeypatch np.random.choice to always pick (1,4)
    def fake_choice(seq, size, replace):
        # Return fixed two distinct indices sorted
        return np.array([1, 4])

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(np.random, "choice", fake_choice)

    ga.two_point_crossover(parent, donor)

    # The positions 1..3 inclusive swapped from donor to parent and vice versa
    expected_parent = ["A", "2", "3", "4", "E"]
    expected_donor = ["1", "B", "C", "D", "5"]

    assert parent.condition._cond == expected_parent
    assert donor.condition._cond == expected_donor

    monkeypatch.undo()


def test_add_classifier_appends_new_classifier(population, action_set, match_set, cfg):
    cl = DummyClassifier(("X", "Y", "Z", "W", "V"), cfg=cfg)

    # Disable subsumption and no old classifier found, so new should be appended
    ga.add_classifier(cl, Perception("whatever"), population, match_set, action_set,
                      do_subsumption=False, theta_exp=0)

    assert cl in population
    assert cl in action_set
    assert cl in match_set


def test_add_classifier_increments_num_if_old_classifier(monkeypatch, action_set, population, cfg):
    cl_new = DummyClassifier(("A", "B", "C", "D", "E"), num=1, tga=0, q=1.0,
                            tav=0, marked=False, cfg=cfg)

    # Patch _find_old_classifier to return an existing classifier from action_set
    old_cl = action_set[0]

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old_cl)

    old_num_before = old_cl.num
    ga.add_classifier(cl_new, Perception("whatever"), population, None, action_set,
                      do_subsumption=True, theta_exp=0)

    # old_cl num incremented by one since not marked
    assert old_cl.num == old_num_before + 1
    # cl_new not appended since old_cl found
    assert cl_new not in population


def test_add_classifier_does_not_increment_if_marked(monkeypatch, action_set, population, cfg):
    cl_new = DummyClassifier(("A", "B", "C", "D", "E"), num=1, tga=0, q=1.0,
                            tav=0, marked=False, cfg=cfg)

    old_cl = DummyClassifier(("A", "B", "C", "D", "E"), num=3, tga=0, q=1.0,
                            tav=0, marked=True, cfg=cfg)

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *a, **k: old_cl)

    old_num_before = old_cl.num

    ga.add_classifier(cl_new, Perception("whatever"), population, None, DummyList([old_cl]),
                      do_subsumption=True, theta_exp=0)

    # Since old_cl is marked, no increment of num
    assert old_cl.num == old_num_before
    assert cl_new not in population


def test_delete_classifiers_removes_single_classifier(monkeypatch, cfg):
    # Setup population, match_set, action_set
    cl1 = DummyClassifier(("A", "B", "C", "D", "E"), num=1, tga=0, q=0.5,
                        tav=0, marked=False, cfg=cfg)
    cl2 = DummyClassifier(("F", "G", "H", "I", "J"), num=1, tga=0, q=0.3,
                        tav=0, marked=False, cfg=cfg)
    population = DummyList([cl1, cl2])
    match_set = DummyList([cl1, cl2])
    action_set = DummyList([cl1, cl2])

    insize = 2
    theta_as = 3

    # We monkeypatch random.random to always return <0.3 so deletions occur on first iteration
    monkeypatch.setattr(random, "random", lambda: 0.0)

    # Because insize (2) + sum nums (2) = 4 > theta_as (3), should delete one
    ga.delete_classifiers(population, match_set, action_set, insize, theta_as)

    # Exactly one classifier removed or num decreased by one
    total_num_after = sum(cl.num for cl in action_set)
    assert total_num_after + insize <= theta_as


def test_delete_classifiers_decrements_num_if_num_greater_than_one(monkeypatch, cfg):
    cl1 = DummyClassifier(("A", "B", "C", "D", "E"), num=2, tga=0, q=0.5,
                        tav=0, marked=False, cfg=cfg)
    population = DummyList([cl1])
    match_set = DummyList([cl1])
    action_set = DummyList([cl1])

    insize = 2
    theta_as = 3

    monkeypatch.setattr(random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set, insize, theta_as)

    # num decremented (from 2 to 1)
    assert cl1.num == 1
    # classifier still in lists
    assert cl1 in population and cl1 in match_set and cl1 in action_set


@pytest.mark.parametrize("q_del, q_cl, marked_del, marked_cl, tav_del, tav_cl, expected", [
    (0.5, 0.3, False, False, 5, 10, True),   # cl q worse enough
    (0.5, 0.6, True, False, 5, 4, False),   # q better, but cl_del not marked and cl is marked
    (0.5, 0.55, True, False, 5, 4, True),   # q closer, cl is marked cl_del not, tav ignored
    (0.5, 0.45, False, False, 10, 15, True), # equal markings, cl tav higher
    (0.5, 0.45, False, True, 10, 15, False),  # cl_del marked, cl not marked, tav cl_del higher
])
def test_is_preferred_to_delete_variations(q_del, q_cl, marked_del, marked_cl, tav_del, tav_cl, expected, cfg):
    cl_del = DummyClassifier(("A", "B", "C", "D", "E"), q=q_del, marked=marked_del,
                             tav=tav_del, cfg=cfg)
    cl = DummyClassifier(("A", "B", "C", "D", "E"), q=q_cl, marked=marked_cl,
                         tav=tav_cl, cfg=cfg)
    result = ga._is_preferred_to_delete(cl_del, cl)
    assert result is expected


def test_find_similar_returns_classifier_or_none(cfg):
    cl1 = DummyClassifier(("A", "B", "C", "D", "E"), cfg=cfg)
    cl2 = DummyClassifier(("F", "G", "H", "I", "J"), cfg=cfg)
    population = [cl1]

    # Equal to cl1, should return cl1
    assert ga._find_similar(cl1, population) == cl1

    # Different classifier, returns None
    assert ga._find_similar(cl2, population) is None


def test_find_old_classifier_subsumption(monkeypatch, cfg):
    cl = DummyClassifier(("A", "B", "C", "D", "E"), cfg=cfg)
    pop = [cl]

    # Subsumers returns one item
    monkeypatch.setattr(ga, "find_subsumers", lambda cl_, pop_, th: [cl])
    found = ga._find_old_classifier(pop, cl, True, 0)
    assert found == cl

    # Subsumers empty, fallback to _find_similar called (we patch it)
    monkeypatch.setattr(ga, "find_subsumers", lambda cl_, pop_, th: [])
    monkeypatch.setattr(ga, "_find_similar", lambda other, pop: cl)
    found2 = ga._find_old_classifier(pop, cl, True, 0)
    assert found2 == cl

    # Without subsumption
    monkeypatch.setattr(ga, "_find_similar", lambda other, pop: cl)
    found3 = ga._find_old_classifier(pop, cl, False, 0)
    assert found3 == cl


def test_weighted_random_choice(monkeypatch):
    keys = ["a", "b", "c"]
    weights = {"a": 0.1, "b": 0.2, "c": 0.7}

    # Force random.uniform to a fixed place resulting in selecting "b"
    # cumulative sums: a=0.1, b=0.3, c=1.0
    def fake_uniform(a, b):
        return 0.25

    monkeypatch.setattr(random, "uniform", fake_uniform)
    chosen = ga._weighted_random_choice(weights)
    assert chosen == "b"

    monkeypatch.setattr(random, "uniform", lambda a, b: 0.9)
    chosen2 = ga._weighted_random_choice(weights)
    assert chosen2 == "c"


def test_weighted_random_choice_single_element():
    choices = {"only": 5.0}
    chosen = ga._weighted_random_choice(choices)
    assert chosen == "only"
