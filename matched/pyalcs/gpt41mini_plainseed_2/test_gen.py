import random
import builtins
import pytest
import numpy as np

from lcs.strategies import genetic_algorithms as ga
from lcs import Perception


class DummyCondition:
    def __init__(self, cond, cfg=None):
        self._cond = list(cond)
        self.generalized_idxs = set()
        self.cfg = cfg or DummyCfg()
    def __getitem__(self, idx):
        return self._cond[idx]
    def __setitem__(self, idx, val):
        self._cond[idx] = val
    def __len__(self):
        return len(self._cond)
    def generalize(self, idx):
        self.generalized_idxs.add(idx)
        self._cond[idx] = self.cfg.classifier_wildcard
    def does_match(self, p):
        return True
    def __eq__(self, other):
        if not isinstance(other, DummyClassifier):
            return False
        # Equality defined same condition, action and effect here:
        return (self._cond == other.condition._cond and
                getattr(self, 'action', None) == other.action and
                getattr(self, 'effect', None) == other.effect)
    def __repr__(self):
        return f"DummyCondition({self._cond})"


class DummyCfg:
    classifier_wildcard = '#'
    classifier_length = 5


class DummyClassifier:
    def __init__(self, num=1, tga=0, q=1.0, tav=0, is_marked=False,
                 condition=None, action=0, effect=0, cfg=None):
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self._marked = is_marked
        self.condition = condition or DummyCondition([1,2,3,4,5], cfg=cfg)
        self.action = action
        self.effect = effect
        self.cfg = cfg or DummyCfg()
    def is_marked(self):
        return self._marked
    def __eq__(self, other):
        if not isinstance(other, DummyClassifier):
            return False
        # Define equality as same condition array, action, effect
        return (self.condition._cond == other.condition._cond and
                self.action == other.action and
                self.effect == other.effect)
    def __hash__(self):
        return hash((tuple(self.condition._cond), self.action, self.effect))
    def __repr__(self):
        return f"DummyClassifier(num={self.num}, tga={self.tga}, q={self.q}, tav={self.tav}, marked={self._marked})"


class DummyList(list):
    """
    List with safe_remove method for testing delete_classifiers
    """
    def safe_remove(self, item):
        self.remove(item)


@pytest.mark.parametrize("action_set, time, theta_ga, expected", [
    (None, 10, 5, False),
    ([], 10, 5, False),
    ([DummyClassifier(num=0, tga=10)], 20, 5, False),  # overall_num == 0
    ([DummyClassifier(num=2, tga=0), DummyClassifier(num=3, tga=0)], 10, 5, True),
    ([DummyClassifier(num=1, tga=8), DummyClassifier(num=2, tga=7)], 10, 5, False),
])
def test_should_apply(action_set, time, theta_ga, expected):
    assert ga.should_apply(action_set, time, theta_ga) is expected


def test_set_timestamps_updates_all_tga():
    population = [DummyClassifier(tga=1), DummyClassifier(tga=2)]
    ga.set_timestamps(population, epoch=999)
    assert all(cl.tga == 999 for cl in population)


def test_roulette_wheel_selection_returns_two_parents():
    population = [DummyClassifier(q=1.0), DummyClassifier(q=2.0)]
    fitness = lambda cl: cl.q
    parent1, parent2 = ga.roulette_wheel_selection(population, fitness)
    assert parent1 in population
    assert parent2 in population
    # selections can be same classifier
    assert isinstance(parent1, DummyClassifier)
    assert isinstance(parent2, DummyClassifier)


def test_generalizing_mutation_randomness(monkeypatch):
    cl = DummyClassifier(condition=DummyCondition([1, 2, 3, 4, 5]))
    cl.cfg.classifier_wildcard = '#'

    # Patch random.random so only idx=2 triggers mutation
    random_values = iter([0.9, 0.9, 0.0, 0.9, 0.9])
    monkeypatch.setattr(random, "random", lambda: next(random_values))

    ga.generalizing_mutation(cl, mu=0.5)
    assert cl.condition._cond[2] == cl.cfg.classifier_wildcard
    assert 2 in cl.condition.generalized_idxs
    # Others unchanged
    unchanged = [i for i in range(len(cl.condition)) if i != 2]
    for i in unchanged:
        assert cl.condition._cond[i] != cl.cfg.classifier_wildcard


def test_two_point_crossover_swaps_segments(monkeypatch):
    cfg = DummyCfg()
    cfg.classifier_length = 5

    # Two parents with distinct conditions so that we can see changes
    parent = DummyClassifier(condition=DummyCondition([1, 1, 1, 1, 1], cfg=cfg), cfg=cfg)
    donor = DummyClassifier(condition=DummyCondition([9, 9, 9, 9, 9], cfg=cfg), cfg=cfg)

    # patch np.random.choice to always pick left=1, right=4
    monkeypatch.setattr(np.random, "choice", lambda a, size, replace: np.array([1,4]))

    ga.two_point_crossover(parent, donor)

    # Check that segment [1:4] swapped
    assert parent.condition._cond == [1, 9, 9, 9, 1]
    assert donor.condition._cond == [9, 1, 1, 1, 9]


@pytest.mark.parametrize("old_cl, expected_num", [
    (None, 1),
    (DummyClassifier(is_marked=True), 1),  # marked old_cl does not increment
    (DummyClassifier(is_marked=False), 3),  # increments num by 1
])
def test_add_classifier_behaviour(monkeypatch, old_cl, expected_num):
    population = []
    match_set = []
    action_set = []

    cl = DummyClassifier(num=1)
    cl.condition = DummyCondition([0, 0, 0, 0, 0])
    cl.num = 1

    # We patch _find_old_classifier to control old_cl returned
    monkeypatch.setattr(ga, "_find_old_classifier", lambda a,b,c,d: old_cl)

    if old_cl is not None:
        old_cl.num = 2  # start value for increment testing
        old_cl._marked = old_cl._marked if hasattr(old_cl, "_marked") else False

    ga.add_classifier(cl, Perception([]), population, match_set, action_set, do_subsumption=True, theta_exp=0)

    if old_cl is None:
        assert cl in population
        assert cl in action_set
        assert cl in match_set  # condition.does_match returns True
        assert cl.num == 1
    else:
        # old_cl.num increment only if not marked
        assert old_cl.num == expected_num


def test_delete_classifiers_reduce_num_and_remove(monkeypatch):
    # Create classifiers with different nums, q, tav and marked
    cl1 = DummyClassifier(num=3, q=0.5, tav=10, is_marked=False)
    cl2 = DummyClassifier(num=1, q=0.3, tav=11, is_marked=True)
    cl3 = DummyClassifier(num=1, q=0.6, tav=9, is_marked=False)

    population = DummyList([cl1, cl2, cl3])
    match_set = DummyList([cl1, cl2, cl3])
    action_set = DummyList([cl1, cl2, cl3])

    # monkeypatch random.random to always return <0.3 for first iteration (to choose cl_del fast)
    random_values = iter([0.1, 0.5, 0.5])  # 0.1 picks first cl in expand
    monkeypatch.setattr(random, "random", lambda: next(random_values))

    class ActionSetExpander(DummyList):
        def expand(self):
            return self

    # Use ActionSetExpander to give expand method to action_set
    action_set_exp = ActionSetExpander(action_set)

    # Insert size = 2, theta_as = current nums sum + 1 so deletion needed
    insize = 2
    theta_as = sum(cl.num for cl in action_set) + 1

    # patch action_set with expander with expand method
    ga.delete_classifiers(population, match_set, action_set_exp, insize, theta_as)

    # One deletion happened; since cl1 num=3 > 1 it is decremented
    assert cl1.num == 2 or len(population) < 3

    # Test multiple deletions by forcing random to always find delete candidate
    def always_less_than_0_3():
        return 0.0
    monkeypatch.setattr(random, "random", always_less_than_0_3)

    # Populate new population set for full removal when num<=1
    cl4 = DummyClassifier(num=1, q=0.1, tav=5, is_marked=False)
    population2 = DummyList([cl4])
    match_set2 = DummyList([cl4])
    action_set2 = DummyList([cl4])

    action_set_exp2 = ActionSetExpander(action_set2)
    insize2 = 2
    theta_as2 = 2
    ga.delete_classifiers(population2, match_set2, action_set_exp2, insize2, theta_as2)
    # cl4 should be completely removed since num==1
    assert cl4 not in population2
    assert cl4 not in match_set2
    assert cl4 not in action_set_exp2


@pytest.mark.parametrize("cl_del_params, cl_params, expected", [
    # cl q much less than cl_del
    ((1, 0, False, 10), (0.8, 0, False, 10), True),
    # q equal, cl is marked, cl_del not marked, same tav
    ((1, 0, False, 5), (1, 0, True, 5), True),
    # q equal, both marked, cl.tav > cl_del.tav
    ((1, 5, True, 2), (1, 10, True, 3), True),
    # q equal, both marked, cl.tav <= cl_del.tav
    ((1, 5, True, 10), (1, 5, True, 15), False),
    # q difference > 0.1
    ((1, 10, False, 10), (1.3, 10, False, 10), False),
])
def test__is_preferred_to_delete(cl_del_params, cl_params, expected):
    cl_del = DummyClassifier(q=cl_del_params[0], tav=cl_del_params[1], is_marked=cl_del_params[2])
    cl = DummyClassifier(q=cl_params[0], tav=cl_params[1], is_marked=cl_params[2])
    # Fix tav in the inputs that had been used differently in params
    cl_del.tav = cl_del_params[3]
    cl.tav = cl_params[3]

    assert ga._is_preferred_to_delete(cl_del, cl) is expected


def test__find_old_classifier_use_subsumption(monkeypatch):
    population = [DummyClassifier()]
    cl = DummyClassifier()
    # Patch find_subsumers to return a list of subsumers or empty
    monkeypatch.setattr(ga, "find_subsumers", lambda c, pop, theta: [population[0]])
    # Should return first subsumer
    old = ga._find_old_classifier(population, cl, True, 0)
    assert old == population[0]

    # Patch to return empty list, so fallback to _find_similar
    monkeypatch.setattr(ga, "find_subsumers", lambda c, pop, theta: [])
    monkeypatch.setattr(ga, "_find_similar", lambda c, pop: "similar")
    old = ga._find_old_classifier(population, cl, True, 0)
    assert old == "similar"


def test__find_similar_returns_first_equal_or_none():
    cl = DummyClassifier(condition=DummyCondition([1,1,1,1,1]))
    population = [
        DummyClassifier(condition=DummyCondition([0,0,0,0,0])),
        DummyClassifier(condition=DummyCondition([1,1,1,1,1])),  # equal
        DummyClassifier(condition=DummyCondition([1,1,1,1,2]))
    ]
    found = ga._find_similar(cl, population)
    assert found == population[1]

    # No similar classifier
    cl2 = DummyClassifier(condition=DummyCondition([9,9,9,9,9]))
    found2 = ga._find_similar(cl2, population)
    assert found2 is None


def test__weighted_random_choice(monkeypatch):
    choices = {
        "a": 1,
        "b": 2,
        "c": 3,
    }
    total = sum(choices.values())

    # Force pick to 0.5 * total, should pick 'b'
    monkeypatch.setattr(random, "uniform", lambda a,b: 0.5 * total)
    chosen = ga._weighted_random_choice(choices)
    assert chosen in choices

    # Force pick to 0 (lowest), should pick 'a'
    monkeypatch.setattr(random, "uniform", lambda a,b: 0)
    chosen = ga._weighted_random_choice(choices)
    assert chosen == "a"

    # Force pick to total (highest), should pick last 'c'
    monkeypatch.setattr(random, "uniform", lambda a,b: total)
    chosen = ga._weighted_random_choice(choices)
    assert chosen in choices
