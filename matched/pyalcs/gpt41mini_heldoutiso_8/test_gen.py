import random
import types

import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga
from lcs import Perception
from lcs.strategies.subsumption import find_subsumers


class DummyCfg:
    classifier_length = 5
    classifier_wildcard = '#'


class DummyCondition:
    def __init__(self, values, cfg=None):
        self.values = list(values)
        self.cfg = cfg or DummyCfg()

    def __getitem__(self, idx):
        return self.values[idx]

    def __setitem__(self, idx, val):
        self.values[idx] = val

    def __len__(self):
        return len(self.values)

    def generalize(self, idx):
        # Mark condition position as wildcard
        self.values[idx] = self.cfg.classifier_wildcard

    def __eq__(self, other):
        if not isinstance(other, DummyCondition):
            return False
        return self.values == other.values

    def does_match(self, perception):
        # Simple match logic: all positions equal or wildcard match
        for v, p in zip(self.values, perception):
            if v != self.cfg.classifier_wildcard and v != p:
                return False
        return True


class DummyClassifier:
    def __init__(self, num=1, tga=0, condition=None, action=0,
                 effect=0, q=1.0, tav=0, cfg=None):
        self.num = num
        self.tga = tga
        self.condition = condition or DummyCondition(['a', 'b', 'c', 'd', 'e'], cfg)
        self.action = action
        self.effect = effect
        self.q = q
        self.tav = tav
        self._marked = False
        self.cfg = cfg or DummyCfg()

    def __eq__(self, other):
        if not isinstance(other, DummyClassifier):
            return False
        # Define equality on condition, action and effect as _find_similar requires
        return (self.condition == other.condition and
                self.action == other.action and
                self.effect == other.effect)

    def is_marked(self):
        return self._marked

    def mark(self):
        self._marked = True

    def safe_remove(self, cl):
        # Remove cl from self if present
        if cl in self:
            self.remove(cl)


def test_should_apply_none_action_set():
    assert not ga.should_apply(None, time=10, theta_ga=5)


def test_should_apply_empty_action_set():
    assert not ga.should_apply([], time=10, theta_ga=5)


def test_should_apply_zero_num_in_action_set():
    cl = DummyClassifier(num=0, tga=5)
    assert not ga.should_apply([cl], time=10, theta_ga=3)


def test_should_apply_returns_true_when_avg_delay_exceeds_theta():
    cl1 = DummyClassifier(num=2, tga=4)
    cl2 = DummyClassifier(num=3, tga=6)
    # weighted avg = (2*4 + 3*6)/5 = (8 + 18)/5 = 26/5 = 5.2
    # time - avg_tga = 10 - 5.2 = 4.8 > theta_ga=4
    assert ga.should_apply([cl1, cl2], time=10, theta_ga=4)


def test_should_apply_returns_false_when_avg_delay_below_theta():
    cl1 = DummyClassifier(num=1, tga=6)
    cl2 = DummyClassifier(num=1, tga=8)
    # average= (6+8)/2=7, time=10, diff=3, theta_ga=4 => False
    assert not ga.should_apply([cl1, cl2], time=10, theta_ga=4)


def test_set_timestamps_updates_all_tga():
    cl1 = DummyClassifier(tga=1)
    cl2 = DummyClassifier(tga=5)
    action_set = [cl1, cl2]
    ga.set_timestamps(action_set, epoch=20)
    for cl in action_set:
        assert cl.tga == 20


def test_roulette_wheel_selection_returns_tuple_of_two_parents():
    pop = [DummyClassifier(q=i+1.0) for i in range(5)]
    # Define fitnessfunc as q squared
    fitnessfunc = lambda cl: cl.q ** 2

    parent1, parent2 = ga.roulette_wheel_selection(pop, fitnessfunc)

    assert parent1 in pop
    assert parent2 in pop
    # May be equal, should be tuple
    assert isinstance((parent1, parent2), tuple)
    assert len((parent1, parent2)) == 2


def test_generalizing_mutation_mutates_condition_items_with_probability(monkeypatch):
    random_values = iter([0.1, 0.9, 0.05, 0.99, 0.0])
    monkeypatch.setattr(random, "random", lambda: next(random_values))

    cl = DummyClassifier(condition=DummyCondition(['a', 'b', 'c', 'd', 'e']))
    mu = 0.2
    ga.generalizing_mutation(cl, mu)
    # Position 0: 0.1 < 0.2 mutated
    # Position 1: 0.9 > 0.2 not mutated
    # Position 2: 0.05 < 0.2 mutated
    # Position 3: 0.99 > 0.2 not mutated
    # Position 4: 0.0 < 0.2 mutated
    expected = ['#', 'b', '#', 'd', '#']
    assert cl.condition.values == expected


def test_two_point_crossover_flips_condition_segments(monkeypatch):
    # Patch np.random.choice to return fixed crossover points: 1 and 4
    monkeypatch.setattr(np.random, "choice", lambda a, size, replace: np.array([1, 4]))

    cfg = DummyCfg()
    parent = DummyClassifier(condition=DummyCondition(['a', 'b', 'c', 'd', 'e'], cfg), cfg=cfg)
    donor = DummyClassifier(condition=DummyCondition(['1', '2', '3', '4', '5'], cfg), cfg=cfg)

    ga.two_point_crossover(parent, donor)

    # Expected after flip segment [1:4] (indices 1,2,3)
    # parent: ['a', '2', '3', '4', 'e']
    # donor: ['1', 'b', 'c', 'd', '5']
    assert parent.condition.values == ['a', '2', '3', '4', 'e']
    assert donor.condition.values == ['1', 'b', 'c', 'd', '5']


def test_add_classifier_adds_new_when_no_old_classifier(monkeypatch):
    population = []
    match_set = []
    action_set = []
    p = Perception(['a', 'b', 'c', 'd', 'e'])
    cl = DummyClassifier(condition=DummyCondition(['a', 'b', 'c', 'd', 'e']))
    # Patch _find_old_classifier to return None
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *a, **k: None)

    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=True, theta_exp=10)

    assert cl in population
    assert cl in action_set
    assert cl in match_set


def test_add_classifier_increments_num_of_old_classifier(monkeypatch):
    old = DummyClassifier(num=1)
    population = [old]
    match_set = []
    action_set = [old]
    p = Perception(['a', 'b', 'c', 'd', 'e'])
    newcl = DummyClassifier()
    # Patch _find_old_classifier to return old classifier
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *a, **k: old)
    # Also monkeypatch is_marked to False
    old._marked = False

    ga.add_classifier(newcl, p, population, match_set, action_set, do_subsumption=True, theta_exp=10)

    assert old.num == 2
    # New classifier not added
    assert newcl not in population


def test_add_classifier_does_not_increment_marked_classifier(monkeypatch):
    old = DummyClassifier(num=1)
    old.mark()
    population = [old]
    match_set = []
    action_set = [old]
    p = Perception(['a', 'b', 'c', 'd', 'e'])
    newcl = DummyClassifier()
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *a, **k: old)

    ga.add_classifier(newcl, p, population, match_set, action_set, do_subsumption=True, theta_exp=10)

    # num should not increase
    assert old.num == 1
    assert newcl not in population


def test_delete_classifiers_removes_classifiers(monkeypatch):
    random_values = iter([0.5, 0.2, 0.4, 0.1])

    def fake_random():
        return next(random_values)

    monkeypatch.setattr(random, "random", fake_random)

    # We'll create classifiers with varying q and num
    cl1 = DummyClassifier(q=1.0, num=1)
    cl2 = DummyClassifier(q=0.5, num=2)
    cl3 = DummyClassifier(q=1.5, num=1)
    cl3.mark()
    # Using lists with safe_remove implemented
    class ListWithSafeRemove(list):
        def safe_remove(self, cl):
            if cl in self:
                self.remove(cl)

    population = ListWithSafeRemove([cl1, cl2, cl3])
    match_set = ListWithSafeRemove([cl1, cl2, cl3])
    action_set = ListWithSafeRemove([cl1, cl2, cl3])

    # Add expand method for action_set as used in code
    def expand(self):
        return list(self)
    action_set.expand = types.MethodType(expand, action_set)

    # insize=3, sum num = 4, so loop runs while 7 > theta_as=6
    ga.delete_classifiers(population, match_set, action_set, insize=3, theta_as=6)

    # As random returns <0.3 sometimes, some classifier should be deleted or decremented
    # After deletion length should decrease or sum num should decrease
    total_num = sum(cl.num for cl in action_set)
    assert total_num + 3 <= 6  # After deletion this condition holds


@pytest.mark.parametrize("q_del, q_cl, marked_del, marked_cl, tav_del, tav_cl, expected", [
    (1.0, 0.7, False, False, 10, 20, True),    # cl.q < cl_del.q - 0.1 => True
    (1.0, 1.05, True, False, 10, 15, True),   # equal q diff, cl_del unmarked, cl marked => True
    (1.0, 1.05, False, False, 10, 20, True),  # equal q, none marked, tav_cl > tav_del => True
    (1.0, 1.05, False, True, 10, 5, False),   # equal q, cl_del marked, cl marked=False => False
    (1.0, 1.2, False, False, 10, 5, False),   # difference > 0.1 positive => False
])
def test__is_preferred_to_delete_cases(q_del, q_cl, marked_del, marked_cl, tav_del, tav_cl, expected):
    cl_del = DummyClassifier(q=q_del, tav=tav_del)
    cl_del._marked = marked_del
    cl = DummyClassifier(q=q_cl, tav=tav_cl)
    cl._marked = marked_cl

    result = ga._is_preferred_to_delete(cl_del, cl)
    assert result == expected


def test__find_old_classifier_uses_subsumers(monkeypatch):
    population = [DummyClassifier()]
    cl = DummyClassifier()
    # Create subsumer list with one classifier
    subsumer = DummyClassifier()
    monkeypatch.setattr(ga, "find_subsumers", lambda c, p, t: [subsumer])
    old_cl = ga._find_old_classifier(population, cl, use_subsumption=True, theta_exp=10)
    assert old_cl is subsumer


def test__find_old_classifier_ignores_empty_subsumers(monkeypatch):
    population = [DummyClassifier()]
    cl = DummyClassifier()
    # Return empty subsumer list
    monkeypatch.setattr(ga, "find_subsumers", lambda c, p, t: [])
    # Patch _find_similar to return None
    monkeypatch.setattr(ga, "_find_similar", lambda c, p: None)

    old_cl = ga._find_old_classifier(population, cl, use_subsumption=True, theta_exp=10)
    assert old_cl is None


def test__find_old_classifier_falls_back_to_similar(monkeypatch):
    population = [DummyClassifier()]
    cl = DummyClassifier()
    monkeypatch.setattr(ga, "find_subsumers", lambda c, p, t: [])
    similar = DummyClassifier()
    monkeypatch.setattr(ga, "_find_similar", lambda o, p: similar)
    old_cl = ga._find_old_classifier(population, cl, use_subsumption=True, theta_exp=10)
    assert old_cl is similar


def test__find_similar_finds_first_equal():
    cl1 = DummyClassifier(condition=DummyCondition(['a', 'b', 'c', 'd', 'e']), action=1, effect=1)
    cl2 = DummyClassifier(condition=DummyCondition(['a', 'b', 'c', 'd', 'e']), action=1, effect=1)
    cl3 = DummyClassifier(condition=DummyCondition(['x', 'y', 'z', 'd', 'e']), action=1, effect=1)
    pop = [cl3, cl1, cl2]
    found = ga._find_similar(cl2, pop)
    assert found == cl1  # first that equals cl2 in pop is cl1


def test_weighted_random_choice_selects_key(monkeypatch):
    # Setup choices with deterministic random.uniform
    choices = {1: 0.1, 2: 0.2, 3: 0.7}
    random_calls = []

    def fake_uniform(a, b):
        # Return a value that falls within cumulative sum to choose key 2
        val = 0.25
        random_calls.append(val)
        return val

    monkeypatch.setattr(random, "uniform", fake_uniform)
    chosen = ga._weighted_random_choice(choices)
    assert chosen == 2
