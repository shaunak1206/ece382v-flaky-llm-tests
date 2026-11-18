import random
import pytest
import numpy as np

from lcs.strategies import genetic_algorithms as ga
from lcs import Perception


# Mocks for classifiers and condition because we need properties and methods used in GA code

class DummyCondition:
    def __init__(self, cond_list, wildcard="_"):
        self._condlist = list(cond_list)
        self.classifier_wildcard = wildcard

    def __getitem__(self, item):
        return self._condlist[item]

    def __setitem__(self, key, value):
        self._condlist[key] = value

    def __len__(self):
        return len(self._condlist)

    def generalize(self, idx):
        # mark condition at idx as wildcard
        self._condlist[idx] = self.classifier_wildcard

    def does_match(self, p):
        # just return True for tests as placeholder
        return True

    def __repr__(self):
        return f"DummyCondition({self._condlist})"

    def __iter__(self):
        return iter(self._condlist)


class DummyClassifier:
    def __init__(self, tga=0, num=1, q=0.5, tav=0, condition=None, action=0,
                 effect=0, marked=False, cfg=None):
        self.tga = tga
        self.num = num
        self.q = q
        self.tav = tav
        self.action = action
        self.effect = effect
        self._marked = marked
        self._cfg = cfg or DummyCfg()
        # condition must support __getitem__, __setitem__, len, generalize
        self.condition = condition or DummyCondition(["A", "B", "C", "D"], self._cfg.classifier_wildcard)

    @property
    def cfg(self):
        return self._cfg

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        # Equal if condition, action, effect same
        if not isinstance(other, DummyClassifier):
            return NotImplemented
        return (self.condition._condlist == other.condition._condlist and
                self.action == other.action and
                self.effect == other.effect)

    def __hash__(self):
        # To allow use in dict keys for roulette_wheel_selection
        return hash((tuple(self.condition._condlist), self.action, self.effect))


class DummyCfg:
    def __init__(self):
        self.classifier_wildcard = "_"
        self.classifier_length = 4


class DummyList(list):
    """
    A list subclass with additional 'expand' and 'safe_remove' methods used in code.
    """

    def expand(self):
        for cl in self:
            for _ in range(cl.num):
                yield cl

    def safe_remove(self, item):
        if item in self:
            self.remove(item)


# The imports use lcs.Perception, so to test 'does_match' in add_classifier properly,
# we create a dummy Perception object (empty class)

class DummyPerception:
    pass


@pytest.mark.parametrize("action_set, time, theta_ga, expected", [
    (None, 10, 5, False),  # None action_set returns False
    ([], 10, 5, False),  # Empty list returns False (overall_num == 0)
    ([DummyClassifier(tga=10, num=1)], 20, 5, True),  # time - avg_tga > theta_ga
    ([DummyClassifier(tga=18, num=2), DummyClassifier(tga=22, num=3)], 30, 5, True),
    ([DummyClassifier(tga=10, num=1)], 12, 5, False),  # time - avg_tga == 2 <= theta_ga - no
])
def test_should_apply(action_set, time, theta_ga, expected):
    assert ga.should_apply(action_set, time, theta_ga) is expected


def test_set_timestamps_sets_tga():
    classifiers = [DummyClassifier(tga=0), DummyClassifier(tga=5)]
    new_epoch = 42
    ga.set_timestamps(classifiers, new_epoch)
    for cl in classifiers:
        assert cl.tga == new_epoch


def test_roulette_wheel_selection_returns_two_parents_and_in_population():
    pop = [DummyClassifier(q=0.1), DummyClassifier(q=0.9)]
    # Define fitness function to cube q
    fitness = lambda c: c.q ** 3
    p1, p2 = ga.roulette_wheel_selection(pop, fitness)
    assert p1 in pop and p2 in pop
    assert isinstance(p1, DummyClassifier) and isinstance(p2, DummyClassifier)


def test_generalizing_mutation_generalizes_some_conditions(monkeypatch):
    cl = DummyClassifier(condition=DummyCondition(["A", "B", "C", "D"]))
    # override random.random to control mutation probability:
    mutations = [0.2, 0.8, 0.1, 0.9]

    def fake_random():
        return mutations.pop(0)

    monkeypatch.setattr(random, "random", fake_random)
    mu = 0.5
    ga.generalizing_mutation(cl, mu)
    # Index 0: cond 'A' wildcard? random=0.2<0.5 => generalized => '_'
    assert cl.condition[0] == cl.cfg.classifier_wildcard
    # Index 1: cond 'B' random=0.8>mu no change
    assert cl.condition[1] == "B"
    # Index 2: cond 'C' random=0.1 < mu generalized '_'
    assert cl.condition[2] == cl.cfg.classifier_wildcard
    # Index 3: cond 'D' random=0.9>mu no change
    assert cl.condition[3] == "D"


def test_two_point_crossover_swaps_condition_slices(monkeypatch):
    # Force np.random.choice to pick fixed indices (left=1, right=3)
    monkeypatch.setattr(np.random, "choice", lambda a, size, replace: np.array([3, 1]))
    # sorted will reorder them as left=1, right=3
    p = DummyClassifier(condition=DummyCondition(["A", "B", "C", "D"]))
    d = DummyClassifier(condition=DummyCondition(["W", "X", "Y", "Z"]))

    ga.two_point_crossover(p, d)

    # after crossover at indices 1,2:
    # p.condition parts [1:3] replaced with d.condition parts [1:3] = ['X', 'Y']
    # d.condition parts [1:3] replaced with p.condition parts [1:3] = ['B', 'C']

    assert p.condition._condlist == ["A", "X", "Y", "D"]
    assert d.condition._condlist == ["W", "B", "C", "Z"]


def test_add_classifier_adds_new_and_increments_existing(monkeypatch):
    population = DummyList()
    match_set = DummyList()
    action_set = DummyList()
    p = DummyPerception()

    new_cl = DummyClassifier()
    # Replace _find_old_classifier to None (simulate classifier not found)
    monkeypatch.setattr(ga, "_find_old_classifier", lambda a1, a2, a3, a4: None)
    ga.add_classifier(new_cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=0)
    assert new_cl in population and new_cl in action_set and new_cl in match_set

    # Now test incrementing numerosity on existing classifier (old_cl)
    old_cl = DummyClassifier(num=2, marked=False)
    monkeypatch.setattr(ga, "_find_old_classifier", lambda a1, a2, a3, a4: old_cl)
    # Also test that if old_cl is not marked, num incremented
    ga.add_classifier(new_cl, p, population, match_set, action_set, do_subsumption=True, theta_exp=3)
    assert old_cl.num == 3

    # If old_cl is marked, num must NOT increment
    old_cl_marked = DummyClassifier(num=4, marked=True)
    monkeypatch.setattr(ga, "_find_old_classifier", lambda a1, a2, a3, a4: old_cl_marked)
    ga.add_classifier(new_cl, p, population, match_set, action_set, do_subsumption=True, theta_exp=3)
    assert old_cl_marked.num == 4  # unchanged


def test_delete_classifiers_removes_until_threshold(monkeypatch):
    # create population, match_set, action_set with dummy classifiers
    cl1 = DummyClassifier(num=2, q=0.9, tav=3)
    cl2 = DummyClassifier(num=1, q=0.2, tav=7)
    cl3 = DummyClassifier(num=1, q=0.5, tav=4, marked=True)
    population = DummyList([cl1, cl2, cl3])
    match_set = DummyList([cl1, cl2, cl3])
    action_set = DummyList([cl1, cl2, cl3])

    # patch random.random to have predictable behavior, initially choose cl2 for deletion
    random_values = iter([0.1, 0.4, 0.2, 0.05])  # multiple calls as needed

    monkeypatch.setattr(random, "random", lambda: next(random_values))

    # Current total in action_set = 2+1+1=4
    # insize=2, theta_as=5, so 2+4=6 > 5 => need to delete
    ga.delete_classifiers(population, match_set, action_set, insize=2, theta_as=5)

    # Deletion should reduce total numerosity in action_set below or equal threshold
    total_num = sum(cl.num for cl in action_set)
    assert total_num + 2 <= 5  # insize + total_num now <= theta_as

    # cl_del with num>1 decremented num rather than deleted from lists
    assert cl1.num == 1 or cl2.num == 0 or cl3.num == 0


@pytest.mark.parametrize(
    "cl_del_params, cl_params, expected",
    [
        ({"q": 0.6, "tav": 2, "marked": False}, {"q": 0.4, "tav": 2, "marked": False}, True),
        ({"q": 0.5, "tav": 5, "marked": True}, {"q": 0.5, "tav": 10, "marked": True}, True),
        ({"q": 0.5, "tav": 10, "marked": False}, {"q": 0.5, "tav": 5, "marked": True}, False),
        ({"q": 0.9, "tav": 10, "marked": False}, {"q": 1.0, "tav": 15, "marked": False}, False)
    ]
)
def test_is_preferred_to_delete_behavior(cl_del_params, cl_params, expected):
    cl_del = DummyClassifier(**cl_del_params)
    cl = DummyClassifier(**cl_params)
    result = ga._is_preferred_to_delete(cl_del, cl)
    assert result is expected


def test_find_old_classifier_with_subsumption(monkeypatch):
    population = [DummyClassifier(action=1), DummyClassifier(action=2)]
    cl = DummyClassifier(action=1)
    fake_subsumers = [population[1]]

    monkeypatch.setattr(ga, "find_subsumers", lambda cl_, pop, th: fake_subsumers)

    # Should pick first subsumer
    old_cl = ga._find_old_classifier(population, cl, True, 0)
    assert old_cl == fake_subsumers[0]

    # Test with empty subsumers triggers _find_similar
    monkeypatch.setattr(ga, "find_subsumers", lambda cl_, pop, th: [])
    monkeypatch.setattr(ga, "_find_similar", lambda cl_, pop: "sim")

    old_cl2 = ga._find_old_classifier(population, cl, True, 0)
    assert old_cl2 == "sim"

    # If use_subsumption is False, just calls _find_similar
    old_cl3 = ga._find_old_classifier(population, cl, False, 0)
    assert old_cl3 is None or old_cl3 == ga._find_similar(cl, population)


def test_find_similar_returns_equal_classifier_or_none():
    cl1 = DummyClassifier(condition=DummyCondition(["A", "B", "C"]), action=1, effect=5)
    cl2 = DummyClassifier(condition=DummyCondition(["A", "B", "C"]), action=1, effect=5)
    cl3 = DummyClassifier(condition=DummyCondition(["X", "Y", "Z"]), action=1, effect=5)
    pop = [cl2, cl3]

    # Same condition, action, effect found returns cl2
    found = ga._find_similar(cl1, pop)
    assert found == cl2

    # Different classifier returns None
    cl4 = DummyClassifier(condition=DummyCondition(["Q", "W", "E"]), action=2, effect=1)
    found2 = ga._find_similar(cl4, pop)
    assert found2 is None


def test_weighted_random_choice_returns_expected(monkeypatch):
    cl1 = DummyClassifier()
    cl2 = DummyClassifier()
    choices = {cl1: 0, cl2: 10}

    # random.uniform always returns 5: sum=10
    monkeypatch.setattr(random, "uniform", lambda a, b: 5)

    chosen = ga._weighted_random_choice(choices)
    assert chosen == cl2

    choices2 = {cl1: 5, cl2: 5}
    monkeypatch.setattr(random, "uniform", lambda a, b: 3)
    chosen2 = ga._weighted_random_choice(choices2)
    # Because 3 is less than 5, first key returned
    assert chosen2 == cl1

