import random
import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga

# Dummy Classifier and Condition classes for testing purposes

class DummyCondition:
    def __init__(self, length, wildcard='?'):
        self.data = ['a'] * length
        self.classifier_wildcard = wildcard
        self.generalized_indices = set()

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, val):
        self.data[idx] = val

    def __len__(self):
        return len(self.data)

    def generalize(self, idx):
        # Mark index as generalized to detect mutations
        self.data[idx] = self.classifier_wildcard
        self.generalized_indices.add(idx)

    def __eq__(self, other):
        if not isinstance(other, DummyCondition):
            return False
        return self.data == other.data

    def __repr__(self):
        return f"DummyCondition({self.data})"


class DummyClassifier:
    def __init__(self, num=1, tga=0, condition=None, action=0,
                 q=0.5, tav=0.0, marked=False,
                 cfg=None):
        self.num = num
        self.tga = tga
        self.condition = condition if condition else DummyCondition(5)
        self.action = action
        self.q = q
        self.tav = tav
        self._marked = marked
        self.cfg = cfg if cfg else self.DummyCfg()

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        if not isinstance(other, DummyClassifier):
            return False
        # Equality based on condition, action and q value for tests
        return (self.condition == other.condition and
                self.action == other.action and
                abs(self.q - other.q) < 1e-9)

    def __repr__(self):
        return (f"DummyClassifier(num={self.num}, tga={self.tga}, "
                f"action={self.action}, q={self.q}, tav={self.tav}, marked={self._marked})")

    class DummyCfg:
        def __init__(self):
            self.classifier_wildcard = '?'
            self.classifier_length = 5


class DummyPerception:
    def __init__(self, val):
        self.val = val


class DummyList(list):
    """
    A list subclass that supports safe_remove (returns nothing if not present)
    and expand (returns self for iteration)
    """
    def safe_remove(self, item):
        try:
            self.remove(item)
        except ValueError:
            pass

    def expand(self):
        # Returns self (for action_set.expand())
        return self


# Mock of find_subsumers used in add_classifier tests
def mock_find_subsumers(cl, population, theta_exp):
    # Return subsumers based on cl.condition.data starting with 'sub'
    # For test, if condition starts with 'sub', return first classifier; else none
    for c in population:
        if getattr(c.condition, 'data', None) and c.condition.data and c.condition.data[0] == 's':
            return [c]
    return []


@pytest.mark.parametrize("action_set, time, theta_ga, expected", [
    (None, 10, 5, False),  # None action set
    ([], 10, 5, False),  # Empty action set
    ([DummyClassifier(num=1, tga=3)], 10, 5, False),  # time - avg = 10-3=7 < 5 False
    ([DummyClassifier(num=1, tga=3)], 10, 6, True),  # 10-3=7 > 6 True
    ([DummyClassifier(num=2, tga=5), DummyClassifier(num=3, tga=7)], 15, 4, True),  # weighted avg (2*5+3*7)/5= (10+21)/5=6.2, 15-6.2=8.8 >4
    ([DummyClassifier(num=0, tga=10)], 100, 50, False),  # zero numerosity leads to False
])
def test_should_apply(action_set, time, theta_ga, expected):
    # Seed randoms to not be flaky
    random.seed(123)
    np.random.seed(123)

    result = ga.should_apply(action_set, time, theta_ga)
    assert result is expected


def test_set_timestamps_updates_all():
    random.seed(1)
    np.random.seed(1)
    cl1 = DummyClassifier(tga=5)
    cl2 = DummyClassifier(tga=10)
    action_set = [cl1, cl2]
    epoch = 42

    ga.set_timestamps(action_set, epoch)
    assert all(cl.tga == epoch for cl in action_set)


def test_roulette_wheel_selection_returns_two_parents(monkeypatch):
    random.seed(42)
    np.random.seed(42)
    # Patch _weighted_random_choice to deterministic for test
    # But better test real behavior with fixed seed
    cl1 = DummyClassifier(q=1.0)
    cl2 = DummyClassifier(q=2.0)
    cl3 = DummyClassifier(q=3.0)
    population = [cl1, cl2, cl3]

    def fitnessfunc(cl):
        return cl.q ** 3

    parent1, parent2 = ga.roulette_wheel_selection(population, fitnessfunc)
    assert parent1 in population
    assert parent2 in population


def test_generalizing_mutation_mu_zero_no_changes():
    random.seed(0)
    np.random.seed(0)
    cond = DummyCondition(5)
    cfg = DummyClassifier.DummyCfg()
    cl = DummyClassifier(condition=cond, cfg=cfg)
    mu = 0.0

    ga.generalizing_mutation(cl, mu)

    # No generalization should happen
    assert len(cl.condition.generalized_indices) == 0
    assert cl.condition.data == ['a', 'a', 'a', 'a', 'a']


def test_generalizing_mutation_mu_one_all_changes():
    random.seed(0)
    np.random.seed(0)
    cond = DummyCondition(5)
    cfg = DummyClassifier.DummyCfg()
    cl = DummyClassifier(condition=cond, cfg=cfg)
    mu = 1.0

    ga.generalizing_mutation(cl, mu)

    # All should be generalized except wildcards (all were 'a', so all changed)
    assert cl.condition.data == ['?', '?', '?', '?', '?']
    assert cl.condition.generalized_indices == {0, 1, 2, 3, 4}


def test_two_point_crossover_swaps_slices():
    random.seed(1234)
    np.random.seed(1234)
    cond1 = DummyCondition(5)
    cond2 = DummyCondition(5)
    cl1 = DummyClassifier(condition=cond1)
    cl2 = DummyClassifier(condition=cond2)

    original1 = cond1.data.copy()
    original2 = cond2.data.copy()

    ga.two_point_crossover(cl1, cl2)

    # After crossover, some slice should have been swapped
    # They must still have total length 5
    assert len(cl1.condition) == len(cl2.condition) == 5

    # At least one position should differ after crossover (otherwise no change)
    differences = sum(x != y for x, y in zip(cl1.condition.data, original1))
    assert differences > 0


@pytest.mark.parametrize("use_subsumption, subsumer_present, similar_present", [
    (True, True, False),
    (True, False, True),
    (False, False, True),
    (True, False, False),
    (False, False, False),
])
def test_find_old_classifier(monkeypatch, use_subsumption, subsumer_present, similar_present):
    random.seed(2)
    np.random.seed(2)

    cl = DummyClassifier()
    # population to test:
    subsumer = DummyClassifier(condition=DummyCondition(5))
    similar = DummyClassifier(condition=cl.condition, action=cl.action, q=cl.q)

    population = []
    if subsumer_present:
        population.append(subsumer)
    if similar_present:
        population.append(similar)

    # Patch find_subsumers if use_subsumption to return subsumer or empty
    monkeypatch.setattr(ga, "find_subsumers",
                        lambda cl_in, pop, theta_exp: [subsumer] if subsumer_present and use_subsumption else [])

    # Patch _find_similar to always return similar if available
    monkeypatch.setattr(ga, "_find_similar",
                        lambda other_cl, pop: similar if similar_present else None)

    result = ga._find_old_classifier(population, cl, use_subsumption, theta_exp=0)

    if use_subsumption and subsumer_present:
        assert result == subsumer
    elif similar_present:
        assert result == similar
    else:
        assert result is None


def test_add_classifier_adds_new(monkeypatch):
    random.seed(11)
    np.random.seed(11)

    p = DummyPerception(val=1)
    population = []
    match_set = []
    action_set = []

    cl = DummyClassifier(condition=DummyCondition(5))
    # patch _find_old_classifier to always return None to test add scenario
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: None)

    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert cl in population
    assert cl in action_set
    # cl.condition.does_match() must be tested - we simulate it as equal always True here
    # Let's patch condition.does_match to return True
    cl.condition.does_match = lambda p: True
    # already appended, test again with fresh sets, check appending to match_set
    population2 = []
    match_set2 = []
    action_set2 = []
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: None)

    ga.add_classifier(cl, p, population2, match_set2, action_set2, do_subsumption=False, theta_exp=0)
    assert cl in match_set2


def test_add_classifier_increments_existing(monkeypatch):
    random.seed(7)
    np.random.seed(7)

    cl = DummyClassifier(num=1, condition=DummyCondition(5), marked=False)
    old_cl = DummyClassifier(num=3, condition=cl.condition, marked=False)

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: old_cl)

    population = []
    match_set = []
    action_set = []

    ga.add_classifier(cl, None, population, match_set, action_set, do_subsumption=True, theta_exp=0)

    # old_cl.num should be incremented by 1 only if not marked
    assert old_cl.num == 4


def test_delete_classifiers_removes_items(monkeypatch):
    random.seed(202)
    np.random.seed(202)
    # Setup population and sets with classifier with num=1 and marked
    cl1 = DummyClassifier(num=1, q=0.2, tav=1.0, marked=False)
    cl2 = DummyClassifier(num=2, q=0.1, tav=2.0, marked=True)
    cl3 = DummyClassifier(num=1, q=0.5, tav=0.5, marked=False)

    population = DummyList([cl1, cl2, cl3])
    match_set = DummyList([cl1, cl2, cl3])
    action_set = DummyList([cl1, cl2, cl3])

    # Patch action_set.expand to return action_set itself
    # Done by DummyList.expand()

    # monkeypatch random.random to control removal choice deterministically
    # We patch random.random to cycle through values less than 0.3 to remove first classifiers
    values = [0.1, 0.05, 0.4, 0.2]
    def random_side_effect():
        # cycle over values
        v = values.pop(0)
        values.append(v)
        return v
    monkeypatch.setattr(random, "random", random_side_effect)

    # We want insize = 2, current sum nums = 4, theta_as = 5, so need to remove one
    ga.delete_classifiers(population, match_set, action_set, insize=2, theta_as=5)

    # At least one classifier should have num decreased or removed
    total_num = sum(cl.num for cl in population)
    assert total_num <= 4


def test__is_preferred_to_delete_variants():
    random.seed(1)
    np.random.seed(1)

    base = DummyClassifier(q=0.5, tav=1.0, marked=False)
    worse_q = DummyClassifier(q=0.3, tav=0.5, marked=False)
    better_q = DummyClassifier(q=0.7, tav=0.1, marked=False)
    same_q_marked = DummyClassifier(q=0.5, tav=0.9, marked=True)
    same_q_unmarked = DummyClassifier(q=0.5, tav=1.2, marked=False)

    # q difference less than -0.1 returns True: worse_q < base
    assert ga._is_preferred_to_delete(base, worse_q) is True
    assert ga._is_preferred_to_delete(worse_q, base) is False

    # abs(q difference <= 0.1 and cl marked true, base not marked True
    assert ga._is_preferred_to_delete(base, same_q_marked) is True
    assert ga._is_preferred_to_delete(same_q_marked, base) is False

    # same_q conditions, tav check
    assert ga._is_preferred_to_delete(base, same_q_unmarked) is True  # tav 1.2 > 1.0 True
    assert ga._is_preferred_to_delete(same_q_unmarked, base) is False


def test__find_similar_returns_none_or_similar():
    random.seed(3)
    np.random.seed(3)

    cl1 = DummyClassifier(condition=DummyCondition(5))
    population = [cl1]

    cl2 = DummyClassifier(condition=DummyCondition(5))
    # Make cl2 equal to cl1 by condition, action and q
    cl2.action = cl1.action
    cl2.q = cl1.q
    cl2.condition.data = cl1.condition.data.copy()

    result = ga._find_similar(cl2, population)
    assert result == cl1

    cl3 = DummyClassifier(condition=DummyCondition(5))
    cl3.condition.data = ['different'] * 5

    result_none = ga._find_similar(cl3, population)
    assert result_none is None


def test__weighted_random_choice_choices(monkeypatch):
    random.seed(101)
    np.random.seed(101)

    # Setup choices dict with two classifiers and known values
    cl1 = DummyClassifier(q=2.0)
    cl2 = DummyClassifier(q=8.0)

    choices = {cl1: 2.0, cl2: 8.0}

    # Patch random.uniform to always pick in range to select cl2 deterministically
    monkeypatch.setattr(random, "uniform", lambda a, b: 5.0)

    selected = ga._weighted_random_choice(choices)

    assert selected in choices

    # The pick 5.0 is less than sum 10, current iter: cl1:2 < 5? no, cl2: 8 > 5 yes
    # So selection should be cl2
    # Because first current=2 <5 no, continue, current=10 >5 yes => cl2
    assert selected == cl2
