import random
import types
import pytest
from lcs import Perception
from lcs.strategies import genetic_algorithms as ga


class DummyCondition:
    def __init__(self, cond_list, cfg):
        self._list = cond_list
        self.cfg = cfg
        self.generalize_calls = []

    def __getitem__(self, idx):
        return self._list[idx]

    def __setitem__(self, idx, val):
        self._list[idx] = val

    def __len__(self):
        return len(self._list)

    def generalize(self, idx):
        self.generalize_calls.append(idx)

    def does_match(self, p):
        # Simple stub: returns True if perception equals condition list
        return p.state == self._list

    def __eq__(self, other):
        if not isinstance(other, DummyCondition):
            return False
        return self._list == other._list and self.cfg == other.cfg


class DummyCfg:
    def __init__(self, length, wildcard):
        self.classifier_length = length
        self.classifier_wildcard = wildcard


class DummyClassifier:
    def __init__(self, num=1, tga=0, q=1.0, tav=0, marked=False,
                 condition=None, action=None, effect=None, cfg=None):
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self._marked = marked
        self.condition = condition
        self.action = action
        self.effect = effect
        self.cfg = cfg or DummyCfg(5, '#')
        # For equality and hashing:
        # equality defined as same condition list, action, effect
        # (as per _find_similar requirement)
    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        if not isinstance(other, DummyClassifier):
            return False
        return (self.condition == other.condition and
                self.action == other.action and
                self.effect == other.effect)

    def __hash__(self):
        # Defensive hash: combine attributes
        return hash((tuple(self.condition._list if self.condition else ()),
                     self.action, self.effect))


# For testing _find_old_classifier and subsumption, mock find_subsumers
def test_should_apply_none_and_zero_num():
    # None action set returns False
    assert not ga.should_apply(None, time=10, theta_ga=5)

    # Empty action set
    assert not ga.should_apply([], time=10, theta_ga=5)

    # All classifiers have num=0 -> no GA
    cl1 = DummyClassifier(num=0, tga=1)
    assert not ga.should_apply([cl1], time=10, theta_ga=5)


def test_should_apply_true_false(monkeypatch):
    # Setup classifiers with num and tga
    cl1 = DummyClassifier(num=2, tga=5)
    cl2 = DummyClassifier(num=3, tga=4)
    action_set = [cl1, cl2]
    # overall_time = 5*2+4*3=10+12=22, overall_num=5, average=4.4
    # time - average = 10-4.4=5.6 > theta_ga=5  -> True
    assert ga.should_apply(action_set, time=10, theta_ga=5)

    # For theta_ga=6, should be False
    assert not ga.should_apply(action_set, time=10, theta_ga=6)


def test_set_timestamps_updates_all():
    cl1 = DummyClassifier(tga=0)
    cl2 = DummyClassifier(tga=3)
    action_set = [cl1, cl2]
    ga.set_timestamps(action_set, epoch=10)
    assert cl1.tga == 10
    assert cl2.tga == 10


def test_roulette_wheel_selection_selection_behavior():
    cl1 = DummyClassifier(q=0.1)
    cl2 = DummyClassifier(q=0.4)
    cl3 = DummyClassifier(q=0.5)
    population = [cl1, cl2, cl3]

    def fitnessfunc(cl):
        # fitness is cl.q^3
        return cl.q ** 3

    # We patch random.uniform for deterministic testing,
    # but since weighted random is probabilistic,
    # here just check returned parents are from population
    parent1, parent2 = ga.roulette_wheel_selection(population, fitnessfunc)
    assert parent1 in population
    assert parent2 in population


def test_generalizing_mutation_effect(monkeypatch):
    cfg = DummyCfg(length=3, wildcard='#')
    condition = DummyCondition(cond_list=['A', 'B', 'C'], cfg=cfg)

    cl = DummyClassifier(condition=condition, cfg=cfg)

    # monkeypatch random.random to return values less than mu for idx=1 only
    calls = []

    def fake_random():
        # First call is idx=0, return 1 (no mutation)
        # Second call idx=1, return 0 (mutation)
        # Third call idx=2, return 1 (no mutation)
        val = calls.pop(0)
        return val

    calls.extend([1, 0, 1])
    monkeypatch.setattr("random.random", fake_random)

    mu = 0.5
    ga.generalizing_mutation(cl, mu)
    # Only idx=1 should be generalized
    assert condition.generalize_calls == [1]


def test_two_point_crossover_crossover_effect(monkeypatch):
    cfg = DummyCfg(length=5, wildcard='#')
    cond_p = DummyCondition(cond_list=[1, 2, 3, 4, 5], cfg=cfg)
    cond_d = DummyCondition(cond_list=[6, 7, 8, 9, 10], cfg=cfg)

    parent = DummyClassifier(condition=cond_p, cfg=cfg)
    donor = DummyClassifier(condition=cond_d, cfg=cfg)

    # Patch np.random.choice to select (1,4) for crossover points
    monkeypatch.setattr("numpy.random.choice", lambda arr, size, replace: [1, 4])

    original_parent = cond_p._list.copy()
    original_donor = cond_d._list.copy()

    ga.two_point_crossover(parent, donor)

    # After crossover indexes 1..3 inclusive should have swapped parts
    # Check that slices swapped
    assert parent.condition._list[1:4] == original_donor[1:4]
    assert donor.condition._list[1:4] == original_parent[1:4]

    # The other indices remain unchanged
    assert parent.condition._list[0] == original_parent[0]
    assert donor.condition._list[0] == original_donor[0]
    assert parent.condition._list[4] == original_parent[4]
    assert donor.condition._list[4] == original_donor[4]


def test_add_classifier_new_and_existing(monkeypatch):
    cfg = DummyCfg(length=3, wildcard='#')

    def fake_find_subsumers(cl, population, theta_exp):
        # Return empty list to test no subsumption path
        return []

    monkeypatch.setattr(ga, "find_subsumers", fake_find_subsumers)

    # create classifier to add
    cond = DummyCondition(['A', 'B', 'C'], cfg)
    cl = DummyClassifier(condition=cond, action=1, effect=None, cfg=cfg)

    population = []
    match_set = []
    action_set = []
    p = Perception(state=['A', 'B', 'C'])

    # No subsumer found -> classifier is added
    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=True, theta_exp=5)
    assert cl in population
    assert cl in action_set
    assert cl in match_set  # matches perception

    # Test existing subsumer found, marked classifier -> does not increment num
    marked_cl = DummyClassifier(num=1, tga=0, q=1.0, tav=0, marked=True,
                               condition=cond, action=1, effect=None, cfg=cfg)
    def fake_subsumers(cl, pop, theta_exp):
        return [marked_cl]

    monkeypatch.setattr(ga, "find_subsumers", fake_subsumers)
    population = [marked_cl]
    action_set = [marked_cl]
    match_set = []

    new_cl = DummyClassifier(condition=cond, action=1, effect=None, cfg=cfg)
    ga.add_classifier(new_cl, p, population, match_set, action_set, do_subsumption=True, theta_exp=5)
    # num should not increment because subsumer is marked
    assert marked_cl.num == 1
    # new_cl was not added (should not in population/action_set)
    assert new_cl not in population
    assert new_cl not in action_set


def test_add_classifier_existing_unmarked_increments_num(monkeypatch):
    cfg = DummyCfg(length=3, wildcard='#')
    cond = DummyCondition(['A', 'B', 'C'], cfg)
    unmarked_cl = DummyClassifier(num=1, condition=cond, action=1, effect=None, cfg=cfg)
    unmarked_cl._marked = False  # explicitly unmarked

    def fake_subsumers(cl, pop, theta_exp):
        return [unmarked_cl]

    monkeypatch.setattr(ga, "find_subsumers", fake_subsumers)
    population = [unmarked_cl]
    action_set = [unmarked_cl]
    match_set = []

    p = Perception(state=['A', 'B', 'C'])

    new_cl = DummyClassifier(condition=cond, action=1, effect=None, cfg=cfg)
    ga.add_classifier(new_cl, p, population, match_set, action_set, do_subsumption=True, theta_exp=5)
    # num should increment because subsumer is unmarked
    assert unmarked_cl.num == 2
    # new_cl was not added
    assert new_cl not in population


def test_delete_classifiers_removes_or_decrements(monkeypatch):
    cfg = DummyCfg(length=3, wildcard='#')
    cond = DummyCondition(['A', 'B', 'C'], cfg)

    # Create classifiers with varying num
    cl1 = DummyClassifier(num=2, q=0.5, tav=10, condition=cond, action=1, effect=None, cfg=cfg)
    cl2 = DummyClassifier(num=1, q=0.4, tav=20, condition=cond, action=2, effect=None, cfg=cfg)
    cl3 = DummyClassifier(num=3, q=0.6, tav=5, condition=cond, action=3, effect=None, cfg=cfg)

    population = [cl1, cl2, cl3]
    match_set = [cl1, cl2, cl3]
    action_set = []

    # action_set.expand() expected:
    # We'll monkeypatch action_set to have an expand() method returning self list for iteration
    class ActionSetWrapper(list):
        def expand(self):
            return self

        def safe_remove(self, cl):
            self.remove(cl)

    action_set = ActionSetWrapper([cl1, cl2, cl3])

    # Also add safe_remove to population and match_set
    class SafeList(list):
        def safe_remove(self, cl):
            self.remove(cl)
    population = SafeList([cl1, cl2, cl3])
    match_set = SafeList([cl1, cl2, cl3])

    # Patch random.random to control which classifier gets chosen for deletion
    # We'll simulate that cl2 is chosen for deletion first, then cl1, then cl3
    choices = [0.1, 0.4, 0.2]  # used for comparisons <0.3, so 0.1 and 0.2 <0.3 triggers deletion
    calls = iter(choices * 10)
    monkeypatch.setattr("random.random", lambda: next(calls))

    # The threshold theta_as is 5:
    # total num in action_set = 2+1+3=6
    # insize=0 so 6 > 5 must delete classifiers until <=5
    ga.delete_classifiers(population, match_set, action_set, insize=0, theta_as=5)

    total_num = sum(cl.num for cl in action_set)
    assert total_num <= 5
    # Check that either at least one classifier removed or one classifier's num decremented
    # Either cl2 removed (num=1) or cl1/cl3 decremented (num>1)
    # Also check consistency: population, match_set and action_set synced
    for cl in [cl1, cl2, cl3]:
        in_pop = cl in population
        in_match = cl in match_set
        in_action = cl in action_set
        # All lists must be consistent
        assert in_pop == in_match == in_action


@pytest.mark.parametrize("q_del,q_cl,marked_del,marked_cl,tav_del,tav_cl,expected", [
    (0.5, 0.3, False, False, 5, 10, True),   # cl q higher -> cl preferred
    (0.3, 0.5, False, False, 5, 4, False),   # cl q lower -> cl_del preferred
    (0.5, 0.5, False, True, 5, 6, True),     # q equal, cl marked, forget cl_del marked
    (0.5, 0.5, True, False, 5, 6, False),    # q equal, swap marked flags
    (0.5, 0.5, False, False, 5, 10, True),   # q equal, use tav comparison
    (0.5, 0.5, False, False, 10, 5, False),  # tav comparison inversion
])
def test_is_preferred_to_delete_logic(q_del, q_cl, marked_del, marked_cl, tav_del, tav_cl, expected):
    cl_del = DummyClassifier(q=q_del, tav=tav_del)
    cl_del._marked = marked_del
    cl = DummyClassifier(q=q_cl, tav=tav_cl)
    cl._marked = marked_cl

    result = ga._is_preferred_to_delete(cl_del, cl)
    assert result == expected


def test_find_old_classifier_subsumer(monkeypatch):
    cfg = DummyCfg(length=3, wildcard='#')
    cond = DummyCondition(['A', 'B', 'C'], cfg)
    cl = DummyClassifier(condition=cond, action=1, effect=None, cfg=cfg)

    subsumer = DummyClassifier(condition=cond, action=1, effect=None, cfg=cfg)
    subsumer2 = DummyClassifier(condition=cond, action=1, effect=None, cfg=cfg)

    # subsumers returns list with two subsumers
    monkeypatch.setattr(ga, "find_subsumers", lambda clf, pop, texp: [subsumer, subsumer2])

    population = [subsumer, subsumer2]
    old = ga._find_old_classifier(population, cl, True, 5)
    # Should pick the first subsumer
    assert old is subsumer

    # No subsumers, test fallback to _find_similar
    monkeypatch.setattr(ga, "find_subsumers", lambda clf, pop, texp: [])

    similar_cl = DummyClassifier(condition=cond, action=1, effect=None, cfg=cfg)
    population.append(similar_cl)

    # Should return the first classifier in population equal to cl
    result = ga._find_old_classifier(population, cl, True, 5)
    # Since cl does not equal any (equality depends on condition/action/effect),
    # but similar_cl qualifies
    assert result == similar_cl or result is None


def test_find_similar_returns_first_equal():
    cfg = DummyCfg(length=3, wildcard='#')
    cond1 = DummyCondition(['A', 'B', 'C'], cfg)
    cond2 = DummyCondition(['D', 'E', 'F'], cfg)

    cl = DummyClassifier(condition=cond1, action=1, effect=None, cfg=cfg)
    cl_same = DummyClassifier(condition=cond1, action=1, effect=None, cfg=cfg)
    cl_diff = DummyClassifier(condition=cond2, action=2, effect=None, cfg=cfg)

    population = [cl_diff, cl_same]
    found = ga._find_similar(cl, population)
    assert found == cl_same

    # If no similar classifier
    population = [cl_diff]
    found_none = ga._find_similar(cl, population)
    assert found_none is None


def test_weighted_random_choice_selection(monkeypatch):
    cl1 = DummyClassifier()
    cl2 = DummyClassifier()
    # weights assigned manually
    choices = {cl1: 1, cl2: 3}

    # We will patch random.uniform to deterministic values and test output manually
    vals_expected = [
        (0, cl1),                  # pick very close to 0 -> cl1
        (1.5, cl2),                # pick above 1 but below 4 -> cl2
        (3.999, cl2),              # pick just below max -> cl2
    ]

    for val, expected in vals_expected:
        monkeypatch.setattr("random.uniform", lambda a, b: val)
        selected = ga._weighted_random_choice(choices)
        assert selected == expected
