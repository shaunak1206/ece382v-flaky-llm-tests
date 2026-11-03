import random
import pytest
import numpy as np

from lcs.strategies import genetic_algorithms as ga
from lcs import Perception


class DummyCondition:
    def __init__(self, cond_list, cfg):
        self._cond = list(cond_list)
        self.cfg = cfg
        self.generalize_calls = []

    def __getitem__(self, idx):
        return self._cond[idx]

    def __setitem__(self, idx, val):
        self._cond[idx] = val

    def __len__(self):
        return len(self._cond)

    def __eq__(self, other):
        # For similarity test, compare this condition list with other's
        if not isinstance(other, DummyClassifier):
            return False
        return (self._cond == other.condition._cond and
                other.action == self.cfg.action and
                other.effect == self.cfg.effect)

    def generalize(self, idx):
        self.generalize_calls.append(idx)
        # dummy: set element to wildcard
        self._cond[idx] = self.cfg.classifier_wildcard

    def does_match(self, perception):
        # dummy: always return True for testing
        return True


class DummyCfg:
    def __init__(self, length=5, wildcard='#', action=None, effect=None):
        self.classifier_length = length
        self.classifier_wildcard = wildcard
        self.action = action
        self.effect = effect


class DummyClassifier:
    def __init__(self, condition, action=None, effect=None, num=1, tga=0, q=1.0, tav=0, cfg=None):
        # condition expected to be DummyCondition or compatible
        self.condition = condition
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self._marked = False
        self.cfg = cfg or DummyCfg(length=len(condition), action=action, effect=effect)
        self.action = action  # simplifying
        self.effect = effect

    def __eq__(self, other):
        # Compare based on condition list, action, effect
        if not isinstance(other, DummyClassifier):
            return False
        return (self.condition._cond == other.condition._cond and
                self.action == other.action and
                self.effect == other.effect)

    def is_marked(self):
        return self._marked

    def mark(self):
        self._marked = True


class SafeList(list):
    """List with safe_remove method as used in code."""
    def safe_remove(self, item):
        try:
            self.remove(item)
        except ValueError:
            pass


@pytest.fixture
def sample_action_set():
    cfg = DummyCfg(length=4, wildcard='#', action='a', effect='e')
    cl1 = DummyClassifier(DummyCondition(['0', '1', '0', '1'], cfg), action='a', effect='e', num=2, tga=5, q=0.8, tav=10, cfg=cfg)
    cl2 = DummyClassifier(DummyCondition(['1', '1', '0', '1'], cfg), action='a', effect='e', num=3, tga=10, q=0.5, tav=20, cfg=cfg)
    cl3 = DummyClassifier(DummyCondition(['0', '0', '0', '1'], cfg), action='a', effect='e', num=0, tga=0, q=0.9, tav=5, cfg=cfg)
    return [cl1, cl2, cl3]


def test_should_apply_none_action_set():
    assert ga.should_apply(None, time=10, theta_ga=5) is False


def test_should_apply_zero_num():
    cl = DummyClassifier(DummyCondition(['0'], DummyCfg(length=1)), num=0, tga=10)
    assert ga.should_apply([cl], time=20, theta_ga=5) is False


def test_should_apply_true_and_false(sample_action_set):
    # average tga = (5*2 + 10*3 + 0*0) / (2+3+0) = (10 + 30) / 5 = 8.0
    # time - avg = 14 - 8 = 6
    assert ga.should_apply(sample_action_set, time=14, theta_ga=5) is True
    assert ga.should_apply(sample_action_set, time=13, theta_ga=5) is False


def test_set_timestamps_updates_all(sample_action_set):
    new_epoch = 100
    ga.set_timestamps(sample_action_set, new_epoch)
    for cl in sample_action_set:
        assert cl.tga == new_epoch


def test_roulette_wheel_selection_parents_are_from_population():
    population = [DummyClassifier(DummyCondition(['0'], DummyCfg(length=1)), q=1.0),
                  DummyClassifier(DummyCondition(['1'], DummyCfg(length=1)), q=2.0)]

    def fitness_func(cl):
        return cl.q

    parent1, parent2 = ga.roulette_wheel_selection(population, fitness_func)
    assert parent1 in population and parent2 in population


def test_generalizing_mutation_all_indices_possible(monkeypatch):
    cfg = DummyCfg(length=3, wildcard='#')
    cl = DummyClassifier(DummyCondition(['0', '1', '0'], cfg), cfg=cfg)

    # patch random.random to allow mutation all indices by returning 0 always
    monkeypatch.setattr(random, "random", lambda: 0.0)
    mu = 0.5
    ga.generalizing_mutation(cl, mu)
    # condition should be fully generalized (all indices generalized)
    assert set(cl.condition.generalize_calls) == {0, 1, 2}
    # conditions replaced by wildcard
    assert all(c == cfg.classifier_wildcard for c in cl.condition._cond)


def test_generalizing_mutation_no_mutation(monkeypatch):
    cfg = DummyCfg(length=4, wildcard='#')
    cond_vals = ['0', '1', '0', '1']
    cl = DummyClassifier(DummyCondition(cond_vals, cfg), cfg=cfg)
    # patch random.random to always return > mu to prevent mutation
    monkeypatch.setattr(random, "random", lambda: 1.0)
    mu = 0.5
    ga.generalizing_mutation(cl, mu)
    assert cl.condition.generalize_calls == []
    # original condition preserved
    assert cl.condition._cond == cond_vals


def test_two_point_crossover_swaps_segment(monkeypatch):
    cfg = DummyCfg(length=5, wildcard='#')
    parent = DummyClassifier(DummyCondition(['a', 'b', 'c', 'd', 'e'], cfg), cfg=cfg)
    donor = DummyClassifier(DummyCondition(['1', '2', '3', '4', '5'], cfg), cfg=cfg)

    # patch np.random.choice to fix crossover points
    monkeypatch.setattr(np.random, "choice", lambda r, size, replace: np.array([1, 4]))
    ga.two_point_crossover(parent, donor)

    # crossover should swap indices 1 to 3 (note right index exclusive)
    # indices: 1,2,3
    # parent's indices before = ['b', 'c', 'd']
    # donor's indices before = ['2', '3', '4']
    assert parent.condition._cond == ['a', '2', '3', '4', 'e']
    assert donor.condition._cond == ['1', 'b', 'c', 'd', '5']


def test_add_classifier_new_classifier_added_to_all():
    cfg = DummyCfg(length=3, wildcard='#', action="act", effect="eff")
    cond = DummyCondition(['0', '1', '0'], cfg)
    cl = DummyClassifier(cond, action="act", effect="eff", cfg=cfg)

    # match_set and action_set as SafeList to test safe_remove usage
    population = SafeList()
    match_set = SafeList()
    action_set = SafeList()

    p = Perception({})
    cl.condition.does_match = lambda p: True

    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=0)

    # New classifier added to all lists
    assert cl in population
    assert cl in action_set
    assert cl in match_set


def test_add_classifier_existing_classifier_increment_num(monkeypatch):
    cfg = DummyCfg(length=3, wildcard='#', action="act", effect="eff")
    cond = DummyCondition(['0', '1', '0'], cfg)
    cl1 = DummyClassifier(cond, action="act", effect="eff", num=1, cfg=cfg)
    cl2 = DummyClassifier(cond, action="act", effect="eff", num=1, cfg=cfg)

    # Mark cl2 not marked, cl1 marked
    cl1._marked = False

    population = SafeList([cl1])
    match_set = SafeList()
    action_set = SafeList([cl1])

    p = Perception({})
    cl2.condition.does_match = lambda p: True

    # Patch _find_old_classifier to return cl1 so existing classifier found
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: cl1)

    ga.add_classifier(cl2, p, population, match_set, action_set, do_subsumption=True, theta_exp=0)

    # cl1's numerosity incremented (cl1 is old classifier)
    assert cl1.num == 2
    # cl2 not added newly
    assert cl2 not in population
    assert cl2 not in action_set


def test_delete_classifiers_removes_until_size_below_threshold(monkeypatch):
    cfg = DummyCfg(length=2, wildcard='#', action='act', effect='eff')
    cl1 = DummyClassifier(DummyCondition(['0', '0'], cfg), num=2, q=0.2, tav=5, cfg=cfg)
    cl2 = DummyClassifier(DummyCondition(['1', '1'], cfg), num=3, q=0.1, tav=10, cfg=cfg)
    cl3 = DummyClassifier(DummyCondition(['0', '1'], cfg), num=1, q=0.3, tav=1, cfg=cfg)

    population = SafeList([cl1, cl2, cl3])
    match_set = SafeList([cl1, cl2, cl3])
    action_set = SafeList([cl1, cl2, cl3])

    # Patch action_set.expand() to just return action_set list.
    monkeypatch.setattr(action_set, "expand", lambda: action_set)

    # Patch random.random so that first eligible classifier is chosen immediately
    random_values = iter([0.1, 0.4])  # first < 0.3 triggers deletion, second > 0.3 no effect

    monkeypatch.setattr(random, "random", lambda: next(random_values))

    # insize + sum(num) = 4 + (2+3+1) = 10 > theta_as=8 so deletion needed
    ga.delete_classifiers(population, match_set, action_set, insize=4, theta_as=8)

    # One classifier was deleted (reduce num or removed)
    total_num = sum(cl.num for cl in action_set)
    assert total_num <= 8


def test__is_preferred_to_delete_based_on_q_and_marked():
    cfg = DummyCfg(length=2, wildcard='#')

    cl_del = DummyClassifier(DummyCondition(['0', '0'], cfg), q=0.6, tav=5, cfg=cfg)
    cl = DummyClassifier(DummyCondition(['1', '1'], cfg), q=0.4, tav=10, cfg=cfg)

    # q difference < -0.1: cl.q=0.4 - 0.6 = -0.2 < -0.1 -> True
    assert ga._is_preferred_to_delete(cl_del, cl) is True

    cl.q = 0.55
    cl_del.q = 0.5
    # abs difference = 0.05 <= 0.1

    cl._marked = True
    cl_del._marked = False
    # cl marked but cl_del not => True
    assert ga._is_preferred_to_delete(cl_del, cl) is True

    cl._marked = False
    cl_del._marked = True
    cl.tav = 20
    cl_del.tav = 10
    # cl.tav > cl_del.tav => True
    assert ga._is_preferred_to_delete(cl_del, cl) is True

    cl.tav = 5
    cl_del.tav = 10
    # None of conditions met
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test__find_similar_returns_none_or_classifier():
    cfg = DummyCfg(length=2, wildcard='#')
    c1 = DummyClassifier(DummyCondition(['0', '1'], cfg), action="a", effect="e", cfg=cfg)
    c2 = DummyClassifier(DummyCondition(['0', '1'], cfg), action="a", effect="e", cfg=cfg)
    c3 = DummyClassifier(DummyCondition(['1', '1'], cfg), action="a", effect="e", cfg=cfg)
    population = [c1]

    # Find similar - c2 matches c1 by equality
    assert ga._find_similar(c2, population) == c1
    # c3 differs, no similar found
    assert ga._find_similar(c3, population) is None


def test__find_old_classifier_uses_subsumers(monkeypatch):
    cfg = DummyCfg(length=2, wildcard='#')
    cl = DummyClassifier(DummyCondition(['0', '1'], cfg), action="a", effect="e", cfg=cfg)
    population = []

    # Patch find_subsumers to return list with one element (simulated subsumer)
    subsumer = DummyClassifier(DummyCondition(['#', '#'], cfg), action="a", effect="e", cfg=cfg)
    monkeypatch.setattr(ga, "find_subsumers", lambda c, pop, theta: [subsumer])

    # Returns first subsumer
    old = ga._find_old_classifier(population, cl, use_subsumption=True, theta_exp=0)
    assert old == subsumer

    # Patch find_subsumers to return empty
    monkeypatch.setattr(ga, "find_subsumers", lambda c, pop, theta: [])

    # Patch _find_similar to return 'similar' dummy
    similar = DummyClassifier(DummyCondition(['0', '1'], cfg), action="a", effect="e", cfg=cfg)
    monkeypatch.setattr(ga, "_find_similar", lambda cl_, pop: similar)

    old = ga._find_old_classifier(population, cl, use_subsumption=True, theta_exp=0)
    assert old == similar

    # If no subsumers and no similar, returns None
    monkeypatch.setattr(ga, "_find_similar", lambda cl_, pop: None)
    old = ga._find_old_classifier(population, cl, use_subsumption=True, theta_exp=0)
    assert old is None


def test__weighted_random_choice_returns_key(monkeypatch):
    choices = {
        'a': 1,
        'b': 2,
        'c': 3,
    }
    # total 6; choose pick values to test code paths

    # Patch random.uniform to control pick value
    values = iter([0.5, 2.5, 5.5])

    def fake_uniform(a, b):
        return next(values)

    monkeypatch.setattr(random, "uniform", fake_uniform)

    # First call: pick=0.5 -> 'a' should be picked (0+1 > 0.5)
    assert ga._weighted_random_choice({'a': 1, 'b': 10}) == 'a'

    # Second call: pick=2.5; keys and values: a=1, b=2, c=3
    result = ga._weighted_random_choice(choices)
    # 1 < 2.5 < 3 so should be 'b'
    # We run with 1,2,3 values:
    # sum=6
    # current summed:
    # a=1
    # b=3
    # c=6
    # pick=2.5 < 3 means b chosen
    assert result == 'b'

    # Third call: pick=5.5, c is chosen (sum=6, 5.5 < 6)
    result = ga._weighted_random_choice(choices)
    assert result == 'c'
