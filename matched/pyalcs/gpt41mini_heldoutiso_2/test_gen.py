import random
import types
import pytest
import numpy as np

from lcs.strategies import genetic_algorithms as ga
from lcs import Perception


class DummyCondition:
    def __init__(self, length=5, cond=None, cfg=None):
        self.cfg = cfg or types.SimpleNamespace(classifier_length=length,
                                                classifier_wildcard='#')
        if cond is None:
            self._cond = ['0'] * length
        else:
            self._cond = cond

        # Spy for generalize calls
        self.generalize_calls = []

    def __getitem__(self, idx):
        return self._cond[idx]

    def __setitem__(self, idx, value):
        self._cond[idx] = value

    def __len__(self):
        return len(self._cond)

    def generalize(self, idx):
        self.generalize_calls.append(idx)
        self._cond[idx] = self.cfg.classifier_wildcard

    def does_match(self, perception):
        # Let's say matching if all positions equal perception attributes
        return all(p == c or c == self.cfg.classifier_wildcard
                   for p, c in zip(perception.attributes, self._cond))

    def __eq__(self, other):
        # Equality defined as same condition list
        return isinstance(other, DummyCondition) and self._cond == other._cond


class DummyClassifier:
    def __init__(self,
                 num=1,
                 tga=0,
                 q=1.0,
                 tav=0,
                 condition=None,
                 cfg=None,
                 marked=False):
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self.cfg = cfg or types.SimpleNamespace(classifier_length=5,
                                                classifier_wildcard='#')
        self.condition = condition or DummyCondition(cfg=self.cfg)
        self._marked = marked

    def is_marked(self):
        return self._marked

    def __eq__(self, other):
        if other is None:
            return False
        return (self.num == other.num and
                self.tga == other.tga and
                self.q == other.q and
                self.tav == other.tav and
                self.condition == other.condition)

    def __hash__(self):
        return id(self)


class DummyList(list):
    """
    Implements expand and safe_remove used in genetic_algorithms.delete_classifiers.
    """

    def expand(self):
        # Return all classifier instances (simulate expand)
        return self

    def safe_remove(self, cl):
        if cl in self:
            self.remove(cl)


@pytest.mark.parametrize(
    "action_set,time,theta_ga,expected",
    [
        (None, 100, 10, False),
        ([], 100, 10, False),
        ([DummyClassifier(num=0, tga=10)], 100, 10, False),
        ([DummyClassifier(num=1, tga=90)], 100, 15, False),
        ([DummyClassifier(num=1, tga=60)], 100, 10, True),
        ([DummyClassifier(num=2, tga=60), DummyClassifier(num=1, tga=40)], 100, 10, True),
        ([DummyClassifier(num=1, tga=90), DummyClassifier(num=1, tga=85)], 100, 5, True),
    ]
)
def test_should_apply_various(action_set, time, theta_ga, expected):
    assert ga.should_apply(action_set, time, theta_ga) is expected


def test_set_timestamps_updates_all_classifiers():
    cls = [DummyClassifier(tga=0), DummyClassifier(tga=5)]
    ga.set_timestamps(cls, epoch=123)
    for cl in cls:
        assert cl.tga == 123


def test_roulette_wheel_selection_returns_two_from_population(monkeypatch):
    cls1 = DummyClassifier(q=0.1)
    cls2 = DummyClassifier(q=0.5)
    cls3 = DummyClassifier(q=1.0)
    population = [cls1, cls2, cls3]

    def fitness(cl):
        return cl.q

    # monkeypatch _weighted_random_choice to always pick first and last element respectively to avoid randomness
    calls = []

    def fake_weighted_choice(choices):
        # Return different parent each call for test
        if not calls:
            calls.append(True)
            return cls1
        else:
            return cls3

    monkeypatch.setattr(ga, "_weighted_random_choice", fake_weighted_choice)

    p1, p2 = ga.roulette_wheel_selection(population, fitness)

    assert p1 == cls1
    assert p2 == cls3
    assert p1 in population and p2 in population


def test_generalizing_mutation_generalizes_correct_indices(monkeypatch):
    cfg = types.SimpleNamespace(classifier_length=5, classifier_wildcard='#')
    cl = DummyClassifier(condition=DummyCondition(cfg=cfg), cfg=cfg)

    # set condition so indexes 1 and 3 are not wildcards
    cl.condition._cond = ['0', '1', '#', '2', '#']

    # Patch random.random to control randomness for mutation
    random_calls = [0.1, 0.5, 0.05, 0.7, 0.01]
    iter_calls = iter(random_calls)

    monkeypatch.setattr(random, "random", lambda: next(iter_calls))

    ga.generalizing_mutation(cl, mu=0.2)

    # indexes with random.random < 0.2 and not wildcard: idx0(0.1), idx2('#'), idx4('#'), idx2,4 no mutation
    # idx0 = '0' and 0.1 < 0.2 => generalized (wildcard)
    # idx1 = '1' and 0.5 > 0.2 => no
    # idx3 = '2' and 0.7 > 0.2 => no

    # Only idx0 generalized
    assert 0 in cl.condition.generalize_calls
    assert 1 not in cl.condition.generalize_calls
    assert 3 not in cl.condition.generalize_calls
    # Check that condition is updated at index 0
    assert cl.condition[0] == cl.cfg.classifier_wildcard


def test_two_point_crossover_swaps_slices(monkeypatch):
    cfg = types.SimpleNamespace(classifier_length=6, classifier_wildcard='#')

    parent_cond = DummyCondition(length=6, cond=list('ABCDEF'), cfg=cfg)
    donor_cond = DummyCondition(length=6, cond=list('UVWXYZ'), cfg=cfg)
    parent = DummyClassifier(condition=parent_cond, cfg=cfg)
    donor = DummyClassifier(condition=donor_cond, cfg=cfg)

    # Patch np.random.choice to return fixed crossover points 2 and 5
    monkeypatch.setattr(np.random, "choice", lambda a, size, replace: np.array([2, 5]))

    ga.two_point_crossover(parent, donor)

    # Slices 2 to 5 swapped
    assert parent.condition._cond == ['A', 'B', 'W', 'X', 'Y', 'F']
    assert donor.condition._cond == ['U', 'V', 'C', 'D', 'E', 'Z']


def test_add_classifier_appends_when_no_old(monkeypatch):
    population = []
    match_set = []
    action_set = []
    cl = DummyClassifier(num=1)
    cl.condition._cond[0] = 'A'

    # Subsumption returns None to force adding new
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: None)

    p = Perception(attributes=['A', '0', '0', '0', '0'])

    ga.add_classifier(cl, p, population, match_set, action_set, do_subsumption=False, theta_exp=0)

    assert cl in population
    assert cl in action_set
    # Because condition does_match returns True for matching first attr
    assert cl in match_set


def test_add_classifier_increments_num_when_old_found_not_marked(monkeypatch):
    population = []
    match_set = []
    action_set = []

    old_cl = DummyClassifier(num=3, marked=False)
    cl = DummyClassifier(num=1)

    # Return old_cl from _find_old_classifier forcing increment num
    monkeypatch.setattr(ga, "_find_old_classifier", lambda *a, **kw: old_cl)

    ga.add_classifier(cl, Perception(attributes=[]),
                      population, match_set, action_set,
                      do_subsumption=True, theta_exp=0)

    # Old classifier num incremented
    assert old_cl.num == 4
    # No new classifier added
    assert cl not in population
    assert cl not in action_set


def test_add_classifier_does_not_increment_if_marked(monkeypatch):
    old_cl = DummyClassifier(num=5, marked=True)
    cl = DummyClassifier(num=1)

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *a, **kw: old_cl)

    population = []
    match_set = []
    action_set = []

    ga.add_classifier(cl, Perception(attributes=[]),
                      population, match_set, action_set,
                      do_subsumption=True, theta_exp=0)

    # num should remain same for marked classifier
    assert old_cl.num == 5


def test__is_preferred_to_delete_returns_true_for_lower_q():
    cl_del = DummyClassifier(q=0.9)
    cl = DummyClassifier(q=0.7)
    # q - q_del = -0.2 < -0.1 => True
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test__is_preferred_to_delete_returns_false_for_higher_q():
    cl_del = DummyClassifier(q=0.8)
    cl = DummyClassifier(q=0.95)
    # q - q_del = 0.15 > -0.1 => False
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test__is_preferred_to_delete_considers_marked_and_tav():
    # q close enough, abs diff <= 0.1
    cl_del = DummyClassifier(q=0.9, tav=5, marked=False)
    cl = DummyClassifier(q=0.85, tav=10, marked=True)
    cl._marked = True

    # cl is marked, cl_del not, so returns True
    assert ga._is_preferred_to_delete(cl_del, cl) is True

    # flip marked flags
    cl_del._marked = True
    cl._marked = False
    cl.tav = 10
    cl_del.tav = 5

    # cl is not marked but cl_del is, so will test tav
    # cl tav > cl_del tav => True
    assert ga._is_preferred_to_delete(cl_del, cl) is True

    cl.tav = 3
    # cl tav < cl_del tav => False
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test__find_similar_returns_first_equal_or_none():
    cl1 = DummyClassifier(condition=DummyCondition(cond=list("ABCDE")))
    cl2 = DummyClassifier(condition=DummyCondition(cond=list("ABCDE")))
    cl3 = DummyClassifier(condition=DummyCondition(cond=list("XYZWE")))

    population = [cl3, cl2]

    # cl1 equal to cl2 only
    found = ga._find_similar(cl1, population)
    assert found == cl2

    # No match
    cl4 = DummyClassifier(condition=DummyCondition(cond=list("FFFFF")))
    found_none = ga._find_similar(cl4, population)
    assert found_none is None


def test_delete_classifiers_removes_one_when_needed(monkeypatch):
    cfg = types.SimpleNamespace()
    # Create classifiers with num=1 (one will be removed completely)
    cl1 = DummyClassifier(num=1, q=0.3)
    cl2 = DummyClassifier(num=2, q=0.4)  # will be decremented first
    cl3 = DummyClassifier(num=1, q=0.9)
    action_set = DummyList([cl1, cl2, cl3])
    population = DummyList([cl1, cl2, cl3])
    match_set = DummyList([cl1, cl3])

    random_choice_vals = [0.4, 0.1, 0.2]

    def fake_random():
        # Return values cycling through list, all below 0.3 to select classifiers on first iteration
        while True:
            for v in random_choice_vals:
                yield v

    gen = fake_random()
    monkeypatch.setattr(random, "random", lambda: next(gen))

    # Setup insize and theta_as so we must delete classifiers
    insize = 1
    theta_as = 3  # Current sum cl.num=4, insize=1 => 5 > 3 need deletions

    # Call deletion
    ga.delete_classifiers(population, match_set, action_set, insize, theta_as)

    # One classifier should be modified/deleted:
    # cl2 should have num decremented first from 2 to 1
    # possibly cl1 removed next (num=1) from all lists afterwards

    # Since random.random is <0.3 for multiple tries, it will eventually pick cl1 or cl2 etc.
    # Check that sum of nums in action_set is <= theta_as - insize
    total_num = sum(cl.num for cl in action_set)
    assert total_num + insize <= theta_as

    # Check cl2 num decremented or cl1 removed
    # cl1 removed from lists means cl1 not in population anymore
    if cl1 not in population:
        assert cl1 not in match_set and cl1 not in action_set
    else:
        # if cl1 remain, cl2 num must be 1 now
        assert cl2.num == 1


def test_should_apply_returns_false_on_zero_overall_num():
    # All classifiers with num=0 means no GA
    cl1 = DummyClassifier(num=0, tga=10)
    cl2 = DummyClassifier(num=0, tga=20)
    action_set = [cl1, cl2]
    assert ga.should_apply(action_set, time=50, theta_ga=5) is False


def test__find_old_classifier_prioritizes_subsumers(monkeypatch):
    cl = DummyClassifier()
    population = [DummyClassifier(), DummyClassifier()]

    # Mock find_subsumers to return a list
    monkeypatch.setattr(ga, "find_subsumers", lambda cl_, pop, theta_exp: [population[1]])
    monkeypatch.setattr(ga, "_find_similar", lambda cl_, pop: population[0])

    old = ga._find_old_classifier(population, cl, use_subsumption=True, theta_exp=0)
    # Should return first subsumer, not similar
    assert old == population[1]

    # When no subsumers found, fallback to similar
    monkeypatch.setattr(ga, "find_subsumers", lambda cl_, pop, theta_exp: [])
    old2 = ga._find_old_classifier(population, cl, use_subsumption=True, theta_exp=0)
    assert old2 == population[0]
