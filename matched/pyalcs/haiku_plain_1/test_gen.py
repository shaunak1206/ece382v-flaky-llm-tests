"""Unit tests for lcs.strategies.genetic_algorithms."""
import random

import numpy as np

from lcs import Perception
from lcs.agents.acs import Classifier, Configuration
from lcs.strategies.genetic_algorithms import (
    _find_similar,
    _is_preferred_to_delete,
    add_classifier,
    delete_classifiers,
    generalizing_mutation,
    roulette_wheel_selection,
    set_timestamps,
    should_apply,
    two_point_crossover,
)


class _Stub:
    """Minimal stand-in exposing the attributes the GA helpers read."""

    def __init__(self, name="cl", num=1, q=0.5, tav=0.0, tga=0, marked=False):
        self.name = name
        self.num = num
        self.q = q
        self.tav = tav
        self.tga = tga
        self._marked = marked

    def is_marked(self):
        return self._marked

    def __repr__(self):
        return f"_Stub({self.name!r})"


class _Bag(list):
    """List with the `expand` and `safe_remove` helpers used by delete_classifiers."""

    def expand(self):
        return list(self)

    def safe_remove(self, item):
        try:
            self.remove(item)
        except ValueError:
            pass


class _NumClassifier(Classifier):
    """
    Real Classifier that also carries the numerosity, time stamp and
    experience attributes read by the GA code. The base class defines
    __slots__, so this subclass (which has no __slots__) provides __dict__.
    """


def _make_cfg():
    return Configuration(classifier_length=4, number_of_possible_actions=2)


def _make_cl(cfg, condition="####", action=0, effect="####", q=0.5,
             tav=0.0, num=1, tga=0, exp=0):
    cl = _NumClassifier(condition=condition, action=action, effect=effect,
                        quality=q, tav=tav, cfg=cfg)
    cl.num = num
    cl.tga = tga
    cl.exp = exp
    return cl


# ---------------------------------------------------------------- should_apply

def test_should_apply_false_for_missing_or_empty_action_set():
    assert should_apply(None, 5, 1) is False
    assert should_apply([], 5, 1) is False


def test_should_apply_true_when_last_ga_is_older_than_threshold():
    cl = _Stub(num=1, tga=0)
    assert should_apply([cl], time=10, theta_ga=5) is True


def test_should_apply_is_strict_at_threshold():
    cl = _Stub(num=1, tga=5)
    assert should_apply([cl], time=10, theta_ga=5) is False
    assert should_apply([cl], time=10, theta_ga=4) is True


def test_should_apply_weights_delay_by_numerosity():
    a = _Stub("a", num=1, tga=0)
    b = _Stub("b", num=3, tga=10)
    # Weighted average of tga is (0*1 + 10*3) / 4 = 7.5, so the delay is 4.5.
    # An unweighted average (5.0) would give a delay of 7 and different results.
    assert should_apply([a, b], time=12, theta_ga=5) is False
    assert should_apply([a, b], time=12, theta_ga=4) is True


# -------------------------------------------------------------- set_timestamps

def test_set_timestamps_updates_every_classifier():
    members = [_Stub("a", tga=0), _Stub("b", tga=3), _Stub("c", tga=9)]
    assert set_timestamps(members, 42) is None
    assert all(cl.tga == 42 for cl in members)


# ----------------------------------------------------------- generalizing_mutation

def test_generalizing_mutation_zero_probability_is_noop():
    cfg = _make_cfg()
    cl = _make_cl(cfg, condition="1#01")
    generalizing_mutation(cl, mu=0.0)
    assert list(cl.condition) == ["1", "#", "0", "1"]


def test_generalizing_mutation_full_probability_generalizes_all_specified():
    cfg = _make_cfg()
    cl = _make_cl(cfg, condition="1#0#")
    generalizing_mutation(cl, mu=1.0)
    assert list(cl.condition) == ["#", "#", "#", "#"]


def test_generalizing_mutation_never_specifies_a_wildcard():
    random.seed(123)
    cfg = _make_cfg()
    cl = _make_cl(cfg, condition="10#1")
    before = list(cl.condition)

    generalizing_mutation(cl, mu=0.5)

    after = list(cl.condition)
    assert len(after) == len(before)
    for old, new in zip(before, after):
        if old == "#":
            assert new == "#"
        else:
            assert new in (old, "#")


# ------------------------------------------------------------ two_point_crossover

def test_two_point_crossover_swaps_selected_segment(monkeypatch):
    cfg = _make_cfg()
    parent = _make_cl(cfg, condition="1100")
    donor = _make_cl(cfg, condition="0011")
    # Cut points (unsorted on purpose; the function sorts them): 1 and 3.
    monkeypatch.setattr(np.random, "choice", lambda a, size, replace: [3, 1])

    two_point_crossover(parent, donor)

    assert list(parent.condition) == ["1", "0", "1", "0"]
    assert list(donor.condition) == ["0", "1", "0", "1"]


def test_two_point_crossover_preserves_position_pairs():
    np.random.seed(7)
    cfg = _make_cfg()
    parent = _make_cl(cfg, condition="1#01")
    donor = _make_cl(cfg, condition="0#10")
    before = list(zip(parent.condition, donor.condition))

    two_point_crossover(parent, donor)

    after = list(zip(parent.condition, donor.condition))
    for (p0, d0), (p1, d1) in zip(before, after):
        # Each position either stays put or swaps between parent and donor.
        assert sorted([p0, d0]) == sorted([p1, d1])


# ------------------------------------------------------- roulette_wheel_selection

def test_roulette_wheel_returns_members_of_population():
    random.seed(0)
    population = [_Stub("a"), _Stub("b"), _Stub("c")]

    for _ in range(50):
        parent1, parent2 = roulette_wheel_selection(
            population, fitnessfunc=lambda cl: 1.0)
        assert parent1 in population
        assert parent2 in population


def test_roulette_wheel_never_picks_zero_fitness_member():
    random.seed(1)
    zero = _Stub("zero")
    positive = _Stub("positive")

    def fitness(cl):
        return 0.0 if cl is zero else 1.0

    for _ in range(200):
        parent1, parent2 = roulette_wheel_selection([zero, positive], fitness)
        assert parent1 is positive
        assert parent2 is positive


def test_roulette_wheel_selection_tracks_fitness_ratio():
    random.seed(2024)
    heavy = _Stub("heavy")
    light = _Stub("light")

    def fitness(cl):
        return 3.0 if cl is heavy else 1.0

    draws = 4000
    heavy_count = 0
    for _ in range(draws):
        parent1, _parent2 = roulette_wheel_selection([heavy, light], fitness)
        if parent1 is heavy:
            heavy_count += 1

    # Expected share is 3 / (3 + 1) = 0.75.
    assert 0.70 < heavy_count / draws < 0.80


# ------------------------------------------------------------- delete_classifiers

def test_delete_classifiers_decrements_numerosity_before_removal():
    random.seed(0)
    cl = _Stub("only", num=3)
    population = _Bag([cl])
    match_set = _Bag([cl])
    action_set = _Bag([cl])

    delete_classifiers(population, match_set, action_set,
                       insize=1, theta_as=2)

    assert cl.num == 1
    assert cl in population
    assert cl in match_set
    assert cl in action_set


def test_delete_classifiers_removes_singleton_from_all_lists():
    random.seed(0)
    doomed = _Stub("doomed", num=1)
    keep = _Stub("keep", num=1)
    population = _Bag([doomed, keep])
    match_set = _Bag([doomed])
    action_set = _Bag([doomed])

    delete_classifiers(population, match_set, action_set,
                       insize=1, theta_as=1)

    assert doomed not in population
    assert doomed not in match_set
    assert doomed not in action_set
    assert keep in population
    assert len(population) == 1


def test_delete_classifiers_trims_action_set_to_threshold():
    random.seed(11)
    members = [_Stub(f"c{i}", num=1, q=0.1 * i) for i in range(1, 7)]
    population = _Bag(members)
    match_set = _Bag(members)
    action_set = _Bag(members)
    insize, theta_as = 2, 4

    delete_classifiers(population, match_set, action_set,
                       insize=insize, theta_as=theta_as)

    assert insize + sum(cl.num for cl in action_set) <= theta_as
    for cl in members:
        if cl not in action_set:
            assert cl not in population
            assert cl not in match_set


# ------------------------------------------------------- _is_preferred_to_delete
# Note: the source's second branch tests `cl.is_marked` without calling it,
# which is always truthy. The cases below avoid the one input pattern where
# that changes the outcome (candidate unmarked, cl_del marked, equal quality),
# so they hold under either reading.

def test_is_preferred_to_delete_lower_quality_candidate():
    low = _Stub("low", q=0.2)
    high = _Stub("high", q=0.6)

    assert _is_preferred_to_delete(high, low) is True
    assert _is_preferred_to_delete(low, high) is False


def test_is_preferred_to_delete_marked_candidate_within_tolerance():
    marked = _Stub("marked", q=0.5, marked=True)
    unmarked = _Stub("unmarked", q=0.55, marked=False)

    assert _is_preferred_to_delete(unmarked, marked) is True


def test_is_preferred_to_delete_higher_tav_within_tolerance():
    older = _Stub("older", q=0.5, tav=5.0)
    newer = _Stub("newer", q=0.5, tav=10.0)

    assert _is_preferred_to_delete(older, newer) is True
    assert _is_preferred_to_delete(newer, older) is False


# ------------------------------------------------------------------ _find_similar

def test_find_similar_returns_equal_classifier_or_none():
    cfg = _make_cfg()
    target = _make_cl(cfg, condition="1#0#", action=1, effect="##1#")
    same_triple = _make_cl(cfg, condition="1#0#", action=1, effect="##1#")
    other_action = _make_cl(cfg, condition="1#0#", action=0, effect="##1#")

    assert _find_similar(target, [other_action, same_triple]) is same_triple
    assert _find_similar(target, [other_action]) is None


# ------------------------------------------------------------------ add_classifier

def test_add_classifier_appends_new_and_registers_only_matching_in_match_set():
    cfg = _make_cfg()
    p = Perception("1000")
    population, action_set, match_set = [], [], []

    matching = _make_cl(cfg, condition="1###", action=0, effect="####")
    add_classifier(matching, p, population, match_set, action_set,
                   do_subsumption=False, theta_exp=20)

    non_matching = _make_cl(cfg, condition="0###", action=0, effect="####")
    add_classifier(non_matching, p, population, match_set, action_set,
                   do_subsumption=False, theta_exp=20)

    assert population == [matching, non_matching]
    assert action_set == [matching, non_matching]
    assert match_set == [matching]


def test_add_classifier_increments_numerosity_of_similar_classifier():
    cfg = _make_cfg()
    p = Perception("1000")
    existing = _make_cl(cfg, condition="1#0#", action=1, effect="##1#", num=2)
    population = [existing]
    action_set = [existing]
    match_set = [existing]

    newcomer = _make_cl(cfg, condition="1#0#", action=1, effect="##1#")
    add_classifier(newcomer, p, population, match_set, action_set,
                   do_subsumption=False, theta_exp=20)

    assert existing.num == 3
    assert population == [existing]
    assert action_set == [existing]


def test_add_classifier_leaves_marked_similar_classifier_untouched():
    cfg = _make_cfg()
    p = Perception("1000")
    existing = _make_cl(cfg, condition="1#0#", action=1, effect="##1#", num=2)
    existing.mark[0].add("1")
    assert existing.is_marked()
    population = [existing]
    action_set = [existing]

    newcomer = _make_cl(cfg, condition="1#0#", action=1, effect="##1#")
    add_classifier(newcomer, p, population, [], action_set,
                   do_subsumption=False, theta_exp=20)

    assert existing.num == 2
    assert population == [existing]


def test_add_classifier_subsumption_prefers_most_general_subsumer():
    cfg = _make_cfg()
    p = Perception("1011")
    # Both are experienced and reliable (q > theta_r = 0.9), unmarked, and
    # subsume the newcomer "1011" with the same action and effect.
    general = _make_cl(cfg, condition="1###", action=0, effect="####",
                       q=0.95, exp=50)
    specific = _make_cl(cfg, condition="10##", action=0, effect="####",
                        q=0.95, exp=50)
    action_set = [specific, general]
    population = [specific, general]

    newcomer = _make_cl(cfg, condition="1011", action=0, effect="####")
    add_classifier(newcomer, p, population, [], action_set,
                   do_subsumption=True, theta_exp=20)

    assert general.num == 2
    assert specific.num == 1
    assert len(action_set) == 2
    assert len(population) == 2
