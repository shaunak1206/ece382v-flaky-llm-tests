import random

import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga


SEED = 12345


@pytest.fixture(autouse=True)
def _restore_global_rng_state():
    """Restore the global RNG states so seeding in a test leaks nothing."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    yield
    random.setstate(py_state)
    np.random.set_state(np_state)


class FakeCfg:
    def __init__(self, classifier_wildcard="#", classifier_length=4):
        self.classifier_wildcard = classifier_wildcard
        self.classifier_length = classifier_length


class FakeCondition(list):
    """List of attribute values with the methods used by the GA code."""

    def __init__(self, values, wildcard="#"):
        super().__init__(values)
        self.wildcard = wildcard

    def generalize(self, idx):
        self[idx] = self.wildcard

    def does_match(self, p):
        return all(c == self.wildcard or c == x for c, x in zip(self, p))


class FakeClassifier:
    """Minimal classifier stand-in; equality and hashing are identity-based."""

    def __init__(self, condition, q=0.5, tga=0, tav=0, num=1,
                 marked=False, cfg=None):
        self.cfg = cfg or FakeCfg()
        self.condition = FakeCondition(condition, self.cfg.classifier_wildcard)
        self.q = q
        self.tga = tga
        self.tav = tav
        self.num = num
        self._marked = marked

    def is_marked(self):
        return self._marked


class FakeList(list):
    """List with the extra methods used on population/match/action sets."""

    def expand(self):
        expanded = []
        for cl in self:
            expanded.extend([cl] * 1)
        return expanded

    def safe_remove(self, item):
        for idx, cl in enumerate(self):
            if cl is item:
                del self[idx]
                return


def test_should_apply_returns_false_for_none_action_set():
    random.seed(SEED)
    np.random.seed(SEED)

    assert ga.should_apply(None, time=10, theta_ga=5) is False


def test_should_apply_returns_false_for_empty_action_set():
    random.seed(SEED)
    np.random.seed(SEED)

    assert ga.should_apply(FakeList(), time=10, theta_ga=5) is False


def test_should_apply_true_when_average_delay_exceeds_theta():
    random.seed(SEED)
    np.random.seed(SEED)

    action_set = FakeList([FakeClassifier([0], tga=0, num=1)])

    assert ga.should_apply(action_set, time=10, theta_ga=5) is True


def test_should_apply_false_when_delay_equals_theta():
    random.seed(SEED)
    np.random.seed(SEED)

    # Comparison is strict: delay must be greater than theta_ga.
    action_set = FakeList([FakeClassifier([0], tga=0, num=1)])

    assert ga.should_apply(action_set, time=5, theta_ga=5) is False


def test_should_apply_weights_average_delay_by_numerosity():
    random.seed(SEED)
    np.random.seed(SEED)

    # Weighted average tga = (0*1 + 10*3) / 4 = 7.5
    action_set = FakeList([
        FakeClassifier([0], tga=0, num=1),
        FakeClassifier([1], tga=10, num=3),
    ])

    assert ga.should_apply(action_set, time=10, theta_ga=2) is True   # 2.5 > 2
    assert ga.should_apply(action_set, time=9, theta_ga=2) is False   # 1.5 > 2 is False


def test_set_timestamps_updates_tga_only():
    random.seed(SEED)
    np.random.seed(SEED)

    cl1 = FakeClassifier([0], tga=1, num=2)
    cl2 = FakeClassifier([1], tga=2, num=3)

    result = ga.set_timestamps([cl1, cl2], epoch=7)

    assert result is None
    assert cl1.tga == 7 and cl2.tga == 7
    assert cl1.num == 2 and cl2.num == 3


def test_roulette_wheel_selection_only_picks_positive_fitness():
    random.seed(SEED)
    np.random.seed(SEED)

    a = FakeClassifier([0])
    b = FakeClassifier([1])
    c = FakeClassifier([2])
    fitness = {id(a): 0.0, id(b): 1.0, id(c): 0.0}

    # Total fitness is 1.0 and random pick lies in [0, 1), so only b can be
    # selected for every possible random outcome.
    parent1, parent2 = ga.roulette_wheel_selection(
        [a, b, c], lambda cl: fitness[id(cl)])

    assert parent1 is b
    assert parent2 is b


def test_weighted_random_choice_never_returns_zero_weight_key():
    random.seed(SEED)
    np.random.seed(SEED)

    choices = {"a": 0.0, "b": 2.0, "c": 0.0}

    for _ in range(100):
        assert ga._weighted_random_choice(choices) == "b"


def test_generalizing_mutation_with_zero_mu_changes_nothing():
    random.seed(SEED)
    np.random.seed(SEED)

    cl = FakeClassifier(["0", "1", "#", "0"])

    ga.generalizing_mutation(cl, mu=0.0)

    assert list(cl.condition) == ["0", "1", "#", "0"]


def test_generalizing_mutation_with_full_mu_generalizes_all_specified_positions():
    random.seed(SEED)
    np.random.seed(SEED)

    cl = FakeClassifier(["0", "1", "#", "0"])

    # random.random() is in [0, 1), so it is always < 1.0.
    ga.generalizing_mutation(cl, mu=1.0)

    assert list(cl.condition) == ["#", "#", "#", "#"]


def test_two_point_crossover_swaps_values_only_within_segment():
    random.seed(SEED)
    np.random.seed(SEED)

    parent = FakeClassifier(["0", "0", "0", "0"])
    donor = FakeClassifier(["1", "1", "1", "1"])

    ga.two_point_crossover(parent, donor)

    # Every position must still hold exactly the two original values,
    # one in each classifier, regardless of where the cut points fell.
    for p_val, d_val in zip(parent.condition, donor.condition):
        assert {p_val, d_val} == {"0", "1"}


def test_two_point_crossover_identical_conditions_unchanged():
    random.seed(SEED)
    np.random.seed(SEED)

    parent = FakeClassifier(["0", "1", "0", "1"])
    donor = FakeClassifier(["0", "1", "0", "1"])

    ga.two_point_crossover(parent, donor)

    assert list(parent.condition) == ["0", "1", "0", "1"]
    assert list(donor.condition) == ["0", "1", "0", "1"]


def test_add_classifier_appends_to_population_and_action_set_when_no_old(
        monkeypatch):
    random.seed(SEED)
    np.random.seed(SEED)

    monkeypatch.setattr(ga, "_find_old_classifier",
                        lambda *args, **kwargs: None)
    population, match_set, action_set = FakeList(), FakeList(), FakeList()
    cl = FakeClassifier(["0", "#"])

    ga.add_classifier(cl, "01", population, match_set, action_set,
                      do_subsumption=False, theta_exp=5)

    assert cl in population
    assert cl in action_set


def test_add_classifier_adds_to_match_set_only_when_matching(monkeypatch):
    random.seed(SEED)
    np.random.seed(SEED)

    monkeypatch.setattr(ga, "_find_old_classifier",
                        lambda *args, **kwargs: None)
    population, action_set = FakeList(), FakeList()
    match_set = FakeList()
    matching = FakeClassifier(["0", "#"])
    non_matching = FakeClassifier(["1", "#"])

    ga.add_classifier(matching, "01", population, match_set, action_set,
                      do_subsumption=False, theta_exp=5)
    ga.add_classifier(non_matching, "01", population, match_set, action_set,
                      do_subsumption=False, theta_exp=5)

    assert matching in match_set
    assert non_matching not in match_set
    assert non_matching in population


def test_add_classifier_increments_numerosity_of_unmarked_old(monkeypatch):
    random.seed(SEED)
    np.random.seed(SEED)

    old = FakeClassifier(["0", "#"], num=2, marked=False)
    monkeypatch.setattr(ga, "_find_old_classifier",
                        lambda *args, **kwargs: old)
    population, match_set, action_set = FakeList(), FakeList(), FakeList()
    cl = FakeClassifier(["0", "#"])

    ga.add_classifier(cl, "01", population, match_set, action_set,
                      do_subsumption=False, theta_exp=5)

    assert old.num == 3
    assert cl not in population
    assert cl not in action_set


def test_add_classifier_keeps_numerosity_of_marked_old(monkeypatch):
    random.seed(SEED)
    np.random.seed(SEED)

    old = FakeClassifier(["0", "#"], num=2, marked=True)
    monkeypatch.setattr(ga, "_find_old_classifier",
                        lambda *args, **kwargs: old)
    population, match_set, action_set = FakeList(), FakeList(), FakeList()
    cl = FakeClassifier(["0", "#"])

    ga.add_classifier(cl, "01", population, match_set, action_set,
                      do_subsumption=False, theta_exp=5)

    assert old.num == 2
    assert cl not in population


def test_delete_classifiers_reduces_action_set_to_threshold():
    random.seed(SEED)
    np.random.seed(SEED)

    a = FakeClassifier([0], num=1)
    b = FakeClassifier([1], num=1)
    c = FakeClassifier([2], num=1)
    population = FakeList([a, b, c])
    match_set = FakeList([a, b, c])
    action_set = FakeList([a, b, c])

    # Sum of nums is 3, insize 1, theta_as 2: two classifiers must go.
    ga.delete_classifiers(population, match_set, action_set,
                          insize=1, theta_as=2)

    assert len(action_set) == 1
    assert len(population) == 1
    assert len(match_set) == 1
    assert all(cl in population for cl in action_set)


def test_delete_classifiers_decrements_numerosity_before_removing():
    random.seed(SEED)
    np.random.seed(SEED)

    cl = FakeClassifier([0], num=3)
    population = FakeList()
    action_set = FakeList([cl])

    # Only one classifier exists, so it is always the chosen victim.
    # Its num drops from 3 to 2, which satisfies 2 + 0 <= 2.
    ga.delete_classifiers(population, None, action_set,
                          insize=0, theta_as=2)

    assert cl.num == 2
    assert cl in action_set


def test_is_preferred_to_delete_true_for_clearly_worse_classifier():
    random.seed(SEED)
    np.random.seed(SEED)

    cl_del = FakeClassifier([0], q=0.5)
    cl = FakeClassifier([1], q=0.3)

    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_false_for_clearly_better_classifier():
    random.seed(SEED)
    np.random.seed(SEED)

    cl_del = FakeClassifier([0], q=0.5)
    cl = FakeClassifier([1], q=0.9)

    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_marked_classifier_preferred_when_similar_quality():
    random.seed(SEED)
    np.random.seed(SEED)

    cl_del = FakeClassifier([0], q=0.5, marked=False)
    cl = FakeClassifier([1], q=0.5, marked=True)

    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_find_similar_returns_first_equal_item_or_none():
    random.seed(SEED)
    np.random.seed(SEED)

    assert ga._find_similar(3, [1, 2, 3, 3]) == 3
    assert ga._find_similar(9, [1, 2]) is None


def test_find_old_classifier_prefers_most_general_subsumer(monkeypatch):
    random.seed(SEED)
    np.random.seed(SEED)

    general, specific = "general", "specific"
    monkeypatch.setattr(ga, "find_subsumers",
                        lambda cl, population, theta_exp: [general, specific])

    result = ga._find_old_classifier([1, 2, 3], 2,
                                     use_subsumption=True, theta_exp=5)

    assert result == general


def test_find_old_classifier_falls_back_to_similar_when_no_subsumer(monkeypatch):
    random.seed(SEED)
    np.random.seed(SEED)

    monkeypatch.setattr(ga, "find_subsumers",
                        lambda cl, population, theta_exp: [])

    result = ga._find_old_classifier([1, 2, 3], 2,
                                     use_subsumption=True, theta_exp=5)

    assert result == 2
