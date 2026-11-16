import numpy as np
import pytest

from lcs.strategies import genetic_algorithms as ga


class FakeCfg:
    classifier_wildcard = '#'
    classifier_length = 4


class FakeCondition(list):
    def __init__(self, values=(), matches=True):
        super().__init__(values)
        self.matches = matches
        self.generalized = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = '#'

    def does_match(self, p):
        return self.matches


class FakeCl:
    def __init__(self, condition=None, action=0, q=0.5, tga=0, num=1,
                 tav=0, marked=False, cfg=None):
        self.condition = condition if condition is not None else FakeCondition(['0'])
        self.action = action
        self.q = q
        self.tga = tga
        self.num = num
        self.tav = tav
        self._marked = marked
        self.cfg = cfg if cfg is not None else FakeCfg()

    def is_marked(self):
        return self._marked

    def _key(self):
        return (tuple(self.condition), self.action)

    def __eq__(self, other):
        return isinstance(other, FakeCl) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())


class FakeSet(list):
    def expand(self):
        return list(self)

    def safe_remove(self, cl):
        for i, item in enumerate(self):
            if item is cl:
                del self[i]
                return


# ---------------- should_apply ----------------

def test_should_apply_returns_false_for_none():
    assert ga.should_apply(None, time=100, theta_ga=1) is False


def test_should_apply_returns_false_for_empty_action_set():
    assert ga.should_apply([], time=100, theta_ga=1) is False


def test_should_apply_true_when_average_delay_exceeds_threshold():
    action_set = [FakeCl(tga=0, num=1)]
    assert ga.should_apply(action_set, time=10, theta_ga=5) is True


def test_should_apply_false_when_average_delay_below_threshold():
    action_set = [FakeCl(tga=0, num=1)]
    assert ga.should_apply(action_set, time=3, theta_ga=5) is False


def test_should_apply_uses_numerosity_weighted_average():
    # weighted avg = (0*1 + 10*3) / 4 = 7.5
    action_set = [FakeCl(tga=0, num=1), FakeCl(tga=10, num=3)]
    assert ga.should_apply(action_set, time=12, theta_ga=5) is False
    assert ga.should_apply(action_set, time=13, theta_ga=5) is True


def test_should_apply_threshold_comparison_is_strict():
    action_set = [FakeCl(tga=0, num=1)]
    assert ga.should_apply(action_set, time=5, theta_ga=5) is False


# ---------------- set_timestamps ----------------

def test_set_timestamps_updates_all_classifiers():
    action_set = [FakeCl(tga=1), FakeCl(tga=2, condition=FakeCondition(['1']))]
    ga.set_timestamps(action_set, epoch=42)
    assert all(cl.tga == 42 for cl in action_set)


# ---------------- roulette_wheel_selection ----------------

def _roulette_population():
    a = FakeCl(condition=FakeCondition(['0']), q=1.0)
    b = FakeCl(condition=FakeCondition(['1']), q=2.0)
    c = FakeCl(condition=FakeCondition(['2']), q=3.0)
    return a, b, c


def test_roulette_wheel_selection_picks_by_cumulative_fitness(monkeypatch):
    a, b, c = _roulette_population()
    picks = iter([0.5, 2.5])
    monkeypatch.setattr(ga.random, "uniform", lambda lo, hi: next(picks))

    parent1, parent2 = ga.roulette_wheel_selection([a, b, c], lambda cl: cl.q)

    assert parent1 is a
    assert parent2 is b


def test_roulette_wheel_selection_picks_last_for_high_pick(monkeypatch):
    a, b, c = _roulette_population()
    monkeypatch.setattr(ga.random, "uniform", lambda lo, hi: 5.5)

    parent1, parent2 = ga.roulette_wheel_selection([a, b, c], lambda cl: cl.q)

    assert parent1 is c
    assert parent2 is c


# ---------------- generalizing_mutation ----------------

def test_generalizing_mutation_generalizes_non_wildcard_positions(monkeypatch):
    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    cond = FakeCondition(['0', '1', '#'])
    cl = FakeCl(condition=cond, cfg=FakeCfg())

    ga.generalizing_mutation(cl, mu=0.5)

    assert cond.generalized == [0, 1]
    assert list(cond) == ['#', '#', '#']


def test_generalizing_mutation_does_nothing_when_mu_is_zero(monkeypatch):
    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    cond = FakeCondition(['0', '1', '0'])
    cl = FakeCl(condition=cond, cfg=FakeCfg())

    ga.generalizing_mutation(cl, mu=0.0)

    assert cond.generalized == []
    assert list(cond) == ['0', '1', '0']


# ---------------- two_point_crossover ----------------

def test_two_point_crossover_swaps_segment_between_cut_points(monkeypatch):
    monkeypatch.setattr(ga.np.random, "choice",
                        lambda a, size, replace: np.array([1, 3]))
    parent = FakeCl(condition=['p0', 'p1', 'p2', 'p3'], cfg=FakeCfg())
    donor = FakeCl(condition=['d0', 'd1', 'd2', 'd3'], cfg=FakeCfg())

    ga.two_point_crossover(parent, donor)

    assert list(parent.condition) == ['p0', 'd1', 'd2', 'p3']
    assert list(donor.condition) == ['d0', 'p1', 'p2', 'd3']


@pytest.mark.parametrize("cuts", [[1, 3], [3, 1]])
def test_two_point_crossover_is_independent_of_cut_order(monkeypatch, cuts):
    monkeypatch.setattr(ga.np.random, "choice",
                        lambda a, size, replace: np.array(cuts))
    parent = FakeCl(condition=['p0', 'p1', 'p2', 'p3'], cfg=FakeCfg())
    donor = FakeCl(condition=['d0', 'd1', 'd2', 'd3'], cfg=FakeCfg())

    ga.two_point_crossover(parent, donor)

    assert list(parent.condition) == ['p0', 'd1', 'd2', 'p3']
    assert list(donor.condition) == ['d0', 'p1', 'p2', 'd3']


def test_two_point_crossover_samples_cut_points_from_full_range(monkeypatch):
    calls = []

    def fake_choice(a, size, replace):
        calls.append((list(a), size, replace))
        return np.array([0, 2])

    monkeypatch.setattr(ga.np.random, "choice", fake_choice)
    parent = FakeCl(condition=['p0', 'p1', 'p2', 'p3'], cfg=FakeCfg())
    donor = FakeCl(condition=['d0', 'd1', 'd2', 'd3'], cfg=FakeCfg())

    ga.two_point_crossover(parent, donor)

    assert calls == [([0, 1, 2, 3, 4], 2, False)]


# ---------------- add_classifier ----------------

def test_add_classifier_appends_new_classifier_to_all_lists():
    new = FakeCl(condition=FakeCondition(['0', '1'], matches=True), action=1)
    population, match_set, action_set = [], [], []

    ga.add_classifier(new, None, population, match_set, action_set,
                      do_subsumption=False, theta_exp=0)

    assert population == [new]
    assert action_set == [new]
    assert match_set == [new]


def test_add_classifier_does_not_add_to_match_set_when_not_matching():
    new = FakeCl(condition=FakeCondition(['0', '1'], matches=False), action=1)
    population, match_set, action_set = [], [], []

    ga.add_classifier(new, None, population, match_set, action_set,
                      do_subsumption=False, theta_exp=0)

    assert population == [new]
    assert action_set == [new]
    assert match_set == []


def test_add_classifier_increments_numerosity_of_similar_classifier():
    existing = FakeCl(condition=FakeCondition(['0', '1']), action=1, num=1)
    new = FakeCl(condition=FakeCondition(['0', '1']), action=1, num=1)
    population = [existing]
    action_set = [existing]
    match_set = []

    ga.add_classifier(new, None, population, match_set, action_set,
                      do_subsumption=False, theta_exp=0)

    assert existing.num == 2
    assert len(population) == 1
    assert population[0] is existing
    assert all(x is not new for x in action_set)


def test_add_classifier_does_not_increment_marked_similar_classifier():
    existing = FakeCl(condition=FakeCondition(['0', '1']), action=1,
                      num=1, marked=True)
    new = FakeCl(condition=FakeCondition(['0', '1']), action=1, num=1)
    population = [existing]
    action_set = [existing]
    match_set = []

    ga.add_classifier(new, None, population, match_set, action_set,
                      do_subsumption=False, theta_exp=0)

    assert existing.num == 1
    assert population == [existing]


def test_add_classifier_without_match_set_does_not_fail():
    new = FakeCl(condition=FakeCondition(['0'], matches=True), action=0)
    population, action_set = [], []

    ga.add_classifier(new, None, population, None, action_set,
                      do_subsumption=False, theta_exp=0)

    assert population == [new]
    assert action_set == [new]


# ---------------- delete_classifiers ----------------

def test_delete_classifiers_decrements_numerosity_of_chosen_classifier(monkeypatch):
    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    cl = FakeCl(condition=FakeCondition(['0']), num=3)
    population = FakeSet([cl])
    action_set = FakeSet([cl])

    ga.delete_classifiers(population, None, action_set,
                          insize=0, theta_as=2)

    assert cl.num == 2
    assert population == [cl]
    assert action_set == [cl]


def test_delete_classifiers_removes_worst_classifier_from_all_lists(monkeypatch):
    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    good = FakeCl(condition=FakeCondition(['0']), q=0.5, num=1)
    bad = FakeCl(condition=FakeCondition(['1']), q=0.1, num=1)
    population = FakeSet([good, bad])
    match_set = FakeSet([good, bad])
    action_set = FakeSet([good, bad])

    ga.delete_classifiers(population, match_set, action_set,
                          insize=0, theta_as=1)

    assert len(population) == 1 and population[0] is good
    assert len(match_set) == 1 and match_set[0] is good
    assert len(action_set) == 1 and action_set[0] is good


# ---------------- _is_preferred_to_delete ----------------

def test_is_preferred_to_delete_true_when_candidate_has_much_lower_quality():
    cl_del = FakeCl(q=0.5)
    cl = FakeCl(condition=FakeCondition(['1']), q=0.2)

    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_true_for_similar_quality_marked_candidate():
    cl_del = FakeCl(condition=FakeCondition(['0']), q=0.5, marked=False)
    cl = FakeCl(condition=FakeCondition(['1']), q=0.55, marked=True)

    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_compares_time_average_for_similar_quality():
    cl_del = FakeCl(condition=FakeCondition(['0']), q=0.5, tav=1)
    newer = FakeCl(condition=FakeCondition(['1']), q=0.5, tav=5)
    older = FakeCl(condition=FakeCondition(['2']), q=0.5, tav=0)

    assert ga._is_preferred_to_delete(cl_del, newer) is True
    assert ga._is_preferred_to_delete(cl_del, older) is False
