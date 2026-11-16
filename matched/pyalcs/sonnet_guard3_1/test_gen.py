import random
from types import SimpleNamespace

import numpy as np

import lcs.strategies.genetic_algorithms as ga

WILD = '#'


class FakeCondition(list):
    def __init__(self, items, matches=True):
        super().__init__(items)
        self.matches = matches
        self.generalized = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = WILD

    def does_match(self, p):
        return self.matches


class FakeCl:
    def __init__(self, key=0, num=1, tga=0, q=0.5, tav=0.0, marked=False,
                 condition=None, length=6):
        self.key = key
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self.marked = marked
        self.condition = condition if condition is not None else \
            FakeCondition(['a'] * length)
        self.cfg = SimpleNamespace(classifier_wildcard=WILD,
                                   classifier_length=length)

    def is_marked(self):
        return self.marked

    def __eq__(self, other):
        return isinstance(other, FakeCl) and self.key == other.key

    def __hash__(self):
        return hash(self.key)


class FakeSet(list):
    def expand(self):
        out = []
        for cl in self:
            out.extend([cl] * cl.num)
        return out

    def safe_remove(self, cl):
        for i, el in enumerate(self):
            if el is cl:
                del self[i]
                return


def test_should_apply_none_action_set():
    assert ga.should_apply(None, 100, 10) is False


def test_should_apply_empty_action_set():
    assert ga.should_apply([], 100, 10) is False


def test_should_apply_zero_numerosity():
    action_set = [FakeCl(key=1, num=0, tga=0)]
    assert ga.should_apply(action_set, 1000, 1) is False


def test_should_apply_weighted_average_threshold():
    action_set = [FakeCl(key=1, num=1, tga=0), FakeCl(key=2, num=3, tga=10)]
    # weighted average tga = 7.5; time 20 -> delay 12.5
    assert ga.should_apply(action_set, 20, 12) is True
    assert ga.should_apply(action_set, 20, 13) is False


def test_should_apply_delay_equal_to_threshold_is_false():
    action_set = [FakeCl(key=1, num=2, tga=5)]
    assert ga.should_apply(action_set, 15, 10) is False
    assert ga.should_apply(action_set, 16, 10) is True


def test_set_timestamps_updates_all():
    action_set = [FakeCl(key=i, tga=0) for i in range(4)]
    ga.set_timestamps(action_set, 42)
    assert all(cl.tga == 42 for cl in action_set)


def test_weighted_random_choice_with_patched_uniform(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    choices = {"a": 1, "b": 2, "c": 3}

    monkeypatch.setattr(ga.random, "uniform", lambda lo, hi: 0.5)
    assert ga._weighted_random_choice(choices) == "a"
    monkeypatch.setattr(ga.random, "uniform", lambda lo, hi: 1.5)
    assert ga._weighted_random_choice(choices) == "b"
    monkeypatch.setattr(ga.random, "uniform", lambda lo, hi: 5.5)
    assert ga._weighted_random_choice(choices) == "c"


def test_weighted_random_choice_passes_total_as_upper_bound(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    seen = {}

    def fake_uniform(lo, hi):
        seen['args'] = (lo, hi)
        return 0.0

    monkeypatch.setattr(ga.random, "uniform", fake_uniform)
    ga._weighted_random_choice({"x": 2.5, "y": 1.5})
    assert seen['args'][0] == 0
    assert abs(seen['args'][1] - 4.0) < 1e-9


def test_roulette_wheel_selection_single_weighted_candidate():
    random.seed(1)
    np.random.seed(1)
    cls = [FakeCl(key=i, q=0.0) for i in range(4)]
    cls[2].q = 1.0
    for _ in range(20):
        p1, p2 = ga.roulette_wheel_selection(cls, lambda c: c.q ** 3)
        assert p1 is cls[2]
        assert p2 is cls[2]


def test_roulette_wheel_selection_returns_population_members():
    random.seed(2)
    np.random.seed(2)
    cls = [FakeCl(key=i, q=0.2 + 0.1 * i) for i in range(5)]
    for _ in range(30):
        p1, p2 = ga.roulette_wheel_selection(cls, lambda c: c.q)
        assert any(p1 is c for c in cls)
        assert any(p2 is c for c in cls)


def test_generalizing_mutation_probability_zero_changes_nothing():
    random.seed(3)
    np.random.seed(3)
    cl = FakeCl(condition=FakeCondition(['a', 'b', 'c', 'd']), length=4)
    ga.generalizing_mutation(cl, 0.0)
    assert list(cl.condition) == ['a', 'b', 'c', 'd']
    assert cl.condition.generalized == []


def test_generalizing_mutation_probability_one_generalizes_all_specific():
    random.seed(4)
    np.random.seed(4)
    cl = FakeCl(condition=FakeCondition(['a', WILD, 'c', 'd']), length=4)
    ga.generalizing_mutation(cl, 1.0)
    assert list(cl.condition) == [WILD] * 4
    # wildcard position (index 1) must not be generalized again
    assert cl.condition.generalized == [0, 2, 3]


def test_two_point_crossover_swaps_contiguous_segment():
    random.seed(5)
    np.random.seed(5)
    length = 8
    for _ in range(25):
        parent = FakeCl(key=1, condition=FakeCondition(
            ['p%d' % i for i in range(length)]), length=length)
        donor = FakeCl(key=2, condition=FakeCondition(
            ['d%d' % i for i in range(length)]), length=length)
        ga.two_point_crossover(parent, donor)

        swapped = []
        for i in range(length):
            if parent.condition[i] == 'd%d' % i:
                assert donor.condition[i] == 'p%d' % i
                swapped.append(i)
            else:
                assert parent.condition[i] == 'p%d' % i
                assert donor.condition[i] == 'd%d' % i

        assert len(swapped) >= 1
        assert swapped == list(range(swapped[0], swapped[-1] + 1))
        assert len(parent.condition) == length
        assert len(donor.condition) == length


def test_add_classifier_new_classifier_added_everywhere():
    random.seed(6)
    np.random.seed(6)
    new = FakeCl(key=10, condition=FakeCondition(['a'], matches=True),
                 length=1)
    population, match_set, action_set = FakeSet(), FakeSet(), FakeSet()
    ga.add_classifier(new, "perception", population, match_set, action_set,
                      False, 20)
    assert len(population) == 1 and population[0] is new
    assert len(action_set) == 1 and action_set[0] is new
    assert len(match_set) == 1 and match_set[0] is new


def test_add_classifier_not_matching_skips_match_set():
    random.seed(7)
    np.random.seed(7)
    new = FakeCl(key=10, condition=FakeCondition(['a'], matches=False),
                 length=1)
    population, match_set, action_set = FakeSet(), FakeSet(), FakeSet()
    ga.add_classifier(new, "perception", population, match_set, action_set,
                      False, 20)
    assert len(population) == 1
    assert len(action_set) == 1
    assert len(match_set) == 0


def test_add_classifier_match_set_none():
    random.seed(8)
    np.random.seed(8)
    new = FakeCl(key=10)
    population, action_set = FakeSet(), FakeSet()
    ga.add_classifier(new, "perception", population, None, action_set,
                      False, 20)
    assert len(population) == 1
    assert len(action_set) == 1


def test_add_classifier_similar_increases_numerosity():
    random.seed(9)
    np.random.seed(9)
    old = FakeCl(key=5, num=2)
    new = FakeCl(key=5, num=1)
    population, match_set, action_set = FakeSet([old]), FakeSet([old]), \
        FakeSet([old])
    ga.add_classifier(new, "perception", population, match_set, action_set,
                      False, 20)
    assert old.num == 3
    assert len(population) == 1
    assert len(action_set) == 1


def test_add_classifier_similar_marked_not_incremented():
    random.seed(10)
    np.random.seed(10)
    old = FakeCl(key=5, num=2, marked=True)
    new = FakeCl(key=5, num=1)
    population, match_set, action_set = FakeSet([old]), FakeSet([old]), \
        FakeSet([old])
    ga.add_classifier(new, "perception", population, match_set, action_set,
                      False, 20)
    assert old.num == 2
    assert len(population) == 1


def test_add_classifier_uses_subsumer_when_enabled(monkeypatch):
    random.seed(11)
    np.random.seed(11)
    subsumer = FakeCl(key=100, num=1)
    new = FakeCl(key=200, num=1)
    calls = []

    def fake_find_subsumers(cl, population, theta_exp):
        calls.append((cl, theta_exp))
        return [subsumer]

    monkeypatch.setattr(ga, "find_subsumers", fake_find_subsumers)
    population, match_set, action_set = FakeSet([subsumer]), FakeSet(), \
        FakeSet([subsumer])
    ga.add_classifier(new, "perception", population, match_set, action_set,
                      True, 20)
    assert subsumer.num == 2
    assert len(population) == 1
    assert calls == [(new, 20)]


def test_delete_classifiers_no_deletion_when_room():
    random.seed(12)
    np.random.seed(12)
    a, b = FakeCl(key=1, num=2), FakeCl(key=2, num=1)
    population, match_set, action_set = FakeSet([a, b]), FakeSet([a, b]), \
        FakeSet([a, b])
    ga.delete_classifiers(population, match_set, action_set, 2, 5)
    assert a.num == 2 and b.num == 1
    assert len(population) == 2
    assert len(action_set) == 2


def test_delete_classifiers_decrements_numerosity():
    random.seed(13)
    np.random.seed(13)
    a = FakeCl(key=1, num=3)
    population, match_set, action_set = FakeSet([a]), FakeSet([a]), \
        FakeSet([a])
    ga.delete_classifiers(population, match_set, action_set, 1, 3)
    assert a.num == 2
    assert len(population) == 1
    assert len(action_set) == 1


def test_delete_classifiers_removes_single_numerosity_from_all_lists():
    random.seed(14)
    np.random.seed(14)
    a = FakeCl(key=1, num=1, q=0.2)
    b = FakeCl(key=2, num=1, q=0.9)
    population, match_set, action_set = FakeSet([a, b]), FakeSet([a, b]), \
        FakeSet([a, b])
    ga.delete_classifiers(population, match_set, action_set, 1, 2)

    assert len(action_set) == 1
    assert len(population) == 1
    assert len(match_set) == 1
    survivor = action_set[0]
    assert survivor is population[0]
    assert survivor is match_set[0]
    assert survivor is a or survivor is b


def test_is_preferred_to_delete_much_lower_quality():
    cl_del = FakeCl(q=0.8)
    cl = FakeCl(q=0.5)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_much_higher_quality():
    cl_del = FakeCl(q=0.5)
    cl = FakeCl(q=0.9)
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_similar_quality_marked():
    cl_del = FakeCl(q=0.5, marked=False, tav=10.0)
    cl = FakeCl(q=0.52, marked=True, tav=1.0)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_similar_quality_uses_tav():
    cl_del = FakeCl(q=0.5, tav=5.0)
    higher = FakeCl(q=0.52, tav=9.0)
    lower = FakeCl(q=0.52, tav=1.0)
    assert ga._is_preferred_to_delete(cl_del, higher) is True
    assert ga._is_preferred_to_delete(cl_del, lower) is False


def test_find_similar_returns_first_match_or_none():
    first = FakeCl(key=1)
    second = FakeCl(key=1)
    other = FakeCl(key=2)
    population = [other, first, second]
    assert ga._find_similar(FakeCl(key=1), population) is first
    assert ga._find_similar(FakeCl(key=99), population) is None


def test_find_old_classifier_without_subsumption_uses_similar():
    old = FakeCl(key=3)
    result = ga._find_old_classifier([old], FakeCl(key=3), False, 20)
    assert result is old
    assert ga._find_old_classifier([old], FakeCl(key=4), False, 20) is None


def test_find_old_classifier_subsumption_falls_back_to_similar(monkeypatch):
    old = FakeCl(key=3)
    monkeypatch.setattr(ga, "find_subsumers", lambda cl, pop, theta: [])
    result = ga._find_old_classifier([old], FakeCl(key=3), True, 20)
    assert result is old
