import random

import numpy as np

from lcs.strategies import genetic_algorithms as ga


class FakeCfg:
    def __init__(self, length=5, wildcard='#'):
        self.classifier_length = length
        self.classifier_wildcard = wildcard


class FakeCondition(list):
    def __init__(self, items, matches=True):
        super().__init__(items)
        self.matches = matches
        self.generalized = []

    def generalize(self, idx):
        self.generalized.append(idx)
        self[idx] = '#'

    def does_match(self, p):
        return self.matches


class FakeCl:
    def __init__(self, key='k', num=1, tga=0, q=0.5, tav=0.0, marked=False,
                 condition=None, cfg=None):
        self.key = key
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self.marked = marked
        self.condition = condition if condition is not None else FakeCondition([])
        self.cfg = cfg if cfg is not None else FakeCfg()

    def is_marked(self):
        return self.marked

    def __eq__(self, other):
        return isinstance(other, FakeCl) and self.key == other.key

    def __hash__(self):
        return id(self)


class FakeSet(list):
    def expand(self):
        result = []
        for cl in self:
            result.extend([cl] * cl.num)
        return result

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
    action_set = [FakeCl(num=0, tga=0)]
    assert ga.should_apply(action_set, 100, 10) is False


def test_should_apply_true_when_delay_exceeds_threshold():
    action_set = [FakeCl(num=1, tga=10), FakeCl(num=1, tga=20)]
    # average tga = 15, delay = 25 > 20
    assert ga.should_apply(action_set, 40, 20) is True


def test_should_apply_false_when_delay_equal_to_threshold():
    action_set = [FakeCl(num=2, tga=10)]
    assert ga.should_apply(action_set, 30, 20) is False


def test_should_apply_uses_numerosity_weighted_average():
    action_set = [FakeCl(num=3, tga=0), FakeCl(num=1, tga=100)]
    # weighted average = 25
    assert ga.should_apply(action_set, 50, 24) is True
    assert ga.should_apply(action_set, 50, 25) is False


def test_set_timestamps_updates_all():
    action_set = [FakeCl(key='a', tga=1), FakeCl(key='b', tga=2)]
    ga.set_timestamps(action_set, 77)
    assert [cl.tga for cl in action_set] == [77, 77]


def test_weighted_random_choice_deterministic_picks(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    choices = {'a': 1, 'b': 2, 'c': 3}

    monkeypatch.setattr(ga.random, 'uniform', lambda lo, hi: 0.5)
    assert ga._weighted_random_choice(choices) == 'a'

    monkeypatch.setattr(ga.random, 'uniform', lambda lo, hi: 1.5)
    assert ga._weighted_random_choice(choices) == 'b'

    monkeypatch.setattr(ga.random, 'uniform', lambda lo, hi: 5.0)
    assert ga._weighted_random_choice(choices) == 'c'


def test_roulette_wheel_selection_single_nonzero_weight():
    random.seed(1)
    np.random.seed(1)
    cls = [FakeCl(key=str(i), q=0.0) for i in range(4)]
    cls[2].q = 1.0
    for _ in range(20):
        p1, p2 = ga.roulette_wheel_selection(cls, lambda c: c.q)
        assert p1 is cls[2]
        assert p2 is cls[2]


def test_roulette_wheel_selection_returns_population_members():
    random.seed(2)
    np.random.seed(2)
    cls = [FakeCl(key=str(i), q=0.5 + i) for i in range(3)]
    for _ in range(20):
        p1, p2 = ga.roulette_wheel_selection(cls, lambda c: c.q)
        assert any(p1 is c for c in cls)
        assert any(p2 is c for c in cls)


def test_generalizing_mutation_probability_one_generalizes_all_specific():
    random.seed(3)
    np.random.seed(3)
    cl = FakeCl(condition=FakeCondition(['0', '#', '1', '1']))
    ga.generalizing_mutation(cl, 1.0)
    assert list(cl.condition) == ['#', '#', '#', '#']
    # wildcard position is never touched
    assert cl.condition.generalized == [0, 2, 3]


def test_generalizing_mutation_probability_zero_changes_nothing():
    random.seed(4)
    np.random.seed(4)
    cl = FakeCl(condition=FakeCondition(['0', '#', '1', '1']))
    ga.generalizing_mutation(cl, 0.0)
    assert list(cl.condition) == ['0', '#', '1', '1']
    assert cl.condition.generalized == []


def test_two_point_crossover_swaps_contiguous_nonempty_segment():
    random.seed(5)
    np.random.seed(5)
    length = 8
    cfg = FakeCfg(length=length)
    for _ in range(30):
        parent = FakeCl(condition=FakeCondition(['a'] * length), cfg=cfg)
        donor = FakeCl(condition=FakeCondition(['b'] * length), cfg=cfg)
        ga.two_point_crossover(parent, donor)

        assert len(parent.condition) == length
        assert len(donor.condition) == length
        swapped = [i for i, c in enumerate(parent.condition) if c == 'b']
        assert len(swapped) >= 1
        # contiguous segment
        assert swapped == list(range(swapped[0], swapped[-1] + 1))
        # donor is the exact complement
        for i in range(length):
            if i in swapped:
                assert donor.condition[i] == 'a'
            else:
                assert parent.condition[i] == 'a'
                assert donor.condition[i] == 'b'


def test_two_point_crossover_identical_parents_unchanged():
    random.seed(6)
    np.random.seed(6)
    cfg = FakeCfg(length=4)
    parent = FakeCl(condition=FakeCondition(['1', '0', '1', '0']), cfg=cfg)
    donor = FakeCl(condition=FakeCondition(['1', '0', '1', '0']), cfg=cfg)
    ga.two_point_crossover(parent, donor)
    assert list(parent.condition) == ['1', '0', '1', '0']
    assert list(donor.condition) == ['1', '0', '1', '0']


def test_add_classifier_new_is_added_to_all_sets():
    random.seed(7)
    np.random.seed(7)
    population, match_set, action_set = [], [], []
    new = FakeCl(key='new', condition=FakeCondition(['1'], matches=True))
    ga.add_classifier(new, None, population, match_set, action_set,
                      False, 20)
    assert population == [new]
    assert action_set == [new]
    assert match_set == [new]


def test_add_classifier_not_matching_skips_match_set():
    random.seed(8)
    np.random.seed(8)
    population, match_set, action_set = [], [], []
    new = FakeCl(key='new', condition=FakeCondition(['1'], matches=False))
    ga.add_classifier(new, None, population, match_set, action_set,
                      False, 20)
    assert population == [new]
    assert action_set == [new]
    assert match_set == []


def test_add_classifier_without_match_set():
    random.seed(9)
    np.random.seed(9)
    population, action_set = [], []
    new = FakeCl(key='new')
    ga.add_classifier(new, None, population, None, action_set, False, 20)
    assert population == [new]
    assert action_set == [new]


def test_add_classifier_similar_increments_numerosity():
    random.seed(10)
    np.random.seed(10)
    existing = FakeCl(key='same', num=2)
    population, match_set, action_set = [existing], [existing], [existing]
    new = FakeCl(key='same')
    ga.add_classifier(new, None, population, match_set, action_set,
                      False, 20)
    assert existing.num == 3
    assert population == [existing]
    assert action_set == [existing]
    assert match_set == [existing]


def test_add_classifier_similar_marked_does_not_increment():
    random.seed(11)
    np.random.seed(11)
    existing = FakeCl(key='same', num=2, marked=True)
    population, match_set, action_set = [existing], [existing], [existing]
    ga.add_classifier(FakeCl(key='same'), None, population, match_set,
                      action_set, False, 20)
    assert existing.num == 2
    assert len(population) == 1


def test_find_similar_returns_first_equal_or_none():
    first = FakeCl(key='x')
    second = FakeCl(key='x')
    other = FakeCl(key='y')
    population = [other, first, second]
    assert ga._find_similar(FakeCl(key='x'), population) is first
    assert ga._find_similar(FakeCl(key='z'), population) is None
    assert ga._find_similar(FakeCl(key='x'), []) is None


def test_is_preferred_to_delete_rules():
    cl_del = FakeCl(q=0.5, tav=1.0)
    # much worse quality -> preferred to delete
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.1)) is True
    # much better quality -> not preferred
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.9)) is False
    # similar quality, larger tav -> preferred
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.55, tav=2.0)) is True
    # similar quality, smaller tav -> not preferred
    assert ga._is_preferred_to_delete(cl_del, FakeCl(q=0.55, tav=0.5)) is False
    # similar quality, marked vs unmarked -> preferred
    assert ga._is_preferred_to_delete(
        cl_del, FakeCl(q=0.5, tav=0.0, marked=True)) is True


def test_delete_classifiers_no_deletion_when_room_available():
    random.seed(12)
    np.random.seed(12)
    cl = FakeCl(key='a', num=2)
    population, match_set, action_set = [cl], [cl], FakeSet([cl])
    ga.delete_classifiers(population, match_set, action_set, 1, 5)
    assert cl.num == 2
    assert population == [cl]
    assert list(action_set) == [cl]


def test_delete_classifiers_decrements_numerosity():
    random.seed(13)
    np.random.seed(13)
    cl = FakeCl(key='a', num=3)
    population, match_set, action_set = [cl], [cl], FakeSet([cl])
    ga.delete_classifiers(population, match_set, action_set, 1, 3)
    assert cl.num == 2
    assert population == [cl]
    assert list(action_set) == [cl]


def test_delete_classifiers_removes_worst_classifier_everywhere(monkeypatch):
    random.seed(14)
    np.random.seed(14)
    # make every classifier a deletion candidate
    monkeypatch.setattr(ga.random, 'random', lambda: 0.0)
    good = FakeCl(key='good', num=1, q=0.9)
    bad = FakeCl(key='bad', num=1, q=0.2)
    population = FakeSet([good, bad])
    match_set = FakeSet([good, bad])
    action_set = FakeSet([good, bad])
    ga.delete_classifiers(population, match_set, action_set, 1, 2)
    assert list(population) == [good]
    assert list(match_set) == [good]
    assert list(action_set) == [good]
