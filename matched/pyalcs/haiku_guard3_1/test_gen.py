import random

import numpy as np

from lcs.strategies import genetic_algorithms as ga


class FakeCfg:
    classifier_wildcard = '#'
    classifier_length = 4


CFG = FakeCfg()


class FakeCondition:
    def __init__(self, values, matches=True):
        self._values = list(values)
        self._matches = matches

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, key):
        return self._values[key]

    def __setitem__(self, idx, value):
        self._values[idx] = value

    def generalize(self, idx):
        self._values[idx] = CFG.classifier_wildcard

    def does_match(self, p):
        return self._matches


class FakeClassifier:
    def __init__(self, condition="0000", action=0, q=0.5, num=1, tga=0,
                 tav=0, marked=False, matches=True):
        self.cfg = CFG
        self.condition = FakeCondition(condition, matches)
        self.action = action
        self.q = q
        self.num = num
        self.tga = tga
        self.tav = tav
        self._marked = marked

    def is_marked(self):
        return self._marked

    def _key(self):
        return (tuple(self.condition), self.action)

    def __eq__(self, other):
        if not isinstance(other, FakeClassifier):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())


class FakeList(list):
    def expand(self):
        return list(self)

    def safe_remove(self, cl):
        for i, x in enumerate(self):
            if x is cl:
                del self[i]
                return


SEED = 12345


# ---------------- should_apply ----------------

def test_should_apply_returns_false_for_none():
    random.seed(SEED)
    np.random.seed(SEED)
    assert ga.should_apply(None, time=100, theta_ga=1) is False


def test_should_apply_true_when_average_delay_exceeds_threshold():
    random.seed(SEED)
    np.random.seed(SEED)
    action_set = [FakeClassifier(tga=0, num=1)]
    assert ga.should_apply(action_set, time=10, theta_ga=3) is True


def test_should_apply_false_when_delay_equals_threshold():
    random.seed(SEED)
    np.random.seed(SEED)
    action_set = [FakeClassifier(tga=0, num=1)]
    assert ga.should_apply(action_set, time=3, theta_ga=3) is False


def test_should_apply_uses_numerosity_weighted_average():
    random.seed(SEED)
    np.random.seed(SEED)
    # weighted average of tga: (0*1 + 8*3) / 4 = 6
    action_set = [FakeClassifier(tga=0, num=1, condition="0000"),
                  FakeClassifier(tga=8, num=3, condition="1111")]
    assert ga.should_apply(action_set, time=10, theta_ga=3) is True
    assert ga.should_apply(action_set, time=9, theta_ga=3) is False


# ---------------- set_timestamps ----------------

def test_set_timestamps_sets_epoch_on_all_classifiers():
    random.seed(SEED)
    np.random.seed(SEED)
    action_set = [FakeClassifier(tga=1, condition="0000"),
                  FakeClassifier(tga=2, condition="1111"),
                  FakeClassifier(tga=3, condition="0101")]
    ga.set_timestamps(action_set, epoch=7)
    assert all(cl.tga == 7 for cl in action_set)


# ---------------- roulette_wheel_selection ----------------

def test_roulette_returns_two_members_of_population():
    random.seed(SEED)
    np.random.seed(SEED)
    population = [FakeClassifier(condition="0000", q=0.2),
                  FakeClassifier(condition="1111", q=0.5),
                  FakeClassifier(condition="0101", q=0.3)]
    # Holds for every random outcome, so checked over several draws.
    for _ in range(20):
        parent1, parent2 = ga.roulette_wheel_selection(
            population, lambda cl: cl.q)
        assert any(parent1 is cl for cl in population)
        assert any(parent2 is cl for cl in population)


def test_roulette_picks_only_classifier_with_positive_fitness():
    random.seed(SEED)
    np.random.seed(SEED)
    zero = FakeClassifier(condition="0000", q=0.0)
    winner = FakeClassifier(condition="1111", q=1.0)
    parent1, parent2 = ga.roulette_wheel_selection(
        [zero, winner], lambda cl: cl.q)
    assert parent1 is winner
    assert parent2 is winner


# ---------------- generalizing_mutation ----------------

def test_generalizing_mutation_mu_zero_changes_nothing():
    random.seed(SEED)
    np.random.seed(SEED)
    cl = FakeClassifier(condition="01#1")
    ga.generalizing_mutation(cl, mu=0.0)
    assert list(cl.condition) == ["0", "1", "#", "1"]


def test_generalizing_mutation_mu_one_generalizes_all_non_wildcards():
    random.seed(SEED)
    np.random.seed(SEED)
    cl = FakeClassifier(condition="01#1")
    ga.generalizing_mutation(cl, mu=1.0)
    assert list(cl.condition) == ["#", "#", "#", "#"]


# ---------------- two_point_crossover ----------------

def test_two_point_crossover_swaps_one_contiguous_block():
    random.seed(SEED)
    np.random.seed(SEED)
    parent = FakeClassifier(condition="abcd")
    donor = FakeClassifier(condition="wxyz")
    before_p = list("abcd")
    before_d = list("wxyz")

    ga.two_point_crossover(parent, donor)

    p_after = list(parent.condition)
    d_after = list(donor.condition)
    swapped = [i for i in range(4) if p_after[i] != before_p[i]]

    # Letters are all distinct, so swapped positions are exactly the changed ones.
    assert len(swapped) >= 1
    assert swapped == list(range(swapped[0], swapped[-1] + 1))
    for i in range(4):
        if i in swapped:
            assert p_after[i] == before_d[i]
            assert d_after[i] == before_p[i]
        else:
            assert p_after[i] == before_p[i]
            assert d_after[i] == before_d[i]


def test_two_point_crossover_reproducible_with_same_seed():
    def run():
        random.seed(SEED)
        np.random.seed(SEED)
        parent = FakeClassifier(condition="abcd")
        donor = FakeClassifier(condition="wxyz")
        ga.two_point_crossover(parent, donor)
        return list(parent.condition), list(donor.condition)

    assert run() == run()


# ---------------- add_classifier ----------------

def test_add_classifier_appends_new_when_no_similar_exists():
    random.seed(SEED)
    np.random.seed(SEED)
    population = []
    action_set = []
    new_cl = FakeClassifier(condition="0000")
    ga.add_classifier(new_cl, None, population, None, action_set,
                      do_subsumption=False, theta_exp=0)
    assert len(population) == 1 and population[0] is new_cl
    assert len(action_set) == 1 and action_set[0] is new_cl


def test_add_classifier_appends_to_match_set_only_when_matching():
    random.seed(SEED)
    np.random.seed(SEED)
    population = []
    action_set = []
    match_set = []
    matching = FakeClassifier(condition="0000", matches=True)
    ga.add_classifier(matching, None, population, match_set, action_set,
                      do_subsumption=False, theta_exp=0)
    assert match_set == [matching]

    not_matching = FakeClassifier(condition="1111", matches=False)
    ga.add_classifier(not_matching, None, population, match_set, action_set,
                      do_subsumption=False, theta_exp=0)
    assert len(population) == 2
    assert match_set == [matching]


def test_add_classifier_increments_numerosity_of_equal_unmarked():
    random.seed(SEED)
    np.random.seed(SEED)
    existing = FakeClassifier(condition="0000", action=1, num=1)
    population = [existing]
    action_set = [existing]
    new_cl = FakeClassifier(condition="0000", action=1)
    ga.add_classifier(new_cl, None, population, None, action_set,
                      do_subsumption=False, theta_exp=0)
    assert existing.num == 2
    assert len(population) == 1
    assert len(action_set) == 1


def test_add_classifier_does_not_increment_marked_classifier():
    random.seed(SEED)
    np.random.seed(SEED)
    existing = FakeClassifier(condition="0000", action=1, num=1, marked=True)
    population = [existing]
    action_set = [existing]
    new_cl = FakeClassifier(condition="0000", action=1)
    ga.add_classifier(new_cl, None, population, None, action_set,
                      do_subsumption=False, theta_exp=0)
    assert existing.num == 1
    assert len(population) == 1


def test_add_classifier_uses_subsumer_found_by_find_subsumers(monkeypatch):
    random.seed(SEED)
    np.random.seed(SEED)
    subsumer = FakeClassifier(condition="0#00", num=1)
    population = [subsumer]
    action_set = [subsumer]
    monkeypatch.setattr(ga, "find_subsumers",
                        lambda cl, pop, theta_exp: [subsumer])
    new_cl = FakeClassifier(condition="0100")
    ga.add_classifier(new_cl, None, population, None, action_set,
                      do_subsumption=True, theta_exp=0)
    assert subsumer.num == 2
    assert population == [subsumer]


# ---------------- _find_old_classifier / _find_similar ----------------

def test_find_old_classifier_falls_back_to_similar_when_no_subsumer(monkeypatch):
    random.seed(SEED)
    np.random.seed(SEED)
    similar = FakeClassifier(condition="1010", action=2)
    population = [FakeClassifier(condition="0001"), similar]
    monkeypatch.setattr(ga, "find_subsumers",
                        lambda cl, pop, theta_exp: [])
    query = FakeClassifier(condition="1010", action=2)
    result = ga._find_old_classifier(population, query,
                                     use_subsumption=True, theta_exp=0)
    assert result is similar


def test_find_similar_returns_first_equal_or_none():
    random.seed(SEED)
    np.random.seed(SEED)
    first = FakeClassifier(condition="0000", action=0)
    second = FakeClassifier(condition="0000", action=0)
    population = [FakeClassifier(condition="1111", action=0), first, second]

    query = FakeClassifier(condition="0000", action=0)
    assert ga._find_similar(query, population) is first

    missing = FakeClassifier(condition="1010", action=0)
    assert ga._find_similar(missing, population) is None


# ---------------- _is_preferred_to_delete ----------------

def test_is_preferred_to_delete_true_when_clearly_lower_fitness():
    random.seed(SEED)
    np.random.seed(SEED)
    cl_del = FakeClassifier(q=0.5)
    cl = FakeClassifier(q=0.2)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_false_when_clearly_higher_fitness():
    random.seed(SEED)
    np.random.seed(SEED)
    cl_del = FakeClassifier(q=0.5)
    cl = FakeClassifier(q=0.9)
    assert ga._is_preferred_to_delete(cl_del, cl) is False


def test_is_preferred_to_delete_marked_candidate_with_equal_fitness():
    random.seed(SEED)
    np.random.seed(SEED)
    cl_del = FakeClassifier(q=0.5, marked=False)
    cl = FakeClassifier(q=0.5, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_higher_tav_with_equal_fitness():
    random.seed(SEED)
    np.random.seed(SEED)
    cl_del = FakeClassifier(q=0.5, tav=2)
    cl = FakeClassifier(q=0.5, tav=5)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


# ---------------- delete_classifiers ----------------

def test_delete_classifiers_decrements_numerosity_before_removal():
    random.seed(SEED)
    np.random.seed(SEED)
    cl = FakeClassifier(num=5)
    population = FakeList([cl])
    action_set = FakeList([cl])
    # Total would be 1 + 5 = 6 > 5, so exactly one numerosity unit is removed.
    ga.delete_classifiers(population, None, action_set, insize=1, theta_as=5)
    assert cl.num == 4
    assert population == [cl]
    assert action_set == [cl]


def test_delete_classifiers_removes_single_classifier_with_num_one():
    random.seed(SEED)
    np.random.seed(SEED)
    cl = FakeClassifier(num=1)
    population = FakeList([cl])
    action_set = FakeList([cl])
    ga.delete_classifiers(population, None, action_set, insize=1, theta_as=1)
    assert len(population) == 0
    assert len(action_set) == 0
