import pytest

import lcs.strategies.genetic_algorithms as ga


class DummyCfg:
    def __init__(self, classifier_wildcard="*", classifier_length=4):
        self.classifier_wildcard = classifier_wildcard
        self.classifier_length = classifier_length


class DummyCondition:
    def __init__(self, values, wildcard="*"):
        self.values = list(values)
        self.wildcard = wildcard
        self.generalized = []

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.values[item]
        return self.values[item]

    def __setitem__(self, idx, value):
        self.values[idx] = value

    def generalize(self, idx):
        self.generalized.append(idx)
        self.values[idx] = self.wildcard

    def does_match(self, perception):
        return self.values == list(perception)

    def __eq__(self, other):
        return isinstance(other, DummyCondition) and self.values == other.values


class DummyCl:
    def __init__(self, condition, action="a", effect="e", num=1, tga=0, q=0.0, tav=0.0, marked=False):
        self.condition = condition
        self.action = action
        self.effect = effect
        self.num = num
        self.tga = tga
        self.q = q
        self.tav = tav
        self.marked = marked
        self.cfg = DummyCfg(classifier_wildcard=condition.wildcard, classifier_length=len(condition))

    def is_marked(self):
        return self.marked

    def __eq__(self, other):
        return (
            isinstance(other, DummyCl)
            and self.condition == other.condition
            and self.action == other.action
            and self.effect == other.effect
        )

    def __hash__(self):
        return id(self)


class DummySet(list):
    def expand(self):
        expanded = []
        for cl in self:
            expanded.extend([cl] * cl.num)
        return expanded

    def safe_remove(self, cl):
        self.remove(cl)


def test_should_apply_returns_false_for_none_action_set():
    assert ga.should_apply(None, time=100, theta_ga=10) is False


def test_should_apply_returns_false_when_population_numerosity_is_zero():
    action_set = DummySet([DummyCl(DummyCondition([1]), num=0, tga=0)])
    assert ga.should_apply(action_set, time=100, theta_ga=10) is False


def test_should_apply_returns_true_when_average_delay_exceeds_threshold():
    action_set = DummySet(
        [DummyCl(DummyCondition([1]), num=2, tga=10), DummyCl(DummyCondition([2]), num=1, tga=20)]
    )
    # avg tga = (10*2 + 20*1)/3 = 13.333... ; delay at time 30 = 16.666... > 10
    assert ga.should_apply(action_set, time=30, theta_ga=10) is True


def test_set_timestamps_updates_all_classifiers():
    action_set = DummySet([DummyCl(DummyCondition([1]), tga=1), DummyCl(DummyCondition([2]), tga=2)])
    ga.set_timestamps(action_set, epoch=99)
    assert [cl.tga for cl in action_set] == [99, 99]


def test_roulette_wheel_selection_uses_weighted_choice_twice(monkeypatch):
    population = [DummyCl(DummyCondition([1])), DummyCl(DummyCondition([2]))]

    calls = []

    def fake_weighted_random_choice(choices):
        calls.append(choices)
        return population[len(calls) - 1]

    monkeypatch.setattr(ga, "_weighted_random_choice", fake_weighted_random_choice)
    p1, p2 = ga.roulette_wheel_selection(population, fitnessfunc=lambda cl: 1.0)

    assert p1 is population[0]
    assert p2 is population[1]
    assert len(calls) == 2
    assert set(calls[0].keys()) == set(population)


def test_generalizing_mutation_generalizes_only_selected_attributes(monkeypatch):
    cl = DummyCl(DummyCondition(["x", "y", "z"], wildcard="?"))

    values = iter([0.05, 0.5, 0.01])  # mutate 0 and 2 when mu=0.1
    monkeypatch.setattr(ga.random, "random", lambda: next(values))

    ga.generalizing_mutation(cl, mu=0.1)

    assert cl.condition.generalized == [0, 2]
    assert cl.condition.values == ["?", "y", "?"]


def test_generalizing_mutation_skips_wildcard_values(monkeypatch):
    cl = DummyCl(DummyCondition(["*", "b"], wildcard="*"))

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    ga.generalizing_mutation(cl, mu=1.0)

    assert cl.condition.generalized == [1]
    assert cl.condition.values == ["*", "*"]


def test_two_point_crossover_swaps_middle_segment(monkeypatch):
    parent = DummyCl(DummyCondition([1, 2, 3, 4]))
    donor = DummyCl(DummyCondition([9, 8, 7, 6]))

    monkeypatch.setattr(ga.np.random, "choice", lambda *args, **kwargs: [1, 3])
    ga.two_point_crossover(parent, donor)

    assert parent.condition.values == [1, 8, 7, 4]
    assert donor.condition.values == [9, 2, 3, 6]


def test_add_classifier_appends_new_classifier_and_match_set_when_no_old(monkeypatch):
    cl = DummyCl(DummyCondition([1, 2]))
    population = DummySet()
    match_set = DummySet()
    action_set = DummySet()

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: None)
    ga.add_classifier(cl, p=[1, 2], population=population, match_set=match_set, action_set=action_set,
                       do_subsumption=False, theta_exp=0)

    assert population == [cl]
    assert action_set == [cl]
    assert match_set == [cl]


def test_add_classifier_increments_numerosity_for_existing_unmarked_classifier(monkeypatch):
    existing = DummyCl(DummyCondition([1, 2]), num=3, marked=False)
    population = DummySet([existing])
    match_set = DummySet([existing])
    action_set = DummySet([existing])
    new_cl = DummyCl(DummyCondition([1, 2]))

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: existing)
    ga.add_classifier(new_cl, p=[1, 2], population=population, match_set=match_set, action_set=action_set,
                       do_subsumption=True, theta_exp=0)

    assert existing.num == 4
    assert new_cl not in population


def test_add_classifier_does_not_increment_marked_existing_classifier(monkeypatch):
    existing = DummyCl(DummyCondition([1, 2]), num=3, marked=True)
    population = DummySet([existing])
    action_set = DummySet([existing])
    new_cl = DummyCl(DummyCondition([1, 2]))

    monkeypatch.setattr(ga, "_find_old_classifier", lambda *args, **kwargs: existing)
    ga.add_classifier(new_cl, p=[1, 2], population=population, match_set=None, action_set=action_set,
                       do_subsumption=False, theta_exp=0)

    assert existing.num == 3
    assert new_cl not in population


def test_find_similar_returns_first_equal_classifier():
    target = DummyCl(DummyCondition([1, 2]), action="a", effect="e")
    other = DummyCl(DummyCondition([9, 9]), action="a", effect="e")
    population = DummySet([other, target])

    assert ga._find_similar(target, population) is target


def test_find_old_classifier_prefers_subsumers_when_present(monkeypatch):
    target = DummyCl(DummyCondition([1, 2]))
    subsumer = DummyCl(DummyCondition([1, "*"]))
    population = DummySet([subsumer, target])

    monkeypatch.setattr(ga, "find_subsumers", lambda cl, pop, theta_exp: [subsumer])
    monkeypatch.setattr(ga, "_find_similar", lambda other, pop: target)

    assert ga._find_old_classifier(population, target, use_subsumption=True, theta_exp=5) is subsumer


def test_is_preferred_to_delete_prefers_lower_quality():
    cl_del = DummyCl(DummyCondition([1]), q=0.8, tav=1.0)
    cl = DummyCl(DummyCondition([2]), q=0.6, tav=1.0)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_is_preferred_to_delete_prefers_marked_and_older_when_qualities_similar():
    cl_del = DummyCl(DummyCondition([1]), q=0.5, tav=1.0, marked=False)
    cl = DummyCl(DummyCondition([2]), q=0.55, tav=2.0, marked=True)
    assert ga._is_preferred_to_delete(cl_del, cl) is True


def test_delete_classifiers_removes_singleton_from_all_sets(monkeypatch):
    cl = DummyCl(DummyCondition([1]), num=1, q=0.1, tav=0.0)
    population = DummySet([cl])
    match_set = DummySet([cl])
    action_set = DummySet([cl])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=1)

    assert population == []
    assert match_set == []
    assert action_set == []


def test_delete_classifiers_decrements_numerosity_when_more_than_one(monkeypatch):
    cl = DummyCl(DummyCondition([1]), num=2, q=0.1, tav=0.0)
    population = DummySet([cl])
    match_set = DummySet([cl])
    action_set = DummySet([cl])

    monkeypatch.setattr(ga.random, "random", lambda: 0.0)
    ga.delete_classifiers(population, match_set, action_set, insize=1, theta_as=1)

    assert cl.num == 1
    assert population == [cl]
    assert match_set == [cl]
    assert action_set == [cl]
