import random

import pytest

import lcs.strategies.genetic_algorithms as ga
from lcs import Perception
from lcs.agents.acs2 import Classifier, ClassifiersList, Configuration


def make_cfg(**overrides):
    kwargs = dict(classifier_length=4, number_of_possible_actions=2)
    kwargs.update(overrides)
    return Configuration(**kwargs)


@pytest.fixture
def cfg():
    return make_cfg()


def make_cl(cfg, condition="####", action=0, effect=None, quality=0.5,
            numerosity=1, experience=1, tga=0, tav=0.0):
    return Classifier(condition=condition, action=action, effect=effect,
                      quality=quality, numerosity=numerosity,
                      experience=experience, tga=tga, tav=tav, cfg=cfg)


# should_apply

def test_should_apply_false_for_none_action_set():
    assert ga.should_apply(None, time=500, theta_ga=100) is False


def test_should_apply_false_for_empty_action_set():
    assert ga.should_apply(ClassifiersList(), time=500, theta_ga=100) is False


def test_should_apply_true_when_delay_exceeds_threshold(cfg):
    action_set = ClassifiersList(make_cl(cfg, tga=0, numerosity=1))
    assert ga.should_apply(action_set, time=101, theta_ga=100) is True


def test_should_apply_false_at_exact_threshold(cfg):
    action_set = ClassifiersList(make_cl(cfg, tga=0, numerosity=1))
    assert ga.should_apply(action_set, time=100, theta_ga=100) is False


def test_should_apply_weights_average_delay_by_numerosity(cfg):
    action_set = ClassifiersList(
        make_cl(cfg, condition="0000", tga=0, numerosity=3),
        make_cl(cfg, condition="1111", tga=50, numerosity=1),
    )
    # weighted average of last GA times is (0*3 + 50*1) / 4 = 12.5
    assert ga.should_apply(action_set, time=120, theta_ga=100) is True
    assert ga.should_apply(action_set, time=110, theta_ga=100) is False


# set_timestamps

def test_set_timestamps_stamps_every_classifier(cfg):
    action_set = ClassifiersList(
        make_cl(cfg, condition="0000", tga=3),
        make_cl(cfg, condition="1111", tga=7),
    )
    ga.set_timestamps(action_set, epoch=42)
    assert [cl.tga for cl in action_set] == [42, 42]


# roulette_wheel_selection

def test_roulette_wheel_never_selects_zero_fitness_classifier(cfg):
    dead = make_cl(cfg, condition="0000", quality=0.0)
    alive = make_cl(cfg, condition="1111", quality=1.0)
    population = ClassifiersList(dead, alive)

    for _ in range(50):
        parent1, parent2 = ga.roulette_wheel_selection(
            population, lambda cl: cl.q)
        assert parent1 is alive
        assert parent2 is alive


def test_roulette_wheel_uses_cumulative_fitness(cfg, monkeypatch):
    a = make_cl(cfg, condition="0000", quality=0.2)
    b = make_cl(cfg, condition="1111", quality=0.8)
    draws = iter([0.1, 0.5])
    monkeypatch.setattr(random, "uniform", lambda low, high: next(draws))

    parent1, parent2 = ga.roulette_wheel_selection(
        ClassifiersList(a, b), lambda cl: cl.q)

    assert parent1 is a
    assert parent2 is b


# generalizing_mutation

def test_generalizing_mutation_full_probability_generalizes_specified(cfg):
    cl = make_cl(cfg, condition="01#1")
    ga.generalizing_mutation(cl, mu=1.0)
    assert "".join(cl.condition) == "####"


def test_generalizing_mutation_zero_probability_changes_nothing(cfg):
    cl = make_cl(cfg, condition="01#1")
    ga.generalizing_mutation(cl, mu=0.0)
    assert "".join(cl.condition) == "01#1"


# two_point_crossover

def test_two_point_crossover_swaps_segment_between_cut_points(cfg,
                                                              monkeypatch):
    parent = make_cl(cfg, condition="1111")
    donor = make_cl(cfg, condition="0000")
    monkeypatch.setattr(ga.np.random, "choice",
                        lambda *args, **kwargs: [3, 1])

    ga.two_point_crossover(parent, donor)

    assert "".join(parent.condition) == "1001"
    assert "".join(donor.condition) == "0110"


def test_two_point_crossover_keeps_each_position_pair(cfg):
    for _ in range(30):
        parent = make_cl(cfg, condition="0101")
        donor = make_cl(cfg, condition="1100")
        before = [sorted((p, d))
                  for p, d in zip(parent.condition, donor.condition)]

        ga.two_point_crossover(parent, donor)

        after = [sorted((p, d))
                 for p, d in zip(parent.condition, donor.condition)]
        assert after == before


# add_classifier

def test_add_classifier_inserts_new_classifier_into_all_lists(cfg):
    population = ClassifiersList()
    action_set = ClassifiersList()
    match_set = ClassifiersList()
    new_cl = make_cl(cfg, condition="00##")

    ga.add_classifier(new_cl, Perception("0011"), population, match_set,
                      action_set, do_subsumption=False, theta_exp=20)

    assert len(population) == 1 and population[0] is new_cl
    assert len(action_set) == 1 and action_set[0] is new_cl
    assert len(match_set) == 1 and match_set[0] is new_cl


def test_add_classifier_skips_match_set_when_condition_mismatches(cfg):
    population = ClassifiersList()
    action_set = ClassifiersList()
    match_set = ClassifiersList()
    new_cl = make_cl(cfg, condition="11##")

    ga.add_classifier(new_cl, Perception("0011"), population, match_set,
                      action_set, do_subsumption=False, theta_exp=20)

    assert len(population) == 1
    assert len(action_set) == 1
    assert len(match_set) == 0


def test_add_classifier_accepts_missing_match_set(cfg):
    population = ClassifiersList()
    action_set = ClassifiersList()
    new_cl = make_cl(cfg, condition="00##")

    ga.add_classifier(new_cl, Perception("0011"), population, None,
                      action_set, do_subsumption=False, theta_exp=20)

    assert len(population) == 1 and population[0] is new_cl


def test_add_classifier_increments_numerosity_of_similar_classifier(cfg):
    existing = make_cl(cfg, condition="00##", action=1)
    population = ClassifiersList(existing)
    action_set = ClassifiersList(existing)
    twin = make_cl(cfg, condition="00##", action=1)

    ga.add_classifier(twin, Perception("0011"), population, None,
                      action_set, do_subsumption=False, theta_exp=20)

    assert existing.num == 2
    assert len(population) == 1
    assert len(action_set) == 1


def test_add_classifier_does_not_grow_marked_similar_classifier(cfg):
    existing = make_cl(cfg, condition="00##", action=1)
    existing.mark[0].add("1")
    population = ClassifiersList(existing)
    action_set = ClassifiersList(existing)
    twin = make_cl(cfg, condition="00##", action=1)

    ga.add_classifier(twin, Perception("0011"), population, None,
                      action_set, do_subsumption=False, theta_exp=20)

    assert existing.num == 1
    assert len(population) == 1
    assert len(action_set) == 1


def test_add_classifier_reliable_subsumer_absorbs_specific_classifier(cfg):
    general = make_cl(cfg, condition="00##", action=0,
                      quality=0.95, experience=30)
    population = ClassifiersList(general)
    action_set = ClassifiersList(general)
    specific = make_cl(cfg, condition="0011", action=0)

    ga.add_classifier(specific, Perception("0011"), population, None,
                      action_set, do_subsumption=True, theta_exp=20)

    assert general.num == 2
    assert len(population) == 1
    assert len(action_set) == 1


def test_add_classifier_ignores_subsumer_that_is_not_reliable(cfg):
    general = make_cl(cfg, condition="00##", action=0,
                      quality=0.5, experience=30)
    population = ClassifiersList(general)
    action_set = ClassifiersList(general)
    specific = make_cl(cfg, condition="0011", action=0)

    ga.add_classifier(specific, Perception("0011"), population, None,
                      action_set, do_subsumption=True, theta_exp=20)

    assert general.num == 1
    assert len(population) == 2
    assert len(action_set) == 2


# delete_classifiers

def test_delete_classifiers_is_noop_when_under_threshold(cfg):
    cl = make_cl(cfg, numerosity=2)
    population = ClassifiersList(cl)
    action_set = ClassifiersList(cl)

    ga.delete_classifiers(population, None, action_set,
                          insize=1, theta_as=3)

    assert cl.num == 2
    assert len(population) == 1


def test_delete_classifiers_decrements_numerosity_before_removing(
        cfg, monkeypatch):
    cl = make_cl(cfg, numerosity=3)
    population = ClassifiersList(cl)
    action_set = ClassifiersList(cl)
    monkeypatch.setattr(random, "random", lambda: 0.0)

    ga.delete_classifiers(population, None, action_set,
                          insize=0, theta_as=2)

    assert cl.num == 2
    assert len(population) == 1
    assert len(action_set) == 1


def test_delete_classifiers_removes_least_fit_classifier_when_full(
        cfg, monkeypatch):
    fit = make_cl(cfg, condition="0000", quality=0.9)
    weak = make_cl(cfg, condition="1111", quality=0.1)
    population = ClassifiersList(fit, weak)
    match_set = ClassifiersList(fit, weak)
    action_set = ClassifiersList(fit, weak)
    monkeypatch.setattr(random, "random", lambda: 0.0)

    ga.delete_classifiers(population, match_set, action_set,
                          insize=0, theta_as=1)

    assert len(population) == 1 and population[0] is fit
    assert len(match_set) == 1 and match_set[0] is fit
    assert len(action_set) == 1 and action_set[0] is fit


# _is_preferred_to_delete

def test_is_preferred_to_delete_follows_quality_gap(cfg):
    high = make_cl(cfg, condition="0000", quality=0.9)
    low = make_cl(cfg, condition="1111", quality=0.1)

    assert ga._is_preferred_to_delete(high, low) is True
    assert ga._is_preferred_to_delete(low, high) is False


def test_is_preferred_to_delete_breaks_quality_ties_by_mark_then_tav(cfg):
    cl_del = make_cl(cfg, condition="0000", quality=0.5, tav=1.0)

    marked = make_cl(cfg, condition="1111", quality=0.5, tav=1.0)
    marked.mark[0].add("1")
    assert ga._is_preferred_to_delete(cl_del, marked) is True

    newer = make_cl(cfg, condition="1111", quality=0.5, tav=2.0)
    assert ga._is_preferred_to_delete(cl_del, newer) is True

    older = make_cl(cfg, condition="1111", quality=0.5, tav=0.5)
    assert ga._is_preferred_to_delete(cl_del, older) is False
