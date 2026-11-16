import copy as copy_module

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import hausdorff_distance


# ----------------------------- Interval -----------------------------

def test_interval_construction_and_properties():
    i = Interval(1.0, 3.0)
    assert i.lb == 1.0
    assert i.ub == 3.0
    assert i.middle == 2.0
    assert i.width == 2.0
    assert str(i) == "[1.0; 3.0]"
    assert repr(i) == str(i)


def test_interval_wrong_bounds_raises_and_fix_swaps():
    with pytest.raises(IntervalExceptions.WrongBoundsException, match="Improper interval"):
        Interval(2.0, 1.0)
    fixed = Interval(2.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 2.0


def test_interval_from_point():
    p = Interval.from_point(4.5)
    assert p.lb == 4.5
    assert p.ub == 4.5
    assert p.width == 0


def test_interval_bound_setters_validate():
    i = Interval(1.0, 3.0)
    i.lb = 2.0
    i.ub = 5.0
    assert (i.lb, i.ub) == (2.0, 5.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 6.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 1.0
    assert (i.lb, i.ub) == (2.0, 5.0)


def test_interval_copy_is_independent():
    i = Interval(1.0, 2.0)
    c = copy_module.copy(i)
    c.ub = 10.0
    assert i.ub == 2.0
    assert c.ub == 10.0


def test_interval_addition_and_negation():
    s = Interval(1.0, 2.0) + Interval(10.0, 20.0)
    assert (s.lb, s.ub) == (11.0, 22.0)
    s2 = 1.0 + Interval(1.0, 2.0)
    assert (s2.lb, s2.ub) == (2.0, 3.0)
    n = -Interval(1.0, 3.0)
    assert (n.lb, n.ub) == (-3.0, -1.0)


def test_interval_subtraction_and_reverse_subtraction():
    d = Interval(1.0, 3.0) - Interval(0.0, 1.0)
    assert (d.lb, d.ub) == (0.0, 3.0)
    r = 5.0 - Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (3.0, 4.0)


def test_interval_multiplication():
    m = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (m.lb, m.ub) == (-4.0, 8.0)
    m2 = Interval(1.0, 2.0) * -2.0
    assert (m2.lb, m2.ub) == (-4.0, -2.0)
    m3 = 3.0 * Interval(1.0, 2.0)
    assert (m3.lb, m3.ub) == (3.0, 6.0)


def test_interval_division():
    q = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert q.lb == pytest.approx(0.25)
    assert q.ub == pytest.approx(1.0)
    q2 = Interval(2.0, 4.0) / 2.0
    assert (q2.lb, q2.ub) == (1.0, 2.0)
    q3 = 1.0 / Interval(2.0, 4.0)
    assert q3.lb == pytest.approx(0.25)
    assert q3.ub == pytest.approx(0.5)


def test_interval_inversion_with_zero_in_range():
    inv_zero_lb = ~Interval(0.0, 2.0)
    assert inv_zero_lb.lb == 0.5
    assert inv_zero_lb.ub == float("inf")
    inv_zero_ub = ~Interval(-2.0, 0.0)
    assert inv_zero_ub.lb == float("-inf")
    assert inv_zero_ub.ub == -0.5
    inv_both = ~Interval(-1.0, 1.0)
    assert inv_both.lb == float("-inf")
    assert inv_both.ub == float("inf")


def test_interval_power_even_and_odd():
    even = Interval(-2.0, 3.0) ** 2
    assert (even.lb, even.ub) == (0.0, 9.0)
    even_pos = Interval(1.0, 2.0) ** 2
    assert (even_pos.lb, even_pos.ub) == (1.0, 4.0)
    odd = Interval(-2.0, 3.0) ** 3
    assert (odd.lb, odd.ub) == (-8.0, 27.0)


def test_interval_equality_and_inequality():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 3.0)
    assert Interval.from_point(1.0) == 1.0
    assert Interval(1.0, 2.0) != 1.0
    with pytest.raises(NotImplementedError):
        Interval(1.0, 2.0) == "abc"


def test_interval_ordering_compares_lower_bounds():
    assert Interval(1.0, 2.0) < Interval(2.0, 3.0)
    assert Interval(2.0, 3.0) > Interval(1.0, 10.0)
    assert Interval(1.0, 2.0) <= 1.0
    assert Interval(1.0, 2.0) >= 1.0
    assert Interval(1.0, 2.0) < 1.5
    assert not (Interval(1.0, 2.0) > 1.5)


def test_interval_inner_subtraction():
    r = Interval.inner_subtraction(Interval(1.0, 5.0), Interval(0.0, 2.0))
    assert (r.lb, r.ub) == (1.0, 3.0)
    r2 = Interval(1.0, 5.0) << Interval(0.0, 2.0)
    assert (r2.lb, r2.ub) == (1.0, 3.0)


def test_interval_reduction_of_degenerate_results(monkeypatch):
    result = Interval.from_point(1.0) + 2.0
    assert not isinstance(result, Interval)
    assert result == 3.0

    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    not_reduced = Interval.from_point(1.0) + 2.0
    assert isinstance(not_reduced, Interval)
    assert (not_reduced.lb, not_reduced.ub) == (3.0, 3.0)


# ------------------------------- Box --------------------------------

def test_box_bounds_middle_and_width():
    b = Box(Interval(0.0, 1.0), 2.0, Interval(-4.0, 0.0))
    assert list(b.lb) == [0.0, 2.0, -4.0]
    assert list(b.ub) == [1.0, 2.0, 0.0]
    assert list(b.middle) == [0.5, 2.0, -2.0]
    indices, w = b.width
    assert indices == [2]
    assert w == 4.0


def test_box_width_reports_all_max_indices():
    b = Box(Interval(0.0, 2.0), Interval(5.0, 7.0), Interval(0.0, 1.0))
    indices, w = b.width
    assert indices == [0, 1]
    assert w == 2.0


def test_box_equality_str_and_copy():
    b1 = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    b2 = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    b3 = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))
    assert b1 == b2
    assert b1 != b3
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 2.0, 3.0)
    with pytest.raises(NotImplementedError):
        b1 == [1, 2]
    assert str(b1) == "[0.0; 1.0] x [2.0; 3.0]"

    c = copy_module.copy(b1)
    assert isinstance(c, Box)
    c[0].ub = 10.0
    assert b1[0].ub == 1.0
    assert c[0].ub == 10.0


def test_box_try_to_reduce():
    degenerate = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    reduced = degenerate._try_to_reduce()
    assert list(reduced) == [1.0, 2.0]

    wide = Box(Interval(1.0, 2.0), Interval(2.0, 2.0))
    kept = wide._try_to_reduce()
    assert isinstance(kept, Box)
    assert list(kept.lb) == [1.0, 2.0]
    assert list(kept.ub) == [2.0, 2.0]


# ------------------------ RS_SubinclusionFunction ---------------------

def test_rs_subinclusion_function_bounds_within_true_range():
    np.random.seed(0)
    f = RS_SubinclusionFunction(lambda p: p[0] + p[1], number_of_samples=200)
    result = f(Box(Interval(0.0, 1.0), Interval(2.0, 3.0)))
    assert isinstance(result, Interval)
    assert 2.0 <= result.lb <= result.ub <= 4.0
    assert result.width > 0.5


def test_rs_subinclusion_function_sample_count_and_points_in_box():
    np.random.seed(1)
    calls = []

    def func(p):
        calls.append(np.array(p))
        return 5.0

    f = RS_SubinclusionFunction(func, number_of_samples=17)
    result = f(Box(Interval(-1.0, 1.0), Interval(10.0, 20.0)))
    assert len(calls) == 17
    for p in calls:
        assert -1.0 <= p[0] <= 1.0
        assert 10.0 <= p[1] <= 20.0
    assert (result.lb, result.ub) == (5.0, 5.0)


# -------------------------- hausdorff_distance ------------------------

def test_hausdorff_distance_intervals_and_numbers():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 4.0)) == 3.0
    assert hausdorff_distance(Interval(2.0, 4.0), Interval(0.0, 1.0)) == 3.0
    assert hausdorff_distance(Interval(1.0, 2.0), Interval(1.0, 2.0)) == 0.0
    assert hausdorff_distance(1.0, 4.0) == 3.0
    assert hausdorff_distance(Interval(0.0, 2.0), 5.0) == 5.0
    # nested intervals: distance is the largest endpoint gap
    assert hausdorff_distance(Interval(0.0, 10.0), Interval(3.0, 4.0)) == 6.0
