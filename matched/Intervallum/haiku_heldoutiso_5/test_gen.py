import math
from copy import copy

import numpy as np
import pytest

import intervallum.interval_functions as ifn
from intervallum.box import Box
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions

# Note: hausdorff_distance is defined in intervallum/interval_functions.py
# (not in box.py), so it is tested through the ifn module.


# ---------------------------------------------------------------- Box

def test_box_str_joins_elements_with_x_separator():
    assert str(Box(1.0, 2.0)) == "1.0 x 2.0"
    assert str(Box(Interval(0.0, 1.0), 2.0)) == "[0.0; 1.0] x 2.0"


def test_box_eq_and_ne_compare_elementwise():
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    assert not (Box(1.0, 2.0) != Box(1.0, 2.0))


def test_box_lb_ub_middle_handle_intervals_and_numbers():
    b = Box(Interval(0.0, 2.0), 5.0)
    assert isinstance(b.lb, np.ndarray)
    assert np.array_equal(b.lb, np.array([0.0, 5.0]))
    assert np.array_equal(b.ub, np.array([2.0, 5.0]))
    assert np.array_equal(b.middle, np.array([1.0, 5.0]))


def test_box_width_returns_indices_of_maximal_width():
    indices, max_width = Box(Interval(0.0, 1.0), Interval(0.0, 3.0)).width
    assert indices == [1]
    assert max_width == pytest.approx(3.0)

    indices, max_width = Box(1.0, 2.0).width
    assert indices == [0, 1]
    assert max_width == 0


def test_box_copy_copies_intervals_deeply():
    original = Box(Interval(0.0, 1.0), 2.0)
    duplicate = copy(original)
    assert isinstance(duplicate, Box)
    assert duplicate[0] is not original[0]
    assert duplicate[0].lb == 0.0
    assert duplicate[0].ub == 1.0
    assert duplicate[1] == 2.0


def test_box_try_to_reduce_returns_middle_for_degenerate_box_else_box():
    reduced = Box(1.0, 2.0)._try_to_reduce()
    assert np.array_equal(reduced, np.array([1.0, 2.0]))
    assert not isinstance(reduced, Box)

    wide = Box(Interval(0.0, 1.0), 2.0)
    kept = wide._try_to_reduce()
    assert isinstance(kept, Box)
    assert kept is not wide


# ---------------------------------------------------------------- Interval

def test_interval_construction_validates_bounds_and_setters_enforce_order():
    i = Interval(1.0, 2.0)
    assert i.lb == 1.0
    assert i.ub == 2.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.5
    i.ub = 5.0
    assert i.ub == 5.0


def test_interval_fix_swaps_reversed_bounds():
    fixed = Interval(2.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 2.0


def test_interval_str_from_point_middle_width_and_comparisons():
    i = Interval(1.0, 2.0)
    assert str(i) == "[1.0; 2.0]"
    assert repr(i) == "[1.0; 2.0]"
    p = Interval.from_point(3.0)
    assert p.lb == 3.0 and p.ub == 3.0
    assert i.middle == pytest.approx(1.5)
    assert i.width == pytest.approx(1.0)
    assert Interval(1.0, 5.0) < Interval(2.0, 3.0)
    assert Interval(1.0, 5.0) < 2.0
    assert Interval(3.0, 4.0) >= 3.0
    assert not (Interval(3.0, 4.0) > 3.0)


def test_interval_add_sub_neg():
    s = Interval(1.0, 2.0) + Interval(3.0, 4.0)
    assert (s.lb, s.ub) == (4.0, 6.0)
    s = Interval(1.0, 2.0) + 1.0
    assert (s.lb, s.ub) == (2.0, 3.0)
    n = -Interval(1.0, 2.0)
    assert (n.lb, n.ub) == (-2.0, -1.0)
    d = Interval(1.0, 2.0) - Interval(0.0, 1.0)
    assert (d.lb, d.ub) == (0.0, 2.0)
    d = Interval(1.0, 2.0) - 1.0
    assert (d.lb, d.ub) == (0.0, 1.0)


def test_interval_reflected_add_and_sub():
    r = 5.0 - Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (3.0, 4.0)
    r = 1.0 + Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (2.0, 3.0)


def test_interval_mul_and_rmul():
    m = Interval(1.0, 2.0) * Interval(3.0, 4.0)
    assert (m.lb, m.ub) == (3.0, 8.0)
    m = Interval(-1.0, 2.0) * 2.0
    assert (m.lb, m.ub) == (-2.0, 4.0)
    m = 2.0 * Interval(1.0, 3.0)
    assert (m.lb, m.ub) == (2.0, 6.0)


def test_interval_invert_and_division():
    inv = ~Interval(2.0, 4.0)
    assert (inv.lb, inv.ub) == pytest.approx((0.25, 0.5))

    inv = ~Interval(0.0, 2.0)
    assert inv.lb == pytest.approx(0.5)
    assert inv.ub == math.inf

    q = Interval(2.0, 4.0) / 2.0
    assert (q.lb, q.ub) == pytest.approx((1.0, 2.0))

    r = 1.0 / Interval(2.0, 4.0)
    assert (r.lb, r.ub) == pytest.approx((0.25, 0.5))


def test_interval_pow_even_and_odd_exponents():
    even = Interval(-2.0, 3.0) ** 2
    assert (even.lb, even.ub) == pytest.approx((0.0, 9.0))

    even_positive = Interval(2.0, 3.0) ** 2
    assert (even_positive.lb, even_positive.ub) == pytest.approx((4.0, 9.0))

    odd = Interval(-2.0, 3.0) ** 3
    assert (odd.lb, odd.ub) == pytest.approx((-8.0, 27.0))


def test_interval_equality_uses_admissible_error():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 2.5)
    assert Interval(1.0, 1.0) == 1.0
    assert Interval(1.0, 2.0) != 1.0


def test_interval_reduces_narrow_results_to_floats(monkeypatch):
    reduced = Interval(1.0, 1.0) + 1.0
    assert isinstance(reduced, float)
    assert reduced == pytest.approx(2.0)

    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    kept = Interval(1.0, 1.0) + 1.0
    assert isinstance(kept, Interval)
    assert kept.lb == pytest.approx(2.0)


# ---------------------------------------------------------------- RS_SubinclusionFunction

def test_rs_subinclusion_degenerate_box_gives_exact_value():
    rs = RS_SubinclusionFunction(lambda x: x[0] + x[1], number_of_samples=5)
    result = rs(Box(1.0, 2.0))
    assert isinstance(result, Interval)
    assert result.lb == pytest.approx(3.0)
    assert result.ub == pytest.approx(3.0)


def test_rs_subinclusion_evaluates_requested_number_of_samples():
    calls = []

    def f(x):
        calls.append(x)
        return float(x[0])

    rs = RS_SubinclusionFunction(f, number_of_samples=7)
    rs(Box(Interval(0.0, 1.0), Interval(0.0, 1.0)))
    assert len(calls) == 7


def test_rs_subinclusion_values_stay_within_expected_range():
    np.random.seed(0)
    rs = RS_SubinclusionFunction(lambda x: x[0] * 10.0 + x[1], number_of_samples=200)
    result = rs(Box(Interval(0.0, 1.0), Interval(2.0, 3.0)))
    assert result.lb >= 2.0
    assert result.ub <= 13.0
    assert result.lb <= result.ub


# ---------------------------------------------------------------- interval_functions

def test_split_and_bisect_partition_interval():
    parts = ifn.split(Interval(0.0, 4.0), [1.0, 1.0])
    assert len(parts) == 2
    assert (parts[0].lb, parts[0].ub) == pytest.approx((0.0, 2.0))
    assert (parts[1].lb, parts[1].ub) == pytest.approx((2.0, 4.0))

    weighted = ifn.split(Interval(0.0, 4.0), [1.0, 3.0])
    assert (weighted[1].lb, weighted[1].ub) == pytest.approx((1.0, 4.0))

    halves = ifn.bisect(Interval(0.0, 2.0))
    assert (halves[0].lb, halves[0].ub) == pytest.approx((0.0, 1.0))
    assert (halves[1].lb, halves[1].ub) == pytest.approx((1.0, 2.0))

    assert ifn.split(2.0, [1.0, 1.0, 1.0]) == [2.0, 2.0, 2.0]


def test_constrain_and_shrink():
    c = ifn.constrain(Interval(-5.0, 10.0), 0.0, 5.0)
    assert (c.lb, c.ub) == (0.0, 5.0)
    assert ifn.constrain(7.0, 0.0, 5.0) == 5.0
    assert ifn.constrain(-1.0, 0.0, 5.0) == 0.0
    untouched = ifn.constrain(Interval(1.0, 2.0), 0.0, 5.0)
    assert (untouched.lb, untouched.ub) == (1.0, 2.0)

    s = ifn.shrink(Interval(0.0, 4.0), 0.5)
    assert (s.lb, s.ub) == pytest.approx((1.0, 3.0))
    assert ifn.shrink(3.0, 0.5) == 3.0


def test_sqrt_and_log_monotonic_and_domain_errors():
    r = ifn.sqrt(Interval(4.0, 9.0))
    assert (r.lb, r.ub) == pytest.approx((2.0, 3.0))

    clipped = ifn.sqrt(Interval(-1.0, 4.0))
    assert (clipped.lb, clipped.ub) == pytest.approx((0.0, 2.0))

    assert ifn.sqrt(4.0) == pytest.approx(2.0)

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.sqrt(Interval(-3.0, -1.0))

    lg = ifn.log(Interval(1.0, math.e))
    assert (lg.lb, lg.ub) == pytest.approx((0.0, 1.0))


def test_trig_functions_use_extrema_and_full_period_shortcut():
    s = ifn.sin(Interval(0.0, math.pi / 2))
    assert s.lb == pytest.approx(0.0, abs=1e-12)
    assert s.ub == pytest.approx(1.0)

    c = ifn.cos(Interval(0.0, math.pi))
    assert c.lb == pytest.approx(-1.0)
    assert c.ub == pytest.approx(1.0)

    wide = ifn.sin(Interval(0.0, 10.0))
    assert (wide.lb, wide.ub) == pytest.approx((-1.0, 1.0))

    assert ifn.sin(math.pi / 2) == pytest.approx(1.0)


def test_abs_and_exp_monotonic_evaluation():
    a = ifn.abs(Interval(-3.0, 2.0))
    assert (a.lb, a.ub) == pytest.approx((0.0, 3.0))

    a_pos = ifn.abs(Interval(1.0, 2.0))
    assert (a_pos.lb, a_pos.ub) == pytest.approx((1.0, 2.0))

    e = ifn.exp(Interval(0.0, 1.0))
    assert (e.lb, e.ub) == pytest.approx((1.0, math.e))


def test_hausdorff_distance_cases():
    assert ifn.hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == pytest.approx(0.0)
    assert ifn.hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == pytest.approx(2.0)
    assert ifn.hausdorff_distance(1.0, 3.0) == pytest.approx(2.0)
    assert ifn.hausdorff_distance(Interval(0.0, 4.0), Interval(1.0, 2.0)) == pytest.approx(2.0)

    a, b = Interval(0.0, 1.0), Interval(0.5, 2.0)
    assert ifn.hausdorff_distance(a, b) == pytest.approx(1.0)
    assert ifn.hausdorff_distance(a, b) == pytest.approx(ifn.hausdorff_distance(b, a))
