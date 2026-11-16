import math
from copy import copy

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum import interval_functions as F


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------

def test_interval_stores_bounds():
    i = Interval(1.0, 2.0)
    assert i.lb == 1.0
    assert i.ub == 2.0
    assert i.width == 1.0
    assert i.middle == 1.5


def test_interval_wrong_bounds_raises():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)


def test_interval_fix_swaps_reversed_bounds():
    i = Interval(2.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 2.0


def test_interval_from_point_is_degenerate():
    i = Interval.from_point(3.5)
    assert i.lb == 3.5
    assert i.ub == 3.5
    assert i.width == 0.0


def test_interval_setters_enforce_ordering():
    i = Interval(1.0, 2.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0
    i.lb = 0.5
    i.ub = 2.5
    assert (i.lb, i.ub) == (0.5, 2.5)


def test_interval_equality_with_tolerance_and_inequality():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0 + 1e-9)
    assert Interval(1.0, 2.0) != Interval(1.5, 2.5)
    assert not (Interval(1.0, 2.0) == Interval(1.5, 2.5))


def test_interval_equality_rejects_unsupported_type():
    with pytest.raises(NotImplementedError):
        _ = Interval(1.0, 2.0) == "not an interval"


def test_interval_addition_with_interval_and_scalar():
    assert (Interval(1.0, 2.0) + Interval(3.0, 4.0)) == Interval(4.0, 6.0)
    res = Interval(1.0, 2.0) + 1.0
    assert (res.lb, res.ub) == (2.0, 3.0)
    res_r = 1.0 + Interval(1.0, 2.0)
    assert (res_r.lb, res_r.ub) == (2.0, 3.0)


def test_point_interval_arithmetic_reduces_to_float():
    res = Interval.from_point(1.0) + 2.0
    assert isinstance(res, float)
    assert res == pytest.approx(3.0)


def test_reduction_disabled_keeps_interval(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    res = Interval.from_point(1.0) + 2.0
    assert isinstance(res, Interval)
    assert res.lb == 3.0 and res.ub == 3.0


def test_subtraction_and_negation():
    res = Interval(1.0, 2.0) - Interval(0.5, 1.0)
    assert (res.lb, res.ub) == (0.0, 1.5)
    neg = -Interval(1.0, 2.0)
    assert (neg.lb, neg.ub) == (-2.0, -1.0)


def test_multiplication_with_mixed_signs():
    res = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (res.lb, res.ub) == (-4.0, 8.0)
    res_r = 2.0 * Interval(1.0, 2.0)
    assert (res_r.lb, res_r.ub) == (2.0, 4.0)


def test_power_even_and_odd():
    even = Interval(-2.0, 3.0) ** 2
    assert (even.lb, even.ub) == (0.0, 9.0)
    odd = Interval(-2.0, 3.0) ** 3
    assert (odd.lb, odd.ub) == (-8.0, 27.0)


def test_inversion_and_division():
    inv = ~Interval(2.0, 4.0)
    assert (inv.lb, inv.ub) == pytest.approx((0.25, 0.5))
    unbounded = ~Interval(-1.0, 1.0)
    assert unbounded.lb == -math.inf and unbounded.ub == math.inf
    div = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert (div.lb, div.ub) == pytest.approx((0.25, 1.0))
    div_s = Interval(1.0, 2.0) / 2.0
    assert (div_s.lb, div_s.ub) == pytest.approx((0.5, 1.0))


def test_ordering_comparisons_use_lower_bound():
    assert Interval(1.0, 5.0) < Interval(2.0, 3.0)
    assert Interval(1.0, 5.0) < 2.0
    assert Interval(3.0, 4.0) >= Interval(3.0, 10.0)
    assert Interval(3.0, 4.0) > 0.5


def test_try_to_reduce_small_width_returns_middle():
    reduced = Interval(0.0, 1e-6)._try_to_reduce()
    assert isinstance(reduced, float)
    assert reduced == pytest.approx(5e-7)
    wide = Interval(0.0, 1.0)._try_to_reduce()
    assert isinstance(wide, Interval)


def test_interval_copy_is_independent():
    original = Interval(1.0, 2.0)
    duplicate = copy(original)
    assert duplicate is not original
    assert (duplicate.lb, duplicate.ub) == (1.0, 2.0)


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

def test_box_lb_ub_middle_with_mixed_entries():
    box = Box(Interval(0.0, 1.0), 2.0)
    np.testing.assert_array_equal(box.lb, np.array([0.0, 2.0]))
    np.testing.assert_array_equal(box.ub, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(box.middle, np.array([0.5, 2.0]))


def test_box_width_reports_indices_of_maximum():
    box = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), 2.0)
    indices, max_width = box.width
    assert indices == [1]
    assert max_width == 3.0


def test_box_str_and_equality():
    box = Box(Interval(0.0, 1.0), 2.0)
    assert str(box) == "[0.0; 1.0] x 2.0"
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 3.0)


def test_box_equality_rejects_unsupported_type():
    with pytest.raises(NotImplementedError):
        _ = Box(1.0, 2.0) == [1.0, 2.0]


def test_box_copy_creates_equivalent_new_object():
    box = Box(Interval(0.0, 1.0), 2.0)
    duplicate = copy(box)
    assert duplicate is not box
    assert isinstance(duplicate, Box)
    assert str(duplicate) == str(box)


def test_box_try_to_reduce_behaviour():
    degenerate = Box(1.0, 2.0)._try_to_reduce()
    assert not isinstance(degenerate, Box)
    np.testing.assert_array_equal(degenerate, np.array([1.0, 2.0]))

    wide = Box(Interval(0.0, 2.0), 2.0)._try_to_reduce()
    assert isinstance(wide, Box)


# ---------------------------------------------------------------------------
# RS_SubinclusionFunction
# ---------------------------------------------------------------------------

def test_rs_subinclusion_encloses_function_range():
    np.random.seed(0)
    box = Box(Interval(0.0, 1.0), Interval(-1.0, 1.0))
    rs = RS_SubinclusionFunction(lambda x: x[0], number_of_samples=2000)
    result = rs(box)
    assert isinstance(result, Interval)
    assert result.lb >= 0.0
    assert result.ub <= 1.0
    assert result.width > 0.9


def test_rs_subinclusion_is_deterministic_with_seed():
    box = Box(Interval(0.0, 1.0), Interval(0.0, 1.0))
    rs = RS_SubinclusionFunction(lambda x: x[0] + x[1], number_of_samples=100)
    np.random.seed(42)
    first = rs(box)
    np.random.seed(42)
    second = rs(box)
    assert first.lb == second.lb
    assert first.ub == second.ub


# ---------------------------------------------------------------------------
# interval_functions
# ---------------------------------------------------------------------------

def test_sin_of_float_returns_number():
    res = F.sin(0.0)
    assert isinstance(res, float)
    assert res == pytest.approx(0.0, abs=1e-12)


def test_sin_interval_contains_extrema():
    res = F.sin(Interval(0.0, math.pi))
    assert isinstance(res, Interval)
    assert res.lb == pytest.approx(0.0, abs=1e-9)
    assert res.ub == pytest.approx(1.0)


def test_sin_wide_interval_is_unit_range():
    res = F.sin(Interval(0.0, 10.0))
    assert res.lb == -1.0
    assert res.ub == 1.0


def test_abs_of_interval_crossing_zero():
    res = F.abs(Interval(-2.0, 3.0))
    assert (res.lb, res.ub) == (0.0, 3.0)


def test_exp_of_interval_is_monotonic():
    res = F.exp(Interval(0.0, 1.0))
    assert res.lb == pytest.approx(1.0)
    assert res.ub == pytest.approx(math.e)


def test_sqrt_and_log_domain_handling():
    res = F.sqrt(Interval(-1.0, 4.0))
    assert res.lb == pytest.approx(0.0)
    assert res.ub == pytest.approx(2.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        F.sqrt(Interval(-2.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        F.log(Interval(-1.0, 0.0))


def test_split_by_ratios_two_parts():
    parts = F.split(Interval(0.0, 4.0), [1.0, 3.0])
    assert [(p.lb, p.ub) for p in parts] == pytest.approx([(0.0, 1.0), (1.0, 4.0)])


@pytest.mark.xfail(
    strict=True,
    reason="split() reuses the reassigned loop variable 'i', so with 3+ ratios "
           "the subsequent parts are shifted.",
)
def test_split_by_ratios_three_parts():
    parts = F.split(Interval(0.0, 4.0), [1.0, 1.0, 2.0])
    assert [(p.lb, p.ub) for p in parts] == pytest.approx([(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)])


def test_split_of_number_repeats_it():
    assert F.split(2.0, [1.0, 2.0]) == [2.0, 2.0]


def test_bisect_halves_interval():
    left, right = F.bisect(Interval(0.0, 2.0))
    assert (left.lb, left.ub) == pytest.approx((0.0, 1.0))
    assert (right.lb, right.ub) == pytest.approx((1.0, 2.0))


def test_constrain_and_shrink():
    constrained = F.constrain(Interval(-5.0, 5.0), -1.0, 2.0)
    assert (constrained.lb, constrained.ub) == (-1.0, 2.0)
    assert F.constrain(7.0, 0.0, 1.0) == 1.0

    shrunk = F.shrink(Interval(0.0, 4.0), 0.5)
    assert (shrunk.lb, shrunk.ub) == pytest.approx((1.0, 3.0))
    assert F.shrink(3.0, 0.5) == 3.0


def test_hausdorff_distance_intervals_and_numbers():
    assert F.hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == 0.0
    assert F.hausdorff_distance(Interval(0.0, 1.0), Interval(0.5, 2.0)) == pytest.approx(1.0)
    assert F.hausdorff_distance(1.0, 3.0) == pytest.approx(2.0)
