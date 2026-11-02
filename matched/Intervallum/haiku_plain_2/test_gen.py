import math
from copy import copy

import numpy as np
import pytest

from intervallum import interval_functions as ifn
from intervallum.box import Box
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions


# ---------------------------------------------------------------- Box


def test_box_str_and_repr_join_components():
    box = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    assert str(box) == "[0.0; 1.0] x [2.0; 3.0]"
    assert repr(box) == "[0.0; 1.0] x [2.0; 3.0]"
    assert str(Box(1.0, 2.0)) == "1.0 x 2.0"


def test_box_lb_ub_middle_for_mixed_components():
    box = Box(Interval(0.0, 2.0), 5.0)
    np.testing.assert_array_equal(box.lb, np.array([0.0, 5.0]))
    np.testing.assert_array_equal(box.ub, np.array([2.0, 5.0]))
    np.testing.assert_array_equal(box.middle, np.array([1.0, 5.0]))


def test_box_width_reports_widest_component_indices():
    idx, w = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), 4.0).width
    assert idx == [1]
    assert w == 3.0

    idx, w = Box(Interval(0.0, 2.0), Interval(5.0, 7.0), 1.0).width
    assert idx == [0, 1]
    assert w == 2.0

    idx, w = Box(1.0, 2.0).width
    assert idx == [0, 1]
    assert w == 0


def test_box_eq_and_ne_against_arrays_and_boxes():
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    assert not (Box(1.0, 2.0) == np.array([1.0, 2.0, 3.0]))


def test_box_eq_rejects_plain_lists():
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == [1.0, 2.0]


def test_box_try_to_reduce_and_copy():
    reduced = Box(Interval(2.0, 2.0), 5.0)._try_to_reduce()
    np.testing.assert_array_equal(reduced, np.array([2.0, 5.0]))

    wide = Box(Interval(0.0, 1.0), 5.0)
    kept = wide._try_to_reduce()
    assert isinstance(kept, Box)
    assert kept is not wide
    np.testing.assert_array_equal(kept.lb, np.array([0.0, 5.0]))

    cloned = copy(wide)
    assert isinstance(cloned, Box)
    assert cloned[0] is not wide[0]
    assert (cloned[0].lb, cloned[0].ub) == (0.0, 1.0)


# ---------------------------------------------------------------- Interval


def test_interval_reversed_bounds_raise_unless_fixed():
    with pytest.raises(IntervalExceptions.WrongBoundsException) as excinfo:
        Interval(2.0, 1.0)
    assert "Improper interval" in str(excinfo.value)

    fixed = Interval(2.0, 1.0, fix=True)
    assert (fixed.lb, fixed.ub) == (1.0, 2.0)


def test_interval_from_point_str_middle_and_setters():
    point = Interval.from_point(2.0)
    assert (point.lb, point.ub, point.width) == (2.0, 2.0, 0.0)
    assert str(point) == "[2.0; 2.0]"
    assert str(Interval(1.0, 2.5)) == "[1.0; 2.5]"
    assert Interval(1.0, 3.0).middle == 2.0

    i = Interval(0.0, 1.0)
    i.lb = 0.5
    i.ub = 3.0
    assert (i.lb, i.ub) == (0.5, 3.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 4.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0


def test_interval_addition_and_subtraction():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 5.0)

    s = a + b
    assert (s.lb, s.ub) == (4.0, 7.0)

    shifted = a + 1.0
    assert (shifted.lb, shifted.ub) == (2.0, 3.0)

    radd = 1.0 + a
    assert (radd.lb, radd.ub) == (2.0, 3.0)

    d = a - b
    assert (d.lb, d.ub) == (-4.0, -1.0)

    rsub = 10.0 - a
    assert (rsub.lb, rsub.ub) == (8.0, 9.0)


def test_interval_multiplication_division_and_inversion():
    prod = Interval(-1.0, 2.0) * Interval(-3.0, 4.0)
    assert (prod.lb, prod.ub) == (-6.0, 8.0)

    scaled = 3.0 * Interval(1.0, 2.0)
    assert (scaled.lb, scaled.ub) == (3.0, 6.0)

    inv = ~Interval(2.0, 4.0)
    assert (inv.lb, inv.ub) == (0.25, 0.5)

    half_open = ~Interval(0.0, 2.0)
    assert half_open.lb == 0.5
    assert half_open.ub == math.inf

    unbounded = ~Interval(-1.0, 1.0)
    assert unbounded.lb == -math.inf
    assert unbounded.ub == math.inf

    quot = Interval(1.0, 2.0) / 2.0
    assert (quot.lb, quot.ub) == (0.5, 1.0)

    ratio = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert (ratio.lb, ratio.ub) == (0.25, 1.0)

    recip = 1.0 / Interval(2.0, 4.0)
    assert (recip.lb, recip.ub) == (0.25, 0.5)


def test_interval_even_and_odd_powers():
    even = Interval(-2.0, 3.0) ** 2
    assert (even.lb, even.ub) == (0.0, 9.0)

    even_negative = Interval(-2.0, -1.0) ** 2
    assert (even_negative.lb, even_negative.ub) == (1.0, 4.0)

    odd = Interval(-2.0, 3.0) ** 3
    assert (odd.lb, odd.ub) == (-8.0, 27.0)


def test_interval_equality_uses_tolerance_and_rejects_ints():
    assert Interval(0.0, 1.0) == Interval(0.0, 1.0)
    assert Interval(0.0, 1.0) != Interval(0.0, 1.5)
    assert Interval(2.0, 2.0) == 2.0
    assert Interval(2.0, 2.0) != 2.5
    assert Interval(0.0, 1.0) != 0.5
    with pytest.raises(NotImplementedError):
        Interval(0.0, 1.0) == 1


def test_interval_comparisons_use_lower_bounds():
    assert Interval(1.0, 5.0) < Interval(2.0, 3.0)
    assert not (Interval(3.0, 4.0) < Interval(2.0, 9.0))
    assert Interval(1.0, 5.0) < 2.0
    assert Interval(2.0, 3.0) >= 2.0
    assert Interval(1.0, 5.0) <= Interval(1.0, 2.0)
    assert Interval(4.0, 5.0) > Interval(1.0, 9.0)


def test_reduction_flag_turns_point_results_into_floats(monkeypatch):
    point_sum = Interval(1.0, 1.0) + 2.0
    assert isinstance(point_sum, float)
    assert point_sum == 3.0

    wide_sum = Interval(1.0, 2.0) + 2.0
    assert isinstance(wide_sum, Interval)

    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    unreduced = Interval(1.0, 1.0) + 2.0
    assert isinstance(unreduced, Interval)
    assert (unreduced.lb, unreduced.ub) == (3.0, 3.0)


def test_inner_subtraction_orders_reversed_bounds(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)

    plain = Interval.inner_subtraction(Interval(1.0, 2.0), Interval(0.5, 0.5))
    assert (plain.lb, plain.ub) == (0.5, 1.5)

    swapped = Interval.inner_subtraction(Interval(0.0, 1.0), Interval(0.0, 2.0))
    assert (swapped.lb, swapped.ub) == (-1.0, 0.0)

    via_operator = Interval(0.0, 1.0) << Interval(0.0, 2.0)
    assert (via_operator.lb, via_operator.ub) == (-1.0, 0.0)


# ---------------------------------------------------------------- RS_SubinclusionFunction


def test_rs_subinclusion_fixed_sampler_gives_exact_range(monkeypatch):
    calls = []

    def fake_uniform(low, high):
        calls.append((low, high))
        # alternate between the lower corner and the upper corner of the box
        return low if len(calls) % 2 == 1 else high

    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    box = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))
    rs = RS_SubinclusionFunction(lambda p: float(p[0] + p[1]), number_of_samples=4)
    result = rs(box)

    assert len(calls) == 4
    assert isinstance(result, Interval)
    # lower corner (0, 2) gives 2.0, upper corner (1, 4) gives 5.0
    assert (result.lb, result.ub) == (2.0, 5.0)


def test_rs_subinclusion_stays_inside_true_range():
    box = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))
    rs = RS_SubinclusionFunction(lambda p: p[0] - p[1], number_of_samples=200)
    result = rs(box)

    assert isinstance(result, Interval)
    # the true range of x - y over [0, 1] x [2, 4] is [-4, -1]; uniform samples
    # never leave the box, so the enclosure can never leave that range
    assert -4.0 <= result.lb <= result.ub <= -1.0


# ---------------------------------------------------------------- interval_functions


def test_sin_and_cos_enclosures_cover_extrema():
    assert ifn.sin(math.pi / 2) == pytest.approx(1.0)

    rising = ifn.sin(Interval(0.0, 1.0))
    assert rising.lb == pytest.approx(0.0, abs=1e-12)
    assert rising.ub == pytest.approx(math.sin(1.0))

    peak = ifn.sin(Interval(0.0, math.pi))
    assert peak.lb == pytest.approx(0.0, abs=1e-12)
    assert peak.ub == pytest.approx(1.0)

    swing = ifn.sin(Interval(-math.pi / 2, math.pi / 2))
    assert swing.lb == pytest.approx(-1.0)
    assert swing.ub == pytest.approx(1.0)

    full = ifn.sin(Interval(0.0, 2 * math.pi))
    assert (full.lb, full.ub) == (-1.0, 1.0)

    falling = ifn.cos(Interval(0.0, math.pi))
    assert falling.lb == pytest.approx(-1.0)
    assert falling.ub == pytest.approx(1.0)

    cos_small = ifn.cos(Interval(0.0, 1.0))
    assert cos_small.lb == pytest.approx(math.cos(1.0))
    assert cos_small.ub == pytest.approx(1.0)


def test_abs_and_exp_enclosures():
    crossing = ifn.abs(Interval(-2.0, 3.0))
    assert (crossing.lb, crossing.ub) == (0.0, 3.0)

    negative = ifn.abs(Interval(-3.0, -1.0))
    assert (negative.lb, negative.ub) == (1.0, 3.0)

    assert ifn.abs(-2.5) == 2.5

    grown = ifn.exp(Interval(0.0, 1.0))
    assert grown.lb == pytest.approx(1.0)
    assert grown.ub == pytest.approx(math.e)
    assert ifn.exp(0.0) == pytest.approx(1.0)


def test_sqrt_and_log_enclosures_and_domain_errors():
    root = ifn.sqrt(Interval(4.0, 9.0))
    assert (root.lb, root.ub) == (2.0, 3.0)

    clipped_root = ifn.sqrt(Interval(-1.0, 4.0))
    assert (clipped_root.lb, clipped_root.ub) == (0.0, 2.0)

    assert ifn.sqrt(4.0) == 2.0
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.sqrt(Interval(-4.0, -1.0))

    logged = ifn.log(Interval(1.0, math.e))
    assert logged.lb == pytest.approx(0.0, abs=1e-12)
    assert logged.ub == pytest.approx(1.0)

    at_zero = ifn.log(Interval(0.0, 1.0))
    assert at_zero.lb == -math.inf
    assert at_zero.ub == 0.0

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.log(Interval(-2.0, 0.0))


def test_split_and_bisect_two_ratios():
    left, right = ifn.split(Interval(0.0, 4.0), [1.0, 3.0])
    assert (left.lb, left.ub) == (0.0, 1.0)
    assert (right.lb, right.ub) == (1.0, 4.0)

    lo, hi = ifn.bisect(Interval(2.0, 6.0))
    assert (lo.lb, lo.ub) == (2.0, 4.0)
    assert (hi.lb, hi.ub) == (4.0, 6.0)

    assert ifn.split(3.0, [1.0, 1.0]) == [3.0, 3.0]


@pytest.mark.xfail(
    strict=False,
    reason="split() rebinds its loop variable, so the third and later pieces drift",
)
def test_split_three_ratios_gives_equal_pieces():
    parts = ifn.split(Interval(0.0, 6.0), [1.0, 1.0, 1.0])
    assert [p.lb for p in parts] == pytest.approx([0.0, 2.0, 4.0])
    assert [p.ub for p in parts] == pytest.approx([2.0, 4.0, 6.0])


def test_constrain_and_shrink():
    clipped = ifn.constrain(Interval(-1.0, 5.0), 0.0, 3.0)
    assert (clipped.lb, clipped.ub) == (0.0, 3.0)

    pushed = ifn.constrain(Interval(4.0, 5.0), 0.0, 3.0)
    assert (pushed.lb, pushed.ub) == (3.0, 3.0)

    assert ifn.constrain(7.0, 0.0, 3.0) == 3.0
    assert ifn.constrain(-2.0, 0.0, 3.0) == 0.0
    assert ifn.constrain(1.5, 0.0, 3.0) == 1.5

    shrunk = ifn.shrink(Interval(0.0, 4.0), 0.5)
    assert (shrunk.lb, shrunk.ub) == (1.0, 3.0)

    unchanged = ifn.shrink(Interval(0.0, 4.0), 1.0)
    assert (unchanged.lb, unchanged.ub) == (0.0, 4.0)

    assert ifn.shrink(2.5, 0.5) == 2.5


def test_hausdorff_distance_intervals_points_and_symmetry():
    hd = ifn.hausdorff_distance
    assert hd(Interval(0.0, 1.0), Interval(0.0, 1.0)) == 0.0
    assert hd(Interval(0.0, 1.0), Interval(0.0, 2.0)) == 1.0
    assert hd(Interval(0.0, 2.0), Interval(1.0, 3.0)) == 1.0
    assert hd(Interval(0.0, 1.0), Interval(3.0, 4.0)) == 3.0
    assert hd(1.0, 3.0) == 2.0
    assert hd(Interval(0.0, 1.0), 0.5) == 0.5

    for a, b in [(Interval(0.0, 1.0), Interval(0.5, 4.0)), (Interval(-2.0, 0.0), 1.0)]:
        assert hd(a, b) == hd(b, a)
