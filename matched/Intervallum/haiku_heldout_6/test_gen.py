import math
from copy import copy

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.interval_functions import hausdorff_distance


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------

def test_interval_bounds_width_and_middle():
    i = Interval(1.0, 3.0)
    assert i.lb == 1.0
    assert i.ub == 3.0
    assert i.width == 2.0
    assert i.middle == 2.0


def test_interval_wrong_bounds_raises():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(3.0, 1.0)


def test_interval_fix_swaps_bounds_and_from_point():
    fixed = Interval(3.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 3.0

    point = Interval.from_point(2.5)
    assert isinstance(point, Interval)
    assert point.lb == 2.5
    assert point.ub == 2.5
    assert point.width == 0.0


def test_bound_setters_validate_order():
    i = Interval(1.0, 2.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0
    i.lb = 0.5
    assert i.lb == 0.5
    assert i.ub == 2.0


def test_add_intervals_and_scalars():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 4.0)

    r = a + b
    assert isinstance(r, Interval)
    assert r.lb == 4.0
    assert r.ub == 6.0

    r2 = a + 1.0
    assert isinstance(r2, Interval)
    assert (r2.lb, r2.ub) == (2.0, 3.0)

    r3 = 1.0 + a
    assert isinstance(r3, Interval)
    assert (r3.lb, r3.ub) == (2.0, 3.0)


def test_degenerate_result_reduces_to_float_and_reduction_can_be_disabled(monkeypatch):
    # Degenerate result is reduced to its middle value by default.
    reduced = Interval(1.0, 1.0) + 0.0
    assert isinstance(reduced, float)
    assert reduced == 1.0

    # Width below the reduction threshold is also reduced to the middle.
    nearly_point = Interval(1.0, 1.0 + 1e-6) + 0.0
    assert isinstance(nearly_point, float)
    assert nearly_point == pytest.approx(1.0 + 5e-7)

    # With reduction disabled, the Interval is preserved.
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    kept = Interval(1.0, 1.0) + 0.0
    assert isinstance(kept, Interval)
    assert (kept.lb, kept.ub) == (1.0, 1.0)


def test_mul_with_mixed_signs_and_scalars():
    r = Interval(1.0, 2.0) * Interval(3.0, 4.0)
    assert (r.lb, r.ub) == (3.0, 8.0)

    r = Interval(-1.0, 2.0) * Interval(-3.0, 4.0)
    assert (r.lb, r.ub) == (-6.0, 8.0)

    r = Interval(1.0, 2.0) * -1
    assert (r.lb, r.ub) == (-2.0, -1.0)

    r = 2 * Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (2.0, 4.0)


def test_neg_sub_and_rsub():
    neg = -Interval(1.0, 2.0)
    assert (neg.lb, neg.ub) == (-2.0, -1.0)

    diff = Interval(3.0, 5.0) - Interval(1.0, 2.0)
    assert (diff.lb, diff.ub) == (1.0, 4.0)

    rdiff = 5.0 - Interval(1.0, 2.0)
    assert (rdiff.lb, rdiff.ub) == (3.0, 4.0)


def test_division_and_invert():
    q = Interval(2.0, 4.0) / 2
    assert (q.lb, q.ub) == (1.0, 2.0)

    q = Interval(2.0, 4.0) / Interval(1.0, 2.0)
    assert (q.lb, q.ub) == (1.0, 4.0)

    inv = ~Interval(2.0, 4.0)
    assert (inv.lb, inv.ub) == (0.25, 0.5)

    unbounded = ~Interval(-1.0, 1.0)
    assert unbounded.lb == -math.inf
    assert unbounded.ub == math.inf


def test_pow_even_and_odd_exponents():
    # Even exponent crossing zero: minimum must be 0.
    even = Interval(-2.0, 3.0) ** 2
    assert isinstance(even, Interval)
    assert even.lb == 0.0
    assert even.ub == 9

    # Odd exponent is monotonic.
    odd = Interval(-2.0, 3.0) ** 3
    assert odd.lb == -8.0
    assert odd.ub == 27.0


def test_eq_uses_admissible_error_tolerance():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0 + 1e-9)
    assert not (Interval(1.0, 2.0) != Interval(1.0, 2.0 + 1e-9))
    assert Interval(1.0, 2.0) != Interval(1.0, 3.0)
    assert not (Interval(1.0, 2.0) == Interval(1.0, 3.0))


def test_eq_with_float():
    assert Interval(1.0, 1.0) == 1.0
    assert Interval(1.0, 2.0) != 1.0
    assert not (Interval(1.0, 2.0) == 1.0)


def test_eq_with_unsupported_type_raises():
    with pytest.raises(NotImplementedError):
        _ = Interval(1.0, 2.0) == "not an interval"


def test_ordering_compares_lower_bounds():
    assert Interval(1.0, 5.0) < Interval(2.0, 3.0)
    assert Interval(2.0, 3.0) >= Interval(2.0, 9.0)
    assert Interval(1.0, 2.0) < 1.5
    assert Interval(3.0, 4.0) > 2.0


def test_str_repr_and_copy_are_independent():
    i = Interval(1.0, 2.0)
    assert str(i) == "[1.0; 2.0]"
    assert repr(i) == "[1.0; 2.0]"

    c = copy(i)
    assert c is not i
    assert c.lb == 1.0
    assert c.ub == 2.0
    c.lb = 0.0
    assert i.lb == 1.0


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

def test_box_lb_ub_and_middle_with_intervals_and_floats():
    b = Box(Interval(1.0, 3.0), 5.0)
    np.testing.assert_array_equal(b.lb, np.array([1.0, 5.0]))
    np.testing.assert_array_equal(b.ub, np.array([3.0, 5.0]))
    np.testing.assert_array_equal(b.middle, np.array([2.0, 5.0]))


def test_box_width_reports_indices_of_widest_components():
    b = Box(Interval(1.0, 3.0), Interval(0.0, 1.0), 4.0)
    indices, max_width = b.width
    assert indices == [0]
    assert max_width == 2.0

    # Pure floats have zero width in every component.
    indices, max_width = Box(1.0, 2.0).width
    assert indices == [0, 1]
    assert max_width == 0


def test_box_str_and_equality():
    b = Box(Interval(1.0, 2.0), 3.0)
    assert str(b) == "[1.0; 2.0] x 3.0"

    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 2.5)
    assert not (Box(1.0, 2.0) == Box(1.0, 2.0, 3.0))

    with pytest.raises(NotImplementedError):
        _ = Box(1.0, 2.0) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# RS_SubinclusionFunction
# ---------------------------------------------------------------------------

def test_rs_subinclusion_bounds_enclose_true_range():
    def f(x):
        return x[0] + x[1]

    np.random.seed(0)
    rs = RS_SubinclusionFunction(f, 5000)
    box = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    result = rs(box)

    assert isinstance(result, Interval)
    # True range of x0 + x1 over the box is [2, 4].
    assert result.lb >= 2.0
    assert result.ub <= 4.0
    assert result.lb < 2.1
    assert result.ub > 3.9


def test_rs_calls_function_number_of_samples_times():
    calls = []

    def f(x):
        calls.append(1)
        return 1.0

    rs = RS_SubinclusionFunction(f, 37)
    result = rs(Box(Interval(0.0, 1.0)))

    assert len(calls) == 37
    assert isinstance(result, Interval)
    assert result.lb == 1.0
    assert result.ub == 1.0


# ---------------------------------------------------------------------------
# hausdorff_distance
# ---------------------------------------------------------------------------

def test_hausdorff_distance_disjoint_and_overlapping_intervals_are_symmetric():
    a = Interval(0.0, 1.0)
    b = Interval(2.0, 3.0)
    assert hausdorff_distance(a, b) == pytest.approx(2.0)
    assert hausdorff_distance(b, a) == pytest.approx(2.0)

    c = Interval(0.0, 2.0)
    d = Interval(1.0, 3.0)
    assert hausdorff_distance(c, d) == pytest.approx(1.0)
    assert hausdorff_distance(d, c) == pytest.approx(1.0)

    assert hausdorff_distance(a, Interval(0.0, 1.0)) == pytest.approx(0.0)


def test_hausdorff_distance_with_scalars_and_point_intervals():
    assert hausdorff_distance(1.0, 3.0) == pytest.approx(2.0)
    assert hausdorff_distance(3.0, 1.0) == pytest.approx(2.0)
    assert hausdorff_distance(Interval(0.0, 4.0), 2.0) == pytest.approx(2.0)
    assert hausdorff_distance(2.0, 2.0) == pytest.approx(0.0)
