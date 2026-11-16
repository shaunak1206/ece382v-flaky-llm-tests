from copy import copy

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import hausdorff_distance


# ---------------------------------------------------------------- Interval

def test_interval_basic_properties():
    i = Interval(1.0, 3.0)
    assert i.lb == 1.0
    assert i.ub == 3.0
    assert i.middle == pytest.approx(2.0)
    assert i.width == pytest.approx(2.0)
    assert str(i) == "[1.0; 3.0]"
    assert repr(i) == "[1.0; 3.0]"


def test_interval_wrong_bounds_raises_and_fix_swaps():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    fixed = Interval(2.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 2.0


def test_interval_from_point():
    i = Interval.from_point(4.5)
    assert i.lb == 4.5
    assert i.ub == 4.5
    assert i.width == 0


def test_interval_bound_setters_validate():
    i = Interval(1.0, 2.0)
    i.lb = 0.5
    i.ub = 3.0
    assert (i.lb, i.ub) == (0.5, 3.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 10.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = -10.0
    assert (i.lb, i.ub) == (0.5, 3.0)


def test_interval_copy_is_independent():
    i = Interval(1.0, 2.0)
    c = copy(i)
    assert c is not i
    c.ub = 5.0
    assert i.ub == 2.0
    assert c.ub == 5.0


def test_interval_addition_and_subtraction():
    a = Interval(1.0, 3.0)
    b = Interval(0.0, 1.0)
    s = a + b
    assert (s.lb, s.ub) == (1.0, 4.0)
    d = a - b
    assert (d.lb, d.ub) == (0.0, 3.0)
    shifted = a + 2.0
    assert (shifted.lb, shifted.ub) == (3.0, 5.0)
    radd = 2.0 + a
    assert (radd.lb, radd.ub) == (3.0, 5.0)
    rsub = 5.0 - Interval(1.0, 2.0)
    assert (rsub.lb, rsub.ub) == (3.0, 4.0)


def test_interval_negation():
    n = -Interval(1.0, 2.0)
    assert (n.lb, n.ub) == (-2.0, -1.0)


def test_interval_multiplication():
    m = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (m.lb, m.ub) == (-4.0, 8.0)
    neg = Interval(1.0, 2.0) * -2.0
    assert (neg.lb, neg.ub) == (-4.0, -2.0)
    r = 3.0 * Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (3.0, 6.0)


def test_interval_inversion_cases():
    inv = ~Interval(2.0, 4.0)
    assert inv.lb == pytest.approx(0.25)
    assert inv.ub == pytest.approx(0.5)

    from_zero = ~Interval(0.0, 2.0)
    assert from_zero.lb == pytest.approx(0.5)
    assert from_zero.ub == float("inf")

    to_zero = ~Interval(-2.0, 0.0)
    assert to_zero.lb == float("-inf")
    assert to_zero.ub == pytest.approx(-0.5)

    around_zero = ~Interval(-1.0, 1.0)
    assert around_zero.lb == float("-inf")
    assert around_zero.ub == float("inf")


def test_interval_division():
    by_scalar = Interval(2.0, 4.0) / 2.0
    assert (by_scalar.lb, by_scalar.ub) == (1.0, 2.0)

    by_interval = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert by_interval.lb == pytest.approx(0.25)
    assert by_interval.ub == pytest.approx(1.0)

    reciprocal = 1.0 / Interval(2.0, 4.0)
    assert reciprocal.lb == pytest.approx(0.25)
    assert reciprocal.ub == pytest.approx(0.5)


def test_interval_power_even_and_odd():
    even = Interval(-2.0, 3.0) ** 2
    assert (even.lb, even.ub) == (0.0, 9.0)

    even_pos = Interval(1.0, 2.0) ** 2
    assert (even_pos.lb, even_pos.ub) == (1.0, 4.0)

    odd = Interval(-2.0, 3.0) ** 3
    assert (odd.lb, odd.ub) == (-8.0, 27.0)

    odd_neg = Interval(-2.0, -1.0) ** 3
    assert (odd_neg.lb, odd_neg.ub) == (-8.0, -1.0)


def test_interval_narrow_result_reduced_to_number():
    result = Interval.from_point(1.0) + 1.0
    assert isinstance(result, float)
    assert result == pytest.approx(2.0)

    narrow = Interval(1.0, 1.000001) + 0.0
    assert isinstance(narrow, float)
    assert narrow == pytest.approx(1.0000005)

    wide = Interval(1.0, 2.0) + 0.0
    assert isinstance(wide, Interval)


def test_interval_reduction_can_be_disabled(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    result = Interval.from_point(1.0) + 1.0
    assert isinstance(result, Interval)
    assert result.lb == 2.0
    assert result.ub == 2.0


def test_interval_inner_subtraction():
    d = Interval(1.0, 3.0) << Interval(0.0, 1.0)
    assert (d.lb, d.ub) == (1.0, 2.0)
    # swapped bounds are fixed
    d2 = Interval.inner_subtraction(Interval(0.0, 3.0), Interval(1.0, 1.0 + 0.0))
    assert isinstance(d2, Interval)
    assert (d2.lb, d2.ub) == (-1.0, 2.0)


def test_interval_equality_and_ordering():
    a = Interval(1.0, 2.0)
    assert a == Interval(1.0, 2.0)
    assert a == Interval(1.0, 2.0 + 1e-9)
    assert a != Interval(1.0, 3.0)
    assert Interval.from_point(1.0) == 1.0
    assert a != 1.0
    with pytest.raises(NotImplementedError):
        a == "text"

    assert Interval(1.0, 2.0) < Interval(2.0, 3.0)
    assert Interval(2.0, 3.0) > Interval(1.0, 5.0)
    assert Interval(1.0, 2.0) <= 1.0
    assert Interval(1.0, 2.0) >= 1.0
    assert not (Interval(1.0, 2.0) > 1.5)


# --------------------------------------------------------------------- Box

def test_box_bounds_middle_and_str():
    b = Box(Interval(0.0, 2.0), 5.0, Interval(-1.0, 1.0))
    assert np.array_equal(b.lb, np.array([0.0, 5.0, -1.0]))
    assert np.array_equal(b.ub, np.array([2.0, 5.0, 1.0]))
    assert np.array_equal(b.middle, np.array([1.0, 5.0, 0.0]))
    assert len(b) == 3

    s = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))
    assert str(s) == "[0.0; 1.0] x [2.0; 4.0]"


def test_box_width_returns_widest_indices():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 5.0), 3.0)
    idx, w = b.width
    assert idx == [1]
    assert w == pytest.approx(3.0)

    tie = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    idx, w = tie.width
    assert idx == [0, 1]
    assert w == pytest.approx(1.0)

    points = Box(1.0, 2.0)
    idx, w = points.width
    assert idx == [0, 1]
    assert w == 0


def test_box_equality_and_inequality():
    a = Box(1.0, 2.0)
    assert a == Box(1.0, 2.0)
    assert a != Box(1.0, 3.0)
    assert a == np.array([1.0, 2.0])
    with pytest.raises(NotImplementedError):
        a == 5


def test_box_copy_creates_independent_intervals():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    c = copy(b)
    assert isinstance(c, Box)
    assert c[0] is not b[0]
    c[0].ub = 10.0
    assert b[0].ub == 1.0


def test_box_try_to_reduce():
    degenerate = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    reduced = degenerate._try_to_reduce()
    assert np.array_equal(reduced, np.array([1.0, 2.0]))

    proper = Box(Interval(0.0, 1.0), 2.0)
    kept = proper._try_to_reduce()
    assert isinstance(kept, Box)
    assert kept.width[1] == pytest.approx(1.0)


# ---------------------------------------------------- RS_SubinclusionFunction

def test_rs_subinclusion_result_within_true_range():
    np.random.seed(0)
    f = RS_SubinclusionFunction(lambda p: p[0] + p[1], 200)
    result = f(Box(Interval(0.0, 1.0), Interval(0.0, 1.0)))
    assert isinstance(result, Interval)
    assert 0.0 <= result.lb <= result.ub <= 2.0
    # with 200 samples the estimate should cover a decent part of [0, 2]
    assert result.width > 1.0


def test_rs_subinclusion_calls_function_requested_number_of_times():
    calls = []

    def g(p):
        calls.append(1)
        return 3.0

    f = RS_SubinclusionFunction(g, 17)
    result = f(Box(Interval(2.0, 3.0), Interval(-1.0, 1.0)))
    assert len(calls) == 17
    assert result.lb == 3.0
    assert result.ub == 3.0


def test_rs_subinclusion_degenerate_box_gives_exact_value():
    f = RS_SubinclusionFunction(lambda p: p[0] * 2.0, 5)
    result = f(Box(4.0))
    assert result.lb == pytest.approx(8.0)
    assert result.ub == pytest.approx(8.0)


# ------------------------------------------------------- hausdorff_distance

def test_hausdorff_distance_intervals():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 4.0)) == pytest.approx(3.0)
    assert hausdorff_distance(Interval(2.0, 4.0), Interval(0.0, 1.0)) == pytest.approx(3.0)
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == 0.0
    # nested intervals
    assert hausdorff_distance(Interval(0.0, 4.0), Interval(1.0, 2.0)) == pytest.approx(2.0)


def test_hausdorff_distance_with_numbers():
    assert hausdorff_distance(1.0, 4.0) == pytest.approx(3.0)
    assert hausdorff_distance(4.0, 1.0) == pytest.approx(3.0)
    assert hausdorff_distance(2.0, 2.0) == 0.0
    assert hausdorff_distance(Interval(0.0, 2.0), 1.0) == pytest.approx(1.0)
    assert hausdorff_distance(1.0, Interval(0.0, 2.0)) == pytest.approx(1.0)
