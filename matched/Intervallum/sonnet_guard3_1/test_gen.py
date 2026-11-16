import math

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import hausdorff_distance


# ---------------------------------------------------------------- Interval

def test_interval_construction_and_properties():
    i = Interval(1.0, 4.0)
    assert i.lb == 1.0
    assert i.ub == 4.0
    assert i.middle == pytest.approx(2.5)
    assert i.width == pytest.approx(3.0)
    assert str(i) == "[1.0; 4.0]"
    assert repr(i) == str(i)


def test_interval_wrong_bounds_raise_and_fix_swaps():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    fixed = Interval(2.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 2.0


def test_interval_from_point():
    p = Interval.from_point(3.0)
    assert p.lb == 3.0
    assert p.ub == 3.0
    assert p.width == 0.0


def test_interval_setters_validate_bounds():
    i = Interval(0.0, 1.0)
    i.lb = 0.5
    i.ub = 2.0
    assert (i.lb, i.ub) == (0.5, 2.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0
    assert (i.lb, i.ub) == (0.5, 2.0)


def test_interval_addition_subtraction_negation():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 5.0)
    s = a + b
    assert (s.lb, s.ub) == (pytest.approx(4.0), pytest.approx(7.0))
    d = a - Interval(0.0, 1.0)
    assert (d.lb, d.ub) == (pytest.approx(0.0), pytest.approx(2.0))
    n = -a
    assert (n.lb, n.ub) == (pytest.approx(-2.0), pytest.approx(-1.0))
    r = 10.0 - a
    assert (r.lb, r.ub) == (pytest.approx(8.0), pytest.approx(9.0))
    sc = 1.0 + a
    assert (sc.lb, sc.ub) == (pytest.approx(2.0), pytest.approx(3.0))


def test_interval_multiplication():
    m = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (m.lb, m.ub) == (pytest.approx(-4.0), pytest.approx(8.0))
    neg = Interval(1.0, 2.0) * -2.0
    assert (neg.lb, neg.ub) == (pytest.approx(-4.0), pytest.approx(-2.0))
    rm = 3.0 * Interval(1.0, 2.0)
    assert (rm.lb, rm.ub) == (pytest.approx(3.0), pytest.approx(6.0))


def test_interval_inversion_and_division():
    inv = ~Interval(2.0, 4.0)
    assert (inv.lb, inv.ub) == (pytest.approx(0.25), pytest.approx(0.5))

    inv0 = ~Interval(0.0, 2.0)
    assert inv0.lb == pytest.approx(0.5)
    assert inv0.ub == math.inf

    inv_all = ~Interval(-1.0, 1.0)
    assert inv_all.lb == -math.inf
    assert inv_all.ub == math.inf

    q = Interval(2.0, 4.0) / 2.0
    assert (q.lb, q.ub) == (pytest.approx(1.0), pytest.approx(2.0))
    q2 = Interval(2.0, 4.0) / Interval(1.0, 2.0)
    assert (q2.lb, q2.ub) == (pytest.approx(1.0), pytest.approx(4.0))
    rq = 1.0 / Interval(2.0, 4.0)
    assert (rq.lb, rq.ub) == (pytest.approx(0.25), pytest.approx(0.5))


def test_interval_power_even_and_odd():
    e = Interval(-2.0, 3.0) ** 2
    assert (e.lb, e.ub) == (pytest.approx(0.0), pytest.approx(9.0))
    e2 = Interval(1.0, 2.0) ** 2
    assert (e2.lb, e2.ub) == (pytest.approx(1.0), pytest.approx(4.0))
    o = Interval(-2.0, 3.0) ** 3
    assert (o.lb, o.ub) == (pytest.approx(-8.0), pytest.approx(27.0))


def test_interval_narrow_result_reduced_to_number():
    r = Interval(0.0, 1e-7) + Interval(0.0, 1e-7)
    assert not isinstance(r, Interval)
    assert r == pytest.approx(1e-7)


def test_interval_reduction_can_be_disabled(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    r = Interval(0.0, 1e-7) + Interval(0.0, 1e-7)
    assert isinstance(r, Interval)
    assert r.ub == pytest.approx(2e-7)


def test_interval_inner_subtraction():
    r = Interval(1.0, 3.0) << Interval(0.0, 1.0)
    assert (r.lb, r.ub) == (pytest.approx(1.0), pytest.approx(2.0))
    swapped = Interval(1.0, 2.0) << Interval(0.0, 5.0)
    assert (swapped.lb, swapped.ub) == (pytest.approx(-3.0), pytest.approx(1.0))


def test_interval_equality():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 3.0)
    assert not (Interval(1.0, 2.0) != Interval(1.0, 2.0))
    with pytest.raises(NotImplementedError):
        Interval(1.0, 2.0) == "abc"


def test_interval_ordering_compares_lower_bounds():
    assert Interval(1.0, 2.0) < Interval(1.5, 2.0)
    assert Interval(1.0, 5.0) <= Interval(1.0, 2.0)
    assert Interval(2.0, 3.0) > Interval(1.0, 10.0)
    assert Interval(2.0, 3.0) >= 2.0
    assert Interval(1.0, 2.0) < 3.0
    assert not (Interval(1.0, 2.0) > 3.0)


# --------------------------------------------------------------------- Box

def test_box_properties_with_intervals():
    b = Box(Interval(0.0, 2.0), Interval(1.0, 5.0))
    assert np.allclose(b.lb, [0.0, 1.0])
    assert np.allclose(b.ub, [2.0, 5.0])
    assert np.allclose(b.middle, [1.0, 3.0])


def test_box_properties_with_mixed_numbers_and_intervals():
    b = Box(Interval(0.0, 1.0), 2.0)
    assert np.allclose(b.lb, [0.0, 2.0])
    assert np.allclose(b.ub, [1.0, 2.0])
    assert np.allclose(b.middle, [0.5, 2.0])


def test_box_width_returns_indices_of_widest_components():
    b = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), Interval(1.0, 4.0))
    idx, w = b.width
    assert idx == [1, 2]
    assert w == pytest.approx(3.0)
    idx0, w0 = Box(1.0, 2.0).width
    assert idx0 == [0, 1]
    assert w0 == 0


def test_box_str_and_repr():
    b = Box(Interval(0, 1), Interval(2, 3))
    assert str(b) == "[0; 1] x [2; 3]"
    assert repr(b) == "[0; 1] x [2; 3]"


def test_box_equality_and_inequality():
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    assert Box(1.0, 2.0) != Box(1.0, 2.0, 3.0)
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == 5


def test_box_copy_is_independent_and_try_to_reduce():
    from copy import copy
    original = Box(Interval(0.0, 1.0), 2.0)
    dup = copy(original)
    assert isinstance(dup, Box)
    assert dup == original
    dup[0].ub = 10.0
    assert original[0].ub == 1.0

    # non-degenerate box is copied, not reduced
    reduced = original._try_to_reduce()
    assert isinstance(reduced, Box)
    assert reduced == original

    # degenerate box reduces to the array of middles
    point_box = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    r = point_box._try_to_reduce()
    assert np.allclose(r, [1.0, 2.0])
    assert np.allclose(Box(3.0, 4.0)._try_to_reduce(), [3.0, 4.0])


# ----------------------------------------------------- RS_SubinclusionFunction

def test_rs_subinclusion_result_stays_within_true_range():
    np.random.seed(12345)
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    f = RS_SubinclusionFunction(lambda p: p[0] + p[1], 50)
    result = f(b)
    assert isinstance(result, Interval)
    assert 2.0 <= result.lb <= result.ub <= 4.0


def test_rs_subinclusion_single_sample_and_point_box():
    np.random.seed(7)
    f1 = RS_SubinclusionFunction(lambda p: float(p[0]), 1)
    r1 = f1(Box(Interval(0.0, 1.0)))
    assert r1.lb == r1.ub
    assert 0.0 <= r1.lb <= 1.0

    f2 = RS_SubinclusionFunction(lambda p: float(p[0] + p[1]), 5)
    r2 = f2(Box(1.0, 2.0))
    assert r2.lb == pytest.approx(3.0)
    assert r2.ub == pytest.approx(3.0)


def test_rs_subinclusion_uses_requested_number_of_samples(monkeypatch):
    np.random.seed(0)
    calls = []

    def fake_uniform(low, high):
        calls.append((np.asarray(low).copy(), np.asarray(high).copy()))
        return (np.asarray(low) + np.asarray(high)) / 2.0

    monkeypatch.setattr(np.random, "uniform", fake_uniform)
    f = RS_SubinclusionFunction(lambda p: float(p[0] * 2.0), 7)
    result = f(Box(Interval(0.0, 2.0), Interval(1.0, 3.0)))
    assert len(calls) == 7
    assert np.allclose(calls[0][0], [0.0, 1.0])
    assert np.allclose(calls[0][1], [2.0, 3.0])
    assert result.lb == pytest.approx(2.0)
    assert result.ub == pytest.approx(2.0)


# ------------------------------------------------------- hausdorff_distance

def test_hausdorff_distance_intervals():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == pytest.approx(2.0)
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == pytest.approx(0.0)
    # nested intervals
    a, b = Interval(0.0, 4.0), Interval(1.0, 2.0)
    assert hausdorff_distance(a, b) == pytest.approx(2.0)
    assert hausdorff_distance(b, a) == pytest.approx(2.0)


def test_hausdorff_distance_with_numbers():
    assert hausdorff_distance(1.0, 4.0) == pytest.approx(3.0)
    assert hausdorff_distance(1.0, Interval(0.0, 3.0)) == pytest.approx(2.0)
    assert hausdorff_distance(Interval(0.0, 3.0), 1.0) == pytest.approx(2.0)
    assert hausdorff_distance(2.0, 2.0) == pytest.approx(0.0)
