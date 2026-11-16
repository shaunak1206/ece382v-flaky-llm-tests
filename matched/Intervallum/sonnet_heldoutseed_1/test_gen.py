import math
from copy import copy

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import hausdorff_distance


# ---------------------------------------------------------------- Interval

def test_interval_basic_properties():
    i = Interval(1.0, 4.0)
    assert i.lb == 1.0
    assert i.ub == 4.0
    assert i.width == 3.0
    assert i.middle == 2.5
    assert str(i) == "[1.0; 4.0]"
    assert repr(i) == str(i)


def test_interval_wrong_bounds_raises_and_fix_swaps():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    i = Interval(2.0, 1.0, fix=True)
    assert (i.lb, i.ub) == (1.0, 2.0)


def test_interval_from_point_and_copy():
    p = Interval.from_point(3.0)
    assert p.lb == p.ub == 3.0
    i = Interval(0.0, 1.0)
    c = copy(i)
    c.ub = 5.0
    assert i.ub == 1.0
    assert c.ub == 5.0


def test_interval_bound_setters_validate():
    i = Interval(0.0, 1.0)
    i.lb = 0.5
    i.ub = 2.0
    assert (i.lb, i.ub) == (0.5, 2.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0
    assert (i.lb, i.ub) == (0.5, 2.0)


def test_interval_addition_and_subtraction():
    s = Interval(1.0, 2.0) + Interval(3.0, 5.0)
    assert (s.lb, s.ub) == (4.0, 7.0)
    s = Interval(1.0, 2.0) + 1.0
    assert (s.lb, s.ub) == (2.0, 3.0)
    s = 1.0 + Interval(1.0, 2.0)
    assert (s.lb, s.ub) == (2.0, 3.0)
    d = Interval(1.0, 2.0) - Interval(0.0, 1.0)
    assert (d.lb, d.ub) == (0.0, 2.0)
    r = 10.0 - Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (8.0, 9.0)


def test_interval_negation():
    n = -Interval(1.0, 3.0)
    assert (n.lb, n.ub) == (-3.0, -1.0)


def test_interval_multiplication():
    m = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (m.lb, m.ub) == (-4.0, 8.0)
    m = Interval(1.0, 2.0) * -2.0
    assert (m.lb, m.ub) == (-4.0, -2.0)
    m = 3.0 * Interval(1.0, 2.0)
    assert (m.lb, m.ub) == (3.0, 6.0)


def test_interval_division_and_inversion():
    q = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert q.lb == pytest.approx(0.25)
    assert q.ub == pytest.approx(1.0)

    inv = ~Interval(0.0, 2.0)
    assert inv.lb == 0.5
    assert inv.ub == math.inf

    inv = ~Interval(-1.0, 1.0)
    assert inv.lb == -math.inf
    assert inv.ub == math.inf

    q = Interval(2.0, 4.0) / 2.0
    assert (q.lb, q.ub) == (1.0, 2.0)


def test_interval_power_even_and_odd():
    sq = Interval(-2.0, 3.0) ** 2
    assert (sq.lb, sq.ub) == (0.0, 9.0)
    sq = Interval(2.0, 3.0) ** 2
    assert (sq.lb, sq.ub) == (4.0, 9.0)
    cube = Interval(-2.0, 3.0) ** 3
    assert (cube.lb, cube.ub) == (-8.0, 27.0)


def test_interval_narrow_result_is_reduced_to_number():
    r = Interval.from_point(1.0) + 1.0
    assert isinstance(r, float)
    assert r == 2.0
    r = Interval(1.0, 1.0 + 1e-8) * 1.0
    assert isinstance(r, float)
    assert r == pytest.approx(1.0)


def test_interval_reduction_can_be_disabled(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    r = Interval.from_point(1.0) + 1.0
    assert isinstance(r, Interval)
    assert (r.lb, r.ub) == (2.0, 2.0)


def test_interval_equality_and_inequality():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 3.0)
    assert Interval(1.0, 2.0) != 1.5
    assert Interval.from_point(2.0) == 2.0
    with pytest.raises(NotImplementedError):
        Interval(1.0, 2.0) == "abc"


def test_interval_ordering_uses_lower_bounds():
    a, b = Interval(1.0, 5.0), Interval(2.0, 3.0)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert a < 1.5
    assert not (a > 1.0)
    assert a >= 1.0


def test_inner_subtraction_with_numbers_and_intervals():
    r = Interval.inner_subtraction(5.0, 3.0)
    assert r == pytest.approx(2.0)
    assert isinstance(r, float)
    r = Interval(1.0, 5.0) << Interval(0.0, 1.0)
    assert (r.lb, r.ub) == (1.0, 4.0)


# --------------------------------------------------------------------- Box

def test_box_str_and_lengths():
    b = Box(Interval(0, 1), Interval(2, 3))
    assert len(b) == 2
    assert str(b) == "[0; 1] x [2; 3]"
    f = Box(1.0, 2.0)
    assert str(f) == "1.0 x 2.0"


def test_box_bounds_and_middle_with_mixed_elements():
    b = Box(Interval(0.0, 2.0), 5.0, Interval(-1.0, 1.0))
    np.testing.assert_array_equal(b.lb, [0.0, 5.0, -1.0])
    np.testing.assert_array_equal(b.ub, [2.0, 5.0, 1.0])
    np.testing.assert_array_equal(b.middle, [1.0, 5.0, 0.0])


def test_box_width_returns_indices_of_widest_components():
    b = Box(Interval(0.0, 2.0), Interval(1.0, 3.0), 5.0, Interval(0.0, 1.0))
    indices, w = b.width
    assert indices == [0, 1]
    assert w == 2.0


def test_box_equality_and_inequality():
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    assert Box(Interval(0.0, 1.0), Interval(2.0, 3.0)) == Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == 5


def test_box_copy_is_deep_for_intervals():
    b = Box(Interval(0.0, 1.0), 2.0)
    c = copy(b)
    assert isinstance(c, Box)
    c[0].ub = 10.0
    assert b[0].ub == 1.0
    assert c[0].ub == 10.0


def test_box_try_to_reduce():
    degenerate = Box(Interval(1.0, 1.0), 2.0)
    reduced = degenerate._try_to_reduce()
    np.testing.assert_array_equal(reduced, [1.0, 2.0])

    wide = Box(Interval(0.0, 1.0), 2.0)
    kept = wide._try_to_reduce()
    assert isinstance(kept, Box)
    assert kept[0].lb == 0.0 and kept[0].ub == 1.0


# ------------------------------------------------- RS_SubinclusionFunction

def test_rs_subinclusion_encloses_sampled_values():
    np.random.seed(0)
    f = RS_SubinclusionFunction(lambda p: p[0] + p[1], 200)
    result = f(Box(Interval(0.0, 1.0), Interval(2.0, 3.0)))
    assert isinstance(result, Interval)
    assert 2.0 <= result.lb <= result.ub <= 4.0
    # with 200 samples the estimate should cover most of the true range [2, 4]
    assert result.width > 1.0


def test_rs_subinclusion_calls_function_given_number_of_times():
    calls = []

    def g(p):
        calls.append(p)
        return float(p[0])

    f = RS_SubinclusionFunction(g, 17)
    result = f(Box(Interval(0.0, 1.0), 2.0))
    assert len(calls) == 17
    assert all(len(p) == 2 and p[1] == 2.0 for p in calls)
    assert 0.0 <= result.lb <= result.ub <= 1.0


def test_rs_subinclusion_constant_function_gives_point_interval():
    f = RS_SubinclusionFunction(lambda p: 5.0, 10)
    result = f(Box(Interval(0.0, 1.0), Interval(0.0, 1.0)))
    assert result.lb == 5.0
    assert result.ub == 5.0


# ------------------------------------------------------ hausdorff_distance

def test_hausdorff_distance_intervals():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 3.0)) == 2.0
    assert hausdorff_distance(Interval(0.0, 3.0), Interval(0.0, 1.0)) == 2.0
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 4.0)) == 3.0
    assert hausdorff_distance(Interval(1.0, 2.0), Interval(1.0, 2.0)) == 0.0


def test_hausdorff_distance_with_numbers():
    assert hausdorff_distance(1.0, 4.0) == 3.0
    assert hausdorff_distance(4.0, 1.0) == 3.0
    assert hausdorff_distance(2.0, 2.0) == 0.0
    assert hausdorff_distance(0.0, Interval(-1.0, 3.0)) == 3.0
