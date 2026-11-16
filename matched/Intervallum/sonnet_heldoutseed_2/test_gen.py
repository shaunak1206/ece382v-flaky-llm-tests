import copy as copy_module

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import hausdorff_distance


# ---------------------------------------------------------------- Interval

def test_interval_construction_and_properties():
    i = Interval(1.0, 3.0)
    assert i.lb == 1.0
    assert i.ub == 3.0
    assert i.middle == 2.0
    assert i.width == 2.0
    p = Interval.from_point(2.5)
    assert p.lb == p.ub == 2.5
    assert str(i) == "[1.0; 3.0]"
    assert repr(i) == str(i)


def test_interval_wrong_bounds_and_fix():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(3.0, 1.0)
    fixed = Interval(3.0, 1.0, fix=True)
    assert (fixed.lb, fixed.ub) == (1.0, 3.0)


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


def test_interval_addition_subtraction_negation():
    a = Interval(1.0, 2.0)
    b = Interval(0.0, 1.0)
    s = a + b
    assert (s.lb, s.ub) == (1.0, 3.0)
    s = a + 1.0
    assert (s.lb, s.ub) == (2.0, 3.0)
    s = 1.0 + a
    assert (s.lb, s.ub) == (2.0, 3.0)
    d = a - b
    assert (d.lb, d.ub) == (0.0, 2.0)
    r = 5.0 - a
    assert (r.lb, r.ub) == (3.0, 4.0)
    n = -a
    assert (n.lb, n.ub) == (-2.0, -1.0)


def test_interval_multiplication():
    r = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (r.lb, r.ub) == (-4.0, 8.0)
    r = Interval(1.0, 2.0) * -2.0
    assert (r.lb, r.ub) == (-4.0, -2.0)
    r = 3.0 * Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (3.0, 6.0)


def test_interval_inversion_cases():
    r = ~Interval(2.0, 4.0)
    assert (r.lb, r.ub) == (0.25, 0.5)
    r = ~Interval(-4.0, -2.0)
    assert (r.lb, r.ub) == (-0.5, -0.25)
    r = ~Interval(0.0, 2.0)
    assert r.lb == 0.5
    assert r.ub == float("inf")
    r = ~Interval(-2.0, 0.0)
    assert r.lb == float("-inf")
    assert r.ub == -0.5
    r = ~Interval(-1.0, 1.0)
    assert (r.lb, r.ub) == (float("-inf"), float("inf"))


def test_interval_division():
    r = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert (r.lb, r.ub) == (0.25, 1.0)
    r = Interval(1.0, 2.0) / 2.0
    assert (r.lb, r.ub) == (0.5, 1.0)
    r = 1.0 / Interval(2.0, 4.0)
    assert (r.lb, r.ub) == (0.25, 0.5)


def test_interval_power_even_and_odd():
    r = Interval(-2.0, 3.0) ** 2
    assert (r.lb, r.ub) == (0.0, 9.0)
    r = Interval(1.0, 2.0) ** 2
    assert (r.lb, r.ub) == (1.0, 4.0)
    r = Interval(-2.0, -1.0) ** 3
    assert (r.lb, r.ub) == (-8.0, -1.0)
    r = Interval(1.0, 2.0) ** 3
    assert (r.lb, r.ub) == (1.0, 8.0)


def test_interval_result_reduced_to_number_when_narrow():
    r = Interval(1.0, 1.0 + 1e-7) + 0.0
    assert not isinstance(r, Interval)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_interval_reduction_can_be_disabled(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    r = Interval(1.0, 1.0) + 1.0
    assert isinstance(r, Interval)
    assert (r.lb, r.ub) == (2.0, 2.0)


def test_interval_equality_and_comparisons():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 3.0)
    assert Interval(1.0, 2.0) != 1.0
    with pytest.raises(NotImplementedError):
        Interval(1.0, 2.0) == "abc"
    assert Interval(1.0, 2.0) < Interval(2.0, 3.0)
    assert Interval(1.0, 2.0) <= Interval(1.0, 5.0)
    assert Interval(3.0, 4.0) > 2.0
    assert Interval(3.0, 4.0) >= 3.0


def test_interval_inner_subtraction():
    r = Interval(1.0, 3.0) << Interval(0.0, 1.0)
    assert (r.lb, r.ub) == (1.0, 2.0)
    # reversed bounds get fixed
    r = Interval.inner_subtraction(Interval(0.0, 5.0), Interval(1.0, 1.0 + 0.0))
    assert (r.lb, r.ub) == (-1.0, 4.0)


# --------------------------------------------------------------------- Box

def test_box_properties_with_intervals_and_numbers():
    b = Box(Interval(0.0, 2.0), 5.0, Interval(-1.0, 1.0))
    assert isinstance(b, Box)
    assert len(b) == 3
    np.testing.assert_array_equal(b.lb, [0.0, 5.0, -1.0])
    np.testing.assert_array_equal(b.ub, [2.0, 5.0, 1.0])
    np.testing.assert_array_equal(b.middle, [1.0, 5.0, 0.0])


def test_box_width_returns_widest_indices():
    b = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), 2.0, Interval(5.0, 8.0))
    indices, w = b.width
    assert w == 3.0
    assert indices == [1, 3]


def test_box_width_of_point_box_is_zero():
    indices, w = Box(1.0, 2.0).width
    assert w == 0
    assert indices == [0, 1]


def test_box_str_and_repr():
    b = Box(Interval(1.0, 2.0), Interval(3.0, 4.0))
    assert str(b) == "[1.0; 2.0] x [3.0; 4.0]"
    assert repr(b) == "[1.0; 2.0] x [3.0; 4.0]"


def test_box_equality():
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == 5


def test_box_copy_is_deep_for_intervals():
    b = Box(Interval(1.0, 2.0), Interval(3.0, 4.0))
    c = copy_module.copy(b)
    assert isinstance(c, Box)
    c[0].ub = 10.0
    assert b[0].ub == 2.0
    assert c[0].ub == 10.0


def test_box_try_to_reduce():
    degenerate = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    reduced = degenerate._try_to_reduce()
    np.testing.assert_array_equal(reduced, [1.0, 2.0])
    assert not isinstance(reduced[0], Interval)

    wide = Box(Interval(1.0, 2.0), Interval(2.0, 2.0))
    result = wide._try_to_reduce()
    assert isinstance(result, Box)
    np.testing.assert_array_equal(result.lb, [1.0, 2.0])
    np.testing.assert_array_equal(result.ub, [2.0, 2.0])


# ------------------------------------------------- RS_SubinclusionFunction

def test_rs_subinclusion_bounds_within_true_range():
    np.random.seed(0)
    f = RS_SubinclusionFunction(lambda p: p[0] + p[1], 200)
    result = f(Box(Interval(0.0, 1.0), Interval(0.0, 1.0)))
    assert isinstance(result, Interval)
    assert 0.0 <= result.lb <= result.ub <= 2.0
    # with 200 samples the estimate should cover most of the range
    assert result.width > 1.0


def test_rs_subinclusion_constant_and_single_sample():
    f = RS_SubinclusionFunction(lambda p: 5.0, 10)
    r = f(Box(Interval(0.0, 1.0), Interval(2.0, 3.0)))
    assert (r.lb, r.ub) == (5.0, 5.0)

    g = RS_SubinclusionFunction(lambda p: p[0], 1)
    r = g(Box(Interval(0.0, 1.0)))
    assert r.lb == r.ub


def test_rs_subinclusion_reproducible_and_handles_point_components():
    f = RS_SubinclusionFunction(lambda p: p[0] ** 2 + p[1], 50)
    box = Box(Interval(1.0, 2.0), 3.0)
    np.random.seed(42)
    r1 = f(box)
    np.random.seed(42)
    r2 = f(box)
    assert (r1.lb, r1.ub) == (r2.lb, r2.ub)
    assert 4.0 <= r1.lb <= r1.ub <= 7.0


# -------------------------------------------------------- hausdorff_distance

def test_hausdorff_distance_intervals():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 4.0)) == 3.0
    assert hausdorff_distance(Interval(2.0, 4.0), Interval(0.0, 1.0)) == 3.0
    assert hausdorff_distance(Interval(0.0, 4.0), Interval(1.0, 2.0)) == 2.0
    assert hausdorff_distance(Interval(1.0, 2.0), Interval(1.0, 2.0)) == 0.0


def test_hausdorff_distance_with_numbers():
    assert hausdorff_distance(1.0, 4.0) == 3.0
    assert hausdorff_distance(2.0, 2.0) == 0.0
    assert hausdorff_distance(Interval(0.0, 2.0), 5.0) == 5.0
