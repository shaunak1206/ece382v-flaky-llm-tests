import math

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import hausdorff_distance


def test_interval_str_repr_and_properties():
    i = Interval(-2.5, 3.0)
    assert str(i) == "[-2.5; 3.0]"
    assert repr(i) == "[-2.5; 3.0]"
    assert i.lb == -2.5
    assert i.ub == 3.0
    assert i.middle == 0.25
    assert i.width == 5.5


def test_interval_from_point_and_copy():
    i = Interval.from_point(4.2)
    j = i.__copy__()
    assert i == 4.2
    assert j == i
    assert j is not i
    assert j.lb == 4.2 and j.ub == 4.2


def test_interval_wrong_bounds_raises():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(3.0, 2.0)


def test_interval_setters_validate_bounds():
    i = Interval(1.0, 2.0)
    i.lb = 0.5
    assert i.lb == 0.5
    i.ub = 3.0
    assert i.ub == 3.0

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 4.0

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.1


def test_interval_equality_and_inequality_with_tolerance():
    i1 = Interval(1.0, 1.0)
    i2 = Interval(1.0 + 5e-8, 1.0 + 5e-8)
    assert i1 == i2
    assert not (i1 != i2)

    i3 = Interval(1.0 + 1e-4, 1.0 + 1e-4)
    assert i1 != i3


def test_interval_comparison_operators_use_lower_bound():
    i1 = Interval(1.0, 10.0)
    i2 = Interval(2.0, 2.5)
    assert i1 < i2
    assert i1 <= i2
    assert i2 > i1
    assert i2 >= i1
    assert i1 < 1.5
    assert i1 <= 1.0
    assert i2 > 1.5
    assert i2 >= 2.0


def test_interval_add_sub_mul_div_with_numbers_and_intervals():
    i = Interval(1.0, 2.0)
    assert i + 3.0 == Interval(4.0, 5.0)
    assert 3.0 + i == Interval(4.0, 5.0)
    assert i - 1.0 == Interval(0.0, 1.0)
    assert 5.0 - i == Interval(3.0, 4.0)
    assert i * 2.0 == Interval(2.0, 4.0)
    assert 2.0 * i == Interval(2.0, 4.0)
    assert i / 2.0 == Interval(0.5, 1.0)

    j = Interval(-1.0, 3.0)
    assert i + j == Interval(0.0, 5.0)
    assert i * j == Interval(-2.0, 6.0)


def test_interval_invert_and_division_across_zero():
    positive = Interval(2.0, 4.0)
    inv = ~positive
    assert isinstance(inv, Interval)
    assert inv == Interval(0.25, 0.5)
    assert 1.0 / positive == Interval(0.25, 0.5)

    crosses_zero = Interval(-2.0, 3.0)
    inv2 = ~crosses_zero
    assert isinstance(inv2, Interval)
    assert math.isinf(inv2.lb) and math.isinf(inv2.ub)
    assert inv2.lb < 0 < inv2.ub


def test_interval_power_even_and_odd():
    i = Interval(-2.0, 3.0)
    even = i ** 2
    odd = i ** 3
    assert even == Interval(0.0, 9.0)
    assert odd == Interval(-8.0, 27.0)


def test_interval_power_reduces_to_number_for_small_width(monkeypatch):
    old_reduce = IntervalConstants._reduce_intervals_to_numbers
    old_width = IntervalConstants._reduction_width
    try:
        IntervalConstants._reduce_intervals_to_numbers = True
        IntervalConstants._reduction_width = 1.0
        i = Interval(2.0, 2.5)
        result = i + 0.0
        assert isinstance(result, float)
        assert result == pytest.approx(2.25)
    finally:
        IntervalConstants._reduce_intervals_to_numbers = old_reduce
        IntervalConstants._reduction_width = old_width


def test_box_string_repr_and_equality():
    b1 = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    b2 = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    assert str(b1) == "[0.0; 1.0] x [2.0; 3.0]"
    assert repr(b1) == "[0.0; 1.0] x [2.0; 3.0]"
    assert b1 == b2
    assert not (b1 != b2)


def test_box_properties_lb_ub_middle_width():
    b = Box(Interval(0.0, 1.0), Interval(-2.0, 2.0), 5.0)
    np.testing.assert_array_equal(b.lb, np.array([0.0, -2.0, 5.0]))
    np.testing.assert_array_equal(b.ub, np.array([1.0, 2.0, 5.0]))
    np.testing.assert_array_equal(b.middle, np.array([0.5, 0.0, 5.0]))
    indices, width = b.width
    assert indices == [1]
    assert width == 4.0


def test_box_copy_and_try_to_reduce():
    b = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    b_copy = b.__copy__()
    assert b_copy == b
    assert b_copy is not b
    reduced = b._try_to_reduce()
    assert isinstance(reduced, np.ndarray)
    np.testing.assert_array_equal(reduced, np.array([1.0, 2.0]))

    b2 = Box(Interval(0.0, 1.0), Interval(0.0, 2.0))
    unreduced = b2._try_to_reduce()
    assert isinstance(unreduced, Box)
    assert unreduced == b2


def test_rs_subinclusion_function_uses_sampled_points_and_returns_interval(monkeypatch):
    calls = []

    def fake_uniform(low, high):
        calls.append((np.array(low), np.array(high)))
        return np.array(low) + 0.5 * (np.array(high) - np.array(low))

    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    def f(x):
        return float(np.sum(x))

    box = Box(Interval(0.0, 2.0), Interval(10.0, 14.0))
    rs = RS_SubinclusionFunction(f, number_of_samples=4)
    out = rs(box)

    assert isinstance(out, Interval)
    assert out == Interval(11.0, 11.0)
    assert len(calls) == 4
    for low, high in calls:
        np.testing.assert_array_equal(low, np.array([0.0, 10.0]))
        np.testing.assert_array_equal(high, np.array([2.0, 14.0]))


def test_hausdorff_distance_for_intervals_and_numbers():
    assert hausdorff_distance(Interval(0.0, 2.0), Interval(1.0, 3.0)) == 1.0
    assert hausdorff_distance(Interval(0.0, 2.0), 1.0) == 1.0
    assert hausdorff_distance(2.0, 2.0) == 0.0
    assert hausdorff_distance(Interval(1.0, 1.0), Interval(1.0, 1.0)) == 0.0
