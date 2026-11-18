import math

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    abs as iabs,
    bisect,
    constrain,
    cos,
    exp,
    hausdorff_distance,
    log,
    shrink,
    sin,
    split,
    sqrt,
)


def test_interval_init_and_properties():
    i = Interval(1.5, 3.5)
    assert i.lb == 1.5
    assert i.ub == 3.5
    assert i.middle == 2.5
    assert i.width == 2.0
    assert str(i) == "[1.5; 3.5]"
    assert repr(i) == "[1.5; 3.5]"


def test_interval_from_point_and_equality_to_float():
    i = Interval.from_point(2.0)
    assert i == 2.0
    assert i == Interval(2.0, 2.0)
    assert not (i != 2.0)


def test_interval_wrong_bounds_raises():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(5.0, 2.0)


def test_interval_setters_validate_bounds():
    i = Interval(1.0, 3.0)
    i.lb = 0.5
    i.ub = 4.0
    assert i.lb == 0.5
    assert i.ub == 4.0

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 5.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0


def test_interval_arithmetic_add_sub_mul_div():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(3.0, 5.0)

    assert i1 + i2 == Interval(4.0, 7.0)
    assert i2 - i1 == Interval(1.0, 4.0)
    assert i1 * i2 == Interval(3.0, 10.0)
    assert i2 / 2.0 == Interval(1.5, 2.5)
    assert 2.0 / i2 == Interval(0.4, 0.6666666666666666)


def test_interval_neg_and_inner_subtraction():
    i = Interval(1.0, 4.0)
    neg = -i
    assert isinstance(neg, Interval)
    assert neg == Interval(-4.0, -1.0)

    d = i << Interval(0.5, 1.5)
    assert d == Interval(-0.5, 2.5)


def test_interval_comparison_uses_lower_bound():
    assert Interval(1.0, 10.0) < Interval(2.0, 3.0)
    assert Interval(1.0, 10.0) <= 1.0
    assert Interval(3.0, 4.0) > 2.5
    assert Interval(3.0, 4.0) >= Interval(3.0, 100.0)


def test_interval_power_even_and_odd():
    i = Interval(-2.0, 3.0)
    even = i ** 2
    odd = i ** 3
    assert even == Interval(0.0, 9.0)
    assert odd == Interval(-8.0, 27.0)


def test_interval_reduction_to_number_can_be_disabled_and_enabled():
    old_reduce = IntervalConstants._reduce_intervals_to_numbers
    old_width = IntervalConstants._reduction_width
    try:
        IntervalConstants._reduce_intervals_to_numbers = True
        IntervalConstants._reduction_width = 1e-3
        assert isinstance(Interval(1.0, 1.0001) + 0.0, float)

        IntervalConstants._reduce_intervals_to_numbers = False
        result = Interval(1.0, 1.0001) + 0.0
        assert isinstance(result, Interval)
    finally:
        IntervalConstants._reduce_intervals_to_numbers = old_reduce
        IntervalConstants._reduction_width = old_width


def test_box_basic_properties_and_string_representation():
    b = Box(Interval(1.0, 2.0), Interval(3.0, 5.0))
    assert str(b) == "[1.0; 2.0] x [3.0; 5.0]"
    assert repr(b) == "[1.0; 2.0] x [3.0; 5.0]"
    assert np.allclose(b.lb, np.array([1.0, 3.0]))
    assert np.allclose(b.ub, np.array([2.0, 5.0]))
    assert np.allclose(b.middle, np.array([1.5, 4.0]))


def test_box_width_and_try_to_reduce():
    b = Box(Interval(1.0, 2.0), Interval(3.0, 3.0), 7.0)
    idxs, width = b.width
    assert idxs == [0]
    assert width == 1.0
    reduced = b._try_to_reduce()
    assert isinstance(reduced, Box)
    assert np.array_equal(reduced, b)

    point_box = Box(Interval(2.0, 2.0), Interval(3.0, 3.0))
    reduced_point = point_box._try_to_reduce()
    assert isinstance(reduced_point, np.ndarray)
    assert np.allclose(reduced_point, np.array([2.0, 3.0]))


def test_box_equality_and_copy():
    b1 = Box(Interval(1.0, 2.0), Interval(3.0, 4.0))
    b2 = Box(Interval(1.0, 2.0), Interval(3.0, 4.0))
    b3 = Box(Interval(1.0, 2.0), Interval(3.0, 5.0))
    assert b1 == b2
    assert b1 != b3

    b4 = b1.__copy__()
    assert isinstance(b4, Box)
    assert b4 == b1
    assert b4 is not b1
    assert all(a is not b for a, b in zip(b1, b4))


def test_rs_subinclusion_function_uses_sampled_points(monkeypatch):
    calls = []

    def fake_uniform(low, high):
        calls.append((np.array(low), np.array(high)))
        return np.array(low)

    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    def f(x):
        return float(np.sum(x))

    rs = RS_SubinclusionFunction(f, number_of_samples=4)
    box = Box(Interval(1.0, 2.0), Interval(10.0, 20.0))
    result = rs(box)

    assert len(calls) == 4
    assert result == Interval(11.0, 11.0)


def test_trig_and_elementary_interval_functions():
    assert sin(0.0) == 0.0
    assert cos(0.0) == 1.0
    assert iabs(-3.5) == 3.5
    assert exp(0.0) == 1.0
    assert sqrt(9.0) == 3.0
    assert log(math.e) == 1.0


def test_sqrt_and_log_raise_on_invalid_intervals():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-4.0, -1.0))

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-2.0, 0.0))


def test_split_bisect_constrain_and_shrink():
    i = Interval(0.0, 10.0)
    parts = split(i, [1.0, 2.0, 1.0])
    assert parts == [Interval(0.0, 2.5), Interval(2.5, 7.5), Interval(7.5, 10.0)]

    left, right = bisect(i)
    assert left == Interval(0.0, 5.0)
    assert right == Interval(5.0, 10.0)

    assert constrain(Interval(-2.0, 12.0), 0.0, 10.0) == Interval(0.0, 10.0)
    assert shrink(Interval(0.0, 10.0), 0.5) == Interval(2.5, 7.5)


def test_hausdorff_distance_for_intervals_and_points():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 4.0)) == 2.0
    assert hausdorff_distance(1.0, Interval(2.0, 4.0)) == 3.0
    assert hausdorff_distance(Interval(0.0, 0.0), 0.0) == 0.0
