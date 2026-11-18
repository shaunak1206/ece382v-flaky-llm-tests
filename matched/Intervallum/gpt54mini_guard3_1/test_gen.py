import math

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    abs as interval_abs,
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


def test_box_string_repr_and_basic_properties():
    b = Box(Interval(1.0, 2.0), Interval(-1.0, 3.0), 5.0)
    assert str(b) == "[1.0; 2.0] x [-1.0; 3.0] x 5.0"
    assert repr(b) == "[1.0; 2.0] x [-1.0; 3.0] x 5.0"
    np.testing.assert_allclose(b.lb, np.array([1.0, -1.0, 5.0]))
    np.testing.assert_allclose(b.ub, np.array([2.0, 3.0, 5.0]))
    np.testing.assert_allclose(b.middle, np.array([1.5, 1.0, 5.0]))
    assert b.width == ([1], 4.0)


def test_box_try_to_reduce_returns_middle_for_zero_width_box():
    b = Box(Interval(2.0, 2.0), 4.0)
    reduced = b._try_to_reduce()
    assert isinstance(reduced, np.ndarray)
    np.testing.assert_allclose(reduced, np.array([2.0, 4.0]))


def test_box_copy_and_equality():
    b = Box(Interval(1.0, 2.0), 3.0)
    copied = b.__copy__()
    assert copied == b
    assert copied is not b
    assert isinstance(copied[0], Interval)
    assert copied[0] is not b[0]


def test_interval_init_and_wrong_bounds_exception():
    i = Interval(1.0, 3.0)
    assert i.lb == 1.0
    assert i.ub == 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(3.0, 1.0)


def test_interval_setters_validate_bounds():
    i = Interval(1.0, 3.0)
    i.lb = 1.5
    assert i.lb == 1.5
    i.ub = 4.0
    assert i.ub == 4.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 5.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 1.0


def test_interval_equality_with_float_and_interval():
    assert Interval(1.0, 1.0) == 1.0
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 2.1)
    with pytest.raises(NotImplementedError):
        _ = Interval(1.0, 2.0) == "1.0"


def test_interval_arithmetic_operations():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(3.0, 5.0)
    assert i1 + i2 == Interval(4.0, 7.0)
    assert i2 - i1 == Interval(1.0, 4.0)
    assert i1 * i2 == Interval(3.0, 10.0)
    assert i2 / i1 == Interval(1.5, 5.0)
    assert -i1 == Interval(-2.0, -1.0)


def test_interval_power_even_and_odd():
    i = Interval(-2.0, 3.0)
    even = i ** 2
    odd = i ** 3
    assert even == Interval(0.0, 9.0)
    assert odd == Interval(-8.0, 27.0)


def test_interval_reduction_to_number_can_be_toggled(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", True)
    assert (Interval(1.0, 1.0) + Interval(2.0, 2.0)) == 3.0

    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    result = Interval(1.0, 1.0) + Interval(2.0, 2.0)
    assert isinstance(result, Interval)
    assert result == Interval(3.0, 3.0)


def test_interval_function_sin_cos_on_interval():
    i = Interval(0.0, math.pi / 2.0)
    assert sin(i) == Interval(0.0, 1.0)
    assert cos(i) == Interval(0.0, 1.0)


def test_interval_function_abs_exp_and_shrink():
    i = Interval(-2.0, 3.0)
    assert interval_abs(i) == Interval(0.0, 3.0)
    assert exp(0.0) == 1.0
    shrunk = shrink(i, 0.5)
    assert shrunk == Interval(-0.75, 1.75)


def test_interval_function_sqrt_and_log_domain_errors():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-4.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-2.0, 0.0))

    assert sqrt(Interval(0.0, 4.0)) == Interval(0.0, 2.0)
    assert log(Interval(1.0, math.e)) == Interval(0.0, 1.0)


def test_split_and_bisect_on_interval_and_point():
    i = Interval(0.0, 10.0)
    parts = split(i, [1.0, 3.0])
    assert parts == [Interval(0.0, 2.5), Interval(2.5, 10.0)]
    assert bisect(i) == [Interval(0.0, 5.0), Interval(5.0, 10.0)]

    assert split(4.2, [1.0, 2.0, 3.0]) == [4.2, 4.2, 4.2]


def test_constrain_clamps_interval_and_point():
    assert constrain(Interval(-2.0, 3.0), 0.0, 2.0) == Interval(0.0, 2.0)
    assert constrain(-5.0, -1.0, 1.0) == -1.0
    assert constrain(0.5, -1.0, 1.0) == 0.5


def test_hausdorff_distance_for_intervals_and_points():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 4.0)) == 3.0
    assert hausdorff_distance(1.5, Interval(1.0, 2.0)) == 0.5
    assert hausdorff_distance(Interval(1.0, 2.0), 1.5) == 0.5


def test_rs_subinclusion_function_uses_numpy_uniform_and_returns_interval(monkeypatch):
    calls = []

    def fake_uniform(low, high):
        calls.append((np.array(low, copy=True), np.array(high, copy=True)))
        return np.array([low[0], high[1]], dtype=float)

    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    f = RS_SubinclusionFunction(lambda x: float(np.sum(x)), number_of_samples=3)
    box = Box(Interval(1.0, 2.0), Interval(10.0, 20.0))
    result = f(box)

    assert len(calls) == 3
    assert all(np.allclose(c[0], np.array([1.0, 10.0])) for c in calls)
    assert all(np.allclose(c[1], np.array([2.0, 20.0])) for c in calls)
    assert result == Interval(21.0, 21.0)
