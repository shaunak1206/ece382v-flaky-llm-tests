import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import (
    Interval,
    IntervalConstants,
    IntervalExceptions,
)
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin,
    cos,
    abs as interval_abs,
    exp,
    sqrt,
    log,
    split,
    bisect,
    constrain,
    shrink,
    hausdorff_distance,
)


# Tests for intervallum/box.py: Box class
def test_box_creation_and_str_repr():
    i1 = Interval(0, 1)
    i2 = Interval(2, 3)
    b = Box(i1, i2)
    # __str__ and __repr__ give same output in this implementation
    s = str(b)
    r = repr(b)
    assert s == "[0; 1] x [2; 3]"
    assert r == s


def test_box_equality_and_inequality():
    i1 = Interval(0, 1)
    i2 = Interval(2, 3)
    b1 = Box(i1, i2)
    b2 = Box(Interval(0, 1), Interval(2, 3))
    b3 = Box(Interval(0.1, 1), Interval(2, 3))
    assert b1 == b2
    assert not (b1 != b2)
    assert b1 != b3
    # Equality with non-np.ndarray or Box raises NotImplementedError
    with pytest.raises(NotImplementedError):
        _ = b1 == 123


def test_box_copy_and_properties():
    intervals = [Interval(0, 1), Interval(2, 4)]
    b = Box(*intervals)
    b_copy = b.__copy__()
    assert b == b_copy
    assert b is not b_copy
    # lb, ub, middle properties extract correct floats arrays
    np.testing.assert_array_equal(b.lb, np.array([0, 2]))
    np.testing.assert_array_equal(b.ub, np.array([1, 4]))
    np.testing.assert_array_equal(b.middle, np.array([0.5, 3]))


def test_box_width_property_and_try_to_reduce():
    b = Box(Interval(1, 1), Interval(0, 2), Interval(5, 5))
    indices, width = b.width
    assert isinstance(indices, list)
    assert math.isclose(width, 2.0)
    # If max width is 0 (all intervals degenerate), _try_to_reduce returns middle
    b_all_zero = Box(Interval(1, 1), Interval(5, 5))
    reduced = b_all_zero._try_to_reduce()
    expected_middle = b_all_zero.middle
    np.testing.assert_array_equal(reduced, expected_middle)
    # If width > 0, returns a copy (Box instance)
    not_reduced = b._try_to_reduce()
    assert isinstance(not_reduced, Box)
    assert not np.array_equal(not_reduced, b.middle)


# Tests for intervallum/interval.py: Interval class
def test_interval_creation_and_str_repr():
    i = Interval(1, 2)
    assert i.lb == 1
    assert i.ub == 2
    s = str(i)
    assert s == "[1; 2]"
    assert repr(i) == s


def test_interval_wrong_bounds_exception():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2, 1)
    # fix=True reverses bounds but creates valid interval
    i = Interval(2, 1, fix=True)
    assert i.lb == 1
    assert i.ub == 2


def test_interval_setters_bounds_and_raise():
    i = Interval(0, 1)
    i.lb = 0.5
    assert i.lb == 0.5
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 2
    i.ub = 2
    assert i.ub == 2
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0


def test_interval_properties_middle_width():
    i = Interval(2, 4)
    assert i.middle == 3
    assert i.width == 2


def test_interval_try_to_reduce_behavior():
    i_small = Interval(1, 1 + IntervalConstants._reduction_width / 2)
    val = i_small._try_to_reduce()
    assert isinstance(val, float)
    assert math.isclose(val, i_small.middle)
    i_large = Interval(0, 1)
    val2 = i_large._try_to_reduce()
    assert isinstance(val2, Interval)
    assert val2 == i_large


def test_interval_inner_subtraction_and_lshift():
    i1 = Interval(3, 5)
    i2 = Interval(1, 2)
    result = i1 << i2
    assert isinstance(result, Interval)
    assert result.lb == 2  # 3 - 1
    assert result.ub == 4  # 5 - 2


def test_interval_equality_and_neq():
    i1 = Interval(1, 2)
    i2 = Interval(1.00000001, 2.00000001)
    i3 = Interval(1.1, 2.1)
    # equals within admissible error
    assert i1 == i2
    assert not (i1 != i2)
    assert i1 != i3
    # Equality with unsupported type raises
    with pytest.raises(NotImplementedError):
        _ = (i1 == "invalid")


def test_interval_arithmetic_operations():
    i1 = Interval(1, 2)
    i2 = Interval(3, 4)
    # addition
    a = i1 + i2
    assert a.lb == 4 and a.ub == 6
    # negation
    n = -i1
    assert n.lb == -2 and n.ub == -1
    # subtraction
    s = i2 - i1
    assert s.lb == 1 and s.ub == 3
    # multiplication
    m = i1 * i2
    assert m.lb == 3 and m.ub == 8
    # inverse
    inv = ~i1
    assert math.isclose(inv.lb, 1 / i1.ub)
    assert math.isclose(inv.ub, 1 / i1.lb)
    # division
    d = i1 / i2
    inv_i2 = ~i2
    expected = i1 * inv_i2
    assert d == expected


def test_interval_power_even_and_odd():
    i = Interval(-1, 2)
    # even power includes zero, so points should include 0.0
    p2 = i ** 2
    assert isinstance(p2, Interval)
    # odd power does not use monotonic decorator, returns Interval
    p3 = i ** 3
    assert isinstance(p3, Interval)
    # power on floats
    val = (i ** 2)  # should reduce properly
    assert isinstance(val, (Interval, float))


# Tests for intervallum/inclusion_functions.py: RS_SubinclusionFunction
def test_rs_subinclusion_function_call_returns_interval(monkeypatch):
    # Function that returns sum of inputs
    def f(x):
        return float(np.sum(x))

    rs = RS_SubinclusionFunction(f, number_of_samples=50)
    b = Box(Interval(0, 1), Interval(1, 2))

    # Patch np.random.uniform to always return midpoint for repeatability
    monkeypatch.setattr("numpy.random.uniform", lambda low, high: 0.5 * (low + high))

    result = rs(b)
    assert isinstance(result, Interval)
    assert result.lb <= result.ub
    # Since f sums coordinates approx in [0,1] and [1,2], result should be in [1,3]
    assert 1 <= result.lb <= 3
    assert 1 <= result.ub <= 3


# Tests for intervallum/interval_functions.py
def test_sin_cos_with_float_and_interval():
    f_sin, pts_sin = sin(0.5)
    assert callable(f_sin)
    assert pts_sin == [0.5]
    interval = Interval(0, math.pi / 2)
    res_sin = sin(interval)
    assert isinstance(res_sin, Interval)

    f_cos, pts_cos = cos(0.5)
    assert callable(f_cos)
    assert pts_cos == [0.5]
    res_cos = cos(interval)
    assert isinstance(res_cos, Interval)


def test_abs_exp_with_float_and_interval():
    f_abs, pts_abs = interval_abs(-0.5)
    assert callable(f_abs)
    assert pts_abs == [-0.5]
    i = Interval(-1, 1)
    res_abs = interval_abs(i)
    assert isinstance(res_abs, Interval)
    val_exp_float = exp(1.0)
    assert isinstance(val_exp_float, float)
    val_exp_interval = exp(Interval(0, 1))
    assert isinstance(val_exp_interval, Interval)


def test_sqrt_and_log_behavior_and_exceptions():
    f_sqrt, pts_sqrt = sqrt(4.0)
    assert callable(f_sqrt)
    assert pts_sqrt == [4.0]

    res_sqrt = sqrt(Interval(1,4))
    assert isinstance(res_sqrt, Interval)

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-2, -1))

    f_log, pts_log = log(1.0)
    assert callable(f_log)
    assert pts_log == [1.0]

    res_log = log(Interval(0.1, 2))
    assert isinstance(res_log, Interval)

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-1, 0))


def test_split_bisect_constrain_shrink_behavior():
    i = Interval(0, 10)
    parts = split(i, [1, 2, 1])
    assert len(parts) == 3
    total_width = sum((p.ub - p.lb) for p in parts)
    assert math.isclose(total_width, i.width)

    bisected = bisect(i)
    assert len(bisected) == 2

    # constrain clamps to limits
    c1 = constrain(Interval(-5, 15), 0, 10)
    assert 0 <= c1.lb <= 10
    assert 0 <= c1.ub <= 10
    c2 = constrain(5, 0, 10)
    assert c2 == 5
    c3 = constrain(-1, 0, 10)
    assert c3 == 0
    c4 = constrain(12, 0, 10)
    assert c4 == 10

    # shrink reduces width by alpha
    i2 = Interval(0, 4)
    shrunk = shrink(i2, 0.5)
    expected_width = i2.width * 0.5
    assert math.isclose(shrunk.width, expected_width)
    # shrink for float returns float unchanged
    assert shrink(5, 0.5) == 5


def test_hausdorff_distance_properties():
    i1 = Interval(0, 2)
    i2 = Interval(1, 3)
    dist = hausdorff_distance(i1, i2)
    # distance should be symmetric and >= 0
    dist2 = hausdorff_distance(i2, i1)
    assert dist == dist2
    assert dist >= 0
    # Distance between identical intervals is zero
    assert hausdorff_distance(i1, Interval(0, 2)) == 0
    # Distance between two floats (treated as degenerate intervals)
    fdist = hausdorff_distance(1.0, 2.0)
    assert isinstance(fdist, float)
    assert fdist >= 0
