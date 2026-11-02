import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin, cos, abs as interval_abs, exp, sqrt, log,
    split, bisect, constrain, shrink,
    hausdorff_distance
)


# Interval tests
def test_interval_creation_and_str_repr():
    i = Interval(1.0, 2.0)
    assert str(i) == "[1.0; 2.0]"
    assert repr(i) == str(i)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    # with fix=True should swap
    i_fixed = Interval(2.0, 1.0, fix=True)
    assert i_fixed.lb == 1.0 and i_fixed.ub == 2.0


def test_interval_properties_setters():
    i = Interval(0.0, 1.0)
    i.lb = 0.2
    assert math.isclose(i.lb, 0.2)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 1.5
    i.lb = 0.0
    i.ub = 0.8
    assert math.isclose(i.ub, 0.8)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = -0.2


def test_interval_middle_and_width():
    i = Interval(-1, 3)
    assert math.isclose(i.middle, 1.0)
    assert math.isclose(i.width, 4.0)


def test_interval_try_to_reduce_and_equality_behavior(monkeypatch):
    # Reduction width default is 1e-5
    i = Interval(1.0, 1.0 + 1e-6)
    # Should reduce to float (middle)
    val = i._try_to_reduce()
    assert isinstance(val, float)
    # Disable reduction
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    i2 = Interval(1.0, 1.0 + 1e-6)
    val2 = i2._try_to_reduce()
    assert isinstance(val2, Interval)

    # Equality: distance <= admissible error should be True
    a = Interval(1.0, 2.0)
    b = Interval(1.0 + 1e-8, 2.0 + 1e-8)
    assert (a == b) is True
    # Not equal if distance larger
    c = Interval(1.0, 2.0)
    d = Interval(1.1, 2.1)
    assert (c == d) is False

    # Equality with float
    e = Interval(1.0, 1.0 + 1e-6)
    assert (e == 1.0) is True
    assert (e != 2.0) is True

    # Unsupported equality
    with pytest.raises(NotImplementedError):
        a == "string"


def test_interval_arithmetic_operations_and_power():
    i1 = Interval(1, 2)
    i2 = Interval(3, 4)

    # Addition
    sum_i = i1 + i2
    assert math.isclose(sum_i.lb, 4)
    assert math.isclose(sum_i.ub, 6)

    # Subtraction
    diff = i2 - i1
    assert math.isclose(diff.lb, 1)
    assert math.isclose(diff.ub, 3)

    # Negation
    neg = -i1
    assert math.isclose(neg.lb, -2)
    assert math.isclose(neg.ub, -1)

    # Multiplication
    mul = i1 * i2
    assert mul.lb == min(1*3, 1*4, 2*3, 2*4)
    assert mul.ub == max(1*3, 1*4, 2*3, 2*4)

    # Division
    div = i2 / i1
    assert div.lb <= div.ub
    # Reverse division
    div_r = 10 / i1
    assert div_r.lb <= div_r.ub

    # Power even
    pow_even = Interval(-2, 3) ** 2
    assert isinstance(pow_even, Interval)
    assert pow_even.lb <= pow_even.ub
    # Power odd
    pow_odd = Interval(2, 3) ** 3
    assert isinstance(pow_odd, (Interval, float))
    assert pow_odd.lb <= pow_odd.ub

    # Invert (~) operator tests
    i_pos = Interval(1, 2)
    inv = ~i_pos
    assert math.isclose(inv.lb, 1/2)
    assert math.isclose(inv.ub, 1/1)

    i_neg = Interval(-2, -1)
    inv_neg = ~i_neg
    assert math.isclose(inv_neg.lb, 1/(-1))
    assert math.isclose(inv_neg.ub, 1/(-2))

    i_zero_lb = Interval(0, 2)
    inv_zero_lb = ~i_zero_lb
    assert math.isclose(inv_zero_lb.lb, 1/2)
    assert math.isinf(inv_zero_lb.ub) and inv_zero_lb.ub > 0

    i_zero_ub = Interval(-2, 0)
    inv_zero_ub = ~i_zero_ub
    assert math.isinf(inv_zero_ub.lb) and inv_zero_ub.lb < 0
    assert math.isclose(inv_zero_ub.ub, 1/(-2))

    i_contains_zero = Interval(-1, 1)
    inv_contains_zero = ~i_contains_zero
    assert math.isinf(inv_contains_zero.lb) and math.isinf(inv_contains_zero.ub)


# interval_functions tests
def test_sin_cos_exp_abs_log_sqrt_behavior():
    i = Interval(0, math.pi/2)
    # sin and cos return Interval containing expected bounds
    s = sin(i)
    c = cos(i)
    assert isinstance(s, Interval)
    assert -1.0 <= s.lb <= s.ub <= 1.0
    assert -1.0 <= c.lb <= c.ub <= 1.0

    # abs includes 0 if interval crosses zero
    ia = Interval(-2, 3)
    absi = interval_abs(ia)
    assert absi.lb >= 0

    # exp is increasing
    iexp = exp(i)
    assert iexp.lb >= math.exp(i.lb)
    assert iexp.ub <= math.exp(i.ub)

    # sqrt positive interval
    sq = sqrt(Interval(0.0, 4.0))
    assert sq.lb == 0.0 and sq.ub == 2.0
    # sqrt negative interval raises exception
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-2, -0.1))
    # sqrt crossing zero clamps lower bound at zero
    sq2 = sqrt(Interval(-1, 4))
    assert sq2.lb == 0.0 and sq2.ub == 2.0

    # log positive interval
    l = log(Interval(1, math.e))
    assert math.log(1) <= l.lb <= l.ub <= math.log(math.e)

    # log with zero or negative bound raises exception
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-1, 0))
    # log crossing zero clamps lower bound at zero
    l2 = log(Interval(-1, 2))
    assert l2.lb <= l2.ub


def test_split_and_bisect():
    i = Interval(0, 10)
    parts = split(i, [1, 1, 2])
    # sum of widths should be original interval width
    total_width = sum((part.ub - part.lb) if isinstance(part, Interval) else 0 for part in parts)
    assert math.isclose(total_width, i.width)

    # bisect splits in two halves
    parts_bisect = bisect(i)
    assert len(parts_bisect) == 2
    assert math.isclose(parts_bisect[0].width, parts_bisect[1].width)
    assert math.isclose(parts_bisect[0].lb, i.lb)
    assert math.isclose(parts_bisect[1].ub, i.ub)

    # split with float input returns repeated floats
    splits_float = split(5.0, [1, 2])
    assert len(splits_float) == 2
    assert all(x == 5.0 for x in splits_float)


def test_constrain_and_shrink():
    i = Interval(-5, 5)
    c = constrain(i, 0, 3)
    assert c.lb >= 0 and c.ub <= 3

    c2 = constrain(10.0, 0, 5)
    assert 0 <= c2 <= 5

    s = shrink(i, 0.2)
    expected_radius = 0.5 * 0.2 * i.width
    expected_middle = i.middle
    assert math.isclose(s.lb, expected_middle - expected_radius)
    assert math.isclose(s.ub, expected_middle + expected_radius)

    s_float = shrink(2.5, 0.5)
    assert s_float == 2.5


def test_hausdorff_distance_behaviour():
    a = Interval(0, 5)
    b = Interval(3, 10)

    dist_ab = hausdorff_distance(a, b)
    dist_ba = hausdorff_distance(b, a)
    # Hausdorff distance should be symmetric and non-negative
    assert math.isclose(dist_ab, dist_ba)
    assert dist_ab >= 0

    # Distance with identical intervals is 0
    dist_zero = hausdorff_distance(a, a)
    assert math.isclose(dist_zero, 0)

    # Distance between float intervals and mixed
    dist_mixed = hausdorff_distance(5.0, Interval(3, 7))
    assert dist_mixed >= 0


# Box tests
def test_box_basics_and_properties():
    # Create Box from Intervals
    i1 = Interval(0, 1)
    i2 = Interval(1, 2)
    b = Box(i1, i2)
    assert isinstance(b, Box)
    assert str(b) == "[0; 1] x [1; 2]" or str(b) == "[0.0; 1.0] x [1.0; 2.0]"  # accept float or int string
    repr_str = repr(b)
    assert isinstance(repr_str, str)
    # Equality checks
    b2 = Box(i1, i2)
    assert b == b2
    with pytest.raises(NotImplementedError):
        b == "string"
    b3 = Box(i2, i1)
    assert b != b3

    # Copy returns new Box equal but different object
    b_copy = b.__copy__()
    assert b_copy == b
    assert b_copy is not b

    # Properties lb, ub, middle return np.ndarray
    lbs = b.lb
    ubs = b.ub
    middles = b.middle
    assert isinstance(lbs, np.ndarray)
    assert isinstance(ubs, np.ndarray)
    assert isinstance(middles, np.ndarray)
    assert all(math.isclose(lbs[i], iv.lb) for i, iv in enumerate(b))
    assert all(math.isclose(ubs[i], iv.ub) for i, iv in enumerate(b))
    assert all(math.isclose(middles[i], iv.middle) for i, iv in enumerate(b))

    # width property returns correct indices and max width
    max_indices, max_width = b.width
    assert isinstance(max_indices, list)
    assert isinstance(max_width, float)
    # all widths are <= max_width
    widths = [iv.width for iv in b]
    assert all(w <= max_width for w in widths)
    # indices correspond to max widths
    for idx in max_indices:
        assert math.isclose(widths[idx], max_width)

    # _try_to_reduce returns middle if max width == 0, else copy
    b_constant = Box(Interval(1,1), Interval(2,2))
    reduced = b_constant._try_to_reduce()
    # middle is returned as ndarray
    assert isinstance(reduced, np.ndarray)
    # When width>0 returns Box copy
    assert isinstance(b._try_to_reduce(), Box)


# RS_SubinclusionFunction tests
def test_rs_subinclusionfunction_call(monkeypatch):
    # Seed numpy random for reproducibility
    np.random.seed(12345)

    # Define a simple function f(x): sum(x)
    def f(x):
        return float(np.sum(x))

    rs = RS_SubinclusionFunction(f, number_of_samples=10)

    # Create a Box with float bounds
    i1 = Interval(0, 1)
    i2 = Interval(1, 2)
    box = Box(i1, i2)

    # Seed numpy random generator inside __call__
    monkeypatch.setattr(np.random, "uniform", lambda low, high: (np.array(low) + np.array(high)) / 2)

    result = rs(box)
    assert isinstance(result, Interval)
    # Because we override uniform to midpoint, min and max should be equal
    expected_val = f(np.array([i1.middle, i2.middle]))
    assert math.isclose(result.lb, expected_val)
    assert math.isclose(result.ub, expected_val)

    # Now test without monkeypatch, but seed np.random in test body to avoid flakiness
    np.random.seed(42)
    rs2 = RS_SubinclusionFunction(lambda x: np.sum(x**2), 5)
    box2 = Box(Interval(1, 2), Interval(3, 4))
    interval_result = rs2(box2)
    assert isinstance(interval_result, Interval)
    assert interval_result.lb <= interval_result.ub


# Test __eq__ and __ne__ for different types on Box and Interval
def test_eq_ne_type_errors():
    b = Box(Interval(0, 1), Interval(1, 2))

    # Box equality with unrelated type raises
    with pytest.raises(NotImplementedError):
        b == "invalid"

    i = Interval(1, 2)
    with pytest.raises(NotImplementedError):
        i == [1,2]

# Test Interval __lt__, __le__, __gt__, __ge__ with Interval and float
@pytest.mark.parametrize("op", ["__lt__", "__le__", "__gt__", "__ge__"])
def test_interval_comparison_ops(op):
    i1 = Interval(1.0, 3.0)
    i2 = Interval(2.0, 4.0)
    method = getattr(i1, op)
    assert isinstance(method(i2), bool)
    assert isinstance(method(2.5), bool)

# Test Interval __radd__, __rmul__, __rsub__, __rtruediv__
def test_interval_reverse_operations():
    i = Interval(1, 2)
    assert (3 + i) == (i + 3)
    assert (3 * i) == (i * 3)
    # rsub -> 3 - i = (-i) + 3
    res = 3 - i
    expected = (-i) + 3
    assert res == expected
    # rtruediv -> 3 / i = 3 * (~i)
    res_div = 3 / i
    expected_div = 3 * (~i)
    assert res_div == expected_div
