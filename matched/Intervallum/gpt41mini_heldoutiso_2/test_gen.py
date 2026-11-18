import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin, cos, abs as interval_abs, exp, sqrt, log,
    split, bisect, constrain, shrink, hausdorff_distance
)


# Tests for intervallum/box.py Box class
def test_box_creation_and_str_repr():
    b = Box(Interval(1, 2), Interval(3, 4))
    assert isinstance(b, np.ndarray)
    assert str(b) == "[1; 2] x [3; 4]"
    assert repr(b) == "[1; 2] x [3; 4]"


def test_box_equality_and_inequality():
    b1 = Box(Interval(1, 2), Interval(3, 4))
    b2 = Box(Interval(1, 2), Interval(3, 4))
    b3 = Box(Interval(0, 2), Interval(3, 4))
    assert b1 == b2
    assert b1 != b3
    with pytest.raises(NotImplementedError):
        b1 == "not a box"


def test_box_copy_and_try_to_reduce():
    intervals = [Interval(1.0, 1.0), Interval(5.0, 10.0)]
    b = Box(*intervals)
    b_copy = b.__copy__()
    assert b_copy == b
    # Width of first interval is zero, second is 5; max width = 5
    indices, width = b.width
    assert width == 5
    # _try_to_reduce returns copy when width > 0
    result = b._try_to_reduce()
    assert isinstance(result, Box)
    
    # Box with all zero width intervals returns middles as np.ndarray
    b2 = Box(Interval(2.0, 2.0), Interval(3.0, 3.0))
    indices2, width2 = b2.width
    assert width2 == 0.0
    reduced = b2._try_to_reduce()
    assert isinstance(reduced, np.ndarray)
    assert np.allclose(reduced, np.array([2.0, 3.0]))

def test_box_properties_lb_ub_middle_width():
    intervals = [Interval(1.0, 4.0), 10, Interval(3.0, 5.0)]
    b = Box(*intervals)
    lb = b.lb
    ub = b.ub
    mid = b.middle
    width_indices, max_width = b.width
    assert np.allclose(lb, [1.0, 10, 3.0])
    assert np.allclose(ub, [4.0, 10, 5.0])
    assert np.allclose(mid, [2.5, 10, 4.0])
    assert max_width == 3.0
    assert sorted(width_indices) == [0]



# Tests for intervallum/interval.py Interval class and behavior

def test_interval_creation_and_str_repr():
    i = Interval(1.0, 2.0)
    assert i.lb == 1.0
    assert i.ub == 2.0
    assert str(i) == "[1.0; 2.0]"
    assert repr(i) == "[1.0; 2.0]"


def test_interval_wrong_bounds_exception():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2, 1)
    # fix=True swaps bounds instead of raising
    i = Interval(5, 3, fix=True)
    assert i.lb == 3
    assert i.ub == 5


def test_interval_setters_bounds_validation():
    i = Interval(1, 5)
    i.lb = 0
    assert i.lb == 0
    i.ub = 6
    assert i.ub == 6
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 10  # lb > ub not allowed
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = -1  # ub < lb not allowed


def test_interval_middle_and_width_properties():
    i = Interval(1, 5)
    assert i.middle == 3.0
    assert i.width == 4.0


def test_interval_try_to_reduce_behavior():
    i = Interval(1, 1 + IntervalConstants._reduction_width / 2)
    reduced = i._try_to_reduce()
    # Should reduce to float middle because width < reduction_width
    assert isinstance(reduced, float)
    i2 = Interval(1, 2)
    reduced2 = i2._try_to_reduce()
    assert isinstance(reduced2, Interval)


def test_interval_inner_subtraction_and_shift_operators():
    i1 = Interval(5, 10)
    i2 = Interval(3, 7)
    result = Interval.inner_subtraction(i1, i2)
    assert result.lb == 2  # 5 - 3
    assert result.ub == 3  # 10 - 7
    # Using << operator
    shifted = i1 << i2
    assert shifted == result
    shifted2 = i1 << 4.0
    assert shifted2.lb == 1
    assert shifted2.ub == 6


def test_interval_eq_and_ne_with_float_and_interval():
    i1 = Interval(1.0, 1.00000000001)
    i2 = Interval(1.0, 1.00000000002)
    # Should be equal within admissible error
    assert i1 == i2
    assert not (i1 != i2)
    # With float equal within admissible error
    assert i1 == 1.0
    # With float far away
    i3 = Interval(1, 2)
    assert (i3 != 3.0) is True
    with pytest.raises(NotImplementedError):
        i1 == "string"


def test_interval_comparisons():
    i1 = Interval(1, 2)
    i2 = Interval(3, 4)
    # __lt__
    assert (i1 < i2) is True
    assert (i2 < 2) is False
    # __le__
    assert (i1 <= i2) is True
    assert (i2 <= 3) is True
    # __gt__
    assert (i2 > i1) is True
    assert (i1 > 0.5) is True
    # __ge__
    assert (i2 >= i1) is True
    assert (i1 >= 1) is True


@pytest.mark.parametrize("op, a, b, expected", [
    ('add', Interval(1, 2), Interval(3, 5), (4, 7)),
    ('add', Interval(1, 2), 2, (3, 4)),
    ('neg', Interval(1, 3), None, (-3, -1)),
    ('sub', Interval(3, 5), Interval(1, 2), (1, 4)),
    ('sub', Interval(3, 5), 1, (2, 4)),
    ('mul', Interval(1, 2), Interval(3, 4), (3, 8)),
    ('mul', Interval(-1, 2), 3, (-3, 6)),
    ('invert', Interval(1, 2), None, (0.5, 1.0)),
    ('invert', Interval(-2, -1), None, (-1.0, -0.5)),
])
def test_interval_arithmetic_operations(op, a, b, expected):
    if op == 'add':
        res = a + b
    elif op == 'neg':
        res = -a
    elif op == 'sub':
        res = a - b
    elif op == 'mul':
        res = a * b
    elif op == 'invert':
        res = ~a
    else:
        pytest.skip("Unknown op")
    lb_expected, ub_expected = expected
    assert isinstance(res, (Interval, float))
    # If result is Interval object, check bounds approximately
    if isinstance(res, Interval):
        assert math.isclose(res.lb, lb_expected, rel_tol=1e-9)
        assert math.isclose(res.ub, ub_expected, rel_tol=1e-9)


def test_interval_division():
    a = Interval(1, 2)
    b = Interval(2, 3)
    c = a / b
    # c should be equivalent to a * (~b)
    inv_b = ~b
    expected = a * inv_b
    assert c == expected

    d = a / 2.0
    assert isinstance(d, Interval)
    d2 = 2.0 / a
    assert isinstance(d2, Interval)


def test_interval_power_even_and_odd():
    i_even = Interval(-1, 2)
    # power=2 triggers _power_even with monotonic decorator => Interval/float
    result_even = i_even ** 2
    assert result_even.lb <= result_even.ub
    # power=3 triggers _power_odd
    i_odd = Interval(2, 3)
    result_odd = i_odd ** 3
    assert math.isclose(result_odd.lb, 8)
    assert math.isclose(result_odd.ub, 27)



# Tests for intervallum/inclusion_functions.py RS_SubinclusionFunction

def test_rs_subinclusion_function_returns_interval(monkeypatch):
    # Fix randomness to a known value for reproducibility
    monkeypatch.setattr("intervallum.inclusion_functions.np.random.uniform", lambda low, high: np.array(low) + 0.5 * (np.array(high) - np.array(low)))
    f = lambda x: np.sum(x)
    box = Box(Interval(0, 1), Interval(2, 3))
    rs = RS_SubinclusionFunction(f, 5)
    result = rs(box)
    assert isinstance(result, Interval)
    # Because randomness fixed at midpoint, min and max values == the same sum
    midpoint_sum = sum([0.5 * (dim.lb + dim.ub) if isinstance(dim, Interval) else dim for dim in box])
    assert math.isclose(result.lb, midpoint_sum)
    assert math.isclose(result.ub, midpoint_sum)



# Tests for intervallum/interval_functions.py functions

def test_sin_cos_exp_abs_with_floats_and_intervals():
    i = Interval(0, math.pi)
    s = sin(i)
    c = cos(i)
    e = exp(i)
    a = interval_abs(i)
    # All should be Interval or float and in correct range
    assert s.lb >= -1.0 and s.ub <= 1.0
    assert c.lb >= -1.0 and c.ub <= 1.0
    assert e.lb > 0
    assert a.lb >= 0
    s_float = sin(0.5)
    c_float = cos(1.0)
    e_float = exp(0.0)
    a_float = interval_abs(-1.0)
    assert isinstance(s_float, float)
    assert isinstance(c_float, float)
    assert isinstance(e_float, float)
    assert isinstance(a_float, float)


def test_sqrt_and_log_behavior_and_exceptions():
    i = Interval(1, 4)
    sqrt_i = sqrt(i)
    assert sqrt_i.lb >= 1 and sqrt_i.ub <= 2
    i_negative = Interval(-1, 4)
    sqrt_neg_i = sqrt(i_negative)
    assert sqrt_neg_i.lb == 0.0
    assert sqrt_neg_i.ub <= 2
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-4, -1))
    i_log = Interval(1, math.e)
    log_i = log(i_log)
    assert log_i.lb >= 0.0
    i_log_neg = Interval(-1, 2)
    log_i_neg = log(i_log_neg)
    assert log_i_neg.lb == float('-inf')
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-4, 0))
    # log with float
    lfloat = log(1.0)
    assert isinstance(lfloat, (int,float))


def test_split_and_bisect_intervals():
    i = Interval(0, 10)
    parts = split(i, [1, 1, 2])
    assert len(parts) == 3
    total_width = sum((p.ub - p.lb if isinstance(p, Interval) else 0) for p in parts)
    assert math.isclose(total_width, i.width)
    bisected = bisect(i)
    assert len(bisected) == 2
    assert bisected[0].ub == bisected[1].lb


def test_constrain_and_shrink_functions():
    i = Interval(0, 10)
    constrained = constrain(i, 2, 8)
    assert constrained.lb >= 2 and constrained.ub <= 8
    constrained_float = constrain(15.0, 0, 10)
    assert constrained_float == 10
    shrunk = shrink(i, 0.5)
    expected_width = 0.5 * 0.5 * i.width
    assert math.isclose(shrunk.width, expected_width*2)
    shrunk_float = shrink(5, 0.5)
    assert shrunk_float == 5


def test_hausdorff_distance_behavior():
    i1 = Interval(0, 5)
    i2 = Interval(3, 7)
    dist = hausdorff_distance(i1, i2)
    # Should be distance from i1 to i2 or vice versa
    expected_dist = max(max(0, i2.lb - i1.lb), max(0, i1.ub - i2.ub))
    assert math.isclose(dist, expected_dist)
    d_equal = hausdorff_distance(i1, i1)
    assert d_equal == 0.0
    # Using floats instead of intervals
    dist_floats = hausdorff_distance(0, 5)
    assert dist_floats == 0.0
