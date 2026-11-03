import math
import numpy as np
import pytest
from copy import copy

from intervallum.box import Box
from intervallum.interval import Interval, IntervalExceptions, IntervalConstants
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin, cos, abs as abs_interval, exp, sqrt, log,
    split, bisect, constrain, shrink,
    _hausdorff_distance_part, hausdorff_distance,
)


# Helper for approximate equality of floats with tolerance
def approx_eq(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_box_new_and_properties():
    # Create box from intervals
    ib = Interval(1.0, 2.0)
    jb = Interval(3.0, 5.0)
    b = Box(ib, jb)
    np.testing.assert_array_equal(b.lb, np.array([1.0, 3.0]))
    np.testing.assert_array_equal(b.ub, np.array([2.0, 5.0]))
    middles = b.middle
    assert approx_eq(middles[0], 1.5)
    assert approx_eq(middles[1], 4.0)
    widths_idx, width_val = b.width
    assert width_val == 2.0  # 5-3=2 > 2-1=1, so max width is 2 from second
    assert 1 in widths_idx


def test_box_str_and_repr():
    b = Box(Interval(0, 1), Interval(2, 3))
    s = str(b)
    r = repr(b)
    expected = "[0; 1] x [2; 3]"
    assert s == expected
    assert r == expected


def test_box_eq_and_ne():
    b1 = Box(Interval(0, 1), Interval(2, 3))
    b2 = Box(Interval(0, 1), Interval(2, 3))
    b3 = Box(Interval(1, 2), Interval(2, 3))
    assert b1 == b2
    assert not (b1 != b2)
    assert b1 != b3
    with pytest.raises(NotImplementedError):
        assert (b1 == "not a box")


def test_box_copy_independent():
    b1 = Box(Interval(0, 1), Interval(2, 3))
    b2 = copy(b1)
    assert b1 == b2
    assert b1 is not b2
    for iv1, iv2 in zip(b1, b2):
        assert iv1 == iv2
        assert iv1 is not iv2


def test_box_try_to_reduce_returns_middle_or_copy():
    # Widths 0 => returns middle
    b = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    reduced = b._try_to_reduce()
    np.testing.assert_array_equal(reduced, np.array([1.0, 2.0]))
    # Widths > 0 => returns Box copy
    b2 = Box(Interval(1.0, 2.0), Interval(2.0, 3.0))
    reduced2 = b2._try_to_reduce()
    assert isinstance(reduced2, Box)
    assert reduced2 == b2
    assert reduced2 is not b2


def test_interval_basic_creation_and_properties():
    i = Interval(2.0, 5.0)
    assert i.lb == 2.0
    assert i.ub == 5.0
    assert approx_eq(i.middle, 3.5)
    assert approx_eq(i.width, 3.0)
    assert str(i) == "[2.0; 5.0]"
    assert repr(i) == "[2.0; 5.0]"


def test_interval_constructor_fix_parameter():
    # lower bound > upper bound with fix=True swaps them
    i_fixed = Interval(5.0, 3.0, fix=True)
    assert i_fixed.lb == 3.0
    assert i_fixed.ub == 5.0
    # Without fix=True, raises exception
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(5.0, 3.0)


def test_interval_lb_and_ub_setters():
    i = Interval(1.0, 3.0)
    i.lb = 0.5
    assert i.lb == 0.5
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 4.0  # can't set lb > ub
    i.ub = 4.0
    assert i.ub == 4.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0  # can't set ub < lb


def test_interval_try_to_reduce_behavior(monkeypatch):
    # Below reduction width => returns float middle
    IntervalConstants._reduction_width = 0.1
    i = Interval(1.0, 1.05)
    reduced = i._try_to_reduce()
    assert isinstance(reduced, float)
    assert approx_eq(reduced, 1.025)
    # Above reduction width => returns Interval copy
    i2 = Interval(1.0, 1.2)
    reduced2 = i2._try_to_reduce()
    assert isinstance(reduced2, Interval)
    assert reduced2.lb == i2.lb
    # Restore constant
    IntervalConstants._reduction_width = 1e-5


def test_interval_inner_subtraction_and_shift():
    i1 = Interval(2.0, 5.0)
    i2 = Interval(1.0, 3.0)
    # Seed numpy random in function body (no random here but per instructions)
    np.random.seed(123)
    result = Interval.inner_subtraction(i1, i2)
    # expected [2-1,5-3] = [1,2]
    assert approx_eq(result.lb, 1.0)
    assert approx_eq(result.ub, 2.0)
    # Using __lshift__
    shifted = i1 << i2
    assert shifted == result


def test_interval_equality_and_inequality():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(1.0 + 1e-8, 2.0 + 1e-8)
    # Should be equal within admissible error 1e-7
    assert i1 == i2
    assert not (i1 != i2)
    i3 = Interval(1.0, 2.0)
    val = 1.5
    assert i3 == val
    assert not (i3 != val)
    with pytest.raises(NotImplementedError):
        i3 == "not an interval"
    # Test NaN values in equality (force nan)
    nan_interval = Interval(float('nan'), float('nan'), fix=True)
    # NaNs in lb and ub replaced with 0 in __eq__
    assert nan_interval == nan_interval


def test_interval_add_neg_sub_mul_div_operations(monkeypatch):
    np.random.seed(111)
    # test addition
    i1 = Interval(1, 2)
    i2 = Interval(3, 5)
    sum_ = i1 + i2
    assert approx_eq(sum_.lb, 4)
    assert approx_eq(sum_.ub, 7)
    # addition with float
    s2 = i1 + 2
    assert approx_eq(s2.lb, 3)
    assert approx_eq(s2.ub, 4)
    s3 = 2 + i1
    assert approx_eq(s3.lb, 3)
    assert approx_eq(s3.ub, 4)
    # negation
    neg = -i1
    assert approx_eq(neg.lb, -2)
    assert approx_eq(neg.ub, -1)
    # subtraction
    sub = i1 - i2
    # equivalent to i1 + (-i2)
    assert sub == i1 + (-i2)
    # reverse subtraction
    rsub = 2 - i1
    assert rsub == (-i1) + 2
    # multiplication with intervals
    i3 = Interval(-1, 2)
    mult = i1 * i3
    products = [1*-1, 1*2, 2*-1, 2*2]
    assert approx_eq(mult.lb, min(products))
    assert approx_eq(mult.ub, max(products))
    # multiplication with float
    m2 = i1 * 3
    assert approx_eq(m2.lb, 3)
    assert approx_eq(m2.ub, 6)
    m3 = 3 * i1
    assert approx_eq(m3.lb, 3)
    assert approx_eq(m3.ub, 6)
    # division
    i4 = Interval(1, 2)
    div = i1 / i4
    # Should be equivalent to i1 * (~i4)
    inv = ~i4
    expected = i1 * inv
    assert div == expected
    # division by float
    div2 = i1 / 2
    assert approx_eq(div2.lb, 0.5)
    assert approx_eq(div2.ub, 1)
    # reverse truediv
    rtdiv = 3 / i1
    inv1 = ~i1
    expected2 = 3 * inv1
    assert rtdiv == expected2


def test_interval_inversion_behavior():
    # lb>0 case
    i1 = Interval(2, 3)
    inv = ~i1
    assert approx_eq(inv.lb, 1 / 3)
    assert approx_eq(inv.ub, 1 / 2)
    # ub<0 case
    i2 = Interval(-3, -1)
    inv2 = ~i2
    assert approx_eq(inv2.lb, 1 / -1)
    assert approx_eq(inv2.ub, 1 / -3)
    # zero in bounds lb==0
    i3 = Interval(0, 2)
    inv3 = ~i3
    assert approx_eq(inv3.lb, 1 / 2)
    assert math.isinf(inv3.ub) and inv3.ub > 0
    # zero in ub == 0
    i4 = Interval(-2, 0)
    inv4 = ~i4
    assert math.isinf(inv4.lb) and inv4.lb < 0
    assert approx_eq(inv4.ub, 1 / -2)
    # zero in middle (negative to positive)
    i5 = Interval(-1, 1)
    inv5 = ~i5
    assert math.isinf(inv5.lb) and inv5.lb < 0
    assert math.isinf(inv5.ub) and inv5.ub > 0


def test_interval_power_even_odd():
    i = Interval(-2, 3)
    # even power uses _power_even decorated with reduce_result and monotonic
    res_even = i ** 2
    assert isinstance(res_even, Interval)
    assert res_even.lb <= 0
    assert res_even.ub == max([(-2)**2, 3**2, 0.0**2])
    # odd power uses _power_odd decorated with reduce_result
    res_odd = i ** 3
    assert isinstance(res_odd, Interval)
    assert res_odd.lb == (-2) ** 3
    assert res_odd.ub == 3 ** 3


def test_rs_subinclusionfunction_returns_correct_interval(monkeypatch):
    np.random.seed(42)
    # f(x): sum of components
    f = lambda x: float(np.sum(x))
    b = Box(Interval(0, 1), Interval(1, 2))
    rs = RS_SubinclusionFunction(f, number_of_samples=100)
    # call sets numpy seed inside itself; set seed here for reproducibility
    # monkeypatch np.random.uniform to deterministic
    def mock_uniform(low, high):
        # Always pick low to give predictable min and max
        return low
    monkeypatch.setattr("numpy.random.uniform", mock_uniform)
    result = rs(b)
    # minimum sum is sum of all lb (because uniform returns low)
    expected_min = sum(b.lb)
    # maximum sum >= expected_min
    assert isinstance(result, Interval)
    assert result.lb <= result.ub
    assert approx_eq(result.lb, expected_min)
    # Because mocked uniform returns always lb, max should be equal to min
    assert approx_eq(result.ub, expected_min)


def test_interval_functions_trig_and_abs_exp(monkeypatch):
    monkeypatch.setattr("numpy.random.seed", lambda seed=None: None)  # dummy to follow requirements
    # sin: interval within [pi/2, 3*pi/2]
    from math import pi
    i = Interval(pi/2, 3*pi/2)
    s = sin(i)
    assert isinstance(s, Interval)
    # sin ranges between -1 and 1
    assert s.lb >= -1 and s.ub <= 1
    # cos
    c = cos(i)
    assert isinstance(c, Interval)
    # cos could range between -1 and 1
    assert c.lb >= -1 and c.ub <= 1
    # abs
    i2 = Interval(-1, 2)
    a = abs_interval(i2)
    assert a.lb == 0  # abs min at zero
    assert a.ub == 2
    # exp
    i3 = Interval(0, 1)
    e = exp(i3)
    assert math.exp(i3.lb) >= e.lb - 1e-9 and math.exp(i3.ub) <= e.ub + 1e-9


def test_sqrt_log_behavior_and_exceptions():
    # sqrt valid interval
    i = Interval(0, 4)
    s = sqrt(i)
    assert isinstance(s, Interval)
    assert s.lb == 0
    assert approx_eq(s.ub, 2)
    # sqrt with negative ub => raises
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-3, -1))
    # sqrt with lb < 0 < ub
    s2 = sqrt(Interval(-1, 4))
    assert approx_eq(s2.lb, 0)
    assert approx_eq(s2.ub, 2)
    # log valid interval
    i2 = Interval(1, math.e)
    l = log(i2)
    assert approx_eq(l.lb, 0)
    assert approx_eq(l.ub, 1)
    # log with ub <= 0 => raises
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-1, 0))
    # log with lb < 0 < ub
    l2 = log(Interval(-1, 4))
    assert l2.lb == -math.inf
    assert l2.ub >= math.log(4)


def test_split_bisect_function():
    i = Interval(0, 10)
    intervals = split(i, [1, 3, 1])
    assert len(intervals) == 3
    width_sum = sum([iv.width if isinstance(iv, Interval) else 0 for iv in intervals])
    assert approx_eq(width_sum, 10)
    # bisect is split with [1,1]
    bis = bisect(i)
    assert len(bis) == 2
    assert approx_eq(bis[0].width, 5)
    assert approx_eq(bis[1].width, 5)
    # split on float returns list of that float
    f = 3.14
    split_f = split(f, [1, 2])
    assert split_f == [f, f]


def test_constrain_and_shrink_functions():
    i = Interval(0, 10)
    # constrain clamps interval bounds within min, max
    c1 = constrain(i, 3, 8)
    assert c1.lb >= 3
    assert c1.ub <= 8
    c2 = constrain(5.0, 3, 8)
    assert c2 == 5.0
    c3 = constrain(2.0, 3, 8)
    assert c3 == 3.0
    c4 = constrain(9.0, 3, 8)
    assert c4 == 8.0
    # shrink with alpha = 0 returns middle interval (width 0)
    sh = shrink(i, 0)
    assert approx_eq(sh.lb, sh.ub)
    assert approx_eq(sh.lb, i.middle)
    # shrink with alpha = 1 returns original interval
    sh2 = shrink(i, 1)
    assert approx_eq(sh2.lb, i.lb)
    assert approx_eq(sh2.ub, i.ub)
    # shrink on float returns same float
    f = 5.5
    shf = shrink(f, 0.5)
    assert shf == f


def test_hausdorff_distance_correctness():
    i1 = Interval(0, 4)
    i2 = Interval(1, 3)
    d = hausdorff_distance(i1, i2)
    # hausdorff_distance is max over parts
    part1 = _hausdorff_distance_part(i1, i2)
    part2 = _hausdorff_distance_part(i2, i1)
    assert approx_eq(d, max(part1, part2))
    # It is never negative
    assert d >= 0
    # Distance between identical intervals is 0
    d_equal = hausdorff_distance(i1, i1)
    assert approx_eq(d_equal, 0)
    # Hausdorff distance with float converts floats to point intervals
    d_float = hausdorff_distance(i1, 2.5)
    assert d_float >= 0


def test_hausdorff_distance_part_values():
    # i1 bounds totally inside i2 => distance 0
    i1 = Interval(1, 2)
    i2 = Interval(0, 3)
    d = _hausdorff_distance_part(i1, i2)
    assert d == 0.0
    # i1 lower bound smaller than i2.lb
    i1 = Interval(-1, 1)
    i2 = Interval(0, 2)
    d = _hausdorff_distance_part(i1, i2)
    # d1 = i2.lb - i1.lb = 0 - (-1) = 1 > 0
    assert approx_eq(d, 1.0)
    # i1 upper bound greater than i2.ub
    i1 = Interval(0, 4)
    i2 = Interval(0, 3)
    d = _hausdorff_distance_part(i1, i2)
    # d2 = i1.ub - i2.ub = 4 - 3 = 1 > 0
    assert approx_eq(d, 1.0)
