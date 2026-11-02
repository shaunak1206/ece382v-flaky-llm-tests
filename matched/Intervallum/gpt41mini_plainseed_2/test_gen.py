import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import (
    Interval, IntervalConstants, IntervalExceptions,
    reduce_result, monotonic
)
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin, cos, abs as interval_abs, exp, sqrt, log,
    split, bisect, constrain, shrink,
    hausdorff_distance
)


# ---------- Tests for intervallum/interval.py ----------

def test_interval_creation_and_properties():
    i = Interval(1.0, 2.0)
    assert i.lb == 1.0
    assert i.ub == 2.0
    assert i.middle == 1.5
    assert i.width == 1.0
    assert str(i) == "[1.0; 2.0]"
    assert repr(i) == str(i)


def test_interval_wrong_bounds_exception():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    # fix=True will swap bounds
    i = Interval(2.0, 1.0, fix=True)
    assert i.lb == 1.0 and i.ub == 2.0


def test_lb_ub_setters():
    i = Interval(0, 10)
    i.lb = 2
    assert i.lb == 2
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 20  # lb > ub

    i.ub = 15
    assert i.ub == 15
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 1  # ub < lb


def test_try_to_reduce_reduces_small_width():
    i = Interval(1.0, 1.0 + IntervalConstants._reduction_width / 2)
    reduced = i._try_to_reduce()
    assert isinstance(reduced, float)
    expected = (i.lb + i.ub) / 2
    assert math.isclose(reduced, expected)

    i2 = Interval(1.0, 1.0 + IntervalConstants._reduction_width * 2)
    reduced2 = i2._try_to_reduce()
    assert isinstance(reduced2, Interval)
    assert reduced2.lb == i2.lb and reduced2.ub == i2.ub


def test_inner_subtraction_and_lshift_operator():
    i1 = Interval(2, 5)
    i2 = Interval(1, 3)
    res = i1 << i2
    # Should be interval [2-1,5-3] = [1,2]
    assert isinstance(res, Interval)
    assert math.isclose(res.lb, 1)
    assert math.isclose(res.ub, 2)

    # Also test with float
    res2 = i1 << 1.5
    assert math.isclose(res2.lb, 2 - 1.5)
    assert math.isclose(res2.ub, 5 - 1.5)


def test_interval_equality_and_inequality():
    i1 = Interval(1, 2)
    i2 = Interval(1, 2)
    i3 = Interval(1, 3)
    assert (i1 == i2) is True
    assert (i1 != i3) is True
    with pytest.raises(NotImplementedError):
        i1 == "string"


def test_interval_comparisons():
    i = Interval(1, 5)
    assert (i < 10) is True
    assert (i <= 1) is True
    assert (i > 0) is True
    assert (i >= 1) is True

    i2 = Interval(2, 6)
    assert (i < i2) is True
    assert (i <= i2) is True
    assert (i2 > i) is True
    assert (i2 >= i) is True


def test_interval_arithmetic_operations():
    i1 = Interval(1, 2)
    i2 = Interval(3, 4)

    # Addition
    a = i1 + i2
    assert isinstance(a, float) or isinstance(a, Interval)
    assert math.isclose(a.lb if isinstance(a, Interval) else a, 4)
    # Reverse addition
    b = 1 + i1
    assert math.isclose(b, 2)

    # Negation
    neg = -i1
    assert isinstance(neg, float) or isinstance(neg, Interval)
    assert math.isclose(neg if isinstance(neg, float) else neg.lb, -2)

    # Subtraction
    sub = i2 - i1
    assert math.isclose(sub, 2)

    # Multiplication
    mul = i1 * i2
    assert math.isclose(mul, 4)
    rmul = 2 * i1
    assert math.isclose(rmul, 2)

    # Inversion (__invert__)
    inv = ~Interval(1, 2)
    assert isinstance(inv, float) or isinstance(inv, Interval)
    assert inv.lb <= 1.0 and inv.ub >= 0.5

    inv_zero_lb = ~Interval(0, 2)
    assert inv_zero_lb.lb == 0.5 and inv_zero_lb.ub == math.inf

    inv_zero_ub = ~Interval(-2, 0)
    assert inv_zero_ub.lb == -math.inf and inv_zero_ub.ub == -0.5

    inv_cross = ~Interval(-1,1)
    assert inv_cross.lb == -math.inf and inv_cross.ub == math.inf

    # Division
    div = i2 / i1
    assert isinstance(div, float) or isinstance(div, Interval)

    rdiv = 5 / i2
    assert isinstance(rdiv, float) or isinstance(rdiv, Interval)


def test_interval_power_operations():
    i = Interval(-1, 2)

    # Even power triggers monotonic reduction
    p2 = i ** 2
    assert isinstance(p2, float) or isinstance(p2, Interval)
    # Odd power returns an Interval
    p3 = i ** 3
    assert isinstance(p3, float) or isinstance(p3, Interval)
    # Power with float
    p1 = i ** 1
    assert isinstance(p1, float) or isinstance(p1, Interval)


def test_reduce_result_decorator_behavior():
    @reduce_result
    def returns_interval():
        return Interval(0, 0.000001)

    original_reduction_state = IntervalConstants._reduce_intervals_to_numbers
    IntervalConstants._reduce_intervals_to_numbers = True
    reduced = returns_interval()
    assert isinstance(reduced, float)

    IntervalConstants._reduce_intervals_to_numbers = False
    unreduced = returns_interval()
    assert isinstance(unreduced, Interval)
    IntervalConstants._reduce_intervals_to_numbers = original_reduction_state


def test_monotonic_decorator_behavior():
    @monotonic
    def f():
        return lambda x: x * 2, [1, 2, 3]

    interval = f()
    assert isinstance(interval, Interval)
    assert interval.lb == 2
    assert interval.ub == 6


# ---------- Tests for intervallum/box.py ----------

def test_box_creation_and_str_repr():
    i1 = Interval(1, 2)
    i2 = Interval(3, 4)
    b = Box(i1, i2)
    assert isinstance(b, Box)
    assert str(b) == "[1; 2] x [3; 4]"
    assert repr(b) == "[1; 2] x [3; 4]"


def test_box_eq_and_ne():
    b1 = Box(Interval(1, 2), Interval(3, 4))
    b2 = Box(Interval(1, 2), Interval(3, 4))
    b3 = Box(Interval(0, 1), Interval(3, 4))
    assert b1 == b2
    assert b1 != b3
    with pytest.raises(NotImplementedError):
        b1 == "string"


def test_box_copy_and_properties():
    intervals = [Interval(1,2), Interval(3,4), 5]
    b = Box(*intervals)
    b_copy = b.__copy__()
    assert b == b_copy
    # Modify copy and check original intact
    if isinstance(b_copy[0], Interval):
        b_copy[0].lb = 0
        assert b[0].lb == 1

    # Test lb, ub, middle properties
    lbs = b.lb
    ubs = b.ub
    middles = b.middle
    assert np.allclose(lbs[:2], [1, 3])
    assert lbs[2] == 5
    assert np.allclose(ubs[:2], [2, 4])
    assert ubs[2] == 5
    assert np.allclose(middles[:2], [1.5, 3.5])
    assert middles[2] == 5

def test_box_width_and_try_to_reduce():
    # Width returns indices where width maximal and max_width
    i1 = Interval(0, 1)
    i2 = Interval(0, 2)
    b = Box(i1, i2)
    indices, max_width = b.width
    assert max_width == 2
    assert indices == [1]

    # Try to reduce when max_width > 0 returns copy
    res = b._try_to_reduce()
    assert isinstance(res, Box)

    # When max_width == 0 returns middle (numpy array)
    i3 = Interval(1, 1)
    b2 = Box(i3, i3)
    res2 = b2._try_to_reduce()
    assert isinstance(res2, np.ndarray)
    assert np.allclose(res2, [1.0, 1.0])


# ---------- Tests for intervallum/inclusion_functions.py ----------

def test_rs_subinclusion_function_calls_and_returns_interval():
    call_count = 0

    def f(x):
        nonlocal call_count
        call_count += 1
        return np.sum(x)

    b = Box(Interval(0,1), Interval(1,2))
    rs = RS_SubinclusionFunction(f, 10)
    result = rs(b)
    assert isinstance(result, Interval)
    assert call_count == 10
    # Values in interval should be within possible sums
    assert result.lb <= result.ub
    assert result.lb >= 0
    assert result.ub <= 3


# ---------- Tests for intervallum/interval_functions.py ----------

@pytest.mark.parametrize("func", [sin, cos, interval_abs, exp])
def test_trig_exp_abs_functions_on_float_and_interval(func):
    # On float
    res1 = func(1.0)
    assert isinstance(res1, float) or isinstance(res1, Interval)

    # On interval smaller than 2*pi
    i = Interval(0, math.pi)
    res2 = func(i)
    assert isinstance(res2, float) or isinstance(res2, Interval)
    assert hasattr(res2, "lb") or isinstance(res2, float)


def test_sqrt_log_operations():
    # Sqrt valid interval
    i = Interval(0, 4)
    res = sqrt(i)
    assert isinstance(res, float) or isinstance(res, Interval)
    assert res.lb >= 0

    # Sqrt invalid interval raises exception
    i2 = Interval(-1, -0.5)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(i2)

    # Log invalid interval raises exception
    i3 = Interval(-1, 0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(i3)

    # Log valid interval with lb < 0 but ub > 0
    i4 = Interval(-1, 1)
    res4 = log(i4)
    assert isinstance(res4, float) or isinstance(res4, Interval)


def test_split_and_bisect():
    i = Interval(0, 10)
    # split with ratios
    parts = split(i, [1, 2, 1])
    assert len(parts) == 3
    assert abs(sum(part.width if isinstance(part, Interval) else 0 for part in parts) - i.width) < 1e-10

    # bisect is split with two equal parts
    bis = bisect(i)
    assert len(bis) == 2
    assert math.isclose(bis[0].width, bis[1].width)
    assert bis[0].lb == i.lb
    assert bis[1].ub == i.ub


def test_constrain_shrink():
    i = Interval(1, 5)
    c = constrain(i, 2, 4)
    assert c.lb >= 2 and c.ub <= 4

    c2 = constrain(1.5, 1, 2)
    assert 1 <= c2 <= 2

    shrunk = shrink(i, 0.5)
    assert shrunk.width == pytest.approx(0.5 * i.width)
    shrunk_f = shrink(3.0, 0.5)
    assert shrunk_f == 3.0


def test_hausdorff_distance_calculation():
    i1 = Interval(0, 2)
    i2 = Interval(1, 3)
    dist = hausdorff_distance(i1, i2)
    assert dist >= 0

    # Distance between identical intervals is zero
    dist2 = hausdorff_distance(i1, i1)
    assert math.isclose(dist2, 0.0)

    # Distance with floats
    dist3 = hausdorff_distance(0.0, 1.0)
    assert dist3 == 1.0

