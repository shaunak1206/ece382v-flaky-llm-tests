import math
import pytest
import numpy as np

from intervallum.box import Box
from intervallum.interval import (
    Interval, IntervalExceptions, IntervalConstants
)
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin, cos, abs, exp, sqrt, log,
    split, bisect, constrain, shrink,
    hausdorff_distance
)


# Helper function to compare floats with tolerance
def approx_equal(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_box_creation_and_str_repr():
    intervals = [Interval(1, 2), Interval(3, 4)]
    b = Box(*intervals)
    # Check that Box behaves like ndarray of Intervals
    assert isinstance(b, np.ndarray)
    # __str__ returns joined string representations
    s = str(b)
    r = repr(b)
    for i in intervals:
        assert str(i) in s
        assert repr(i) in r


def test_box_equality_and_inequality():
    b1 = Box(Interval(1, 2), Interval(3, 4))
    b2 = Box(Interval(1, 2), Interval(3, 4))
    b3 = Box(Interval(1, 2), Interval(3, 5))
    assert b1 == b2
    assert b1 != b3
    # Comparing with non-ndarray should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        _ = (b1 == "not a Box or ndarray")


def test_box_copy_independence():
    b1 = Box(Interval(1, 2), Interval(3, 4))
    b2 = b1.__copy__()
    assert b1 == b2
    # Changing b2 intervals does not affect b1
    b2[0].lb = 0
    assert b1[0].lb == 1
    assert b2[0].lb == 0


def test_box_properties_lb_ub_middle_width():
    intervals = [Interval(1, 2), Interval(0, 5), Interval(-1, 4)]
    b = Box(*intervals)
    lbs = b.lb
    ubs = b.ub
    middles = b.middle
    widths_indices, max_width = b.width

    # Check arrays
    assert np.allclose(lbs, [1, 0, -1])
    assert np.allclose(ubs, [2, 5, 4])
    assert np.allclose(middles, [(1+2)/2, (0+5)/2, (-1+4)/2])
    # max width is 5 (0 to 5)
    assert math.isclose(max_width, 5)
    # index of max width should be [1]
    assert widths_indices == [1]

    # Also test when some elements are floats, width 0 expected there
    b_mixed = Box(Interval(1, 3), 5.0, Interval(2, 4))
    indices, max_w = b_mixed.width
    # widths are [2, 0, 2], max is 2, indices [0, 2]
    assert max_w == 2
    assert sorted(indices) == [0, 2]


def test_box_try_to_reduce_returns_middle_or_copy():
    b_zero_width = Box(Interval(1, 1), Interval(2, 2))
    # All intervals with zero width, should return middle points (ndarray)
    reduced = b_zero_width._try_to_reduce()
    assert isinstance(reduced, np.ndarray)
    expected = np.array([1, 2])
    assert np.allclose(reduced, expected)

    b_nonzero_width = Box(Interval(1, 2), Interval(2, 3))
    reduced2 = b_nonzero_width._try_to_reduce()
    # Should return a Box (copy)
    assert isinstance(reduced2, Box)
    # Should be equal but not same object
    assert reduced2 == b_nonzero_width
    assert reduced2 is not b_nonzero_width


def test_interval_creation_and_str_repr():
    i = Interval(1.0, 2.0)
    assert i.lb == 1.0
    assert i.ub == 2.0
    s = str(i)
    r = repr(i)
    assert s == r
    assert "[1.0; 2.0]" == s


def test_interval_wrong_bounds_exception_raise():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    # fix=True swaps bounds
    i = Interval(2.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 2.0


def test_interval_setters_enforce_bounds():
    i = Interval(1.0, 2.0)
    # Setting lb above ub raises
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    # Setting ub below lb raises
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0
    # Valid settings update
    i.lb = 0.0
    assert i.lb == 0.0
    i.ub = 5.0
    assert i.ub == 5.0


def test_interval_equality_and_inequality_with_float_and_interval():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(1.0, 2.0)
    assert i1 == i2
    assert not (i1 != i2)
    # Compare with float within acceptable error
    approx_value = 1.5
    i3 = Interval(1.50000000001, 1.50000000002)
    assert i3 == approx_value
    # Not equal with larger difference
    i4 = Interval(1.4, 1.6)
    assert not (i4 == 1.3)
    # Comparing with unsupported type raises
    with pytest.raises(NotImplementedError):
        _ = (i1 == "something")


def test_interval_add_sub_negation_mul_div_pow_operations():
    i1 = Interval(1, 2)
    i2 = Interval(3, 4)
    # Addition
    r = i1 + i2
    assert math.isclose(r.lb, 4)
    assert math.isclose(r.ub, 6)
    # Subtraction
    r2 = i2 - i1
    assert math.isclose(r2.lb, 1)
    assert math.isclose(r2.ub, 3)
    # Negation
    r3 = -i1
    assert math.isclose(r3.lb, -2)
    assert math.isclose(r3.ub, -1)
    # Multiplication with interval
    r4 = i1 * i2
    # products: [1*3, 1*4, 2*3, 2*4] = [3,4,6,8]
    assert math.isclose(r4.lb, 3)
    assert math.isclose(r4.ub, 8)
    # Division by float
    r5 = i2 / 2
    assert math.isclose(r5.lb, 1.5)
    assert math.isclose(r5.ub, 2)
    # Power even exponent
    i3 = Interval(-2, 3)
    p = i3 ** 2  # should contain 0 and edges
    assert isinstance(p, Interval)
    assert p.lb <= 0
    assert p.ub >= 9
    # Power odd exponent
    p2 = i3 ** 3
    assert isinstance(p2, Interval)
    assert p2.lb <= -8
    assert p2.ub >= 27


def test_interval_inverse_cases_and_truediv():
    # Interval strictly positive
    i = Interval(1, 2)
    inv = ~i
    assert math.isclose(inv.lb, 1/2)
    assert math.isclose(inv.ub, 1/1)
    # Interval strictly negative
    j = Interval(-3, -1)
    inv_j = ~j
    assert math.isclose(inv_j.lb, 1/(-1))
    assert math.isclose(inv_j.ub, 1/(-3))
    # Interval contains zero lb=0
    k = Interval(0, 2)
    inv_k = ~k
    assert math.isclose(inv_k.lb, 1/2)
    assert inv_k.ub == math.inf
    # Interval contains zero ub=0
    m = Interval(-2, 0)
    inv_m = ~m
    assert inv_m.lb == -math.inf
    assert math.isclose(inv_m.ub, 1/(-2))
    # Division
    div = i / j
    # i / j = i * ~j
    expected = i * inv_j
    assert div == expected
    div_r = j / 2
    expected_r = j * (1/2)
    assert div_r == expected_r


def test_reduce_result_decorator_and_try_to_reduce_behavior(monkeypatch):
    # Ensure IntervalConstants._reduce_intervals_to_numbers True returns float when width < threshold
    i = Interval(1.0, 1.0 + IntervalConstants._reduction_width / 2)
    value = i._try_to_reduce()
    assert isinstance(value, float)
    # Patch IntervalConstants to disable reduction
    monkeypatch.setattr("intervallum.interval.IntervalConstants._reduce_intervals_to_numbers", False)
    value2 = i._try_to_reduce()
    assert isinstance(value2, Interval)
    # The decorated method respects this as well
    @Interval.reduce_result
    def f():
        return Interval(1, 1 + IntervalConstants._reduction_width / 2)
    monkeypatch.setattr("intervallum.interval.IntervalConstants._reduce_intervals_to_numbers", True)
    v = f()
    assert isinstance(v, float) or isinstance(v, Interval)
    monkeypatch.setattr("intervallum.interval.IntervalConstants._reduce_intervals_to_numbers", False)
    v2 = f()
    assert isinstance(v2, Interval)


def test_monotonic_decorator_applies_correct_bounds():
    # The monotonic decorator returns interval over min and max values on points
    def f():
        def func(x):
            return x**2
        return func, [-2, 0, 3]
    decorated = Interval.monotonic(f)
    result = decorated()
    assert math.isclose(result.lb, 0)
    assert math.isclose(result.ub, 9)


def test_RS_SubinclusionFunction_behavior(monkeypatch):
    np.random.seed(123)
    import random
    random.seed(123)
    # The function passed returns the first coordinate's square
    def f(x):
        return x[0] ** 2
    box = Box(Interval(1, 2), Interval(3, 4))
    rsf = RS_SubinclusionFunction(f, number_of_samples=100)
    # Seed numpy random explicitly for reproducibility
    np.random.seed(123)
    # Seed python random explicitly
    random.seed(123)
    res = rsf(box)
    # The result should be an Interval between min and max of sampled squared first coords
    assert isinstance(res, Interval)
    # Check that the result contains values between 1 and 4 (since first coord in [1,2])
    assert 1 <= res.lb <= 4
    assert 1 <= res.ub <= 4
    # The lower bound <= upper bound
    assert res.lb <= res.ub


def test_interval_functions_trig_exp_abs_log_sqrt(monkeypatch):
    # Seed to ensure reproducibility (even though no random here, see instructions)
    np.random.seed(42)
    import random
    random.seed(42)
    # sin of a float
    v = sin(math.pi/2)
    assert approx_equal(v, 1.0)
    # sin over interval with width < 2pi
    i = Interval(0, math.pi)
    s = sin(i)
    assert isinstance(s, Interval)
    # cos with large interval width >= 2pi
    large_i = Interval(0, 2*math.pi)
    c = cos(large_i)
    assert isinstance(c, Interval)
    # abs with interval crossing zero
    cross_i = Interval(-1, 1)
    a = abs(cross_i)
    assert a.lb == 0.0
    # exp with float and interval
    e_float = exp(1.0)
    e_int = exp(Interval(0, 1))
    assert isinstance(e_float, float) or isinstance(e_float, Interval)  # reduced to float
    assert isinstance(e_int, Interval)
    # sqrt non-negative interval
    sqrt_i = sqrt(Interval(0, 4))
    assert math.isclose(sqrt_i.lb, 0.0)
    assert math.isclose(sqrt_i.ub, 2.0)
    # sqrt negative ub should raise
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-2, -1))
    # sqrt partially negative should work
    sqrt2 = sqrt(Interval(-1, 4))
    assert sqrt2.lb == 0.0
    # log positive interval
    log_i = log(Interval(1, math.e))
    assert isinstance(log_i, Interval)
    # log zero or negative should raise
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-1, 0))
    # log partially negative
    log2 = log(Interval(-1, 2))
    assert log2.lb <= 0


def test_split_and_bisect_intervals(monkeypatch):
    i = Interval(0, 10)
    parts = split(i, [1, 1, 2])
    total_width = sum(p.width if isinstance(p, Interval) else 0 for p in parts)
    # Total width should be close to original width
    assert approx_equal(total_width, i.width)
    # lengths proportional to ratios: first two should have width 2.5, last 5
    expected_widths = [2.5, 2.5, 5]
    for p, w in zip(parts, expected_widths):
        assert approx_equal(p.width, w)
    # split float returns repeated floats
    fl = 5.0
    parts_f = split(fl, [1,2])
    assert parts_f == [5.0, 5.0]

    # bisect splits evenly
    bis = bisect(i)
    assert len(bis) == 2
    assert approx_equal(bis[0].width, bis[1].width)
    assert approx_equal(sum(p.width for p in bis), i.width)


def test_constrain_and_shrink_interval(monkeypatch):
    i = Interval(-5, 5)
    # constrain clamps lower and upper bounds
    c = constrain(i, -3, 3)
    assert c.lb >= -3 and c.ub <= 3
    # Constrain float clamps values
    c2 = constrain(10, 0, 5)
    assert c2 == 5
    # shrink reduces width
    s = shrink(i, 0.4)
    expected_width = i.width * 0.4
    assert approx_equal(s.width, expected_width)
    # shrink float returns float unchanged
    sh_f = shrink(2.0, 0.5)
    assert sh_f == 2.0


def test_hausdorff_distance_basic():
    i1 = Interval(0, 10)
    i2 = Interval(2, 8)
    dist = hausdorff_distance(i1, i2)
    # Distance should be max( (2-0)_+, (10-8)_+, (0-2)_+, (8-10)_+ ) == 2
    assert math.isclose(dist, 2)
    # Distance zero for equal intervals or points
    i3 = Interval(5, 5)
    dist2 = hausdorff_distance(i3, 5)
    assert approx_equal(dist2, 0)
    # Distance symmetric
    dist3 = hausdorff_distance(i2, i1)
    assert math.isclose(dist, dist3)

