import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalExceptions, IntervalConstants
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin, cos, abs as iabs, exp, sqrt, log,
    split, bisect, constrain, shrink, hausdorff_distance,
)
from intervallum.interval import IntervalNumber


# ==== Tests for Interval class ====


def test_interval_creation_and_properties():
    i = Interval(1.0, 2.0)
    assert i.lb == 1.0
    assert i.ub == 2.0
    assert math.isclose(i.middle, 1.5)
    assert math.isclose(i.width, 1.0)
    assert str(i) == "[1.0; 2.0]"
    assert repr(i) == str(i)


def test_interval_wrong_bounds_raises():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)


def test_interval_wrong_bounds_fix_true():
    i = Interval(2.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 2.0


def test_interval_lb_ub_setter_valid_and_invalid():
    i = Interval(0.0, 1.0)
    i.lb = 0.5
    assert i.lb == 0.5
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 2.0
    i.ub = 2.0
    assert i.ub == 2.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0


def test_interval_eq_ne_with_float_and_interval():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(1.0, 2.0)
    i3 = Interval(1.0, 3.0)
    assert i1 == i2
    assert i1 != i3
    # close enough to float, difference less than admissible error
    i4 = Interval(1.00000000001, 2.0)
    assert i1 == 1.0
    assert not (i1 != 1.0)
    # raises for unsupported type
    with pytest.raises(NotImplementedError):
        i1 == "str"
    with pytest.raises(NotImplementedError):
        i1 != "str"


def test_interval_add_sub_neg_mul_div_pow():
    i1 = Interval(1, 2)
    i2 = Interval(3, 4)
    # Add intervals and floats
    r = i1 + i2
    assert isinstance(r, (Interval, float))
    assert r.lb == 4 and r.ub == 6

    r = i1 + 1.5
    assert math.isclose(r.lb, 2.5)
    # Subtraction
    r = i1 - i2
    assert isinstance(r, (Interval, float))
    # Negation
    neg = -i1
    assert math.isclose(neg.lb, -2)
    # Multiplication
    mul = i1 * i2
    assert math.isclose(mul.lb, 3) and math.isclose(mul.ub, 8)
    mul2 = i1 * 2.0
    assert mul2.lb == 2 and mul2.ub == 4
    # Division
    div = i2 / i1
    # i2 in [3,4], i1 in [1,2], so division range should be approx [1.5,4]
    assert div.lb <= 4 and div.ub >= 1.5
    div2 = 6 / i1
    assert div2.lb <= 6 and div2.ub >= 3
    # Power even
    i_even = Interval(-2, 3)
    p_even = i_even ** 2
    # check properties of returned Interval or float
    if isinstance(p_even, Interval):
        assert p_even.lb >= 0
    # Power odd
    i_odd = Interval(1, 3)
    p_odd = i_odd ** 3
    assert isinstance(p_odd, (Interval, float))
    # Only test for reduced values (float) as well
    small_i = Interval(1, 1 + 1e-6)
    p_small = small_i ** 2
    assert isinstance(p_small, float)


def test_interval_invert_and_inner_subtraction():
    i = Interval(2, 4)
    inv = ~i
    assert inv.lb <= 1/4 and inv.ub >= 1/2
    i_zero_lb = Interval(0, 4)
    inv_zero_lb = ~i_zero_lb
    assert math.isinf(inv_zero_lb.ub)
    i_zero_ub = Interval(-1, 0)
    inv_zero_ub = ~i_zero_ub
    assert math.isinf(inv_zero_ub.lb)
    # Inner subtraction returns Interval and respects bounds fixed=True
    i1 = Interval(5, 7)
    i2 = Interval(1, 3)
    sub = Interval.inner_subtraction(i1, i2)
    assert sub.lb == 4 and sub.ub == 6


def test_interval_try_to_reduce_behavior():
    # Interval with zero width reduced to float
    i = Interval(1.0, 1.0 + 1e-6)
    IntervalConstants._reduction_width = 1e-5
    reduced = i._try_to_reduce()
    assert isinstance(reduced, float)
    # Interval with width greater than threshold remains Interval
    i_wide = Interval(1.0, 1.1)
    not_reduced = i_wide._try_to_reduce()
    assert isinstance(not_reduced, Interval)


# ==== Tests for Box class ====


def create_box_with_intervals_and_floats():
    return Box(Interval(0, 1), Interval(2, 3), 5)


def test_box_str_and_repr_and_equality_and_copy():
    b = create_box_with_intervals_and_floats()
    s = str(b)
    r = repr(b)
    assert isinstance(s, str)
    assert isinstance(r, str)
    b2 = copy(b)
    assert b == b2
    assert b is not b2
    # Equality with np.ndarray of same content
    arr = np.array(b)
    assert b == arr
    # Inequality check
    b3 = create_box_with_intervals_and_floats()
    b3[0] = Interval(10, 20)
    assert b != b3


def test_box_properties_lb_ub_middle_width():
    b = create_box_with_intervals_and_floats()
    lb = b.lb
    ub = b.ub
    middle = b.middle
    assert isinstance(lb, np.ndarray)
    assert isinstance(ub, np.ndarray)
    assert isinstance(middle, np.ndarray)
    expected_lb = np.array([0, 2, 5])
    expected_ub = np.array([1, 3, 5])
    expected_middle = np.array([0.5, 2.5, 5])
    np.testing.assert_array_equal(lb, expected_lb)
    np.testing.assert_array_equal(ub, expected_ub)
    np.testing.assert_array_equal(middle, expected_middle)
    # width returns indices and max width correctly
    indices, w = b.width
    assert isinstance(indices, list)
    assert isinstance(w, float)
    assert math.isclose(w, 1)  # width of Interval(0,1) and Interval(2,3) is 1, float 5 width is 0
    # the indices where width is max include first two intervals (width=1)
    assert set(indices) == {0, 1}


def test_box_try_to_reduce_returns_middle_or_copy():
    # Box with zero max width returns middle np.ndarray
    b = Box(Interval(5, 5), Interval(5, 5))
    indices, max_width = b.width
    assert max_width == 0.0
    result = b._try_to_reduce()
    assert isinstance(result, np.ndarray)
    # Box with positive width returns a Box
    b2 = Box(Interval(0, 1), Interval(0, 2))
    result2 = b2._try_to_reduce()
    assert isinstance(result2, Box)
    assert result2 == b2


# ==== Tests for RS_SubinclusionFunction ====


def test_rs_subinclusionfunction_call_behavior():
    # Define a simple function f: np.ndarray -> float
    def f(x):
        return x[0] + x[1]

    f_wrapper = RS_SubinclusionFunction(f, number_of_samples=10)
    box = Box(Interval(0, 1), Interval(1, 2))
    result = f_wrapper(box)
    assert isinstance(result, Interval)
    # The interval lb <= ub and covers possible sums between bounds
    assert result.lb <= result.ub
    assert result.lb >= 0
    assert result.ub <= 3


# ==== Tests for functions in intervallum.interval_functions ====


@pytest.mark.parametrize("func", [sin, cos, iabs, exp])
def test_monotonic_functions_with_float_and_interval(func):
    # For float inputs returns Interval reduced to float
    result_f = func(0.5)
    assert isinstance(result_f, (float, Interval))
    # For Interval inputs returns Interval
    i = Interval(0, math.pi / 2)
    result_i = func(i)
    assert isinstance(result_i, Interval)
    # The lb should be <= ub
    assert result_i.lb <= result_i.ub


def test_sqrt_and_log_valid_and_invalid():
    # sqrt with positive interval returns Interval
    i_pos = Interval(1, 4)
    r = sqrt(i_pos)
    assert isinstance(r, Interval)
    # sqrt with interval containing zero returns Interval
    i_zero = Interval(-1, 4)
    r = sqrt(i_zero)
    assert isinstance(r, Interval)
    # sqrt with negative interval raises
    i_neg = Interval(-4, -1)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(i_neg)
    # log with positive interval returns Interval
    i_pos = Interval(1, 10)
    r2 = log(i_pos)
    assert isinstance(r2, Interval)
    # log with interval containing zero or negative parts returns Interval
    i_zero_neg = Interval(-1, 10)
    r3 = log(i_zero_neg)
    assert isinstance(r3, Interval)
    # log with interval fully non-positive raises
    i_neg_full = Interval(-5, 0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(i_neg_full)


def test_split_and_bisect_for_interval_and_float():
    i = Interval(0, 10)
    parts = split(i, [1, 2, 1])
    assert len(parts) == 3
    total_width = sum(p.width if isinstance(p, Interval) else 0 for p in parts)
    assert math.isclose(total_width, i.width)
    # bisect returns two parts of equal width
    halves = bisect(i)
    assert len(halves) == 2
    assert math.isclose(halves[0].width, halves[1].width)
    f = 5.0
    parts_float = split(f, [1, 2])
    assert all(p == f for p in parts_float)
    halves_float = bisect(f)
    assert halves_float == [f, f]


def test_constrain_interval_and_float():
    i = Interval(0, 10)
    constrained = constrain(i, 3, 7)
    assert constrained.lb >= 3 and constrained.ub <= 7
    f = 20.0
    constrained_f = constrain(f, 0, 10)
    assert constrained_f == 10


def test_shrink_interval_and_float_behavior():
    i = Interval(2, 6)
    shrunk = shrink(i, 0.5)
    assert isinstance(shrunk, Interval)
    # width after shrink should be half of original's width * alpha
    expected_width = i.width * 0.5 * 0.5
    assert math.isclose(shrunk.width, expected_width)
    f = 5.0
    shrunk_f = shrink(f, 0.5)
    assert shrunk_f == f


def test_hausdorff_distance_symmetry_and_nonnegative():
    i1 = Interval(0, 2)
    i2 = Interval(1, 3)
    d = hausdorff_distance(i1, i2)
    d2 = hausdorff_distance(i2, i1)
    assert math.isclose(d, d2)
    assert d >= 0
    # Distance with float and interval
    d3 = hausdorff_distance(i1, 1.5)
    assert d3 >= 0
    d4 = hausdorff_distance(1.5, i2)
    assert d4 >= 0
    # Distance between identical intervals is zero or close
    assert hausdorff_distance(i1, i1) < 1e-9


# Specific tests for Interval methods that cause NotImplementedError


def test_interval_eq_raises_not_implemented_error_for_unsupported_type():
    i = Interval(0, 1)
    with pytest.raises(NotImplementedError):
        i == [1, 2]
    with pytest.raises(NotImplementedError):
        i != {}


def test_box_eq_raises_not_implemented_error_for_unsupported_type():
    b = Box(Interval(0,1))
    with pytest.raises(NotImplementedError):
        b == "string"
