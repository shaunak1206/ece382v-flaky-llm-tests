import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalExceptions, IntervalConstants
from intervallum.inclusion_functions import RS_SubinclusionFunction
import intervallum.interval_functions as interval_functions


# -------- Tests for Interval class --------

def test_interval_creation_and_properties():
    i = Interval(1.0, 2.0)
    assert i.lb == 1.0
    assert i.ub == 2.0
    assert i.middle == 1.5
    assert i.width == 1.0

def test_interval_creation_with_wrong_bounds_and_fix_false_raises():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(3.0, 2.0, fix=False)

def test_interval_creation_with_wrong_bounds_and_fix_true_swaps_bounds():
    i = Interval(3.0, 2.0, fix=True)
    assert i.lb == 2.0
    assert i.ub == 3.0

def test_interval_lb_setter_valid_and_invalid():
    i = Interval(1.0, 3.0)
    i.lb = 2.0
    assert i.lb == 2.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 4.0  # greater than ub

def test_interval_ub_setter_valid_and_invalid():
    i = Interval(1.0, 3.0)
    i.ub = 4.0
    assert i.ub == 4.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.5  # less than lb

def test_interval_try_to_reduce_and_equality():
    i = Interval(1.00000001, 1.00000002)
    # small width, should reduce to number
    reduced = i._try_to_reduce()
    assert isinstance(reduced, float)
    # equality within admissible error
    i2 = Interval(1.000000015, 1.000000025)
    assert i == i2
    # not equal if difference is large
    i3 = Interval(1.0, 1.1)
    assert i3 != i

def test_interval_addition_and_negation_and_subtraction():
    i1 = Interval(1, 2)
    i2 = Interval(3, 4)
    r = i1 + i2
    assert isinstance(r, (Interval, float))
    assert math.isclose(r.lb, 4)
    assert math.isclose(r.ub, 6)
    r2 = -i1
    assert isinstance(r2, (Interval, float))
    assert math.isclose(r2.lb, -2)
    assert math.isclose(r2.ub, -1)
    r3 = i1 - i2
    assert isinstance(r3, (Interval, float))
    assert math.isclose(r3.lb, -3)
    assert math.isclose(r3.ub, -1)

def test_interval_multiplication_and_division():
    i1 = Interval(2, 3)
    i2 = Interval(4, 5)
    r = i1 * i2
    assert isinstance(r, (Interval, float))
    assert math.isclose(r.lb, 8)
    assert math.isclose(r.ub, 15)
    r2 = i2 / i1
    assert isinstance(r2, (Interval, float))
    assert r2.lb <= r2.ub  # bounds correct
    # division by scalar
    r3 = i2 / 2
    assert math.isclose(r3.lb, 2)
    assert math.isclose(r3.ub, 2.5)

def test_interval_invert_operator():
    # positive interval
    i = Interval(1, 2)
    inv = ~i
    assert isinstance(inv, (Interval, float))
    assert math.isclose(inv.lb, 1/2)
    assert math.isclose(inv.ub, 1/1)
    # interval including zero
    i_zero = Interval(-1, 1)
    inv_zero = ~i_zero
    assert isinstance(inv_zero, (Interval, float))
    assert inv_zero.lb == -math.inf
    assert inv_zero.ub == math.inf

def test_interval_power_even_and_odd():
    i = Interval(-1, 2)
    square = i ** 2
    assert isinstance(square, (Interval, float))
    assert square.lb <= square.ub
    cube = i ** 3
    assert isinstance(cube, (Interval, float))
    assert cube.lb <= cube.ub
    # test single point interval reduces to number
    i_single = Interval(2, 2)
    square_single = i_single ** 2
    assert isinstance(square_single, float)

def test_interval_eq_raises_with_invalid_type():
    i = Interval(1, 2)
    with pytest.raises(NotImplementedError):
        _ = (i == "string")

# -------- Tests for Box class --------

def test_box_creation_and_str_repr_eq():
    i1 = Interval(1, 2)
    i2 = Interval(3, 4)
    b = Box(i1, i2)
    assert isinstance(b, Box)
    s = str(b)
    r = repr(b)
    assert str(i1) in s and str(i2) in s
    assert repr(i1) in r and repr(i2) in r
    b2 = Box(i1, i2)
    assert b == b2
    b3 = Box(1, 2)
    b4 = Box(1, 2)
    assert b3 == b4
    with pytest.raises(NotImplementedError):
        b == "non-box"

def test_box_properties_lb_ub_middle_width():
    i1 = Interval(1, 3)  # width=2
    i2 = Interval(2, 5)  # width=3
    b = Box(i1, i2)
    lbs = b.lb
    ubs = b.ub
    middles = b.middle
    assert np.allclose(lbs, [1, 2])
    assert np.allclose(ubs, [3, 5])
    assert np.allclose(middles, [2, 3.5])
    max_indices, max_width = b.width
    assert max_width == 3
    assert 1 in max_indices  # index 1 has max width

def test_box_try_to_reduce_returns_middle_if_zero_width():
    i1 = Interval(1, 1)
    b = Box(i1, i1)
    # width is zero for both intervals, so width[1] == 0
    reduced = b._try_to_reduce()
    assert isinstance(reduced, np.ndarray)
    # values match middle values (which equal lb and ub)
    assert np.allclose(reduced, b.middle)

def test_box_try_to_reduce_returns_copy_if_nonzero_width():
    i1 = Interval(1, 2)
    b = Box(i1, i1)
    reduced = b._try_to_reduce()
    assert isinstance(reduced, Box)
    assert reduced == b
    assert reduced is not b  # must be a copy

def test_box_copy_independence():
    i1 = Interval(1, 2)
    b = Box(i1)
    b_copy = b.__copy__()
    assert b == b_copy
    b[0].lb = 1.5
    # Original interval lb changed should not affect copy’s lb
    assert b_copy[0].lb == 1

# -------- Tests for RS_SubinclusionFunction --------

def test_rs_subinclusion_function_basic_behavior():
    def f(x):
        return x[0] + 1
    box = Box(Interval(0, 2))
    rs = RS_SubinclusionFunction(f, number_of_samples=10)
    result = rs(box)
    assert isinstance(result, Interval)
    assert result.lb <= result.ub
    assert result.lb >= 1  # minimum f(x) = 0 + 1 =1
    assert result.ub <= 3  # maximum f(x) = 2 + 1 =3

def test_rs_subinclusion_function_multiple_dims():
    dims = 3
    def f(x):
        return sum(x)
    intervals = [Interval(0, 1) for _ in range(dims)]
    box = Box(*intervals)
    rs = RS_SubinclusionFunction(f, number_of_samples=100)
    result = rs(box)
    assert isinstance(result, Interval)
    assert result.lb >= 0
    assert result.ub <= 3


# -------- Tests for interval_functions.py --------

@pytest.mark.parametrize("func, input_val, expected_min, expected_max", [
    (interval_functions.sin, Interval(0, math.pi / 2), math.sin(0), math.sin(math.pi / 2)),
    (interval_functions.cos, Interval(0, math.pi), math.cos(math.pi), math.cos(0)),
    (interval_functions.abs, Interval(-2, 1), 0, 2),
    (interval_functions.exp, Interval(0, 1), math.exp(0), math.exp(1)),
])
def test_basic_interval_functions(func, input_val, expected_min, expected_max):
    result = func(input_val)
    assert isinstance(result, (Interval, float))
    assert result.lb <= result.ub
    assert expected_min - 1e-7 <= result.lb <= expected_max + 1e-7
    assert expected_min - 1e-7 <= result.ub <= expected_max + 1e-7

def test_sqrt_and_log_errors_and_behavior():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        interval_functions.sqrt(Interval(-2, -1))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        interval_functions.log(Interval(-2, -1))
    # sqrt with negative lb but positive ub clamps to zero
    sq = interval_functions.sqrt(Interval(-1, 4))
    assert sq.lb == 0
    assert sq.ub == 2
    # log with negative lb but positive ub clamps to zero
    lg = interval_functions.log(Interval(-1, 4))
    assert math.isinf(lg.lb) == False
    assert lg.ub > 0

def test_split_and_bisect_of_intervals():
    i = Interval(0, 10)
    splits = interval_functions.split(i, [1, 2, 1])
    assert len(splits) == 3
    total_width = sum((s.ub - s.lb) for s in splits)
    assert math.isclose(total_width, i.width)
    bisects = interval_functions.bisect(i)
    assert len(bisects) == 2
    assert bisects[0].width == bisects[1].width

def test_constrain_function_behavior():
    i = Interval(-5, 5)
    constrained = interval_functions.constrain(i, 0, 3)
    assert constrained.lb == 0
    assert constrained.ub == 3
    # float constraining
    val = interval_functions.constrain(-1, 0, 2)
    assert val == 0
    val2 = interval_functions.constrain(3, 0, 2)
    assert val2 == 2
    val3 = interval_functions.constrain(1.5, 0, 2)
    assert val3 == 1.5

def test_shrink_function_behavior():
    i = Interval(0, 4)
    shrunk = interval_functions.shrink(i, 0.5)
    expected_width = 0.5 * 0.5 * i.width  # half of half width = 0.5
    new_width = shrunk.ub - shrunk.lb
    assert math.isclose(new_width, i.width * 0.5)
    assert shrunk.middle == i.middle
    # float input returns the same float
    val = interval_functions.shrink(5.0, 0.5)
    assert val == 5.0

def test_hausdorff_distance_for_intervals_and_floats():
    i1 = Interval(0, 2)
    i2 = Interval(1, 3)
    dist = interval_functions.hausdorff_distance(i1, i2)
    assert dist >= 0
    # symmetric
    dist2 = interval_functions.hausdorff_distance(i2, i1)
    assert math.isclose(dist, dist2)
    # test with floats treated as intervals
    dist_floats = interval_functions.hausdorff_distance(1.0, 2.5)
    assert dist_floats >= 0
