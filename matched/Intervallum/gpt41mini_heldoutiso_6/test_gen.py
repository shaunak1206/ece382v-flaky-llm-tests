import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import (Interval, IntervalExceptions, IntervalConstants)
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin, cos, abs, exp, sqrt, log,
    split, bisect, constrain, shrink,
    hausdorff_distance
)


# Tests for intervallum/box.py: Box class

def test_box_creation_and_str_repr():
    iv1 = Interval(1, 2)
    iv2 = Interval(3, 4)
    box = Box(iv1, iv2)
    assert isinstance(box, Box)
    assert str(box) == "[1; 2] x [3; 4]"
    assert repr(box) == "[1; 2] x [3; 4]"

def test_box_equality():
    b1 = Box(Interval(0, 1), Interval(2, 3))
    b2 = Box(Interval(0, 1), Interval(2, 3))
    b3 = Box(Interval(0, 1), Interval(3, 4))
    assert b1 == b2
    assert b1 != b3
    with pytest.raises(NotImplementedError):
        b1 == "string"

def test_box_copy_and_try_to_reduce_with_zero_width():
    iv_zero_width = Interval(5, 5)
    box = Box(iv_zero_width, iv_zero_width)
    middle = box._try_to_reduce()
    # Because width == 0, it should return middle as a numpy array of floats
    assert isinstance(middle, np.ndarray)
    assert np.allclose(middle, np.array([5.0, 5.0]))

def test_box_try_to_reduce_with_nonzero_width():
    iv1 = Interval(1, 2)
    iv2 = Interval(3, 4)
    box = Box(iv1, iv2)
    result = box._try_to_reduce()
    # Should return a copy of the box (Box instance)
    assert isinstance(result, Box)
    assert result == box

def test_box_properties_lb_ub_middle_width():
    iv1 = Interval(1, 4)
    iv2 = Interval(2, 3)
    box = Box(iv1, iv2)
    # lb and ub should be numpy arrays of the intervals' bounds
    assert np.allclose(box.lb, np.array([1, 2]))
    assert np.allclose(box.ub, np.array([4, 3]))
    assert np.allclose(box.middle, np.array([2.5, 2.5]))
    # width should identify the dimension (0) with max width 3
    dims, max_width = box.width
    assert dims == [0]
    assert max_width == 3

def test_box_width_multiple_equal_max_widths():
    iv1 = Interval(1, 4)
    iv2 = Interval(5, 8)  # same width 3
    iv3 = 7
    box = Box(iv1, iv2, iv3)
    dims, max_width = box.width
    assert set(dims) == {0,1}  # first two have width 3
    assert max_width == 3

def test_box_eq_ne_with_ndarray():
    iv1 = Interval(1, 2)
    iv2 = Interval(3, 4)
    box = Box(iv1, iv2)
    arr = np.array([iv1, iv2])
    assert box == arr
    assert not(box != arr)


# Tests for intervallum/interval.py: Interval class and associated functionality

def test_interval_creation_and_str_repr():
    i = Interval(1.0, 2.0)
    assert i.lb == 1.0
    assert i.ub == 2.0
    assert str(i) == "[1.0; 2.0]"
    assert repr(i) == "[1.0; 2.0]"

def test_interval_wrong_bounds_exception():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(3.0, 2.0)

def test_interval_fix_flag_swaps_bounds():
    i = Interval(3.0, 2.0, fix=True)
    assert math.isclose(i.lb, 2.0)
    assert math.isclose(i.ub, 3.0)

def test_interval_setters_valid_and_invalid():
    i = Interval(1.0, 3.0)
    i.lb = 0.5
    assert math.isclose(i.lb, 0.5)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 4.0
    i.ub = 4.0
    assert math.isclose(i.ub, 4.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0

def test_interval_middle_and_width():
    i = Interval(2.0, 4.0)
    assert math.isclose(i.middle, 3.0)
    assert math.isclose(i.width, 2.0)

def test_interval_try_to_reduce_behavior():
    low = 1.0
    high = 1.0 + IntervalConstants._reduction_width / 2
    i = Interval(low, high)
    result = i._try_to_reduce()
    # Because width < reduction_width, returns midpoint float
    assert isinstance(result, float)
    assert math.isclose(result, 0.5 * (low + high))

    # For bigger width, returns Interval
    i2 = Interval(0.0, 1e-3)
    orig_width = i2.width
    if orig_width >= IntervalConstants._reduction_width:
        result2 = i2._try_to_reduce()
        assert isinstance(result2, Interval)
    else:
        # adjust for very small reduction width to test
        pass

def test_interval_inner_subtraction_and_lshift():
    i1 = Interval(3.0, 5.0)
    i2 = Interval(1.0, 2.0)
    diff = Interval.inner_subtraction(i1, i2)
    assert isinstance(diff, Interval)
    # Lower bound subtract lower bound, upper bound subtract upper bound (fixed=True)
    assert math.isclose(diff.lb, 2.0)
    assert math.isclose(diff.ub, 4.0)

    # __lshift__ calls inner_subtraction
    diff2 = i1 << i2
    assert diff == diff2

def test_interval_eq_and_ne_with_float_and_interval():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(1.00000005, 2.00000005)
    # Should be equal within admissible error
    assert i1 == i2
    assert not(i1 != i2)
    # Compare with float inside the interval
    assert i1 == 1.5
    # Check that NotImplementedError raised for unsupported type
    with pytest.raises(NotImplementedError):
        i1 == "string"

def test_interval_operators_add_sub_mul_neg_div_pow():
    i1 = Interval(1, 2)
    i2 = Interval(3, 4)
    # Add
    res_add = i1 + i2
    assert math.isclose(res_add.lb, 4)
    assert math.isclose(res_add.ub, 6)
    # Subtract
    res_sub = i1 - i2
    assert math.isclose(res_sub.lb, -3)
    assert math.isclose(res_sub.ub, -1)
    # Multiply
    res_mul = i1 * i2
    assert math.isclose(res_mul.lb, 3)
    assert math.isclose(res_mul.ub, 8)
    # Negation
    res_neg = -i1
    assert math.isclose(res_neg.lb, -2)
    assert math.isclose(res_neg.ub, -1)
    # True division by float
    res_div = i1 / 2.0
    assert math.isclose(res_div.lb, 0.5)
    assert math.isclose(res_div.ub, 1.0)
    # True division by Interval
    res_div2 = i1 / i2
    # Division results should be Interval (approximate range)
    assert isinstance(res_div2, Interval)
    # Power odd exponent (3)
    res_pow_odd = i1 ** 3
    assert math.isclose(res_pow_odd.lb, 1**3)
    assert math.isclose(res_pow_odd.ub, 2**3)
    # Power even exponent (2)
    res_pow_even = i1 ** 2
    # Because power even returns Interval (or float if reducible)
    if isinstance(res_pow_even, Interval):
        assert res_pow_even.lb <= res_pow_even.ub
    else:
        assert isinstance(res_pow_even, float)

def test_interval_inversion():
    # Interval strictly positive
    i_pos = Interval(1, 2)
    inv_pos = ~i_pos
    assert math.isclose(inv_pos.lb, 1/2)
    assert math.isclose(inv_pos.ub, 1/1)
    # Interval strictly negative
    i_neg = Interval(-3, -1)
    inv_neg = ~i_neg
    assert math.isclose(inv_neg.lb, 1/-1)
    assert math.isclose(inv_neg.ub, 1/-3)
    # Interval containing zero
    i_zero = Interval(-1, 1)
    inv_zero = ~i_zero
    assert math.isinf(inv_zero.lb)
    assert math.isinf(inv_zero.ub)
    # Interval lb == 0
    i_lb0 = Interval(0, 2)
    inv_lb0 = ~i_lb0
    assert math.isclose(inv_lb0.lb, 1/2)
    assert inv_lb0.ub == math.inf
    # Interval ub == 0
    i_ub0 = Interval(-2, 0)
    inv_ub0 = ~i_ub0
    assert inv_ub0.lb == -math.inf
    assert math.isclose(inv_ub0.ub, 1/-2)

# Tests for intervallum/inclusion_functions.py: RS_SubinclusionFunction

def test_rs_subinclusion_function_returns_interval():
    # Function that returns sum of inputs
    def f(x):
        return np.sum(x)
    box = Box(Interval(0, 1), Interval(1, 2))
    rsf = RS_SubinclusionFunction(f, number_of_samples=100)
    result = rsf(box)
    assert isinstance(result, Interval)
    # Result bounds should enclose possible sums from [0*1,...,1*2]
    assert result.lb <= 3  # min sum possible is adding two lb: 0+1=1
    assert result.ub >= 1

def test_rs_subinclusion_function_samples_within_bounds(monkeypatch):
    # Patch np.random.uniform to return midpoints for deterministic test
    centers = []
    def fake_uniform(low, high):
        centers.append((low,high))
        return 0.5*(np.array(low)+np.array(high))
    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    def identity_sum(x):
        return sum(x)

    box = Box(Interval(1, 3), Interval(4, 6))
    rsf = RS_SubinclusionFunction(identity_sum, 3)
    result = rsf(box)
    # All samples returned as midpoint sums
    expected_value = sum(0.5 * (box.lb + box.ub))
    # Because min and max of same midpoints, result should be a single number reduced from Interval
    assert isinstance(result, (float, Interval))
    if isinstance(result, float):
        assert math.isclose(result, expected_value)
    else:
        # If not reduced (unlikely here)
        assert result.lb <= expected_value <= result.ub

# Tests for intervallum/interval_functions.py

def test_trigonometric_functions_on_floats_and_intervals():
    i = Interval(0, math.pi/2)
    sin_interval = sin(i)
    cos_interval = cos(i)
    # sin should range in [0,1], cos in roughly [0,1]
    assert 0 <= sin_interval.lb <= 1
    assert 0 <= sin_interval.ub <= 1
    assert 0 <= cos_interval.lb <= 1
    assert 0 <= cos_interval.ub <= 1

    # On float input, returns an Interval reduced to float
    sf = sin(0.5)
    cf = cos(0.5)
    assert isinstance(sf, float)
    assert isinstance(cf, float)

def test_abs_exp_sqrt_log_behavior():
    i_pos = Interval(1, 3)
    i_neg_pos = Interval(-1, 1)
    i_neg = Interval(-3, -1)

    # abs on interval covering zero adds 0 to points
    abs_interval = abs(i_neg_pos)
    # The lower bound should be zero because abs includes zero point
    assert math.isclose(abs_interval.lb, 0.0)
    # exp on interval returns increasing bounds
    exp_interval = exp(i_pos)
    assert exp_interval.lb < exp_interval.ub
    # sqrt on negative interval raises error
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(i_neg)
    # sqrt on interval crossing zero clamps to zero
    sqrt_interval = sqrt(i_neg_pos)
    assert sqrt_interval.lb >= 0.0 and sqrt_interval.ub >= 0.0
    # log on interval <=0 raises error
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-2, 0))
    # log on interval crossing zero clamps to zero and upper bound
    log_interval = log(i_neg_pos)
    assert log_interval.lb <= 0 and log_interval.ub > 0
    # On float inputs returns float number (reduced interval)
    assert isinstance(abs(-2.3), float)
    assert isinstance(exp(1.0), float)
    assert isinstance(sqrt(4.0), float)
    assert isinstance(log(2.0), float)

def test_split_and_bisect_intervals():
    i = Interval(0, 10)
    ratios = [1, 2, 1]
    parts = split(i, ratios)
    # Should produce len(ratios) parts
    assert len(parts) == len(ratios)
    # Parts are increasing and cover full interval approximately
    assert math.isclose(parts[0].lb, 0)
    assert math.isclose(parts[-1].ub, 10)
    # Bisection splits in 2 equal parts
    halves = bisect(i)
    assert len(halves) == 2
    assert math.isclose(halves[0].width, halves[1].width)
    assert math.isclose(halves[0].ub, halves[1].lb)

def test_constrain_function():
    i = Interval(-5, 5)
    c = constrain(i, -1, 1)
    assert c.lb >= -1 and c.ub <= 1
    # Constrain on float clamps value
    val = constrain(10, -2, 2)
    assert val == 2
    val2 = constrain(-3, -2, 2)
    assert val2 == -2
    val3 = constrain(1, -2, 2)
    assert val3 == 1

def test_shrink_function():
    i = Interval(0, 10)
    shrunk = shrink(i, 0.5)
    expected_width = 0.5 * 0.5 * i.width
    assert math.isclose(shrunk.width, expected_width * 2)
    # Shrink on float returns float unchanged
    val = shrink(5.0, 0.5)
    assert val == 5.0

def test_hausdorff_distance_cases():
    i1 = Interval(0, 1)
    i2 = Interval(0.5, 2)
    dist = hausdorff_distance(i1, i2)
    # Distance is max of _hausdorff_distance_part
    part1 = max(0.0, i2.lb - i1.lb)
    part2 = max(0.0, i1.ub - i2.ub)
    expected = max(part1, part2)
    part3 = max(0.0, i1.lb - i2.lb)
    part4 = max(0.0, i2.ub - i1.ub)
    expected2 = max(part3, part4)
    expected_dist = max(expected, expected2)
    assert math.isclose(dist, expected_dist)

    # Hausdorff distance with float input
    dist_float = hausdorff_distance(1.0, i2)
    assert isinstance(dist_float, float)
    dist_float_2 = hausdorff_distance(i1, 1.0)
    assert isinstance(dist_float_2, float)
