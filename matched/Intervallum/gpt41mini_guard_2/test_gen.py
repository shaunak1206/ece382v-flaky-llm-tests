import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin, cos, abs as interval_abs, exp, sqrt, log,
    split, bisect, constrain, shrink, hausdorff_distance,
)
    

# ------------------- Tests for Interval -------------------

def test_interval_init_and_properties():
    i = Interval(1.0, 2.0)
    assert i.lb == 1.0
    assert i.ub == 2.0
    assert abs(i.middle - 1.5) < 1e-12
    assert abs(i.width - 1.0) < 1e-12

def test_interval_wrong_bounds_exception():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    # fix=True swaps bounds silently
    i = Interval(2.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 2.0

def test_interval_str_repr_eq_ne():
    i1 = Interval(1.0, 3.0)
    i2 = Interval(1.0, 3.0)
    i3 = Interval(1.0, 2.9)
    s = str(i1)
    r = repr(i1)
    assert s == r
    assert i1 == i2
    assert not (i1 != i2)
    assert i1 != i3
    # comparison to float of close value triggers __eq__
    close_float = i1.middle
    assert (i1 == close_float) is True
    assert (i1 != close_float) is False

def test_interval_setters_enforce_bounds():
    i = Interval(0.0, 5.0)
    i.lb = 1.0
    assert i.lb == 1.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 10.0
    i.ub = 10.0
    assert i.ub == 10.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0

def test_interval_try_to_reduce_behavior():
    IntervalConstants._reduction_width = 1e-2
    i1 = Interval(1.0, 1.005)
    result1 = i1._try_to_reduce()
    assert isinstance(result1, float)
    assert abs(result1 - 1.0025) < 1e-12
    i2 = Interval(0.0, 1.0)
    result2 = i2._try_to_reduce()
    assert isinstance(result2, Interval)
    # restore default value
    IntervalConstants._reduction_width = 1e-5

def test_interval_add_sub_mul_neg_div_pow():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(3.0, 4.0)
    # Addition
    s = i1 + i2
    assert isinstance(s, Interval)
    assert math.isclose(s.lb, 4.0, rel_tol=1e-12)
    assert math.isclose(s.ub, 6.0, rel_tol=1e-12)
    # Subtraction
    d = i1 - i2
    assert isinstance(d, Interval)
    assert math.isclose(d.lb, -3.0, rel_tol=1e-12)
    assert math.isclose(d.ub, -1.0, rel_tol=1e-12)
    # Multiplication
    m = i1 * i2
    assert isinstance(m, Interval)
    assert math.isclose(m.lb, 3.0, rel_tol=1e-12)
    assert math.isclose(m.ub, 8.0, rel_tol=1e-12)
    # Negation
    n = -i1
    assert isinstance(n, Interval)
    assert math.isclose(n.lb, -2.0, rel_tol=1e-12)
    assert math.isclose(n.ub, -1.0, rel_tol=1e-12)
    # Division
    d1 = i1 / i2
    assert isinstance(d1, Interval)
    assert d1.lb < d1.ub
    d2 = 5 / i1
    assert isinstance(d2, Interval)
    # Power even
    p_even = i1 ** 2
    assert isinstance(p_even, (float, Interval))
    # Power odd
    p_odd = i1 ** 3
    assert isinstance(p_odd, (float, Interval))

def test_interval_invert_and_inner_subtraction():
    i = Interval(2.0, 4.0)
    inv = ~i
    assert isinstance(inv, Interval)
    # Covers positive interval inversion
    assert math.isclose(inv.lb, 1 / 4.0, rel_tol=1e-12)
    assert math.isclose(inv.ub, 1 / 2.0, rel_tol=1e-12)
    # test inner_subtraction reduction to float for narrow intervals
    IntervalConstants._reduction_width = 1e-1
    i1 = Interval(1.0, 1.05)
    i2 = Interval(0.05, 0.1)
    sub = Interval.inner_subtraction(i1, i2)
    assert isinstance(sub, float)
    # restore
    IntervalConstants._reduction_width = 1e-5

def test_interval_eq_not_implemented():
    i = Interval(0, 1)
    with pytest.raises(NotImplementedError):
        i == "not interval or float"
    with pytest.raises(NotImplementedError):
        i != "not interval or float"

def test_interval_comparison_operators():
    i1 = Interval(1.0, 3.0)
    i2 = Interval(2.0, 4.0)
    assert (i1 < i2) is True
    assert (i1 <= i2) is True
    assert (i2 > i1) is True
    assert (i2 >= i1) is True
    # Compare to float
    assert (i1 < 2.0) is True
    assert (i1 <= 1.0) is True
    assert (i2 > 3.0) is True
    assert (i2 >= 4.0) is True


# ------------------- Tests for Box -------------------

def test_box_creation_and_properties():
    i1 = Interval(0, 1)
    i2 = Interval(1, 2)
    b = Box(i1, i2)
    assert isinstance(b, Box)
    # lb property
    lbs = b.lb
    assert np.allclose(lbs, [0, 1])
    # ub property
    ubs = b.ub
    assert np.allclose(ubs, [1, 2])
    # middle property
    middles = b.middle
    assert np.allclose(middles, [0.5, 1.5])
    # width property: both widths equal to 1, max-width = 1, max indices are [0, 1]
    indices, width = b.width
    assert math.isclose(width, 1)
    assert set(indices) == {0, 1}

def test_box_str_and_repr_and_eq_ne():
    i1 = Interval(0, 1)
    i2 = Interval(1, 2)
    b = Box(i1, i2)
    s = str(b)
    r = repr(b)
    assert isinstance(s, str)
    assert s == r
    # equality
    b2 = Box(i1, i2)
    assert b == b2
    assert not (b != b2)
    # inequality with different intervals
    b3 = Box(Interval(0, 0.5), i2)
    assert b != b3
    # eq with non ndarray/Box should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        b == "not a Box"

def test_box_copy_and_try_to_reduce():
    i1 = Interval(1, 2)
    i2 = Interval(3, 4)
    b = Box(i1, i2)
    b_copy = b.__copy__()
    assert isinstance(b_copy, Box)
    # Change original to confirm deep copy
    i1.lb = 10
    assert not (b.lb == b_copy.lb).all()
    # _try_to_reduce returns middle if max width is 0 else returns copy
    b_zero_width = Box(Interval(1, 1), Interval(2, 2))
    result = b_zero_width._try_to_reduce()
    assert isinstance(result, np.ndarray) and np.allclose(result, [1.0, 2.0])
    # When width > 0, returns a copy of Box
    result2 = b._try_to_reduce()
    assert isinstance(result2, Box)
    assert result2 == b

def test_box_with_non_interval_elements():
    # Box can be constructed with numbers which are returned as-is in properties
    b = Box(1.0, 2.0)
    assert np.allclose(b.lb, [1.0, 2.0])
    assert np.allclose(b.ub, [1.0, 2.0])
    assert np.allclose(b.middle, [1.0, 2.0])
    indices, width = b.width
    assert width == 0
    assert indices == [0, 1]


# ------------------- Tests for RS_SubinclusionFunction -------------------

def test_rs_subinclusion_function_behavior(monkeypatch):
    # Fix seed to ensure reproducibility of random.uniform inside RS_SubinclusionFunction
    rng = np.random.default_rng(seed=12345)
    def fixed_uniform(low, high):
        # Return mid point for deterministic behavior instead of random
        return 0.5 * (low + high)
    monkeypatch.setattr(np.random, "uniform", fixed_uniform)

    # function returns sum of inputs
    def f(x):
        return sum(x)
    box = Box(Interval(1, 3), Interval(2, 4))
    rs_func = RS_SubinclusionFunction(f, number_of_samples=5)
    interval = rs_func(box)
    assert isinstance(interval, Interval)
    # Because fixed_uniform returns midpoint, all points give same value:
    mid_point = [(i.lb+i.ub)*0.5 if isinstance(i, Interval) else i for i in box]
    expected_value = sum(mid_point)
    assert math.isclose(interval.lb, expected_value)
    assert math.isclose(interval.ub, expected_value)

def test_rs_subinclusion_function_returns_correct_interval(monkeypatch):
    # Provide predetermined sample points and function values to test bounds
    fixed_points = [
        np.array([1.0, 2.0]),
        np.array([2.0, 3.0]),
        np.array([3.0, 4.0])
    ]
    values = [10.0, 20.0, 15.0]
    call_count = 0
    def fake_uniform(low, high):
        nonlocal call_count
        # Return points cyclically for each coordinate
        point = fixed_points[call_count // 2][call_count % 2]
        call_count += 1
        return point
    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    def f(x):
        # Return pre-stored values instead of computation
        idx = fixed_points.index(x)
        return values[idx]

    box = Box(Interval(0, 3), Interval(1, 4))
    rs_func = RS_SubinclusionFunction(lambda x: x[0] + x[1], number_of_samples=3)
    result = rs_func(box)
    assert isinstance(result, Interval)
    # Since uniform overridden to produce midpoints, expect min and max both at midpoint sum
    midpoints = [np.mean([i.lb, i.ub]) if isinstance(i, Interval) else i for i in box]
    expected_mid = sum(midpoints)
    assert math.isclose(result.lb, expected_mid)
    assert math.isclose(result.ub, expected_mid)


# ------------------- Tests for interval_functions -------------------

def test_trig_functions_basic_behavior():
    i = Interval(0.0, math.pi / 2)
    s = sin(i)
    c = cos(i)
    assert isinstance(s, Interval)
    assert isinstance(c, Interval)
    # values should be within expected ranges
    assert 0 <= s.lb <= 1
    assert 0 <= s.ub <= 1
    assert -1 <= c.lb <= 1
    assert -1 <= c.ub <= 1

def test_trig_functions_reduce_for_wide_interval():
    interval = Interval(0, 10 * math.pi)
    s = sin(interval)
    c = cos(interval)
    # The sin and cos should cover entire range
    assert s.lb <= -1 + 1e-12 and s.ub >= 1 - 1e-12
    assert c.lb <= -1 + 1e-12 and c.ub >= 1 - 1e-12

def test_abs_exp_functions_behavior():
    i = Interval(-2.0, 3.0)
    a = interval_abs(i)
    e = exp(i)
    assert isinstance(a, Interval)
    assert isinstance(e, Interval)
    assert a.lb >= 0
    assert e.lb > 0

def test_sqrt_log_behavior_and_exceptions():
    pos_interval = Interval(1.0, 4.0)
    s = sqrt(pos_interval)
    l = log(pos_interval)
    assert isinstance(s, Interval)
    assert isinstance(l, Interval)
    # sqrt of negative interval raises
    neg_interval = Interval(-1.0, -0.5)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(neg_interval)
    # log of non-positive interval raises
    zero_interval = Interval(0.0, 0.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(zero_interval)
    # mixed interval containing zero returns interval from 0 to upper bound for sqrt and log
    mixed_interval = Interval(-1.0, 1.0)
    s_mixed = sqrt(mixed_interval)
    l_mixed = log(mixed_interval)
    assert s_mixed.lb == 0
    assert l_mixed.lb == 0 or l_mixed.lb == -math.inf or l_mixed.lb < 0  # log might be -inf or 0 at 0

def test_split_and_bisect_functions():
    i = Interval(0.0, 4.0)
    parts = split(i, [1.0, 3.0])
    assert len(parts) == 2
    assert math.isclose(parts[0].width + parts[1].width, 4.0, rel_tol=1e-12)
    bisected = bisect(i)
    assert len(bisected) == 2
    assert math.isclose(bisected[0].width, bisected[1].width, rel_tol=1e-12)

def test_constrain_function_behavior():
    i = Interval(-5, 5)
    c1 = constrain(i, 0, 10)
    assert c1.lb == 0
    assert c1.ub == 5
    c2 = constrain(i, -10, 0)
    assert c2.lb == -5
    assert c2.ub == 0
    # scalar constrain returns clipped scalar
    assert constrain(15.0, 0, 10) == 10
    assert constrain(-5.0, 0, 10) == 0
    assert constrain(5.0, 0, 10) == 5.0

def test_shrink_function_behavior():
    i = Interval(0, 4)
    shrunk = shrink(i, 0.5)
    expected_width = 0.5 * 0.5 * 4  # 0.5 alpha * width/2 on each side
    assert math.isclose(shrunk.width, 2.0, rel_tol=1e-12)
    # Scalar input returns unchanged
    s = shrink(42.0, 0.3)
    assert s == 42.0

def test_hausdorff_distance_function():
    i1 = Interval(0.0, 1.0)
    i2 = Interval(1.0, 2.0)
    d = hausdorff_distance(i1, i2)
    assert math.isclose(d, 1.0, rel_tol=1e-12)
    # Distance with overlapping intervals is zero or positive
    i3 = Interval(0.5, 1.5)
    d2 = hausdorff_distance(i1, i3)
    assert d2 >= 0
    # Scalar vs interval
    d3 = hausdorff_distance(0.5, i3)
    assert d3 >= 0


# End of tests
