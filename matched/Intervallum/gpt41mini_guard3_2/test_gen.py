import math
import pytest
import numpy as np

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin, cos, abs as interval_abs, exp, sqrt, log, split, bisect,
    constrain, shrink, hausdorff_distance
)


def test_box_creation_and_str_repr():
    b = Box(Interval(1, 2), Interval(3, 4))
    assert isinstance(b, Box)
    s = str(b)
    r = repr(b)
    assert s == r
    assert s == "[1; 2] x [3; 4]"


def test_box_eq_and_ne():
    b1 = Box(Interval(0, 1), Interval(1, 2))
    b2 = Box(Interval(0, 1), Interval(1, 2))
    b3 = Box(Interval(0, 1), Interval(2, 3))
    assert b1 == b2
    assert not (b1 != b2)
    assert b1 != b3
    with pytest.raises(NotImplementedError):
        _ = (b1 == 123)


def test_box_copy_independence():
    b1 = Box(Interval(0, 1), Interval(1, 2))
    b2 = b1.__copy__()
    assert b1 == b2
    assert b1 is not b2
    # Changing copied intervals doesn't affect original
    b2[0] = Interval(10, 20)
    assert b1 != b2


def test_box_properties_lb_ub_middle_width():
    b = Box(Interval(1, 3), 5, Interval(4, 4))
    lbs = b.lb
    ubs = b.ub
    middles = b.middle
    widths, max_w = b.width
    np.testing.assert_allclose(lbs, [1, 5, 4])
    np.testing.assert_allclose(ubs, [3, 5, 4])
    np.testing.assert_allclose(middles, [2, 5, 4])
    # widths list: Interval widths and 0 for numbers
    assert set(widths) == {0, 2} or set(widths) == {0, 0, 2}
    assert math.isclose(max_w, 2.0)


def test_box_try_to_reduce_with_zero_width_returns_middle():
    b = Box(Interval(5, 5), Interval(2, 3))  # First interval width=0, second =1
    res = b._try_to_reduce()
    # Since width is not zero due to second, result is a copy (Box)
    assert isinstance(res, Box)
    b2 = Box(Interval(1, 1), Interval(2, 2))
    res2 = b2._try_to_reduce()
    # width is zero, expect ndarray with middles
    np.testing.assert_allclose(res2, b2.middle)


def test_interval_creation_and_validation():
    i = Interval(1.0, 2.0)
    assert i.lb == 1.0
    assert i.ub == 2.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        _ = Interval(3.0, 2.0)
    # With fix True, bounds swapped
    i2 = Interval(5.0, 2.0, fix=True)
    assert i2.lb == 2.0
    assert i2.ub == 5.0


def test_interval_setters_and_errors():
    i = Interval(1.0, 3.0)
    i.lb = 0.5
    assert i.lb == 0.5
    i.ub = 4.0
    assert i.ub == 4.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 5.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = -1.0


def test_interval_try_to_reduce_and_width_middle():
    i = Interval(1.0, 1.000001)
    # Reduction width is 1e-5, so this should reduce to float middle
    red = i._try_to_reduce()
    assert isinstance(red, float)
    mid = 0.5 * (1.0 + 1.000001)
    assert math.isclose(red, mid, rel_tol=1e-9)
    wider_i = Interval(1.0, 1.1)
    red2 = wider_i._try_to_reduce()
    assert isinstance(red2, Interval)
    assert red2 == wider_i
    assert math.isclose(wider_i.width, 0.1)
    assert math.isclose(wider_i.middle, 1.05)


def test_interval_arithmetic_add_neg_sub_mul_div_pow():
    IntervalConstants._reduce_intervals_to_numbers = True
    # add
    i1 = Interval(1.0, 2.0)
    i2 = Interval(3.0, 4.0)
    s = i1 + i2
    assert math.isclose(s.lb, 4.0)
    assert math.isclose(s.ub, 6.0)
    s2 = i1 + 1.0
    assert math.isclose(s2.lb, 2.0)
    # neg
    neg = -i1
    assert math.isclose(neg.lb, -2.0)
    assert math.isclose(neg.ub, -1.0)
    # sub
    sub = i2 - i1
    assert math.isclose(sub.lb, 1.0)
    assert math.isclose(sub.ub, 3.0)
    # mul
    mul = i1 * i2
    assert math.isclose(mul.lb, 3.0)
    assert math.isclose(mul.ub, 8.0)
    mul2 = i1 * 2.0
    assert math.isclose(mul2.lb, 2.0)
    # div
    div = i2 / i1
    # intervals: [3,4]/[1,2] = min/max over products [3/1,3/2,4/1,4/2]
    expected_lb = min([3/1,3/2,4/1,4/2])
    expected_ub = max([3/1,3/2,4/1,4/2])
    assert math.isclose(div.lb, expected_lb)
    assert math.isclose(div.ub, expected_ub)
    # pow even
    pe = i1 ** 2
    assert isinstance(pe, Interval)
    # pow odd
    po = i1 ** 3
    assert isinstance(po, Interval)
    # Check pow odd == manual pow
    expected_po = Interval(i1.lb**3, i1.ub**3)
    assert po == expected_po


def test_interval_eq_ne_with_tolerance():
    IntervalConstants._admissible_error = 1e-7
    i1 = Interval(1.0, 1.0 + 5e-8)
    i2 = Interval(1.0, 1.0)
    assert i1 == i2
    i3 = Interval(1.0, 1.0 + 1e-6)
    assert i3 != i2
    # Also compare with float
    assert i2 == 1.0
    assert not (i2 != 1.0)
    with pytest.raises(NotImplementedError):
        _ = i2 == "string"


@pytest.mark.parametrize("func", [sin, cos, interval_abs, exp, sqrt, log])
def test_interval_functions_monotonic_behavior_and_reduce(func):
    # Seed np random for any implicit randomness (none expected here but requested)
    np.random.seed(123)
    # Test over a known interval
    i = Interval(0.0, math.pi)
    try:
        result = func(i)
        # Should be an Interval or float after reduction
        assert (isinstance(result, Interval) or isinstance(result, float))
    except IntervalExceptions.OperationIsNotDefined:
        # sqrt with negative or log with negative range may raise
        pass

    # For float input, result is Interval or float wrapping a single value
    res_float = func(math.pi / 2)
    assert (isinstance(res_float, Interval) or isinstance(res_float, float))


def test_interval_functions_sqrt_log_errors():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        _ = sqrt(Interval(-2.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        _ = log(Interval(-5.0, 0))


def test_split_and_bisect_behavior():
    IntervalConstants._reduce_intervals_to_numbers = True
    i = Interval(0.0, 10.0)
    parts = split(i, [1, 2, 3])
    assert len(parts) == 3
    widths = [p.ub - p.lb if isinstance(p, Interval) else 0 for p in parts]
    # Ratios should be proportional: 1/6*10=~1.6667, 2/6*10=3.3333, 3/6*10=5.0
    expected_widths = [10/6, 20/6, 30/6]
    for w, ew in zip(widths, expected_widths):
        assert math.isclose(w, ew, rel_tol=1e-9)
    # Split on float input returns repeated floats
    fl = 5.0
    res = split(fl, [1.0, 1.0])
    assert res == [5.0, 5.0]
    # bisect splits into two equal intervals
    b = bisect(i)
    assert len(b) == 2
    assert math.isclose(b[0].width, b[1].width, rel_tol=1e-9)


def test_constrain_behaviour():
    i = Interval(-5, 5)
    result = constrain(i, 0, 3)
    assert result.lb == 0
    assert result.ub == 3
    f = 10.0
    res_f = constrain(f, 0, 5)
    assert res_f == 5
    res_f2 = constrain(f, 0, 15)
    assert res_f2 == 10.0


def test_shrink_behavior():
    i = Interval(0, 10)
    shrunk = shrink(i, 0.2)
    expected_width = 10 * 0.2
    assert math.isclose(shrunk.width, expected_width, rel_tol=1e-9)
    float_val = 5.0
    shrunk_float = shrink(float_val, 0.2)
    assert shrunk_float == 5.0


def test_hausdorff_distance_basic_values():
    i1 = Interval(0, 1)
    i2 = Interval(0.5, 1.5)
    dist = hausdorff_distance(i1, i2)
    expected = max( max(0.5 - 0, 0), max(1 - 1.5, 0), max(0 - 0.5, 0), max(1.5 - 1, 0))
    assert math.isclose(dist, 0.5, rel_tol=1e-9)

    # Identical intervals distance is zero
    dist_equal = hausdorff_distance(i1, i1)
    assert math.isclose(dist_equal, 0.0)

    # Distance between float and interval
    f = 0.25
    dist_float = hausdorff_distance(i1, f)
    # Should be non-negative
    assert dist_float >= 0.0


def test_rs_subinclusion_function_behavior(monkeypatch):
    np.random.seed(42)
    # Prepare simple function to use
    f = lambda x: sum(x) if isinstance(x, (list, np.ndarray)) else x
    rs = RS_SubinclusionFunction(f, 5)

    # Patch np.random.uniform to generate deterministic output for repeatability
    def fake_uniform(low, high):
        # low, high are arrays or lists or floats
        arr_low = np.array(low)
        arr_high = np.array(high)
        return (arr_low + arr_high) / 2

    monkeypatch.setattr("numpy.random.uniform", fake_uniform)

    # Build a Box with known intervals
    b = Box(Interval(0, 1), Interval(1, 2))
    result = rs(b)
    # Since fake_uniform returns middles, function evals to sum of middles
    expected_mid = sum(b.middle)
    assert math.isclose(result.lb, expected_mid)
    assert math.isclose(result.ub, expected_mid)
    # Result should be an Interval or float (should reduce because width zero)
    assert isinstance(result, (Interval, float))
