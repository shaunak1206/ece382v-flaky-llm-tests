import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import (
    Interval,
    IntervalConstants,
    IntervalExceptions,
)
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin,
    cos,
    abs as interval_abs,
    exp,
    sqrt,
    log,
    split,
    bisect,
    constrain,
    shrink,
    hausdorff_distance,
)


# Constants for floating point comparison tolerance
FLOAT_TOL = 1e-7


def test_box_new_and_str_repr():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(3.0, 4.0)
    b = Box(i1, i2)
    assert isinstance(b, Box)
    # __str__ and __repr__ produce same output
    assert str(b) == repr(b)
    assert str(b) == f"{str(i1)} x {str(i2)}"


def test_box_eq_and_ne():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(2.0, 3.0)
    b1 = Box(i1, i2)
    b2 = Box(i1, i2)
    b3 = Box(i2, i1)

    assert b1 == b2
    assert b1 != b3
    assert not (b1 != b2)
    # Comparing with numpy array with same elements should also work
    arr = np.array([i1, i2])
    assert b1 == arr
    with pytest.raises(NotImplementedError):
        _ = (b1 == "invalid")


def test_box_copy_and_try_to_reduce():
    i1 = Interval(1.0, 1.0 + IntervalConstants._reduction_width / 2)
    i2 = Interval(2.0, 3.0)
    b = Box(i1, i2)
    # Copy returns a Box equal to original but a different object
    b_copy = b.__copy__()
    assert b_copy == b
    assert b_copy is not b
    # _try_to_reduce returns middle if width is zero for all intervals
    i_zero_width = Interval(5.0, 5.0)
    b_zero = Box(i_zero_width, i_zero_width)
    reduced = b_zero._try_to_reduce()
    # reduced is an ndarray of middles (floats)
    assert isinstance(reduced, np.ndarray)
    for comp in reduced:
        assert isinstance(comp, float)
        assert comp == pytest.approx(5.0)
    # If max width > 0, returns copy (a Box)
    b_reduced = b._try_to_reduce()
    assert isinstance(b_reduced, Box)
    assert b_reduced == b


def test_box_properties_lb_ub_middle_width():
    intervals = [
        Interval(1.0, 3.0),
        Interval(-1.0, 2.0),
        5.0,  # float, should treat as number
    ]
    b = Box(*intervals)

    lbs = b.lb
    ubs = b.ub
    middles = b.middle
    widths_indices, max_width = b.width

    expected_lbs = [1.0, -1.0, 5.0]
    expected_ubs = [3.0, 2.0, 5.0]
    expected_middles = [2.0, 0.5, 5.0]
    expected_widths = [2.0, 3.0, 0.0]

    np.testing.assert_allclose(lbs, expected_lbs)
    np.testing.assert_allclose(ubs, expected_ubs)
    np.testing.assert_allclose(middles, expected_middles)
    assert max_width == pytest.approx(max(expected_widths))
    # max width index should be 1 (width = 3.0)
    assert widths_indices == [1]


def test_interval_init_and_wrong_bounds_exception():
    # Normal initialization works
    i = Interval(1.0, 2.0)
    assert i.lb == 1.0
    assert i.ub == 2.0

    # If lower bound > upper bound and fix=False, raise exception
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        _ = Interval(5.0, 4.0, fix=False)

    # If fix=True, bounds get swapped
    i_fixed = Interval(5.0, 4.0, fix=True)
    assert i_fixed.lb == 4.0
    assert i_fixed.ub == 5.0


def test_interval_str_repr_and_copy():
    i = Interval(1.5, 3.5)
    s = str(i)
    r = repr(i)
    assert s == r
    assert s == "[1.5; 3.5]"
    i_copy = i.__copy__()
    assert i_copy == i
    assert i_copy is not i


def test_interval_width_middle_try_to_reduce_and_equality():
    i = Interval(1.0, 1.0 + IntervalConstants._reduction_width / 2)
    mid = i.middle
    width = i.width
    assert mid == pytest.approx(1.0 + width/2)
    assert width > 0
    reduced = i._try_to_reduce()
    # Because width < _reduction_width, reduced should be float equal to middle
    assert isinstance(reduced, float)
    assert reduced == pytest.approx(mid)

    # Equality comparisons: interval vs interval and float
    i1 = Interval(0.0, 1.0)
    i2 = Interval(0.0 + 1e-8, 1.0 + 1e-8)
    # They should be considered equal because difference within admissible error
    assert i1 == i2
    assert not (i1 != i2)
    # Equal to float close to middle
    assert i1 == (i1.middle + 1e-8)
    # Not equal to sufficiently different float
    assert i1 != (i1.middle + 1e-4)

    # __eq__ should raise NotImplementedError for invalid types
    with pytest.raises(NotImplementedError):
        _ = (i1 == "invalid")


def test_interval_arithmetic_operations():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 4.0)
    # Addition
    c = a + b
    assert isinstance(c, Interval)
    np.testing.assert_allclose([c.lb, c.ub], [4.0, 6.0])
    # Addition with float
    d = a + 1.5
    np.testing.assert_allclose([d.lb, d.ub], [2.5, 3.5])
    # Reverse addition
    e = 1.5 + a
    assert d == e

    # Negation
    neg_a = -a
    np.testing.assert_allclose([neg_a.lb, neg_a.ub], [-2.0, -1.0])

    # Subtraction
    sub_ab = a - b
    np.testing.assert_allclose([sub_ab.lb, sub_ab.ub], [-3.0, -1.0])
    # Reverse subtraction
    sub_ba = b - a
    np.testing.assert_allclose([sub_ba.lb, sub_ba.ub], [1.0, 3.0])

    # Multiplication
    mult_ab = a * b
    expected_products = [1.0*3.0, 1.0*4.0, 2.0*3.0, 2.0*4.0]
    np.testing.assert_allclose([mult_ab.lb, mult_ab.ub], [min(expected_products), max(expected_products)])

    # Multiplication with float
    mult_a_float = a * 2.0
    np.testing.assert_allclose([mult_a_float.lb, mult_a_float.ub], [2.0, 4.0])
    # Reverse multiplication
    mult_float_a = 2.0 * a
    assert mult_a_float == mult_float_a

    # Division by interval
    c = a / b
    div_bounds = [1.0/4.0, 1.0/3.0, 2.0/4.0, 2.0/3.0]
    np.testing.assert_allclose([c.lb, c.ub], [min(div_bounds), max(div_bounds)])

    # Division by float
    c2 = a / 2.0
    np.testing.assert_allclose([c2.lb, c2.ub], [0.5, 1.0])

    # Reverse division
    c3 = 6.0 / a
    div_a_bounds = [6.0/2.0, 6.0/1.0]
    np.testing.assert_allclose([c3.lb, c3.ub], [min(div_a_bounds), max(div_a_bounds)])


def test_interval_pow_and_invert():
    # Even power
    i = Interval(-2.0, 3.0)
    pow_even = i ** 2
    assert isinstance(pow_even, Interval)
    vals = [(-2.0)**2, 0.0, 3.0**2]
    np.testing.assert_allclose([pow_even.lb, pow_even.ub], [min(vals), max(vals)])

    # Odd power
    pow_odd = i ** 3
    expected_bounds = [(-2.0) ** 3, 3.0 ** 3]
    np.testing.assert_allclose([pow_odd.lb, pow_odd.ub], [min(expected_bounds), max(expected_bounds)])

    # Invert ~ operator on Interval no zero crossing
    i_positive = Interval(1.0, 2.0)
    inv_i = ~i_positive
    np.testing.assert_allclose([inv_i.lb, inv_i.ub], [1/2.0, 1/1.0])

    # Interval crossing zero lower bound zero
    i_zero_lb = Interval(0.0, 2.0)
    inv_i_lb = ~i_zero_lb
    assert inv_i_lb.lb == pytest.approx(1/2.0)
    assert math.isinf(inv_i_lb.ub) and inv_i_lb.ub > 0

    # Interval crossing zero upper bound zero
    i_zero_ub = Interval(-2.0, 0.0)
    inv_i_ub = ~i_zero_ub
    assert math.isinf(inv_i_ub.lb) and inv_i_ub.lb < 0
    assert inv_i_ub.ub == pytest.approx(1 / -2.0)

    # Interval spanning zero both sides
    i_spans_zero = Interval(-1.0, 1.0)
    inv_span = ~i_spans_zero
    assert math.isinf(inv_span.lb) and inv_span.lb < 0
    assert math.isinf(inv_span.ub) and inv_span.ub > 0


def test_rs_subinclusion_function_evaluates_bounds(monkeypatch):
    # Define a deterministic function f: sum of coordinates
    f = lambda x: float(np.sum(x))
    n_samples = 5
    rs_func = RS_SubinclusionFunction(f, n_samples)

    # Prepare fixed Box with fixed intervals
    intervals = [Interval(0.0, 1.0), Interval(2.0, 3.0)]
    b = Box(*intervals)

    # Patch np.random.uniform to produce fixed values for testing non-flaky behavior
    uniform_calls = []

    def fake_uniform(low, high):
        # Return midpoint close to deterministic values from bounds to fake variability
        uniform_calls.append((low, high))
        return 0.5 * (low + high)

    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    res = rs_func(b)
    # The values should all be same because of fake_uniform
    expected_value = sum(np.array(b.lb) * 0.5 + np.array(b.ub) * 0.5) / 2
    assert isinstance(res, Interval)
    # The returned interval min and max are equal because returned values are same
    assert res.lb == pytest.approx(min(res.lb, res.ub))
    assert res.lb == pytest.approx(res.ub)


@pytest.mark.parametrize("input_val,expected", [
    (Interval(0.0, 2 * math.pi), (-1.0, 1.0)),
    (Interval(0.0, math.pi / 2), (0.0, 1.0)),
    (0.5, (math.sin(0.5), math.sin(0.5))),
])
def test_interval_functions_sin(input_val, expected):
    res = sin(input_val)
    assert isinstance(res, Interval)
    assert res.lb >= expected[0] - FLOAT_TOL
    assert res.ub <= expected[1] + FLOAT_TOL


@pytest.mark.parametrize("input_val,expected", [
    (Interval(0.0, 2 * math.pi), (-1.0, 1.0)),
    (Interval(0.0, math.pi / 2), (0.0, 1.0)),
    (0.5, (math.cos(0.5), math.cos(0.5))),
])
def test_interval_functions_cos(input_val, expected):
    res = cos(input_val)
    assert isinstance(res, Interval)
    assert res.lb >= expected[0] - FLOAT_TOL
    assert res.ub <= expected[1] + FLOAT_TOL


@pytest.mark.parametrize("input_val,expected", [
    (Interval(-1.0, 1.0), (0.0, 1.0)),
    (Interval(-2.0, -1.0), (1.0, 2.0)),
    (0.5, (0.5, 0.5)),
])
def test_interval_functions_abs(input_val, expected):
    res = interval_abs(input_val)
    assert isinstance(res, Interval)
    assert res.lb >= expected[0] - FLOAT_TOL
    assert res.ub <= expected[1] + FLOAT_TOL


@pytest.mark.parametrize("input_val", [
    Interval(0, 1),
    1.0,
])
def test_interval_functions_exp(input_val):
    res = exp(input_val)
    assert isinstance(res, Interval)
    assert res.lb > 0


def test_sqrt_valid_and_invalid():
    # Valid sqrt
    i = Interval(1.0, 4.0)
    res = sqrt(i)
    assert isinstance(res, Interval)
    assert math.isclose(res.lb, 1.0, abs_tol=FLOAT_TOL)
    assert math.isclose(res.ub, 2.0, abs_tol=FLOAT_TOL)

    # sqrt on interval crossing zero
    i_neg_low = Interval(-1.0, 4.0)
    res2 = sqrt(i_neg_low)
    assert res2.lb == pytest.approx(0.0)
    assert res2.ub == pytest.approx(2.0)

    # sqrt on strictly negative interval raises exception
    i_neg = Interval(-2.0, -0.5)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(i_neg)


def test_log_valid_and_invalid():
    i = Interval(1.0, math.e)
    res = log(i)
    assert isinstance(res, Interval)
    # log(1) = 0, log(e) = 1
    assert math.isclose(res.lb, 0.0, abs_tol=FLOAT_TOL)
    assert math.isclose(res.ub, 1.0, abs_tol=FLOAT_TOL)

    i_neg_low = Interval(-1.0, 2.0)
    res2 = log(i_neg_low)
    # bounds must be >=0 inside domain
    assert res2.lb < res2.ub

    i_neg = Interval(-1.0, 0.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(i_neg)


@pytest.mark.parametrize(
    "i_val,ratios,expected_num",
    [
        (Interval(0.0, 10.0), [1, 1, 2], 3),
        (5.0, [1, 2], 2),
    ],
)
def test_split(i_val, ratios, expected_num):
    parts = split(i_val, ratios)
    assert len(parts) == len(ratios)
    if isinstance(i_val, Interval):
        total_width = sum(p.width if isinstance(p, Interval) else 0 for p in parts)
        assert math.isclose(total_width, i_val.width, rel_tol=1e-6) or math.isclose(total_width, 0)
    else:
        # Non interval input returns repeated values
        assert all(p == i_val for p in parts)


def test_bisect_splits_interval_equally():
    i = Interval(0.0, 4.0)
    parts = bisect(i)
    assert len(parts) == 2
    assert all(isinstance(p, Interval) for p in parts)
    assert math.isclose(parts[0].width, parts[1].width, rel_tol=1e-7)
    assert math.isclose(parts[0].ub, parts[1].lb, rel_tol=1e-7)


@pytest.mark.parametrize(
    "i_val,min_,max_,expected_lb,expected_ub",
    [
        (Interval(-5, 10), 0, 5, 0, 5),
        (Interval(0, 4), 1, 3, 1, 3),
        (5, 1, 10, 5, 5),
        (-1, 0, 10, 0, 0),
    ],
)
def test_constrain(i_val, min_, max_, expected_lb, expected_ub):
    res = constrain(i_val, min_, max_)
    if isinstance(res, Interval):
        assert res.lb >= expected_lb
        assert res.ub <= expected_ub
    else:
        assert expected_lb <= res <= expected_ub


@pytest.mark.parametrize(
    "i_val,alpha",
    [
        (Interval(0.0, 4.0), 0.5),
        (5.0, 0.5),
    ],
)
def test_shrink(i_val, alpha):
    res = shrink(i_val, alpha)
    if isinstance(i_val, Interval):
        expected_width = i_val.width * alpha * 0.5 * 2
        assert math.isclose(res.width, expected_width, rel_tol=1e-7)
        assert math.isclose(res.middle, i_val.middle, rel_tol=1e-7)
    else:
        assert res == i_val


def test_hausdorff_distance_zero_and_nonzero():
    i1 = Interval(1.0, 3.0)
    i2 = Interval(1.0, 3.0)
    dist = hausdorff_distance(i1, i2)
    assert dist == pytest.approx(0.0)

    i3 = Interval(2.0, 5.0)
    dist2 = hausdorff_distance(i1, i3)
    # Distance should be non-negative and at least difference in lower or upper bounds
    assert dist2 >= 0.0
    assert dist2 == max(max(0.0, i3.lb - i1.lb), max(0.0, i1.ub - i3.ub),
                        max(0.0, i1.lb - i3.lb), max(0.0, i3.ub - i1.ub))


def test_hausdorff_distance_with_float_inputs():
    i = Interval(1.0, 3.0)
    f = 2.0
    dist = hausdorff_distance(i, f)
    # Distance between interval and float is max of one sided distances
    expected = max(
        max(0.0, f - i.lb),
        max(0.0, i.ub - f),
        max(0.0, i.lb - f),
        max(0.0, f - i.ub),
    )
    assert dist >= 0.0
    assert math.isclose(dist, expected, rel_tol=1e-7)

