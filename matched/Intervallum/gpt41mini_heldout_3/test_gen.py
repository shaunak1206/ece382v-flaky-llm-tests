import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin, cos, abs as iabs, exp, sqrt, log,
    split, bisect, constrain, shrink, hausdorff_distance,
    _hausdorff_distance_part,
)

# ---------- Tests for intervallum/interval.py and Interval class ----------

def test_interval_creation_and_properties():
    i = Interval(1.0, 2.0)
    assert i.lb == 1.0
    assert i.ub == 2.0
    assert i.middle == 1.5
    assert i.width == 1.0
    assert str(i) == "[1.0; 2.0]"
    assert repr(i) == "[1.0; 2.0]"


def test_interval_wrong_bounds_exception():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(3.0, 2.0, fix=False)
    # fix = True should swap bounds without error
    i = Interval(3.0, 2.0, fix=True)
    assert i.lb == 2.0
    assert i.ub == 3.0


def test_interval_setters():
    i = Interval(1.0, 2.0)
    i.lb = 0.5
    assert i.lb == 0.5
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 2.5  # lb > ub
    i.ub = 3.0
    assert i.ub == 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0  # ub < lb


def test_interval_equality_and_inequality():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(1.0, 2.0)
    i3 = Interval(1.0, 2.0000001)
    f = 1.5
    assert (i1 == i2) is True
    assert (i1 != i3) is True
    assert (i1 == f) is True
    assert (i1 != 3.0) is True
    with pytest.raises(NotImplementedError):
        _ = (i1 == "string")


def test_interval_addition_subtraction():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(3.0, 4.0)
    r = i1 + i2
    assert isinstance(r, Interval)
    assert math.isclose(r.lb, 4.0) and math.isclose(r.ub, 6.0)
    r2 = i1 - i2
    assert isinstance(r2, Interval)
    assert math.isclose(r2.lb, -3.0) and math.isclose(r2.ub, -1.0)

    # with floats
    r3 = i1 + 1.0
    assert isinstance(r3, Interval)
    assert math.isclose(r3.lb, 2.0) and math.isclose(r3.ub, 3.0)
    r4 = 1.0 + i1
    assert r3 == r4


def test_interval_multiplication_division():
    i1 = Interval(1.0, 3.0)
    i2 = Interval(2.0, 4.0)
    r = i1 * i2
    assert isinstance(r, Interval)
    assert math.isclose(r.lb, 2.0)
    assert math.isclose(r.ub, 12.0)

    r2 = i1 / i2
    assert isinstance(r2, Interval)
    # multiplication by reciprocal should have handled division

    r3 = i1 * 2.0
    assert isinstance(r3, Interval)
    assert math.isclose(r3.lb, 2.0)
    assert math.isclose(r3.ub, 6.0)

    r4 = 2.0 * i1
    assert r3 == r4

    # reciprocal of Interval positive range
    i3 = Interval(1.0, 2.0)
    inv = ~i3
    assert math.isclose(inv.lb, 0.5)
    assert math.isclose(inv.ub, 1.0)

    # reciprocal including zero bounds
    i_zero_lb = Interval(0.0, 2.0)
    inv_zero_lb = ~i_zero_lb
    assert inv_zero_lb.lb == 0.5
    assert math.isinf(inv_zero_lb.ub) and inv_zero_lb.ub > 0

    i_zero_ub = Interval(-2.0, 0.0)
    inv_zero_ub = ~i_zero_ub
    assert math.isinf(inv_zero_ub.lb) and inv_zero_ub.lb < 0
    assert math.isclose(inv_zero_ub.ub, -0.5)

    # reciprocal with interval straddling zero
    i_straddle = Interval(-1.0, 1.0)
    inv_straddle = ~i_straddle
    assert math.isinf(inv_straddle.lb) and inv_straddle.lb < 0
    assert math.isinf(inv_straddle.ub) and inv_straddle.ub > 0


def test_interval_negation_power():
    i = Interval(-2.0, 3.0)
    neg_i = -i
    assert math.isclose(neg_i.lb, -3.0)
    assert math.isclose(neg_i.ub, 2.0)

    # powers
    even_pow = i ** 2
    assert isinstance(even_pow, Interval)
    assert even_pow.lb <= 0.0 <= even_pow.ub

    odd_pow = i ** 3
    assert isinstance(odd_pow, Interval)
    assert math.isclose(odd_pow.lb, (-2.0) ** 3)
    assert math.isclose(odd_pow.ub, 3.0 ** 3)


def test_interval_try_to_reduce_and_copy():
    i = Interval(1.0, 1.0 + IntervalConstants._reduction_width / 2)
    reduced = i._try_to_reduce()
    assert isinstance(reduced, float)

    i2 = Interval(1.0, 2.0)
    cop = i2.__copy__()
    assert cop == i2
    assert cop is not i2


# ---------- Tests for intervallum/box.py and Box class ----------

def test_box_creation_and_properties():
    i1 = Interval(1, 2)
    i2 = Interval(3, 4)
    b = Box(i1, i2)
    assert isinstance(b, Box)
    assert str(b) == "[1; 2] x [3; 4]"
    assert repr(b) == "[1; 2] x [3; 4]"
    # lb, ub, middle
    np.testing.assert_array_equal(b.lb, np.array([1, 3]))
    np.testing.assert_array_equal(b.ub, np.array([2, 4]))
    np.testing.assert_array_equal(b.middle, np.array([1.5, 3.5]))
    # width returns indices with max width and max_width
    indices, width = b.width
    assert width == max(i.width for i in b)
    assert all(0 <= idx < len(b) for idx in indices)
    # test equality and inequality
    b2 = Box(i1, i2)
    assert b == b2
    assert not (b != b2)
    # test copy works
    bcopy = b.__copy__()
    assert bcopy == b
    assert bcopy is not b


def test_box_equality_raises_on_other_types():
    b = Box(Interval(1, 2))
    with pytest.raises(NotImplementedError):
        _ = (b == "not a box")
    # inequality fallback
    with pytest.raises(NotImplementedError):
        _ = (b != object())


def test_box_try_to_reduce_returns_middle_when_width_zero():
    b = Box(Interval(1.0, 1.0), Interval(2.0, 3.0))
    # manually make width zero on one component only
    # box width should be max width across components (so >0)
    res = b._try_to_reduce()
    assert isinstance(res, Box)
    b_all_zero = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    res2 = b_all_zero._try_to_reduce()
    # Since width is zero, returns middle
    assert isinstance(res2, np.ndarray)
    np.testing.assert_array_equal(res2, b_all_zero.middle)


# ---------- Tests for intervallum/inclusion_functions.py and RS_SubinclusionFunction ----------

def test_rs_subinclusion_function_returns_interval():
    f = lambda x: np.sum(x)
    number_of_samples = 100
    rsf = RS_SubinclusionFunction(f, number_of_samples)
    b = Box(Interval(0, 1), Interval(1, 2))
    interval = rsf(b)
    assert isinstance(interval, Interval)
    assert interval.lb <= interval.ub
    # values should lie within min and max
    val = f(np.array([b.lb[0], b.lb[1]]))
    assert interval.lb <= val


# ---------- Tests for intervallum/interval_functions.py ----------

def test_trig_functions_single_float_interval():
    for f in [sin, cos]:
        res = f(0.5)
        assert isinstance(res, Interval)
        assert res.lb <= math.sin(0.5) <= res.ub or res.lb <= math.cos(0.5) <= res.ub


def test_trig_functions_interval_covering_2pi_or_more():
    i = Interval(0, 2 * math.pi)
    s = sin(i)
    c = cos(i)
    # The interval returned should be [-1,1]
    for res in [s, c]:
        assert math.isclose(res.lb, -1)
        assert math.isclose(res.ub, 1)


def test_abs_function_with_interval_crossing_zero():
    i = Interval(-1.0, 2.0)
    a = iabs(i)
    assert math.isclose(a.lb, 0.0)
    assert math.isclose(a.ub, 2.0)


def test_exp_function_with_interval():
    i = Interval(0, 1)
    e = exp(i)
    assert e.lb <= math.exp(0) <= e.ub
    assert e.lb <= math.exp(1) <= e.ub


def test_sqrt_function_valid_and_invalid():
    i = Interval(1.0, 4.0)
    s = sqrt(i)
    assert math.isclose(s.lb, 1.0)
    assert math.isclose(s.ub, 2.0)

    i_with_neg = Interval(-1.0, 4.0)
    s_neg = sqrt(i_with_neg)
    assert math.isclose(s_neg.lb, 0.0)
    assert s_neg.ub == s.ub

    i_neg = Interval(-2.0, -1.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(i_neg)


def test_log_function_valid_and_invalid():
    i = Interval(1.0, 10.0)
    l = log(i)
    assert l.lb <= math.log(1.0) <= l.ub
    assert l.lb <= math.log(10.0) <= l.ub

    i_with_neg = Interval(-1.0, 10.0)
    l_neg = log(i_with_neg)
    assert l_neg.lb <= -math.inf <= l_neg.ub

    i_neg = Interval(-2.0, 0.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(i_neg)


def test_split_and_bisect_for_interval_and_float():
    i = Interval(0.0, 2.0)
    ratios = [1, 3, 2]
    parts = split(i, ratios)
    assert len(parts) == len(ratios)
    total_width = sum(p.width if isinstance(p, Interval) else 0.0 for p in parts)
    # Since reduction to float may disallow width checks (width 0), rough check:
    assert all(isinstance(p, (float, Interval)) for p in parts)

    f = 5.0
    parts_float = split(f, [1, 2])
    assert parts_float == [5.0, 5.0]

    bisected = bisect(i)
    assert len(bisected) == 2
    assert abs(bisected[0].width - bisected[1].width) < 1e-10


def test_constrain_function():
    i = Interval(-1.0, 3.0)
    c = constrain(i, 0.0, 2.0)
    assert c.lb >= 0.0 and c.ub <= 2.0

    # Constrain float above max
    cf = constrain(5.0, 0.0, 3.0)
    assert cf == 3.0

    # Constrain float below min
    cf2 = constrain(-1.0, 0.0, 3.0)
    assert cf2 == 0.0


def test_shrink_function():
    i = Interval(0.0, 4.0)
    s = shrink(i, 0.5)
    expected_middle = i.middle
    expected_width = 0.5 * i.width * 0.5
    assert math.isclose(s.middle, expected_middle)
    assert math.isclose(s.width, i.width * 0.5)

    f = 2.0
    sf = shrink(f, 0.5)
    assert sf == 2.0


def test_hausdorff_distance_parts_and_full():
    i1 = Interval(0.0, 2.0)
    i2 = Interval(1.0, 3.0)
    d1 = _hausdorff_distance_part(i1, i2)
    assert d1 == 1.0
    d2 = _hausdorff_distance_part(i2, i1)
    assert d2 == 0.0
    d = hausdorff_distance(i1, i2)
    assert d == 1.0

    # Single floats
    f1 = 1.0
    f2 = 3.0
    d_floats = hausdorff_distance(f1, f2)
    assert d_floats == 2.0

    # Identical intervals return 0
    d_ident = hausdorff_distance(i1, i1)
    assert d_ident == 0.0


def test_monotonic_decorator_behavior():
    from intervallum.interval import monotonic

    def sample_function():
        return (lambda x: 2 * x + 1), [1, 2, 3]
    decorated = monotonic(sample_function)()
    # Should create an Interval covering 3 points applied
    assert isinstance(decorated, Interval)
    assert decorated.lb <= 3 and decorated.ub >= 7


def test_reduce_result_decorator_behavior():
    from intervallum.interval import reduce_result, IntervalConstants

    @reduce_result
    def create_interval():
        return Interval(1.0, 1.0 + IntervalConstants._reduction_width / 2)

    IntervalConstants._reduce_intervals_to_numbers = True
    res = create_interval()
    assert isinstance(res, float)

    IntervalConstants._reduce_intervals_to_numbers = False
    res2 = create_interval()
    assert isinstance(res2, Interval)

    # restore default
    IntervalConstants._reduce_intervals_to_numbers = True
