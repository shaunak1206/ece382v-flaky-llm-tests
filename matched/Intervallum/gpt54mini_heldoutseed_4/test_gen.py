import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    abs as iabs,
    bisect,
    constrain,
    cos,
    exp,
    hausdorff_distance,
    log,
    shrink,
    sin,
    split,
    sqrt,
)


@pytest.fixture(autouse=True)
def restore_interval_constants():
    old_reduce = IntervalConstants._reduce_intervals_to_numbers
    old_width = IntervalConstants._reduction_width
    old_error = IntervalConstants._admissible_error
    yield
    IntervalConstants._reduce_intervals_to_numbers = old_reduce
    IntervalConstants._reduction_width = old_width
    IntervalConstants._admissible_error = old_error


def test_interval_init_and_properties():
    i = Interval(1.5, 4.0)
    assert str(i) == "[1.5; 4.0]"
    assert repr(i) == "[1.5; 4.0]"
    assert i.lb == 1.5
    assert i.ub == 4.0
    assert i.middle == 2.75
    assert i.width == 2.5


def test_interval_init_wrong_bounds_raises():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(5.0, 1.0)


def test_interval_fix_and_setters():
    i = Interval(5.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 5.0
    i.lb = 0.5
    i.ub = 6.0
    assert i.lb == 0.5
    assert i.ub == 6.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 7.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.1


def test_interval_equality_and_inequality():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(1.0, 2.0)
    i3 = Interval(1.0, 2.0000002)
    assert i1 == i2
    assert i1 != i3
    assert i1 == 1.0
    assert not (i1 != 1.0)


def test_interval_comparison_operators():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(3.0, 4.0)
    assert i1 < i2
    assert i1 <= i2
    assert i2 > i1
    assert i2 >= i1
    assert i1 < 2.0
    assert i2 > 2.5


def test_interval_arithmetic_add_sub_mul_div():
    i = Interval(1.0, 3.0)
    j = Interval(-2.0, 4.0)
    assert i + j == Interval(-1.0, 7.0)
    assert i - j == Interval(-3.0, 5.0)
    assert i * j == Interval(-6.0, 12.0)
    assert (i / 2.0) == Interval(0.5, 1.5)
    assert (8.0 / Interval(2.0, 4.0)) == Interval(2.0, 4.0)


def test_interval_negation_and_inverse_cases():
    assert -Interval(1.0, 3.0) == Interval(-3.0, -1.0)
    assert ~Interval(2.0, 4.0) == Interval(0.25, 0.5)
    inv_cross_zero = ~Interval(-2.0, 2.0)
    assert math.isinf(inv_cross_zero.lb) and math.isinf(inv_cross_zero.ub)
    inv_positive_with_zero_lb = ~Interval(0.0, 2.0)
    assert inv_positive_with_zero_lb.lb == 0.5 and math.isinf(inv_positive_with_zero_lb.ub)


def test_interval_power_even_and_odd():
    i = Interval(-2.0, 3.0)
    assert i ** 2 == Interval(0.0, 9.0)
    assert i ** 3 == Interval(-8.0, 27.0)


def test_box_properties_and_str_repr_equality():
    b = Box(Interval(1.0, 2.0), 5.0, Interval(-1.0, 4.0))
    assert str(b) == "[1.0; 2.0] x 5.0 x [-1.0; 4.0]"
    assert repr(b) == "[1.0; 2.0] x 5.0 x [-1.0; 4.0]"
    assert np.array_equal(b.lb, np.array([1.0, 5.0, -1.0]))
    assert np.array_equal(b.ub, np.array([2.0, 5.0, 4.0]))
    assert np.array_equal(b.middle, np.array([1.5, 5.0, 1.5]))
    assert b.width == ([1], 5.0)

    b2 = Box(Interval(1.0, 2.0), 5.0, Interval(-1.0, 4.0))
    assert b == b2
    assert b != Box(Interval(1.0, 2.1), 5.0, Interval(-1.0, 4.0))


def test_box_copy_and_try_to_reduce():
    b = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    c = b.__copy__()
    assert isinstance(c, Box)
    assert c == b
    reduced = b._try_to_reduce()
    assert isinstance(reduced, np.ndarray)
    assert np.array_equal(reduced, np.array([1.0, 2.0]))

    b2 = Box(Interval(1.0, 2.0), Interval(3.0, 5.0))
    not_reduced = b2._try_to_reduce()
    assert isinstance(not_reduced, Box)
    assert not_reduced == b2


def test_rs_subinclusion_function_samples_and_interval_result(monkeypatch):
    seq = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    def fake_uniform(low, high):
        # return deterministic values within the interval box
        return np.array([next(seq), next(seq)])

    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    def f(point):
        return float(point[0] + point[1])

    rs = RS_SubinclusionFunction(f, number_of_samples=3)
    result = rs(Box(Interval(0.0, 1.0), Interval(0.0, 1.0)))
    assert isinstance(result, Interval)
    assert result.lb == pytest.approx(0.1)
    assert result.ub == pytest.approx(0.9)


def test_trigonometric_functions_on_intervals():
    i = Interval(0.0, math.pi / 2)
    s = sin(i)
    c = cos(i)
    assert s == Interval(0.0, 1.0)
    assert c == Interval(0.0, 1.0)


def test_trigonometric_functions_across_full_period_reduce_to_minus_one_one():
    i = Interval(0.0, 2 * math.pi)
    assert sin(i) == Interval(-1.0, 1.0)
    assert cos(i) == Interval(-1.0, 1.0)


def test_abs_exp_sqrt_log_on_intervals():
    assert iabs(Interval(-3.0, 2.0)) == Interval(0.0, 3.0)
    assert exp(Interval(0.0, 1.0)) == Interval(1.0, math.e)
    assert sqrt(Interval(4.0, 9.0)) == Interval(2.0, 3.0)
    assert log(Interval(1.0, math.e)) == Interval(0.0, 1.0)


def test_sqrt_and_log_domain_errors():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-4.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-2.0, 0.0))


def test_split_bisect_constrain_shrink_and_hausdorff_distance():
    i = Interval(0.0, 10.0)
    parts = split(i, [1.0, 2.0, 1.0])
    assert parts == [Interval(0.0, 2.5), Interval(2.5, 7.5), Interval(7.5, 10.0)]
    assert bisect(i) == [Interval(0.0, 5.0), Interval(5.0, 10.0)]

    assert constrain(Interval(-2.0, 7.0), 0.0, 5.0) == Interval(0.0, 5.0)
    assert constrain(3.5, 0.0, 5.0) == 3.5

    assert shrink(Interval(2.0, 8.0), 0.5) == Interval(3.5, 6.5)
    assert hausdorff_distance(Interval(0.0, 2.0), Interval(1.0, 3.0)) == 1.0
    assert hausdorff_distance(1.0, Interval(1.0, 1.0)) == 0.0
