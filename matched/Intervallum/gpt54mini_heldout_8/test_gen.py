import math
from copy import copy

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


def test_interval_construction_and_properties():
    i = Interval(-2.0, 3.0)
    assert str(i) == "[-2.0; 3.0]"
    assert repr(i) == "[-2.0; 3.0]"
    assert i.lb == -2.0
    assert i.ub == 3.0
    assert i.middle == 0.5
    assert i.width == 5.0


def test_interval_from_point_and_copy():
    i = Interval.from_point(4.2)
    j = copy(i)
    assert i == 4.2
    assert j == i
    assert j is not i
    assert j.lb == 4.2 and j.ub == 4.2


def test_interval_wrong_bounds_and_setters():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, -1.0)

    i = Interval(1.0, 2.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0


def test_interval_equality_uses_admissible_error():
    a = Interval(1.0, 2.0)
    b = Interval(1.0 + 1e-8, 2.0 + 1e-8)
    c = Interval(1.0, 2.00001)
    assert a == b
    assert a != c
    assert a == 1.5 is False


def test_interval_order_comparisons():
    a = Interval(1.0, 10.0)
    b = Interval(2.0, 3.0)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert a < 1.5 is True
    assert b >= 2.0 is True


def test_interval_add_sub_neg_mul_div_scalar_and_interval():
    i = Interval(1.0, 3.0)
    j = Interval(-2.0, 4.0)

    assert i + 2.0 == Interval(3.0, 5.0)
    assert 2.0 + i == Interval(3.0, 5.0)
    assert -i == Interval(-3.0, -1.0)
    assert i - 2.0 == Interval(-1.0, 1.0)
    assert 5.0 - i == Interval(2.0, 4.0)

    prod = i * j
    assert isinstance(prod, Interval)
    assert prod == Interval(-6.0, 12.0)

    div = i / 2.0
    assert div == Interval(0.5, 1.5)
    div2 = i / Interval(2.0, 4.0)
    assert isinstance(div2, Interval)
    assert div2.lb <= div2.ub


def test_interval_inversion_handles_sign_cases():
    assert ~Interval(2.0, 4.0) == Interval(0.25, 0.5)
    assert ~Interval(-4.0, -2.0) == Interval(-0.5, -0.25)

    pos_zero = ~Interval(0.0, 2.0)
    assert math.isinf(pos_zero.ub)
    assert pos_zero.lb == 0.5

    neg_zero = ~Interval(-2.0, 0.0)
    assert math.isinf(neg_zero.lb) and neg_zero.lb < 0
    assert neg_zero.ub == -0.5


def test_interval_power_even_and_odd():
    i = Interval(-2.0, 3.0)
    even = i ** 2
    assert isinstance(even, Interval)
    assert even == Interval(0.0, 9.0)

    odd = i ** 3
    assert isinstance(odd, Interval)
    assert odd == Interval(-8.0, 27.0)


def test_box_basic_properties_and_string():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))
    assert str(b) == "[0.0; 1.0] x [2.0; 4.0]"
    assert repr(b) == "[0.0; 1.0] x [2.0; 4.0]"
    np.testing.assert_array_equal(b.lb, np.array([0.0, 2.0]))
    np.testing.assert_array_equal(b.ub, np.array([1.0, 4.0]))
    np.testing.assert_array_equal(b.middle, np.array([0.5, 3.0]))


def test_box_equality_and_copy():
    b1 = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))
    b2 = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))
    b3 = copy(b1)

    assert b1 == b2
    assert b1 == b3
    assert b1 is not b3
    assert b1 != Box(Interval(0.0, 1.0), Interval(2.0, 5.0))


def test_box_width_and_try_to_reduce():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 2.0))
    idx, max_width = b.width
    assert idx == [0]
    assert max_width == 1.0

    reduced = b._try_to_reduce()
    assert isinstance(reduced, Box)
    assert reduced == b

    p = Box(Interval(5.0, 5.0), Interval(7.0, 7.0))
    reduced_p = p._try_to_reduce()
    assert isinstance(reduced_p, np.ndarray)
    np.testing.assert_array_equal(reduced_p, np.array([5.0, 7.0]))


def test_split_and_bisect_interval_and_scalar():
    i = Interval(0.0, 10.0)
    parts = split(i, [1.0, 2.0, 1.0])
    assert parts == [Interval(0.0, 2.5), Interval(2.5, 7.5), Interval(7.5, 10.0)]

    left, right = bisect(i)
    assert left == Interval(0.0, 5.0)
    assert right == Interval(5.0, 10.0)

    scalar_parts = split(3.14, [1.0, 1.0, 1.0])
    assert scalar_parts == [3.14, 3.14, 3.14]


def test_split_reduces_to_numbers_when_enabled():
    IntervalConstants._reduce_intervals_to_numbers = True
    i = Interval(1.0, 1.0 + 1e-6)
    parts = split(i, [1.0, 1.0])
    assert parts == [1.0, 1.0000005]


def test_constraint_and_shrink_on_interval_and_scalar():
    i = Interval(-2.0, 5.0)
    constrained = constrain(i, 0.0, 3.0)
    assert constrained == Interval(0.0, 3.0)

    assert constrain(-5.0, -1.0, 1.0) == -1.0
    assert constrain(0.25, -1.0, 1.0) == 0.25

    shrunk = shrink(i, 0.5)
    assert shrunk == Interval(-1.25, 4.25)
    assert shrink(2.0, 0.1) == 2.0


def test_hausdorff_distance_for_intervals_and_scalars():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 4.0)) == 3.0
    assert hausdorff_distance(0.0, Interval(2.0, 4.0)) == 4.0
    assert hausdorff_distance(Interval(1.0, 3.0), 2.0) == 1.0
    assert hausdorff_distance(1.5, 1.5) == 0.0


def test_standard_interval_functions_on_basic_intervals():
    i = Interval(0.0, math.pi / 2)
    s = sin(i)
    c = cos(i)
    a = iabs(Interval(-2.0, 3.0))
    e = exp(Interval(0.0, 1.0))

    assert s == Interval(0.0, 1.0)
    assert c == Interval(0.0, 1.0)
    assert a == Interval(0.0, 3.0)
    assert e == Interval(1.0, math.e)


def test_sqrt_and_log_valid_and_invalid_domains():
    assert sqrt(Interval(4.0, 9.0)) == Interval(2.0, 3.0)
    assert sqrt(Interval(-1.0, 4.0)) == Interval(0.0, 2.0)

    assert log(Interval(1.0, math.e)) == Interval(0.0, 1.0)
    assert log(Interval(-2.0, 3.0)) == Interval(-math.inf, math.log(3.0))

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-4.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-4.0, -1.0))


def test_random_subinclusion_function_uses_samples_and_bounds(monkeypatch):
    calls = []

    def fake_uniform(low, high):
        calls.append((np.array(low), np.array(high)))
        return np.array([(np.asarray(low)[0] + np.asarray(high)[0]) / 2.0])

    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    f = lambda x: float(np.sum(x))
    rs = RS_SubinclusionFunction(f=f, number_of_samples=4)
    b = Box(Interval(0.0, 2.0), Interval(10.0, 14.0))

    result = rs(b)

    assert len(calls) == 4
    for low, high in calls:
        np.testing.assert_array_equal(low, np.array([0.0, 10.0]))
        np.testing.assert_array_equal(high, np.array([2.0, 14.0]))
    assert result == Interval(6.0, 6.0)
