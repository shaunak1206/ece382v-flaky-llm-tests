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
    abs as interval_abs,
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


def test_interval_init_and_string_repr():
    i = Interval(1.5, 3.0)
    assert i.lb == 1.5
    assert i.ub == 3.0
    assert str(i) == "[1.5; 3.0]"
    assert repr(i) == "[1.5; 3.0]"


def test_interval_init_wrong_bounds_raises():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(3.0, 1.0)


def test_interval_init_fix_swaps_bounds():
    i = Interval(3.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 3.0


def test_interval_equality_within_and_beyond_tolerance():
    a = Interval(1.0, 2.0)
    b = Interval(1.0 + 1e-8, 2.0 - 1e-8)
    c = Interval(1.0 + 1e-3, 2.0)
    assert a == b
    assert a != c


def test_interval_comparison_uses_lower_bound():
    a = Interval(1.0, 10.0)
    b = Interval(2.0, 3.0)
    assert a < b
    assert b > a
    assert a <= 1.0
    assert b >= 2.0


def test_interval_arithmetic_add_sub_mul_div():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 5.0)
    assert a + b == Interval(4.0, 7.0)
    assert b - a == Interval(1.0, 4.0)
    assert a * b == Interval(3.0, 10.0)
    assert a / 2.0 == Interval(0.5, 1.0)


def test_interval_neg_and_rsub():
    i = Interval(1.0, 3.0)
    assert -i == Interval(-3.0, -1.0)
    assert 10.0 - i == Interval(7.0, 9.0)


def test_interval_power_even_and_odd():
    i = Interval(-2.0, 3.0)
    assert i ** 2 == Interval(0.0, 9.0)
    assert Interval(2.0, 4.0) ** 3 == Interval(8.0, 64.0)


def test_interval_inverse_and_division_by_interval():
    i = Interval(2.0, 4.0)
    inv = ~i
    assert inv == Interval(0.25, 0.5)
    assert i / Interval(2.0, 2.0) == Interval(1.0, 2.0)


def test_interval_copy_and_from_point():
    i = Interval.from_point(4.2)
    j = i.__copy__()
    assert i == j
    assert i is not j
    assert i.lb == i.ub == 4.2


def test_box_basic_properties_and_string():
    b = Box(Interval(1.0, 2.0), Interval(3.0, 5.0))
    assert isinstance(b, Box)
    assert np.allclose(b.lb, np.array([1.0, 3.0]))
    assert np.allclose(b.ub, np.array([2.0, 5.0]))
    assert np.allclose(b.middle, np.array([1.5, 4.0]))
    assert b.width == ([1], 2.0)
    assert str(b) == "[1.0; 2.0] x [3.0; 5.0]"


def test_box_copy_and_equality_with_ndarray():
    b1 = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    b2 = b1.__copy__()
    assert b1 == b2
    assert b1 is not b2
    assert isinstance(b2, Box)
    assert b1 != np.array([[0.0, 1.0], [2.0, 3.0]])


def test_box_try_to_reduce_returns_middle_for_zero_width():
    b = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    reduced = b._try_to_reduce()
    assert isinstance(reduced, np.ndarray)
    assert np.allclose(reduced, np.array([1.0, 2.0]))

    b2 = Box(Interval(1.0, 2.0), Interval(2.0, 2.0))
    reduced2 = b2._try_to_reduce()
    assert isinstance(reduced2, Box)


def test_rs_subinclusion_function_uses_sampled_points(monkeypatch):
    values = [1.0, 4.0, 2.0, 3.0]

    def fake_uniform(low, high):
        return np.array([(l + h) / 2.0 for l, h in zip(low, high)])

    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    def f(x):
        return values.pop(0)

    rs = RS_SubinclusionFunction(f, number_of_samples=4)
    result = rs(Box(Interval(0.0, 1.0), Interval(2.0, 4.0)))
    assert result == Interval(1.0, 4.0)


def test_split_and_bisect_on_interval_and_number():
    i = Interval(0.0, 10.0)
    parts = split(i, [1.0, 2.0, 1.0])
    assert parts == [Interval(0.0, 2.5), Interval(2.5, 7.5), Interval(7.5, 10.0)]
    assert bisect(i) == [Interval(0.0, 5.0), Interval(5.0, 10.0)]
    assert split(3.14, [1.0, 2.0]) == [3.14, 3.14]


def test_constrain_and_shrink():
    assert constrain(Interval(-2.0, 5.0), 0.0, 3.0) == Interval(0.0, 3.0)
    assert constrain(2.5, 0.0, 3.0) == 2.5

    i = Interval(0.0, 10.0)
    assert shrink(i, 0.5) == Interval(2.5, 7.5)
    assert shrink(1.23, 0.5) == 1.23


def test_hausdorff_distance_on_intervals_and_points():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(1.0, 2.0)) == 1.0
    assert hausdorff_distance(Interval(0.0, 2.0), Interval(0.5, 1.5)) == 0.5
    assert hausdorff_distance(1.0, 4.0) == 3.0


def test_monotonic_interval_functions_basic():
    assert sin(Interval(0.0, math.pi / 2.0)) == Interval(0.0, 1.0)
    assert cos(Interval(0.0, math.pi)) == Interval(-1.0, 1.0)
    assert interval_abs(Interval(-2.0, 3.0)) == Interval(0.0, 3.0)
    assert exp(Interval(0.0, 1.0)) == Interval(1.0, math.e)


def test_sqrt_and_log_error_paths():
    assert sqrt(Interval(4.0, 9.0)) == Interval(2.0, 3.0)
    assert log(Interval(1.0, math.e)) == Interval(0.0, 1.0)

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-4.0, -1.0))

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-2.0, 0.0))


def test_reduce_result_flag_can_return_number(monkeypatch):
    old_reduce = IntervalConstants._reduce_intervals_to_numbers
    old_width = IntervalConstants._reduction_width
    try:
        IntervalConstants._reduce_intervals_to_numbers = True
        IntervalConstants._reduction_width = 10.0
        assert Interval(1.0, 1.5) + Interval(2.0, 2.5) == 3.0
        assert (Interval(1.0, 1.001) * 2.0) == 2.002
    finally:
        IntervalConstants._reduce_intervals_to_numbers = old_reduce
        IntervalConstants._reduction_width = old_width
