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


def test_interval_init_and_fix_swaps_bounds():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(3.0, 1.0)

    i = Interval(3.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 3.0


def test_interval_from_point_and_str():
    i = Interval.from_point(2.5)
    assert i.lb == 2.5
    assert i.ub == 2.5
    assert str(i) == "[2.5; 2.5]"
    assert repr(i) == "[2.5; 2.5]"


def test_interval_property_setters_validate_bounds():
    i = Interval(1.0, 2.0)
    i.lb = 0.5
    i.ub = 2.5
    assert i.lb == 0.5
    assert i.ub == 2.5

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.1


def test_interval_equality_respects_admissible_error():
    IntervalConstants._admissible_error = 1e-7
    assert Interval(1.0, 1.0) == Interval(1.0 + 5e-8, 1.0 + 5e-8)
    assert Interval(1.0, 1.0) != Interval(1.0 + 1e-4, 1.0 + 1e-4)
    assert Interval(1.0, 1.0) == 1.0
    assert Interval(1.0, 1.0) != 1.1


def test_interval_comparison_uses_lower_bound():
    i1 = Interval(1.0, 5.0)
    i2 = Interval(2.0, 3.0)
    assert i1 < i2
    assert i1 <= i2
    assert i2 > i1
    assert i2 >= i1
    assert i1 < 1.5
    assert i2 >= 2.0


def test_interval_add_neg_sub_mul():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(3.0, 4.0)
    assert i1 + i2 == Interval(4.0, 6.0)
    assert -i1 == Interval(-2.0, -1.0)
    assert i2 - i1 == Interval(1.0, 3.0)
    assert i1 * i2 == Interval(3.0, 8.0)


def test_interval_division_and_inversion():
    i = Interval(2.0, 4.0)
    inv = ~i
    assert isinstance(inv, Interval)
    assert inv == Interval(0.25, 0.5)
    assert i / 2 == Interval(1.0, 2.0)
    assert 8 / i == Interval(2.0, 4.0)


def test_interval_power_even_and_odd():
    i = Interval(-2.0, 3.0)
    even = i ** 2
    odd = i ** 3
    assert isinstance(even, Interval)
    assert even == Interval(0.0, 9.0)
    assert odd == Interval(-8.0, 27.0)


def test_interval_copy_and_reduce_result_behavior():
    IntervalConstants._reduce_intervals_to_numbers = False
    i = Interval(1.0, 1.0 + 1e-6)
    copied = copy(i)
    assert copied == i
    assert copied is not i
    reduced = i._try_to_reduce()
    assert isinstance(reduced, Interval)

    IntervalConstants._reduce_intervals_to_numbers = True
    reduced2 = Interval(1.0, 1.0 + 1e-6)._try_to_reduce()
    assert isinstance(reduced2, float)


def test_box_properties_and_string_representation():
    b = Box(Interval(1.0, 2.0), Interval(3.0, 5.0))
    assert np.allclose(b.lb, np.array([1.0, 3.0]))
    assert np.allclose(b.ub, np.array([2.0, 5.0]))
    assert np.allclose(b.middle, np.array([1.5, 4.0]))
    assert b.width == ([1], 2.0)
    assert str(b) == "[1.0; 2.0] x [3.0; 5.0]"
    assert repr(b) == "[1.0; 2.0] x [3.0; 5.0]"


def test_box_equality_and_copy():
    b1 = Box(Interval(1.0, 2.0), Interval(3.0, 4.0))
    b2 = Box(Interval(1.0, 2.0), Interval(3.0, 4.0))
    b3 = copy(b1)
    assert b1 == b2
    assert b1 == b3
    assert b1 != Box(Interval(1.0, 2.1), Interval(3.0, 4.0))
    with pytest.raises(NotImplementedError):
        _ = b1 == 123


def test_box_try_to_reduce_returns_middle_for_zero_width():
    b = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    reduced = b._try_to_reduce()
    assert isinstance(reduced, np.ndarray)
    assert np.allclose(reduced, np.array([1.0, 2.0]))

    b2 = Box(Interval(1.0, 2.0), Interval(2.0, 2.0))
    reduced2 = b2._try_to_reduce()
    assert isinstance(reduced2, Box)


def test_interval_functions_sin_cos_abs_exp_on_intervals():
    IntervalConstants._reduce_intervals_to_numbers = False
    assert sin(Interval(0.0, math.pi / 2)) == Interval(0.0, 1.0)
    assert cos(Interval(0.0, math.pi)) == Interval(-1.0, 1.0)
    assert iabs(Interval(-2.0, 3.0)) == Interval(0.0, 3.0)
    assert exp(Interval(0.0, 1.0)) == Interval(1.0, math.e)


def test_interval_functions_sqrt_and_log_validate_domain():
    assert sqrt(Interval(1.0, 4.0)) == Interval(1.0, 2.0)
    assert sqrt(Interval(-1.0, 4.0)) == Interval(0.0, 2.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-4.0, -1.0))

    assert log(Interval(1.0, math.e)) == Interval(0.0, 1.0)
    assert log(Interval(-1.0, 2.0)) == Interval(-math.inf, math.log(2.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-2.0, -1.0))


def test_split_bisect_constrain_and_shrink():
    IntervalConstants._reduce_intervals_to_numbers = False
    parts = split(Interval(0.0, 10.0), [1.0, 2.0, 1.0])
    assert parts == [Interval(0.0, 2.5), Interval(2.5, 7.5), Interval(7.5, 10.0)]
    assert bisect(Interval(0.0, 10.0)) == [Interval(0.0, 5.0), Interval(5.0, 10.0)]
    assert constrain(Interval(-2.0, 3.0), 0.0, 2.0) == Interval(0.0, 2.0)
    assert shrink(Interval(0.0, 10.0), 0.2) == Interval(4.0, 6.0)


def test_hausdorff_distance_between_intervals_and_numbers():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == 2.0
    assert hausdorff_distance(Interval(0.0, 3.0), Interval(1.0, 2.0)) == 1.0
    assert hausdorff_distance(1.0, 4.0) == 3.0
    assert hausdorff_distance(Interval(1.0, 1.0), 1.0) == 0.0


def test_rs_subinclusion_function_uses_sampled_points(monkeypatch):
    calls = []

    def fake_uniform(low, high):
        calls.append((np.array(low), np.array(high)))
        return np.array(low) + 0.5 * (np.array(high) - np.array(low))

    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    def f(x):
        return float(np.sum(x))

    rs = RS_SubinclusionFunction(f, number_of_samples=4)
    box = Box(Interval(0.0, 2.0), Interval(10.0, 14.0))
    result = rs(box)

    assert len(calls) == 4
    assert result == Interval(11.0, 11.0)
