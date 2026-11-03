import math

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
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


@pytest.fixture(autouse=True)
def reset_interval_constants():
    old_reduce = IntervalConstants._reduce_intervals_to_numbers
    old_width = IntervalConstants._reduction_width
    old_error = IntervalConstants._admissible_error
    yield
    IntervalConstants._reduce_intervals_to_numbers = old_reduce
    IntervalConstants._reduction_width = old_width
    IntervalConstants._admissible_error = old_error


def test_interval_str_repr_and_from_point():
    i = Interval(1.0, 2.5)
    assert str(i) == "[1.0; 2.5]"
    assert repr(i) == "[1.0; 2.5]"

    p = Interval.from_point(3.0)
    assert p.lb == 3.0
    assert p.ub == 3.0


def test_interval_wrong_bounds_raises_and_fix_swaps():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)

    fixed = Interval(2.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 2.0


def test_interval_property_setters_validate_bounds():
    i = Interval(1.0, 3.0)
    i.lb = 0.5
    i.ub = 4.0
    assert i.lb == 0.5
    assert i.ub == 4.0

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 10.0

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0


def test_interval_equality_and_comparison_with_float_and_interval():
    i = Interval(1.0, 2.0)
    assert i == Interval(1.0, 2.0)
    assert i == 1.0
    assert i != 3.0
    assert i < 3.0
    assert i <= 1.0
    assert i > 0.5
    assert i >= 2.0

    with pytest.raises(NotImplementedError):
        _ = i == "not-a-number"


def test_interval_add_sub_mul_neg_and_r_operations():
    i = Interval(1.0, 2.0)
    j = Interval(3.0, 4.0)

    assert i + j == Interval(4.0, 6.0)
    assert 2.0 + i == Interval(3.0, 4.0)
    assert -i == Interval(-2.0, -1.0)
    assert i - 1.0 == Interval(0.0, 1.0)
    assert 5.0 - i == Interval(3.0, 4.0)
    assert i * j == Interval(3.0, 8.0)
    assert 2.0 * i == Interval(2.0, 4.0)


def test_interval_division_and_inversion():
    i = Interval(2.0, 4.0)
    inv = ~i
    assert inv == Interval(0.25, 0.5)
    assert i / 2.0 == Interval(1.0, 2.0)
    assert 8.0 / i == Interval(2.0, 4.0)

    cross_zero = Interval(-1.0, 2.0)
    inv_cross = ~cross_zero
    assert math.isinf(inv_cross.lb) and math.isinf(inv_cross.ub)


def test_interval_power_even_and_odd():
    i = Interval(-2.0, 3.0)
    assert i ** 2 == Interval(0.0, 9.0)
    assert i ** 3 == Interval(-8.0, 27.0)

    point = Interval.from_point(-2.0)
    assert point ** 4 == 16.0 if IntervalConstants._reduce_intervals_to_numbers else Interval(16.0, 16.0)


def test_interval_reduction_to_number_toggle():
    IntervalConstants._reduce_intervals_to_numbers = True
    IntervalConstants._reduction_width = 1e-5
    reduced = Interval(1.0, 1.0 + 1e-6)
    assert isinstance(reduced._try_to_reduce(), float)

    IntervalConstants._reduce_intervals_to_numbers = False
    not_reduced = Interval(1.0, 1.0 + 1e-6)
    result = not_reduced + 0.0
    assert isinstance(result, Interval)


def test_box_properties_str_repr_equality_and_copy():
    b = Box(Interval(1.0, 2.0), 3.0)
    assert str(b) == "[1.0; 2.0] x 3.0"
    assert repr(b) == "[1.0; 2.0] x 3.0"

    assert np.array_equal(b.lb, np.array([1.0, 3.0]))
    assert np.array_equal(b.ub, np.array([2.0, 3.0]))
    assert np.array_equal(b.middle, np.array([1.5, 3.0]))

    copied = b.__copy__()
    assert copied == b
    assert copied is not b

    with pytest.raises(NotImplementedError):
        _ = b == "invalid"


def test_box_width_and_try_to_reduce():
    b = Box(Interval(1.0, 1.0), Interval(2.0, 5.0), 7.0)
    indices, max_width = b.width
    assert indices == [1]
    assert max_width == 3.0
    reduced = b._try_to_reduce()
    assert isinstance(reduced, Box)

    point_box = Box(Interval(1.0, 1.0), 2.0)
    assert np.array_equal(point_box._try_to_reduce(), np.array([1.0, 2.0]))


def test_rs_subinclusion_function_uses_sampled_points(monkeypatch):
    b = Box(Interval(0.0, 1.0), Interval(10.0, 20.0))

    samples = iter(
        [
            np.array([0.1, 11.0]),
            np.array([0.7, 19.0]),
            np.array([0.4, 12.5]),
        ]
    )

    def fake_uniform(low, high):
        return next(samples)

    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    f = RS_SubinclusionFunction(lambda p: float(p[0] + p[1]), number_of_samples=3)
    result = f(b)
    assert isinstance(result, Interval)
    assert result.lb == pytest.approx(11.1)
    assert result.ub == pytest.approx(19.7)


def test_split_and_bisect_interval_and_number():
    i = Interval(0.0, 10.0)
    parts = split(i, [1.0, 2.0, 1.0])
    assert parts == [Interval(0.0, 2.5), Interval(2.5, 7.5), Interval(7.5, 10.0)]

    left, right = bisect(i)
    assert left == Interval(0.0, 5.0)
    assert right == Interval(5.0, 10.0)

    assert split(3.0, [1.0, 2.0]) == [3.0, 3.0]


def test_constrain_and_shrink():
    i = Interval(-5.0, 12.0)
    assert constrain(i, 0.0, 10.0) == Interval(0.0, 10.0)
    assert constrain(7.0, 0.0, 10.0) == 7.0

    s = shrink(Interval(0.0, 10.0), 0.4)
    assert s == Interval(3.0, 7.0)


def test_trigonometric_functions_cover_extrema():
    s = sin(Interval(0.0, math.pi / 2))
    c = cos(Interval(0.0, math.pi))
    assert s == Interval(0.0, 1.0)
    assert c == Interval(-1.0, 1.0)

    wide = sin(Interval(-10.0, 10.0))
    assert wide == Interval(-1.0, 1.0)


def test_abs_exp_sqrt_log_on_intervals_and_numbers():
    assert interval_abs(Interval(-3.0, 2.0)) == Interval(0.0, 3.0)
    assert exp(0.0) == 1.0
    assert exp(Interval(0.0, 1.0)) == Interval(1.0, math.e)

    assert sqrt(Interval(4.0, 9.0)) == Interval(2.0, 3.0)
    assert log(Interval(1.0, math.e)) == Interval(0.0, 1.0)


def test_sqrt_and_log_raise_for_invalid_domains():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-4.0, -1.0))

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-2.0, 0.0))


def test_hausdorff_distance_for_intervals_and_numbers():
    assert hausdorff_distance(Interval(0.0, 2.0), Interval(1.0, 3.0)) == 1.0
    assert hausdorff_distance(1.0, Interval(2.0, 4.0)) == 3.0
    assert hausdorff_distance(2.0, 2.0) == 0.0
