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
    old_err = IntervalConstants._admissible_error
    yield
    IntervalConstants._reduce_intervals_to_numbers = old_reduce
    IntervalConstants._reduction_width = old_width
    IntervalConstants._admissible_error = old_err


def test_interval_init_and_string_representation():
    i = Interval(1.5, 3.25)
    assert i.lb == 1.5
    assert i.ub == 3.25
    assert str(i) == "[1.5; 3.25]"
    assert repr(i) == "[1.5; 3.25]"


def test_interval_init_wrong_bounds_raises():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)


def test_interval_init_fix_swaps_bounds():
    i = Interval(2.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 2.0


def test_interval_setters_validate_bounds():
    i = Interval(1.0, 2.0)
    i.lb = 0.5
    i.ub = 2.5
    assert i.lb == 0.5
    assert i.ub == 2.5
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.1


def test_interval_equality_tolerates_small_distance():
    a = Interval(1.0, 2.0)
    b = Interval(1.0 + 1e-8, 2.0 + 1e-8)
    assert a == b
    assert not (a != b)


def test_interval_equality_with_float_and_notimplemented_for_other_types():
    a = Interval(3.0, 3.0)
    assert a == 3.0
    with pytest.raises(NotImplementedError):
        _ = a == "3"


def test_interval_arithmetic_add_sub_mul_div_and_power():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 4.0)

    assert a + b == Interval(4.0, 6.0)
    assert a + 2.0 == Interval(3.0, 4.0)
    assert 2.0 + a == Interval(3.0, 4.0)

    assert b - a == Interval(1.0, 3.0)
    assert 10.0 - a == Interval(8.0, 9.0)

    assert a * b == Interval(3.0, 8.0)
    assert a * 3.0 == Interval(3.0, 6.0)
    assert 3.0 * a == Interval(3.0, 6.0)

    assert a / 2.0 == Interval(0.5, 1.0)
    assert 2.0 / a == Interval(1.0, 2.0)

    assert (Interval(-2.0, 3.0) ** 2) == Interval(0.0, 9.0)
    assert (Interval(-2.0, 3.0) ** 3) == Interval(-8.0, 27.0)


def test_interval_comparison_uses_lower_bound():
    a = Interval(1.0, 10.0)
    b = Interval(2.0, 3.0)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert a < 1.5
    assert b >= 2.0


def test_interval_inverse_and_division_by_interval():
    i = Interval(2.0, 4.0)
    inv = ~i
    assert inv == Interval(0.25, 0.5)
    assert Interval(8.0, 8.0) / i == Interval(2.0, 4.0)


def test_interval_inverse_of_interval_crossing_zero_contains_infinities():
    i = Interval(-1.0, 1.0)
    inv = ~i
    assert math.isinf(inv.lb)
    assert math.isinf(inv.ub)

    left = Interval(-2.0, -1.0)
    inv2 = ~left
    assert inv2 == Interval(-1.0, -0.5)


def test_interval_reduction_to_number_toggle():
    IntervalConstants._reduce_intervals_to_numbers = True
    reduced = Interval(1.0, 1.0) + Interval(2.0, 2.0)
    assert isinstance(reduced, float)
    assert reduced == 3.0

    IntervalConstants._reduce_intervals_to_numbers = False
    not_reduced = Interval(1.0, 1.0) + Interval(2.0, 2.0)
    assert isinstance(not_reduced, Interval)
    assert not_reduced == Interval(3.0, 3.0)


def test_box_properties_and_string_representation():
    b = Box(Interval(1.0, 2.0), 3.0)
    assert np.allclose(b.lb, np.array([1.0, 3.0]))
    assert np.allclose(b.ub, np.array([2.0, 3.0]))
    assert np.allclose(b.middle, np.array([1.5, 3.0]))
    assert b.width == ([0], 1.0)
    assert str(b) == "[1.0; 2.0] x 3.0"
    assert repr(b) == "[1.0; 2.0] x 3.0"


def test_box_equality_copy_and_reduce_behavior():
    b = Box(Interval(1.0, 2.0), 3.0)
    c = b.__copy__()
    assert c == b
    assert c is not b
    assert isinstance(c[0], Interval)

    reduced = Box(1.0, 2.0)._try_to_reduce()
    assert isinstance(reduced, np.ndarray)
    assert np.allclose(reduced, np.array([1.0, 2.0]))


def test_interval_functions_trig_exp_abs_sqrt_log():
    i = Interval(0.0, math.pi / 2)
    s = sin(i)
    c = cos(i)
    assert s == Interval(0.0, 1.0)
    assert c == Interval(0.0, 1.0)

    assert exp(Interval(0.0, 1.0)) == Interval(1.0, math.e)
    assert iabs(Interval(-2.0, 3.0)) == Interval(0.0, 3.0)
    assert sqrt(Interval(4.0, 9.0)) == Interval(2.0, 3.0)
    assert log(Interval(1.0, math.e)) == Interval(0.0, 1.0)


def test_sqrt_and_log_invalid_domains_raise():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-4.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-2.0, 0.0))


def test_split_and_bisect_interval_and_point():
    i = Interval(0.0, 10.0)
    parts = split(i, [1.0, 2.0, 1.0])
    assert parts == [Interval(0.0, 2.5), Interval(2.5, 7.5), Interval(7.5, 10.0)]

    left, right = bisect(i)
    assert left == Interval(0.0, 5.0)
    assert right == Interval(5.0, 10.0)

    assert split(3.14, [1.0, 1.0]) == [3.14, 3.14]


def test_constrain_and_shrink():
    i = Interval(-2.0, 5.0)
    constrained = constrain(i, 0.0, 4.0)
    assert constrained == Interval(0.0, 4.0)

    assert constrain(10.0, 0.0, 4.0) == 4.0
    assert shrink(Interval(0.0, 10.0), 0.2) == Interval(4.0, 6.0)


def test_hausdorff_distance_for_intervals_and_points():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.5, 1.5)) == 0.5
    assert hausdorff_distance(1.0, 3.0) == 2.0
    assert hausdorff_distance(Interval(0.0, 1.0), 0.5) == 0.5


def test_rs_subinclusion_function_uses_samples_and_box_bounds(monkeypatch):
    samples = []

    def fake_uniform(low, high):
        samples.append((np.array(low), np.array(high)))
        return np.array([0.25, 0.75])

    monkeypatch.setattr(np.random, "uniform", fake_uniform)

    def f(x):
        return float(np.sum(x))

    rs = RS_SubinclusionFunction(f, number_of_samples=3)
    result = rs(Box(Interval(0.0, 1.0), Interval(2.0, 4.0)))

    assert isinstance(result, Interval)
    assert result == Interval(1.0, 1.0)
    assert len(samples) == 3
    for low, high in samples:
        assert np.allclose(low, np.array([0.0, 2.0]))
        assert np.allclose(high, np.array([1.0, 4.0]))
