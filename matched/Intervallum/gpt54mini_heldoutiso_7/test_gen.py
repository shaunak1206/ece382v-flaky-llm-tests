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


def test_interval_init_and_string_representation():
    i = Interval(1.5, 3.5)
    assert i.lb == 1.5
    assert i.ub == 3.5
    assert str(i) == "[1.5; 3.5]"
    assert repr(i) == "[1.5; 3.5]"


def test_interval_init_raises_on_wrong_bounds_without_fix():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(5.0, 2.0)


def test_interval_init_fix_swaps_bounds():
    i = Interval(5.0, 2.0, fix=True)
    assert i.lb == 2.0
    assert i.ub == 5.0


def test_interval_setters_validate_bounds():
    i = Interval(1.0, 3.0)
    i.lb = 0.5
    i.ub = 4.0
    assert i.lb == 0.5
    assert i.ub == 4.0

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 10.0

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0


def test_interval_equality_with_tolerance_and_inequality():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(1.0 + 1e-8, 2.0 + 1e-8)
    assert i1 == i2
    assert not (i1 != i2)
    assert i1 != Interval(1.1, 2.1)


def test_interval_copy_and_from_point():
    p = Interval.from_point(4.2)
    assert p.lb == 4.2 and p.ub == 4.2
    c = p.__copy__()
    assert c == p
    assert c is not p


def test_interval_arithmetic_add_sub_mul_div_and_neg():
    i = Interval(1.0, 3.0)
    j = Interval(2.0, 5.0)

    assert i + j == Interval(3.0, 8.0)
    assert i + 2.0 == Interval(3.0, 5.0)
    assert 2.0 + i == Interval(3.0, 5.0)

    assert -i == Interval(-3.0, -1.0)
    assert j - i == Interval(-1.0, 4.0)
    assert i - 2.0 == Interval(-1.0, 1.0)
    assert 10.0 - i == Interval(7.0, 9.0)

    assert i * j == Interval(2.0, 15.0)
    assert i * 2.0 == Interval(2.0, 6.0)
    assert 2.0 * i == Interval(2.0, 6.0)

    assert i / 2.0 == Interval(0.5, 1.5)


def test_interval_inverse_and_division_by_interval():
    i = Interval(2.0, 4.0)
    inv = ~i
    assert isinstance(inv, Interval)
    assert inv.lb == pytest.approx(0.25)
    assert inv.ub == pytest.approx(0.5)

    result = i / Interval(1.0, 2.0)
    assert isinstance(result, Interval)
    assert result.lb == pytest.approx(1.0)
    assert result.ub == pytest.approx(4.0)


def test_interval_power_even_and_odd():
    i = Interval(-2.0, 3.0)
    even = i ** 2
    odd = i ** 3

    assert isinstance(even, Interval)
    assert even.lb == 0.0
    assert even.ub == 9.0

    assert isinstance(odd, Interval)
    assert odd.lb == -8.0
    assert odd.ub == 27.0


def test_box_properties_and_string_repr():
    b = Box(Interval(1.0, 2.0), Interval(-1.0, 4.0), 3.0)
    assert isinstance(b, Box)
    assert str(b) == "[1.0; 2.0] x [-1.0; 4.0] x 3.0"
    assert repr(b) == "[1.0; 2.0] x [-1.0; 4.0] x 3.0"
    np.testing.assert_allclose(b.lb, np.array([1.0, -1.0, 3.0]))
    np.testing.assert_allclose(b.ub, np.array([2.0, 4.0, 3.0]))
    np.testing.assert_allclose(b.middle, np.array([1.5, 1.5, 3.0]))


def test_box_width_and_try_to_reduce():
    b = Box(Interval(1.0, 1.0), Interval(2.0, 2.0), 3.0)
    idx, width = b.width
    assert idx == [0, 1, 2]
    assert width == 0.0
    reduced = b._try_to_reduce()
    assert isinstance(reduced, np.ndarray)
    np.testing.assert_allclose(reduced, np.array([1.0, 2.0, 3.0]))

    b2 = Box(Interval(1.0, 2.0), 5.0)
    reduced2 = b2._try_to_reduce()
    assert isinstance(reduced2, Box)
    assert reduced2 == b2


def test_box_equality_and_copy():
    b1 = Box(Interval(1.0, 2.0), 5.0)
    b2 = Box(Interval(1.0, 2.0), 5.0)
    assert b1 == b2
    assert not (b1 != b2)

    c = b1.__copy__()
    assert isinstance(c, Box)
    assert c == b1
    assert c is not b1
    assert c[0] is not b1[0]


def test_rs_subinclusion_function_returns_interval_from_samples(monkeypatch):
    monkeypatch.setattr(np.random, "uniform", lambda low, high: np.array(low, dtype=float))
    f = RS_SubinclusionFunction(lambda x: float(np.sum(x)), number_of_samples=4)
    b = Box(Interval(1.0, 2.0), Interval(10.0, 20.0))
    out = f(b)
    assert isinstance(out, Interval)
    assert out.lb == pytest.approx(11.0)
    assert out.ub == pytest.approx(11.0)


def test_trigonometric_functions_over_intervals_and_points():
    assert sin(0.0) == 0.0
    assert cos(0.0) == 1.0

    i = Interval(0.0, math.pi / 2)
    s = sin(i)
    c = cos(i)
    assert s == Interval(0.0, 1.0)
    assert c == Interval(0.0, 1.0)

    wide = Interval(-10.0, 10.0)
    wsin = sin(wide)
    wcos = cos(wide)
    assert wsin == Interval(-1.0, 1.0)
    assert wcos == Interval(-1.0, 1.0)


def test_abs_exp_sqrt_log_functions():
    assert iabs(-3.0) == 3.0
    assert exp(0.0) == 1.0

    assert iabs(Interval(-2.0, 3.0)) == Interval(0.0, 3.0)
    assert exp(Interval(0.0, 1.0)) == Interval(1.0, math.e)

    assert sqrt(Interval(4.0, 9.0)) == Interval(2.0, 3.0)
    assert log(Interval(1.0, math.e)) == Interval(0.0, 1.0)


def test_sqrt_and_log_raise_on_invalid_domains():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-4.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-2.0, 0.0))


def test_split_bisect_constrain_and_shrink():
    i = Interval(0.0, 10.0)

    parts = split(i, [1.0, 2.0, 1.0])
    assert parts == [Interval(0.0, 2.5), Interval(2.5, 7.5), Interval(7.5, 10.0)]

    b = bisect(i)
    assert b == [Interval(0.0, 5.0), Interval(5.0, 10.0)]

    constrained = constrain(Interval(-5.0, 12.0), 0.0, 10.0)
    assert constrained == Interval(0.0, 10.0)
    assert constrain(3.5, 0.0, 10.0) == 3.5

    shrunk = shrink(i, 0.5)
    assert shrunk == Interval(2.5, 7.5)
    assert shrink(2.0, 0.1) == 2.0


def test_hausdorff_distance_for_intervals_and_numbers():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == 2.0
    assert hausdorff_distance(Interval(0.0, 2.0), Interval(1.0, 3.0)) == 1.0
    assert hausdorff_distance(1.0, Interval(1.0, 1.0)) == 0.0
    assert hausdorff_distance(2.0, 5.0) == 3.0
