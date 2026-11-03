import math
from unittest.mock import patch

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


def test_interval_from_point_and_str():
    i = Interval.from_point(3.5)
    assert i.lb == 3.5
    assert i.ub == 3.5
    assert str(i) == "[3.5; 3.5]"


def test_interval_wrong_bounds_raises():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)


def test_interval_fix_swaps_bounds():
    i = Interval(2.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 2.0


def test_interval_equality_within_admissible_error():
    IntervalConstants._admissible_error = 1e-7
    assert Interval(1.0, 1.0) == Interval(1.0 + 5e-8, 1.0 + 5e-8)
    assert not (Interval(1.0, 1.0) == Interval(1.0 + 1e-4, 1.0 + 1e-4))


def test_interval_comparison_operators():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(3.0, 4.0)
    assert i1 < i2
    assert i1 <= i2
    assert i2 > i1
    assert i2 >= i1
    assert i1 < 2.0
    assert i2 >= 3.0


def test_interval_add_sub_mul_div_with_intervals():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(3.0, 5.0)
    assert i1 + i2 == Interval(4.0, 7.0)
    assert i2 - i1 == Interval(1.0, 4.0)
    assert i1 * i2 == Interval(3.0, 10.0)
    assert i2 / i1 == Interval(1.5, 5.0)


def test_interval_neg_and_inverse():
    i = Interval(2.0, 4.0)
    assert -i == Interval(-4.0, -2.0)
    inv = ~i
    assert isinstance(inv, Interval)
    assert inv == Interval(0.25, 0.5)


def test_interval_power_even_and_odd():
    i = Interval(-2.0, 3.0)
    even = i ** 2
    odd = i ** 3
    assert even == Interval(0.0, 9.0)
    assert odd == Interval(-8.0, 27.0)


def test_interval_copy_and_reduction_to_number():
    IntervalConstants._reduction_width = 0.5
    i = Interval(1.0, 1.1)
    reduced = i._try_to_reduce()
    assert isinstance(reduced, float)
    assert reduced == pytest.approx(1.05)

    copied = i.__copy__()
    assert isinstance(copied, Interval)
    assert copied == i
    assert copied is not i


def test_box_basic_properties_and_string():
    b = Box(Interval(1.0, 2.0), Interval(-1.0, 3.0))
    assert str(b) == "[1.0; 2.0] x [-1.0; 3.0]"
    assert repr(b) == "[1.0; 2.0] x [-1.0; 3.0]"
    assert np.allclose(b.lb, np.array([1.0, -1.0]))
    assert np.allclose(b.ub, np.array([2.0, 3.0]))
    assert np.allclose(b.middle, np.array([1.5, 1.0]))


def test_box_width_and_try_to_reduce():
    b = Box(Interval(1.0, 2.0), Interval(5.0, 5.0))
    idxs, width = b.width
    assert idxs == [0]
    assert width == 1.0
    reduced = b._try_to_reduce()
    assert isinstance(reduced, Box)
    assert reduced == b

    p = Box(Interval(2.0, 2.0), Interval(3.0, 3.0))
    reduced_p = p._try_to_reduce()
    assert isinstance(reduced_p, np.ndarray)
    assert np.allclose(reduced_p, np.array([2.0, 3.0]))


def test_box_copy_and_equality():
    b1 = Box(Interval(1.0, 2.0), Interval(3.0, 4.0))
    b2 = b1.__copy__()
    assert b2 == b1
    assert b2 is not b1
    assert isinstance(b2, Box)


def test_random_subinclusion_function_uses_samples_and_bounds():
    f = RS_SubinclusionFunction(lambda x: float(np.sum(x)), number_of_samples=5)
    box = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))

    with patch("numpy.random.uniform", side_effect=[
        np.array([0.0, 2.0]),
        np.array([1.0, 3.0]),
        np.array([0.5, 4.0]),
        np.array([0.25, 2.5]),
        np.array([0.75, 3.5]),
    ]) as mocked_uniform:
        result = f(box)

    assert mocked_uniform.call_count == 5
    assert isinstance(result, Interval)
    assert result.lb == pytest.approx(2.0)
    assert result.ub == pytest.approx(5.0)


def test_trigonometric_functions_on_intervals_and_floats():
    assert sin(0.0) == 0.0
    assert cos(0.0) == 1.0

    s = sin(Interval(0.0, math.pi / 2))
    c = cos(Interval(0.0, math.pi))
    assert s == Interval(0.0, 1.0)
    assert c == Interval(-1.0, 1.0)


def test_exp_abs_sqrt_log_functions():
    assert exp(0.0) == 1.0
    assert iabs(Interval(-2.0, 3.0)) == Interval(0.0, 3.0)
    assert sqrt(Interval(4.0, 9.0)) == Interval(2.0, 3.0)
    assert log(Interval(1.0, math.e)) == Interval(0.0, 1.0)


def test_sqrt_and_log_invalid_domains_raise():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-4.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-2.0, 0.0))


def test_split_bisect_constrain_shrink_and_hausdorff_distance():
    i = Interval(0.0, 10.0)
    parts = split(i, [1.0, 2.0, 1.0])
    assert parts == [Interval(0.0, 2.5), Interval(2.5, 7.5), Interval(7.5, 10.0)]
    assert bisect(i) == [Interval(0.0, 5.0), Interval(5.0, 10.0)]

    assert constrain(Interval(-5.0, 5.0), 0.0, 3.0) == Interval(0.0, 3.0)
    assert shrink(Interval(0.0, 10.0), 0.5) == Interval(2.5, 7.5)

    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == 2.0
    assert hausdorff_distance(Interval(0.0, 3.0), Interval(1.0, 2.0)) == 1.0
