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


def test_interval_construction_and_string_representation():
    i = Interval(-1.5, 2.25)
    assert i.lb == pytest.approx(-1.5)
    assert i.ub == pytest.approx(2.25)
    assert str(i) == "[-1.5; 2.25]"
    assert repr(i) == "[-1.5; 2.25]"


def test_interval_wrong_bounds_raises():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)


def test_interval_fix_swaps_bounds():
    i = Interval(2.0, 1.0, fix=True)
    assert i.lb == pytest.approx(1.0)
    assert i.ub == pytest.approx(2.0)


def test_interval_property_setters_validate_bounds():
    i = Interval(1.0, 3.0)
    i.lb = 0.5
    i.ub = 3.5
    assert i.lb == pytest.approx(0.5)
    assert i.ub == pytest.approx(3.5)

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 4.0

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0


def test_interval_equality_and_ordering():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(1.0 + 1e-8, 2.0 + 1e-8)
    assert i1 == i2
    assert not (i1 != i2)
    assert Interval(0.0, 1.0) < Interval(1.0, 2.0)
    assert Interval(0.0, 1.0) <= 1.0
    assert Interval(2.0, 3.0) > 1.0
    assert Interval(2.0, 3.0) >= 2.0


def test_interval_add_sub_mul_and_r_operations():
    i = Interval(1.0, 3.0)
    j = Interval(2.0, 4.0)

    s = i + j
    assert isinstance(s, Interval)
    assert s.lb == pytest.approx(3.0)
    assert s.ub == pytest.approx(7.0)

    d = j - i
    assert d.lb == pytest.approx(-1.0)
    assert d.ub == pytest.approx(3.0)

    p = i * j
    assert p.lb == pytest.approx(2.0)
    assert p.ub == pytest.approx(12.0)

    assert (2.0 + i) == (i + 2.0)
    assert (2.0 * i) == (i * 2.0)
    assert (10.0 - i) == (Interval(7.0, 9.0))


def test_interval_division_and_inverse_behaviors():
    i = Interval(2.0, 4.0)
    inv = ~i
    assert isinstance(inv, Interval)
    assert inv.lb == pytest.approx(0.25)
    assert inv.ub == pytest.approx(0.5)

    div = i / 2.0
    assert div.lb == pytest.approx(1.0)
    assert div.ub == pytest.approx(2.0)

    recip = 8.0 / i
    assert recip.lb == pytest.approx(2.0)
    assert recip.ub == pytest.approx(4.0)


def test_interval_power_even_and_odd():
    i = Interval(-2.0, 3.0)

    even = i ** 2
    assert isinstance(even, Interval)
    assert even.lb == pytest.approx(0.0)
    assert even.ub == pytest.approx(9.0)

    odd = i ** 3
    assert isinstance(odd, Interval)
    assert odd.lb == pytest.approx(-8.0)
    assert odd.ub == pytest.approx(27.0)


def test_interval_reduction_to_number_and_no_reduction(monkeypatch):
    tiny = Interval(1.0, 1.0 + 1e-6)

    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", True)
    reduced = tiny._try_to_reduce()
    assert isinstance(reduced, float)
    assert reduced == pytest.approx(1.0000005)

    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    result = tiny + Interval(0.0, 0.0)
    assert isinstance(result, Interval)
    assert result.lb == pytest.approx(1.0)
    assert result.ub == pytest.approx(1.000001)


def test_box_properties_and_string_representation():
    b = Box(Interval(1.0, 2.0), 3.0, Interval(-1.0, 1.0))
    assert str(b) == "[1.0; 2.0] x 3.0 x [-1.0; 1.0]"
    assert repr(b) == "[1.0; 2.0] x 3.0 x [-1.0; 1.0]"
    assert np.array_equal(b.lb, np.array([1.0, 3.0, -1.0]))
    assert np.array_equal(b.ub, np.array([2.0, 3.0, 1.0]))
    assert np.array_equal(b.middle, np.array([1.5, 3.0, 0.0]))


def test_box_width_and_try_to_reduce_for_point_box():
    b = Box(Interval(2.0, 2.0), 5.0, Interval(-1.0, -1.0))
    idxs, width = b.width
    assert idxs == [0, 1, 2]
    assert width == pytest.approx(0.0)

    reduced = b._try_to_reduce()
    assert isinstance(reduced, np.ndarray)
    assert np.array_equal(reduced, np.array([2.0, 5.0, -1.0]))


def test_box_copy_is_independent():
    b1 = Box(Interval(1.0, 2.0), 3.0)
    b2 = b1.__copy__()
    assert isinstance(b2, Box)
    assert b1 == b2

    b2[0].lb = 0.0
    assert b1[0].lb == pytest.approx(1.0)
    assert b2[0].lb == pytest.approx(0.0)


def test_rs_subinclusion_function_samples_deterministic_with_seed(monkeypatch):
    np.random.seed(12345)

    seen_points = []

    def f(x):
        arr = np.asarray(x, dtype=float)
        seen_points.append(arr.copy())
        return float(arr.sum())

    rs = RS_SubinclusionFunction(f, number_of_samples=4)
    box = Box(Interval(0.0, 1.0), Interval(10.0, 20.0))
    out = rs(box)

    assert isinstance(out, Interval)
    assert len(seen_points) == 4
    for p in seen_points:
        assert p.shape == (2,)
        assert 0.0 <= p[0] <= 1.0
        assert 10.0 <= p[1] <= 20.0

    values = [float(p.sum()) for p in seen_points]
    assert out.lb == pytest.approx(min(values))
    assert out.ub == pytest.approx(max(values))


def test_trig_functions_on_intervals_and_points():
    s = sin(Interval(0.0, math.pi / 2))
    c = cos(Interval(0.0, math.pi / 2))
    assert s.lb == pytest.approx(0.0)
    assert s.ub == pytest.approx(1.0)
    assert c.lb == pytest.approx(0.0)
    assert c.ub == pytest.approx(1.0)

    sp = sin(0.0)
    cp = cos(0.0)
    assert sp == pytest.approx(0.0)
    assert cp == pytest.approx(1.0)


def test_abs_exp_sqrt_log_on_intervals():
    a = iabs(Interval(-2.0, 1.0))
    assert a.lb == pytest.approx(0.0)
    assert a.ub == pytest.approx(2.0)

    e = exp(Interval(0.0, 1.0))
    assert e.lb == pytest.approx(1.0)
    assert e.ub == pytest.approx(math.e)

    r = sqrt(Interval(1.0, 4.0))
    assert r.lb == pytest.approx(1.0)
    assert r.ub == pytest.approx(2.0)

    l = log(Interval(1.0, math.e))
    assert l.lb == pytest.approx(0.0)
    assert l.ub == pytest.approx(1.0)


def test_sqrt_and_log_raise_for_invalid_intervals():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        sqrt(Interval(-3.0, -1.0))

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        log(Interval(-2.0, 0.0))


def test_split_bisect_constrain_and_shrink():
    i = Interval(0.0, 10.0)
    parts = split(i, [1.0, 2.0, 1.0])
    assert len(parts) == 3
    assert parts[0].lb == pytest.approx(0.0)
    assert parts[0].ub == pytest.approx(2.5)
    assert parts[1].lb == pytest.approx(2.5)
    assert parts[1].ub == pytest.approx(7.5)
    assert parts[2].lb == pytest.approx(7.5)
    assert parts[2].ub == pytest.approx(10.0)

    left, right = bisect(i)
    assert left.ub == pytest.approx(5.0)
    assert right.lb == pytest.approx(5.0)

    c = constrain(Interval(-2.0, 12.0), 0.0, 10.0)
    assert c.lb == pytest.approx(0.0)
    assert c.ub == pytest.approx(10.0)

    s = shrink(Interval(0.0, 10.0), 0.2)
    assert s.lb == pytest.approx(4.0)
    assert s.ub == pytest.approx(6.0)


def test_hausdorff_distance_matches_expected_values():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 4.0)) == pytest.approx(3.0)
    assert hausdorff_distance(Interval(1.0, 3.0), Interval(2.0, 5.0)) == pytest.approx(1.0)
    assert hausdorff_distance(1.0, Interval(1.5, 2.0)) == pytest.approx(1.0)
    assert hausdorff_distance(2.0, 2.0) == pytest.approx(0.0)
