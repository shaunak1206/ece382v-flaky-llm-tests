import math
from copy import copy

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
import intervallum.interval_functions as F


def _bounds(i):
    return i.lb, i.ub


# ---------------------------------------------------------------- Interval

def test_interval_construction_and_bounds_validation():
    i = Interval(1.0, 2.0)
    assert (i.lb, i.ub) == (1.0, 2.0)
    assert i.middle == pytest.approx(1.5)
    assert i.width == pytest.approx(1.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    fixed = Interval(2.0, 1.0, fix=True)
    assert _bounds(fixed) == (1.0, 2.0)
    p = Interval.from_point(3.0)
    assert _bounds(p) == (3.0, 3.0)


def test_interval_setters_validate():
    i = Interval(1.0, 2.0)
    i.lb = 0.5
    i.ub = 4.0
    assert _bounds(i) == (0.5, 4.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 10.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0
    assert _bounds(i) == (0.5, 4.0)


def test_interval_str_repr_and_copy():
    i = Interval(1.0, 2.0)
    assert str(i) == "[1.0; 2.0]"
    assert repr(i) == "[1.0; 2.0]"
    c = copy(i)
    c.ub = 10.0
    assert i.ub == 2.0
    assert c.ub == 10.0


def test_interval_add_sub_neg():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 5.0)
    s = a + b
    assert _bounds(s) == (pytest.approx(4.0), pytest.approx(7.0))
    s2 = a + 1.0
    assert _bounds(s2) == (pytest.approx(2.0), pytest.approx(3.0))
    d = a - Interval(0.0, 1.0)
    assert _bounds(d) == (pytest.approx(0.0), pytest.approx(2.0))
    r = 5.0 - a
    assert _bounds(r) == (pytest.approx(3.0), pytest.approx(4.0))
    n = -a
    assert _bounds(n) == (pytest.approx(-2.0), pytest.approx(-1.0))


def test_interval_multiplication_with_signs():
    a = Interval(-2.0, 3.0)
    b = Interval(-1.0, 4.0)
    m = a * b
    assert _bounds(m) == (pytest.approx(-8.0), pytest.approx(12.0))
    m2 = a * -2.0
    assert _bounds(m2) == (pytest.approx(-6.0), pytest.approx(4.0))
    m3 = 2.0 * Interval(1.0, 3.0)
    assert _bounds(m3) == (pytest.approx(2.0), pytest.approx(6.0))


def test_interval_division_and_inversion():
    q = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert _bounds(q) == (pytest.approx(0.25), pytest.approx(1.0))
    q2 = Interval(2.0, 4.0) / 2.0
    assert _bounds(q2) == (pytest.approx(1.0), pytest.approx(2.0))
    r = 1.0 / Interval(2.0, 4.0)
    assert _bounds(r) == (pytest.approx(0.25), pytest.approx(0.5))
    inv_zero = ~Interval(-1.0, 1.0)
    assert _bounds(inv_zero) == (-math.inf, math.inf)
    inv_lb0 = ~Interval(0.0, 2.0)
    assert inv_lb0.lb == pytest.approx(0.5)
    assert inv_lb0.ub == math.inf


def test_interval_power_even_and_odd():
    i = Interval(-2.0, 3.0)
    sq = i ** 2
    assert _bounds(sq) == (pytest.approx(0.0), pytest.approx(9.0))
    cube = i ** 3
    assert _bounds(cube) == (pytest.approx(-8.0), pytest.approx(27.0))
    pos = Interval(1.0, 2.0) ** 2
    assert _bounds(pos) == (pytest.approx(1.0), pytest.approx(4.0))


def test_interval_narrow_result_reduced_to_number_by_default():
    res = Interval(1.0, 1.0 + 1e-6) + 0.0
    assert isinstance(res, float)
    assert res == pytest.approx(1.0 + 5e-7)


def test_interval_reduction_can_be_disabled(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    res = Interval(1.0, 1.0 + 1e-6) + 0.0
    assert isinstance(res, Interval)
    assert res.lb == pytest.approx(1.0)


def test_interval_inner_subtraction():
    r = Interval(1.0, 5.0) << Interval(0.0, 2.0)
    assert _bounds(r) == (pytest.approx(1.0), pytest.approx(3.0))
    swapped = Interval.inner_subtraction(Interval(0.0, 1.0), Interval(0.0, 3.0))
    assert _bounds(swapped) == (pytest.approx(-2.0), pytest.approx(0.0))


def test_interval_equality_and_ordering():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 2.5)
    assert Interval(1.0, 1.0) == 1.0
    assert Interval(1.0, 2.0) != 5.0
    with pytest.raises(NotImplementedError):
        Interval(1.0, 2.0) == "a"
    assert Interval(1.0, 2.0) < Interval(3.0, 4.0)
    assert Interval(1.0, 2.0) <= Interval(1.0, 0.5 + 5.0)
    assert Interval(3.0, 4.0) > 2.0
    assert Interval(3.0, 4.0) >= 3.0
    assert not (Interval(3.0, 4.0) < 3.0)


# ------------------------------------------------------------------- Box

def test_box_basic_properties():
    b = Box(Interval(0.0, 1.0), 2.0, Interval(-1.0, 3.0))
    assert np.allclose(b.lb, [0.0, 2.0, -1.0])
    assert np.allclose(b.ub, [1.0, 2.0, 3.0])
    assert np.allclose(b.middle, [0.5, 2.0, 1.0])
    assert len(b) == 3


def test_box_width_returns_indices_of_widest_components():
    b = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), Interval(1.0, 4.0))
    idx, w = b.width
    assert idx == [1, 2]
    assert w == pytest.approx(3.0)
    idx2, w2 = Box(Interval(0.0, 1.0), Interval(0.0, 3.0)).width
    assert idx2 == [1]
    assert w2 == pytest.approx(3.0)


def test_box_str_repr_equality():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    assert str(b) == "[0.0; 1.0] x [2.0; 3.0]"
    assert repr(b) == "[0.0; 1.0] x [2.0; 3.0]"
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == [1.0, 2.0]


def test_box_copy_is_deep_for_intervals():
    b = Box(Interval(0.0, 1.0), 2.0)
    c = copy(b)
    assert isinstance(c, Box)
    c[0].ub = 5.0
    assert b[0].ub == 1.0
    assert c[0].ub == 5.0


def test_box_try_to_reduce():
    degenerate = Box(Interval(1.0, 1.0), Interval(2.0, 2.0), 3.0)
    reduced = degenerate._try_to_reduce()
    assert np.allclose(reduced, [1.0, 2.0, 3.0])
    wide = Box(Interval(0.0, 1.0), 2.0)
    kept = wide._try_to_reduce()
    assert isinstance(kept, Box)
    assert kept[0].lb == 0.0 and kept[0].ub == 1.0


# ----------------------------------------------- inclusion function (RS)

def test_rs_subinclusion_function_bounds_within_true_range():
    np.random.seed(12345)
    box = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    f = RS_SubinclusionFunction(lambda x: x[0] + x[1], 50)
    res = f(box)
    assert isinstance(res, Interval)
    assert 2.0 <= res.lb <= res.ub <= 4.0


def test_rs_subinclusion_function_constant_and_single_sample():
    np.random.seed(7)
    box = Box(Interval(-1.0, 1.0), Interval(0.0, 2.0))
    const = RS_SubinclusionFunction(lambda x: 5.0, 10)(box)
    assert const.lb == pytest.approx(5.0)
    assert const.ub == pytest.approx(5.0)
    single = RS_SubinclusionFunction(lambda x: float(x[0]), 1)(box)
    assert single.lb == single.ub
    assert -1.0 <= single.lb <= 1.0


def test_rs_subinclusion_function_reproducible_with_seed():
    np.random.seed(2024)
    box = Box(Interval(0.0, 1.0), Interval(0.0, 1.0))
    f = RS_SubinclusionFunction(lambda x: float(x[0] * x[1]), 20)
    first = f(box)
    np.random.seed(2024)
    second = f(box)
    assert first.lb == pytest.approx(second.lb)
    assert first.ub == pytest.approx(second.ub)


# ------------------------------------------------- interval_functions

def test_elementary_functions_on_intervals():
    s = F.sin(Interval(0.0, math.pi / 2))
    assert _bounds(s) == (pytest.approx(0.0, abs=1e-9), pytest.approx(1.0))
    c = F.cos(Interval(0.0, math.pi))
    assert c.lb == pytest.approx(-1.0)
    assert c.ub == pytest.approx(1.0)
    wide = F.sin(Interval(0.0, 7.0))
    assert _bounds(wide) == (pytest.approx(-1.0), pytest.approx(1.0))
    e = F.exp(Interval(0.0, 1.0))
    assert _bounds(e) == (pytest.approx(1.0), pytest.approx(math.e))
    a = F.abs(Interval(-3.0, 2.0))
    assert _bounds(a) == (pytest.approx(0.0), pytest.approx(3.0))
    r = F.sqrt(Interval(-1.0, 4.0))
    assert _bounds(r) == (pytest.approx(0.0), pytest.approx(2.0))
    lg = F.log(Interval(1.0, math.e))
    assert _bounds(lg) == (pytest.approx(0.0), pytest.approx(1.0))


def test_functions_on_floats_and_undefined_domains():
    assert F.exp(0.0) == pytest.approx(1.0)
    assert F.sin(0.5) == pytest.approx(math.sin(0.5))
    assert F.sqrt(9.0) == pytest.approx(3.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        F.sqrt(Interval(-2.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        F.log(Interval(-1.0, 0.0))
    lg = F.log(Interval(0.0, 1.0))
    assert lg.lb == -math.inf
    assert lg.ub == pytest.approx(0.0)


def test_split_bisect_constrain_shrink():
    left, right = F.bisect(Interval(1.0, 5.0))
    assert _bounds(left) == (pytest.approx(1.0), pytest.approx(3.0))
    assert _bounds(right) == (pytest.approx(3.0), pytest.approx(5.0))
    p1, p2 = F.split(Interval(0.0, 4.0), [1.0, 3.0])
    assert _bounds(p1) == (pytest.approx(0.0), pytest.approx(1.0))
    assert _bounds(p2) == (pytest.approx(1.0), pytest.approx(4.0))
    assert F.bisect(2.0) == [2.0, 2.0]

    c = F.constrain(Interval(-5.0, 5.0), 0.0, 3.0)
    assert _bounds(c) == (0.0, 3.0)
    assert F.constrain(7.0, 0.0, 3.0) == 3.0
    assert F.constrain(-7.0, 0.0, 3.0) == 0.0
    assert F.constrain(1.5, 0.0, 3.0) == 1.5

    sh = F.shrink(Interval(0.0, 4.0), 0.5)
    assert _bounds(sh) == (pytest.approx(1.0), pytest.approx(3.0))
    assert F.shrink(2.0, 0.5) == 2.0


def test_hausdorff_distance():
    i1 = Interval(0.0, 1.0)
    i2 = Interval(0.0, 3.0)
    assert F.hausdorff_distance(i1, i2) == pytest.approx(2.0)
    assert F.hausdorff_distance(i2, i1) == pytest.approx(2.0)
    assert F.hausdorff_distance(i1, i1) == pytest.approx(0.0)
    assert F.hausdorff_distance(1.0, 4.0) == pytest.approx(3.0)
    assert F.hausdorff_distance(Interval(0.0, 2.0), 5.0) == pytest.approx(5.0)
    assert F.hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 4.0)) == pytest.approx(3.0)
