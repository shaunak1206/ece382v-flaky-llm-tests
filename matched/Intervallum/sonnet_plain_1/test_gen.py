import math
from copy import copy

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum import interval_functions as ifn
from intervallum.interval_functions import hausdorff_distance


# ---------------------------------------------------------------- Interval

def test_interval_construction_and_properties():
    i = Interval(1.0, 3.0)
    assert i.lb == 1.0
    assert i.ub == 3.0
    assert i.middle == 2.0
    assert i.width == 2.0
    assert str(i) == "[1.0; 3.0]"
    assert repr(i) == "[1.0; 3.0]"


def test_interval_wrong_bounds_and_fix():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    fixed = Interval(2.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 2.0


def test_interval_from_point_and_setters():
    p = Interval.from_point(2.5)
    assert p.lb == 2.5 and p.ub == 2.5 and p.width == 0.0

    i = Interval(0.0, 1.0)
    i.lb = -1.0
    i.ub = 4.0
    assert i.lb == -1.0 and i.ub == 4.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 5.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = -2.0
    assert i.lb == -1.0 and i.ub == 4.0


def test_interval_copy_is_independent():
    i = Interval(1.0, 2.0)
    c = copy(i)
    assert c is not i
    c.ub = 10.0
    assert i.ub == 2.0
    assert c.ub == 10.0


def test_interval_addition_and_subtraction():
    r = Interval(1.0, 2.0) + Interval(3.0, 4.0)
    assert isinstance(r, Interval)
    assert (r.lb, r.ub) == (4.0, 6.0)

    r = 3.0 + Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (4.0, 5.0)

    r = Interval(1.0, 2.0) - Interval(0.0, 1.0)
    assert (r.lb, r.ub) == (0.0, 2.0)

    r = 5.0 - Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (3.0, 4.0)

    n = -Interval(1.0, 2.0)
    assert (n.lb, n.ub) == (-2.0, -1.0)


def test_interval_multiplication():
    r = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (r.lb, r.ub) == (-4.0, 8.0)

    r = Interval(1.0, 2.0) * -1.0
    assert (r.lb, r.ub) == (-2.0, -1.0)

    r = 2.0 * Interval(1.0, 3.0)
    assert (r.lb, r.ub) == (2.0, 6.0)


def test_interval_inversion_and_division():
    r = ~Interval(2.0, 4.0)
    assert (r.lb, r.ub) == (0.25, 0.5)

    r = ~Interval(0.0, 2.0)
    assert r.lb == 0.5 and r.ub == math.inf

    r = ~Interval(-1.0, 0.0)
    assert r.lb == -math.inf and r.ub == -1.0

    r = ~Interval(-1.0, 1.0)
    assert r.lb == -math.inf and r.ub == math.inf

    r = Interval(2.0, 4.0) / Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (1.0, 4.0)

    r = Interval(2.0, 4.0) / 2.0
    assert (r.lb, r.ub) == (1.0, 2.0)

    r = 1.0 / Interval(2.0, 4.0)
    assert (r.lb, r.ub) == (0.25, 0.5)


def test_interval_power():
    r = Interval(-2.0, 3.0) ** 2
    assert (r.lb, r.ub) == (0.0, 9.0)

    r = Interval(-3.0, -1.0) ** 2
    assert (r.lb, r.ub) == (1.0, 9.0)

    r = Interval(-2.0, 3.0) ** 3
    assert (r.lb, r.ub) == (-8.0, 27.0)


def test_interval_degenerate_result_reduces_to_number():
    r = Interval(1.0, 1.0) + Interval(2.0, 2.0)
    assert not isinstance(r, Interval)
    assert r == pytest.approx(3.0)

    r = Interval(2.0, 2.0) * Interval(3.0, 3.0)
    assert not isinstance(r, Interval)
    assert r == pytest.approx(6.0)


def test_interval_reduction_can_be_disabled(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    r = Interval(1.0, 1.0) + Interval(2.0, 2.0)
    assert isinstance(r, Interval)
    assert (r.lb, r.ub) == (3.0, 3.0)


def test_interval_comparisons_use_lower_bound():
    a = Interval(1.0, 2.0)
    b = Interval(2.0, 3.0)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert a < 1.5
    assert a <= 1.0
    assert not (a > 1.0)
    assert a >= 1.0


def test_interval_equality_and_inner_subtraction():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 3.0)
    assert Interval(1.0, 2.0) != 1.5
    assert Interval(1.0, 1.0) == 1.0
    with pytest.raises(NotImplementedError):
        Interval(1.0, 2.0) == "abc"

    r = Interval(1.0, 3.0) << Interval(0.0, 1.0)
    assert (r.lb, r.ub) == (1.0, 2.0)


# --------------------------------------------------------------------- Box

def test_box_bounds_middle_and_strings():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))
    assert list(b.lb) == [0.0, 2.0]
    assert list(b.ub) == [1.0, 4.0]
    assert list(b.middle) == [0.5, 3.0]
    assert str(b) == "[0.0; 1.0] x [2.0; 4.0]"
    assert repr(b) == "[0.0; 1.0] x [2.0; 4.0]"


def test_box_width_reports_widest_components():
    b = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), Interval(5.0, 8.0))
    indices, w = b.width
    assert indices == [1, 2]
    assert w == 3.0

    mixed = Box(2.0, Interval(0.0, 2.0))
    indices, w = mixed.width
    assert indices == [1]
    assert w == 2.0

    points = Box(1.0, 2.0)
    assert points.width == ([0, 1], 0)


def test_box_equality_and_inequality():
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    assert Box(1.0, 2.0) != Box(1.0, 2.0, 3.0)
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == 3


def test_box_copy_creates_new_intervals():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    c = copy(b)
    assert isinstance(c, Box)
    assert c[0] is not b[0]
    c[0].ub = 5.0
    assert b[0].ub == 1.0
    assert list(c.ub) == [5.0, 3.0]


def test_box_try_to_reduce():
    degenerate = Box(1.0, 2.0)
    reduced = degenerate._try_to_reduce()
    assert list(reduced) == [1.0, 2.0]

    wide = Box(Interval(0.0, 1.0), 2.0)
    kept = wide._try_to_reduce()
    assert isinstance(kept, Box)
    assert kept is not wide
    assert list(kept.lb) == [0.0, 2.0]
    assert list(kept.ub) == [1.0, 2.0]


# --------------------------------------------------------- interval_functions

def test_trig_functions_on_intervals():
    s = ifn.sin(Interval(0.0, math.pi))
    assert s.lb == pytest.approx(0.0, abs=1e-12)
    assert s.ub == pytest.approx(1.0)

    c = ifn.cos(Interval(0.0, math.pi))
    assert c.lb == pytest.approx(-1.0)
    assert c.ub == pytest.approx(1.0)

    wide = ifn.sin(Interval(0.0, 7.0))
    assert (wide.lb, wide.ub) == (-1.0, 1.0)

    assert ifn.sin(math.pi / 2) == pytest.approx(1.0)
    assert ifn.cos(0.0) == pytest.approx(1.0)


def test_abs_exp_sqrt_log():
    a = ifn.abs(Interval(-3.0, 2.0))
    assert (a.lb, a.ub) == (0.0, 3.0)
    a = ifn.abs(Interval(2.0, 5.0))
    assert (a.lb, a.ub) == (2.0, 5.0)
    assert ifn.abs(-2.5) == 2.5

    e = ifn.exp(Interval(0.0, 1.0))
    assert e.lb == pytest.approx(1.0)
    assert e.ub == pytest.approx(math.e)

    s = ifn.sqrt(Interval(4.0, 9.0))
    assert (s.lb, s.ub) == (2.0, 3.0)
    s = ifn.sqrt(Interval(-1.0, 4.0))
    assert (s.lb, s.ub) == (0.0, 2.0)

    lg = ifn.log(Interval(1.0, math.e))
    assert lg.lb == pytest.approx(0.0, abs=1e-12)
    assert lg.ub == pytest.approx(1.0)
    lg = ifn.log(Interval(-1.0, math.e))
    assert lg.lb == -math.inf
    assert lg.ub == pytest.approx(1.0)


def test_sqrt_and_log_undefined_domains_raise():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.sqrt(Interval(-4.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.log(Interval(-2.0, 0.0))


def test_split_bisect_constrain_shrink():
    parts = ifn.bisect(Interval(0.0, 4.0))
    assert len(parts) == 2
    assert (parts[0].lb, parts[0].ub) == (0.0, 2.0)
    assert (parts[1].lb, parts[1].ub) == (2.0, 4.0)

    parts = ifn.split(Interval(0.0, 6.0), [1.0, 2.0, 3.0])
    assert [(p.lb, p.ub) for p in parts] == [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0)]

    assert ifn.split(3.0, [1.0, 2.0, 3.0]) == [3.0, 3.0, 3.0]

    c = ifn.constrain(Interval(-5.0, 5.0), -1.0, 1.0)
    assert (c.lb, c.ub) == (-1.0, 1.0)
    c = ifn.constrain(Interval(0.0, 3.0), 1.0, 2.0)
    assert (c.lb, c.ub) == (1.0, 2.0)
    assert ifn.constrain(10.0, 0.0, 5.0) == 5.0
    assert ifn.constrain(-3.0, 0.0, 5.0) == 0.0
    assert ifn.constrain(2.0, 0.0, 5.0) == 2.0

    sh = ifn.shrink(Interval(0.0, 4.0), 0.5)
    assert (sh.lb, sh.ub) == (1.0, 3.0)
    assert ifn.shrink(2.0, 0.5) == 2.0


def test_hausdorff_distance():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 3.0)) == 2.0
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 5.0)) == 4.0
    assert hausdorff_distance(Interval(2.0, 5.0), Interval(0.0, 1.0)) == 4.0
    assert hausdorff_distance(Interval(1.0, 2.0), Interval(1.0, 2.0)) == 0.0
    assert hausdorff_distance(1.0, 4.0) == 3.0
    assert hausdorff_distance(Interval(0.0, 2.0), 1.0) == 1.0


# ------------------------------------------------------ RS_SubinclusionFunction

def test_rs_subinclusion_samples_stay_inside_range():
    np.random.seed(0)
    f = RS_SubinclusionFunction(lambda p: p[0] + p[1], 200)
    box = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    result = f(box)
    assert isinstance(result, Interval)
    assert result.lb <= result.ub
    assert result.lb >= 2.0
    assert result.ub <= 4.0


def test_rs_subinclusion_constant_function_and_point_box():
    f = RS_SubinclusionFunction(lambda p: 5.0, 10)
    result = f(Box(Interval(0.0, 1.0), Interval(0.0, 1.0)))
    assert isinstance(result, Interval)
    assert result.lb == 5.0 and result.ub == 5.0

    g = RS_SubinclusionFunction(lambda p: p[0] * 2.0 + p[1], 5)
    point_result = g(Box(1.0, 3.0))
    assert point_result.lb == pytest.approx(5.0)
    assert point_result.ub == pytest.approx(5.0)
