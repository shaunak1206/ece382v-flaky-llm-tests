import math
from copy import copy

import numpy as np
import pytest

from intervallum import interval_functions as ifn
from intervallum.box import Box
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.interval_functions import hausdorff_distance


@pytest.fixture(autouse=True)
def default_constants(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", True)
    monkeypatch.setattr(IntervalConstants, "_reduction_width", 1e-5)
    monkeypatch.setattr(IntervalConstants, "_admissible_error", 1e-7)


# ---------------------------------------------------------------- Interval

def test_interval_construction_and_properties():
    i = Interval(1.0, 4.0)
    assert i.lb == 1.0
    assert i.ub == 4.0
    assert i.middle == pytest.approx(2.5)
    assert i.width == pytest.approx(3.0)


def test_interval_wrong_bounds_and_fix():
    with pytest.raises(IntervalExceptions.WrongBoundsException) as exc:
        Interval(2.0, 1.0)
    assert "Improper interval [2.0; 1.0]" in str(exc.value)

    fixed = Interval(2.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 2.0


def test_interval_bound_setters():
    i = Interval(1.0, 3.0)
    i.lb = 2.0
    i.ub = 5.0
    assert (i.lb, i.ub) == (2.0, 5.0)

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 6.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 1.0
    assert (i.lb, i.ub) == (2.0, 5.0)


def test_interval_from_point_and_string_forms():
    p = Interval.from_point(2.5)
    assert p.lb == 2.5 and p.ub == 2.5
    assert p.width == 0.0

    i = Interval(1.0, 2.0)
    assert str(i) == "[1.0; 2.0]"
    assert repr(i) == "[1.0; 2.0]"


def test_interval_copy_is_independent():
    original = Interval(1.0, 2.0)
    duplicate = copy(original)
    assert duplicate is not original
    assert (duplicate.lb, duplicate.ub) == (1.0, 2.0)
    duplicate.ub = 10.0
    assert original.ub == 2.0


def test_interval_addition_subtraction_and_reflected_ops():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 5.0)

    s = a + b
    assert (s.lb, s.ub) == (4.0, 7.0)

    s = a + 1.5
    assert (s.lb, s.ub) == (2.5, 3.5)

    s = 1.5 + a
    assert (s.lb, s.ub) == (2.5, 3.5)

    n = -a
    assert (n.lb, n.ub) == (-2.0, -1.0)

    d = a - Interval(0.0, 1.0)
    assert (d.lb, d.ub) == (0.0, 2.0)

    d = a - 0.5
    assert (d.lb, d.ub) == (0.5, 1.5)

    d = 5.0 - a
    assert (d.lb, d.ub) == (3.0, 4.0)

    d = Interval(5.0, 10.0) << Interval(1.0, 3.0)
    assert (d.lb, d.ub) == (4.0, 7.0)


def test_interval_multiplication():
    r = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (r.lb, r.ub) == (-4.0, 8.0)

    r = Interval(1.0, 2.0) * -2.0
    assert (r.lb, r.ub) == (-4.0, -2.0)

    r = 3.0 * Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (3.0, 6.0)


def test_interval_inversion_and_division():
    inv = ~Interval(2.0, 4.0)
    assert inv.lb == pytest.approx(0.25)
    assert inv.ub == pytest.approx(0.5)

    inv = ~Interval(0.0, 2.0)
    assert inv.lb == pytest.approx(0.5)
    assert inv.ub == math.inf

    inv = ~Interval(-2.0, 0.0)
    assert inv.lb == -math.inf
    assert inv.ub == pytest.approx(-0.5)

    inv = ~Interval(-1.0, 1.0)
    assert inv.lb == -math.inf and inv.ub == math.inf

    q = Interval(2.0, 4.0) / Interval(1.0, 2.0)
    assert (q.lb, q.ub) == (pytest.approx(1.0), pytest.approx(4.0))

    q = Interval(2.0, 4.0) / 2.0
    assert (q.lb, q.ub) == (pytest.approx(1.0), pytest.approx(2.0))

    q = 1.0 / Interval(2.0, 4.0)
    assert (q.lb, q.ub) == (pytest.approx(0.25), pytest.approx(0.5))


def test_interval_power_even_and_odd():
    r = Interval(-2.0, 3.0) ** 2
    assert (r.lb, r.ub) == (0.0, 9.0)

    r = Interval(1.0, 2.0) ** 2
    assert (r.lb, r.ub) == (1.0, 4.0)

    r = Interval(-2.0, 3.0) ** 3
    assert (r.lb, r.ub) == (-8.0, 27.0)


def test_interval_ordering_compares_lower_bounds():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 4.0)
    assert a < b
    assert a <= Interval(1.0, 3.0)
    assert b > a
    assert not (a >= b)
    assert a < 1.5
    assert a > 0.5
    assert a >= 1.0
    assert a <= 1.0


def test_interval_equality_uses_tolerance():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) == Interval(1.0 + 1e-9, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 2.5)
    assert Interval(1.0, 1.0) == 1.0
    assert Interval(1.0, 2.0) != 1.5
    with pytest.raises(NotImplementedError):
        Interval(1.0, 2.0) == "text"


def test_interval_reduction_to_number_toggle(monkeypatch):
    result = Interval(1.0, 1.0) + Interval(2.0, 2.0)
    assert not isinstance(result, Interval)
    assert result == pytest.approx(3.0)

    narrow = Interval(0.0, 1e-6) + 1.0
    assert not isinstance(narrow, Interval)
    assert narrow == pytest.approx(1.0 + 0.5e-6)

    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    result = Interval(1.0, 1.0) + Interval(2.0, 2.0)
    assert isinstance(result, Interval)
    assert (result.lb, result.ub) == (3.0, 3.0)


# --------------------------------------------------------------------- Box

def test_box_bounds_middle_and_width():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 5.0), 7.0)
    np.testing.assert_allclose(b.lb, [0.0, 2.0, 7.0])
    np.testing.assert_allclose(b.ub, [1.0, 5.0, 7.0])
    np.testing.assert_allclose(b.middle, [0.5, 3.5, 7.0])

    indices, max_width = b.width
    assert indices == [1]
    assert max_width == pytest.approx(3.0)

    tie = Box(Interval(0.0, 2.0), Interval(1.0, 3.0), 4.0)
    indices, max_width = tie.width
    assert indices == [0, 1]
    assert max_width == pytest.approx(2.0)


def test_box_string_equality_and_copy():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))
    assert str(b) == "[0.0; 1.0] x [2.0; 4.0]"
    assert repr(b) == "[0.0; 1.0] x [2.0; 4.0]"

    same = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))
    other = Box(Interval(0.0, 1.0), Interval(2.0, 5.0))
    assert b == same
    assert b != other

    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == 5

    duplicate = copy(b)
    assert isinstance(duplicate, Box)
    assert duplicate is not b
    duplicate[0].ub = 9.0
    assert b[0].ub == 1.0


def test_box_try_to_reduce():
    degenerate = Box(Interval(1.0, 1.0), 2.0)
    reduced = degenerate._try_to_reduce()
    np.testing.assert_allclose(np.asarray(reduced, dtype=float), [1.0, 2.0])

    wide = Box(Interval(0.0, 1.0), 2.0)
    kept = wide._try_to_reduce()
    assert isinstance(kept, Box)
    assert kept is not wide
    np.testing.assert_allclose(kept.lb, [0.0, 2.0])
    np.testing.assert_allclose(kept.ub, [1.0, 2.0])


# ------------------------------------------------- RS_SubinclusionFunction

def test_rs_subinclusion_degenerate_box_is_exact():
    f = RS_SubinclusionFunction(lambda p: float(p[0] + p[1]), 5)
    result = f(Box(Interval(1.0, 1.0), Interval(2.0, 2.0)))
    assert isinstance(result, Interval)
    assert result.lb == pytest.approx(3.0)
    assert result.ub == pytest.approx(3.0)


def test_rs_subinclusion_samples_stay_inside_range_and_count_calls():
    np.random.seed(1234)
    calls = []

    def func(p):
        calls.append(p)
        return float(p[0] + p[1])

    f = RS_SubinclusionFunction(func, 50)
    result = f(Box(Interval(0.0, 1.0), Interval(2.0, 3.0)))

    assert len(calls) == 50
    for p in calls:
        assert 0.0 <= p[0] <= 1.0
        assert 2.0 <= p[1] <= 3.0
    assert 2.0 <= result.lb <= result.ub <= 4.0
    assert result.lb == pytest.approx(min(float(p[0] + p[1]) for p in calls))
    assert result.ub == pytest.approx(max(float(p[0] + p[1]) for p in calls))


# --------------------------------------------------------- hausdorff / misc

def test_hausdorff_distance_intervals_and_numbers():
    a = Interval(0.0, 1.0)
    assert hausdorff_distance(a, Interval(0.0, 1.0)) == 0.0
    assert hausdorff_distance(a, Interval(0.0, 3.0)) == pytest.approx(2.0)
    assert hausdorff_distance(Interval(0.0, 3.0), a) == pytest.approx(2.0)
    assert hausdorff_distance(a, Interval(2.0, 3.0)) == pytest.approx(2.0)
    assert hausdorff_distance(1.0, 4.0) == pytest.approx(3.0)
    assert hausdorff_distance(0.5, a) == pytest.approx(0.5)
    assert hausdorff_distance(a, 0.5) == pytest.approx(0.5)


def test_split_bisect_and_numbers():
    halves = ifn.bisect(Interval(2.0, 6.0))
    assert len(halves) == 2
    assert (halves[0].lb, halves[0].ub) == (pytest.approx(2.0), pytest.approx(4.0))
    assert (halves[1].lb, halves[1].ub) == (pytest.approx(4.0), pytest.approx(6.0))

    parts = ifn.split(Interval(0.0, 4.0), [1.0, 3.0])
    assert (parts[0].lb, parts[0].ub) == (pytest.approx(0.0), pytest.approx(1.0))
    assert (parts[1].lb, parts[1].ub) == (pytest.approx(1.0), pytest.approx(4.0))

    assert ifn.split(2.0, [1.0, 1.0, 1.0]) == [2.0, 2.0, 2.0]
    assert ifn.bisect(5.0) == [5.0, 5.0]


def test_constrain_and_shrink():
    c = ifn.constrain(Interval(-5.0, 5.0), -1.0, 1.0)
    assert (c.lb, c.ub) == (-1.0, 1.0)
    c = ifn.constrain(Interval(0.0, 5.0), -1.0, 1.0)
    assert (c.lb, c.ub) == (0.0, 1.0)
    assert ifn.constrain(3.0, 0.0, 2.0) == 2.0
    assert ifn.constrain(-3.0, 0.0, 2.0) == 0.0
    assert ifn.constrain(1.0, 0.0, 2.0) == 1.0

    s = ifn.shrink(Interval(0.0, 10.0), 0.5)
    assert (s.lb, s.ub) == (pytest.approx(2.5), pytest.approx(7.5))
    assert ifn.shrink(3.0, 0.5) == 3.0


def test_sin_and_cos_ranges():
    s = ifn.sin(Interval(0.0, math.pi))
    assert s.lb == pytest.approx(0.0, abs=1e-12)
    assert s.ub == pytest.approx(1.0)

    s = ifn.sin(Interval(0.1, 0.2))
    assert s.lb == pytest.approx(math.sin(0.1))
    assert s.ub == pytest.approx(math.sin(0.2))

    wide = ifn.sin(Interval(0.0, 7.0))
    assert (wide.lb, wide.ub) == (-1.0, 1.0)

    c = ifn.cos(Interval(0.0, math.pi))
    assert c.lb == pytest.approx(-1.0)
    assert c.ub == pytest.approx(1.0)

    c = ifn.cos(Interval(-1.0, 1.0))
    assert c.lb == pytest.approx(math.cos(1.0))
    assert c.ub == pytest.approx(1.0)

    assert ifn.cos(0.0) == pytest.approx(1.0)
    assert ifn.sin(0.0) == pytest.approx(0.0, abs=1e-12)


def test_abs_and_exp():
    r = ifn.abs(Interval(-3.0, 2.0))
    assert (r.lb, r.ub) == (0.0, 3.0)
    r = ifn.abs(Interval(-3.0, -1.0))
    assert (r.lb, r.ub) == (1.0, 3.0)
    r = ifn.abs(Interval(1.0, 2.0))
    assert (r.lb, r.ub) == (1.0, 2.0)
    assert ifn.abs(-2.5) == pytest.approx(2.5)

    e = ifn.exp(Interval(0.0, 1.0))
    assert e.lb == pytest.approx(1.0)
    assert e.ub == pytest.approx(math.e)
    assert ifn.exp(0.0) == pytest.approx(1.0)


def test_sqrt_and_log_domains():
    r = ifn.sqrt(Interval(4.0, 9.0))
    assert (r.lb, r.ub) == (pytest.approx(2.0), pytest.approx(3.0))
    r = ifn.sqrt(Interval(-1.0, 4.0))
    assert (r.lb, r.ub) == (pytest.approx(0.0), pytest.approx(2.0))
    assert ifn.sqrt(9.0) == pytest.approx(3.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined) as exc:
        ifn.sqrt(Interval(-3.0, -1.0))
    assert "sqrt" in str(exc.value)

    r = ifn.log(Interval(1.0, math.e))
    assert (r.lb, r.ub) == (pytest.approx(0.0), pytest.approx(1.0))
    r = ifn.log(Interval(-1.0, math.e))
    assert r.lb == -math.inf
    assert r.ub == pytest.approx(1.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.log(Interval(-2.0, 0.0))
