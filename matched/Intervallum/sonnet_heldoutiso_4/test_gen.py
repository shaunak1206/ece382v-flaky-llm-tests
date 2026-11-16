import copy

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import hausdorff_distance


# ---------------------------------------------------------------- Interval

def test_interval_basic_properties():
    i = Interval(1.0, 4.0)
    assert i.lb == 1.0
    assert i.ub == 4.0
    assert i.middle == pytest.approx(2.5)
    assert i.width == pytest.approx(3.0)
    assert str(i) == "[1.0; 4.0]"
    assert repr(i) == str(i)


def test_interval_wrong_bounds_raises_and_fix_swaps():
    with pytest.raises(IntervalExceptions.WrongBoundsException) as exc:
        Interval(2.0, 1.0)
    assert "Improper interval" in str(exc.value)

    fixed = Interval(2.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 2.0


def test_interval_from_point_and_copy():
    p = Interval.from_point(3.0)
    assert p.lb == 3.0 and p.ub == 3.0
    assert p.width == 0.0

    i = Interval(1.0, 2.0)
    c = copy.copy(i)
    assert c is not i
    c.lb = 0.0
    assert i.lb == 1.0
    assert c.lb == 0.0


def test_interval_bound_setters_validate():
    i = Interval(1.0, 2.0)
    i.lb = 1.5
    i.ub = 3.0
    assert (i.lb, i.ub) == (1.5, 3.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 10.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0
    assert (i.lb, i.ub) == (1.5, 3.0)


def test_interval_addition_and_subtraction():
    s = Interval(1.0, 2.0) + Interval(3.0, 5.0)
    assert (s.lb, s.ub) == (4.0, 7.0)

    s = Interval(1.0, 2.0) + 1.0
    assert (s.lb, s.ub) == (2.0, 3.0)

    s = 1.0 + Interval(1.0, 2.0)
    assert (s.lb, s.ub) == (2.0, 3.0)

    d = Interval(1.0, 2.0) - Interval(0.0, 1.0)
    assert (d.lb, d.ub) == (0.0, 2.0)

    d = 5.0 - Interval(1.0, 2.0)
    assert (d.lb, d.ub) == (3.0, 4.0)

    n = -Interval(1.0, 2.0)
    assert (n.lb, n.ub) == (-2.0, -1.0)


def test_interval_multiplication():
    m = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (m.lb, m.ub) == (-4.0, 8.0)

    m = 2.0 * Interval(1.0, 3.0)
    assert (m.lb, m.ub) == (2.0, 6.0)

    m = Interval(1.0, 3.0) * -1.0
    assert (m.lb, m.ub) == (-3.0, -1.0)


def test_interval_division():
    q = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert q.lb == pytest.approx(0.25)
    assert q.ub == pytest.approx(1.0)

    q = Interval(2.0, 4.0) / 2.0
    assert (q.lb, q.ub) == (1.0, 2.0)

    q = 1.0 / Interval(2.0, 4.0)
    assert q.lb == pytest.approx(0.25)
    assert q.ub == pytest.approx(0.5)


def test_interval_inversion_with_zero():
    inv = ~Interval(0.0, 2.0)
    assert inv.lb == pytest.approx(0.5)
    assert inv.ub == float("inf")

    inv = ~Interval(-2.0, 0.0)
    assert inv.lb == float("-inf")
    assert inv.ub == pytest.approx(-0.5)

    inv = ~Interval(-1.0, 1.0)
    assert inv.lb == float("-inf")
    assert inv.ub == float("inf")


def test_interval_power_even_and_odd():
    p = Interval(-2.0, 3.0) ** 2
    assert (p.lb, p.ub) == (0.0, 9.0)

    p = Interval(-3.0, -2.0) ** 2
    assert (p.lb, p.ub) == (4.0, 9.0)

    p = Interval(2.0, 3.0) ** 3
    assert (p.lb, p.ub) == (8.0, 27.0)

    p = Interval(-2.0, 3.0) ** 3
    assert (p.lb, p.ub) == (-8.0, 27.0)


def test_interval_results_reduced_to_numbers_when_narrow():
    r = Interval(1.0, 1.0) + Interval(1.0, 1.0)
    assert isinstance(r, float)
    assert r == pytest.approx(2.0)

    r = Interval(3.0, 3.0) ** 2
    assert isinstance(r, float)
    assert r == pytest.approx(9.0)

    wide = Interval(1.0, 2.0) + 1.0
    assert isinstance(wide, Interval)


def test_interval_reduction_can_be_disabled(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    r = Interval(1.0, 1.0) + Interval(1.0, 1.0)
    assert isinstance(r, Interval)
    assert (r.lb, r.ub) == (2.0, 2.0)


def test_interval_equality_and_inequality():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert not (Interval(1.0, 2.0) != Interval(1.0, 2.0))
    assert Interval(1.0, 2.0) != Interval(1.0, 3.0)
    assert Interval(1.0, 2.0) == Interval(1.0 + 1e-9, 2.0 - 1e-9)

    with pytest.raises(NotImplementedError):
        Interval(1.0, 2.0) == "abc"


def test_interval_ordering_uses_lower_bounds():
    a, b = Interval(1.0, 10.0), Interval(2.0, 3.0)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert a < 1.5
    assert a >= 1.0
    assert not (a > 1.0)


def test_interval_inner_subtraction():
    d = Interval(3.0, 5.0) << Interval(1.0, 2.0)
    assert (d.lb, d.ub) == (2.0, 3.0)

    d = Interval.inner_subtraction(Interval(1.0, 2.0), Interval(1.0, 2.0))
    assert isinstance(d, float)
    assert d == 0.0


# --------------------------------------------------------------------- Box

def test_box_bounds_and_middle():
    b = Box(Interval(0.0, 2.0), Interval(1.0, 5.0), 3.0)
    np.testing.assert_allclose(b.lb, [0.0, 1.0, 3.0])
    np.testing.assert_allclose(b.ub, [2.0, 5.0, 3.0])
    np.testing.assert_allclose(b.middle, [1.0, 3.0, 3.0])


def test_box_width_returns_indices_of_widest_components():
    b = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), 5.0)
    idx, w = b.width
    assert idx == [1]
    assert w == pytest.approx(3.0)

    b2 = Box(Interval(0.0, 2.0), Interval(1.0, 3.0))
    idx, w = b2.width
    assert idx == [0, 1]
    assert w == pytest.approx(2.0)


def test_box_equality():
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    assert Box(1.0, 2.0) != Box(1.0, 2.0, 3.0)
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == [1.0, 2.0]


def test_box_str_and_copy():
    b = Box(Interval(1.0, 2.0), Interval(3.0, 4.0))
    assert str(b) == "[1.0; 2.0] x [3.0; 4.0]"

    c = copy.copy(b)
    assert isinstance(c, Box)
    assert c is not b
    c[0].lb = 0.0
    assert b[0].lb == 1.0
    assert c[0].lb == 0.0


def test_box_try_to_reduce():
    degenerate = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    reduced = degenerate._try_to_reduce()
    np.testing.assert_allclose(reduced, [1.0, 2.0])

    proper = Box(Interval(1.0, 2.0), Interval(2.0, 2.0))
    result = proper._try_to_reduce()
    assert isinstance(result, Box)
    assert result is not proper
    np.testing.assert_allclose(result.lb, [1.0, 2.0])
    np.testing.assert_allclose(result.ub, [2.0, 2.0])


# ------------------------------------------------- RS_SubinclusionFunction

def test_rs_subinclusion_encloses_samples_within_true_range():
    np.random.seed(42)
    f = RS_SubinclusionFunction(lambda p: p[0] + p[1], number_of_samples=200)
    result = f(Box(Interval(0.0, 1.0), Interval(2.0, 3.0)))
    assert isinstance(result, Interval)
    assert 2.0 <= result.lb <= result.ub <= 4.0
    assert result.width > 0.5


def test_rs_subinclusion_calls_function_requested_number_of_times():
    calls = []

    def func(p):
        calls.append(p)
        assert 0.0 <= p[0] <= 1.0
        assert 5.0 <= p[1] <= 6.0
        return 7.0

    f = RS_SubinclusionFunction(func, number_of_samples=13)
    result = f(Box(Interval(0.0, 1.0), Interval(5.0, 6.0)))
    assert len(calls) == 13
    assert result.lb == 7.0
    assert result.ub == 7.0


# ------------------------------------------------------ hausdorff_distance

def test_hausdorff_distance_between_intervals():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == pytest.approx(2.0)
    assert hausdorff_distance(Interval(0.0, 10.0), Interval(4.0, 5.0)) == pytest.approx(5.0)
    assert hausdorff_distance(Interval(1.0, 2.0), Interval(1.0, 2.0)) == 0.0


def test_hausdorff_distance_with_numbers_and_symmetry():
    assert hausdorff_distance(1.0, 4.0) == pytest.approx(3.0)
    assert hausdorff_distance(4.0, 1.0) == pytest.approx(3.0)

    a, b = Interval(0.0, 2.0), 5.0
    assert hausdorff_distance(a, b) == pytest.approx(hausdorff_distance(b, a))
    assert hausdorff_distance(a, b) == pytest.approx(5.0)
