import copy as copy_module

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
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


def test_interval_wrong_bounds_raises():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)


def test_interval_fix_swaps_bounds():
    i = Interval(5.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 5.0


def test_interval_from_point():
    i = Interval.from_point(2.5)
    assert i.lb == 2.5
    assert i.ub == 2.5
    assert i.width == 0.0


def test_interval_bound_setters():
    i = Interval(1.0, 2.0)
    i.lb = 0.0
    i.ub = 10.0
    assert (i.lb, i.ub) == (0.0, 10.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 11.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = -1.0
    assert (i.lb, i.ub) == (0.0, 10.0)


def test_interval_copy_is_independent():
    i = Interval(1.0, 2.0)
    c = copy_module.copy(i)
    c.ub = 5.0
    assert i.ub == 2.0
    assert c.ub == 5.0


def test_interval_addition_and_subtraction():
    a = Interval(1.0, 2.0)
    b = Interval(0.0, 1.0)
    s = a + b
    assert (s.lb, s.ub) == (1.0, 3.0)
    d = a - b
    assert (d.lb, d.ub) == (0.0, 2.0)
    r = 1.0 + a
    assert (r.lb, r.ub) == (2.0, 3.0)
    rs = 5.0 - a
    assert (rs.lb, rs.ub) == (3.0, 4.0)


def test_interval_negation():
    n = -Interval(1.0, 3.0)
    assert (n.lb, n.ub) == (-3.0, -1.0)


def test_interval_multiplication():
    p = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (p.lb, p.ub) == (-4.0, 8.0)
    q = Interval(1.0, 2.0) * -2.0
    assert (q.lb, q.ub) == (-4.0, -2.0)
    r = 3.0 * Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (3.0, 6.0)


def test_interval_division_and_inversion():
    inv = ~Interval(2.0, 4.0)
    assert inv.lb == pytest.approx(0.25)
    assert inv.ub == pytest.approx(0.5)

    zero_lb = ~Interval(0.0, 2.0)
    assert zero_lb.lb == pytest.approx(0.5)
    assert zero_lb.ub == float("inf")

    through_zero = ~Interval(-1.0, 1.0)
    assert through_zero.lb == float("-inf")
    assert through_zero.ub == float("inf")

    q = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert q.lb == pytest.approx(0.25)
    assert q.ub == pytest.approx(1.0)

    s = Interval(2.0, 4.0) / 2.0
    assert (s.lb, s.ub) == (1.0, 2.0)


def test_interval_power():
    even = Interval(-2.0, 3.0) ** 2
    assert (even.lb, even.ub) == (0.0, 9.0)
    even_pos = Interval(1.0, 2.0) ** 2
    assert (even_pos.lb, even_pos.ub) == (1.0, 4.0)
    odd = Interval(-2.0, 3.0) ** 3
    assert (odd.lb, odd.ub) == (-8.0, 27.0)


def test_interval_results_reduced_to_number_when_narrow():
    narrow = Interval(1.0, 1.0 + 1e-6)
    result = narrow + 0.0
    assert isinstance(result, float)
    assert result == pytest.approx(1.0 + 0.5e-6)

    wide = Interval(1.0, 2.0) + 0.0
    assert isinstance(wide, Interval)


def test_interval_reduction_can_be_disabled(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    result = Interval(1.0, 1.0) + Interval(2.0, 2.0)
    assert isinstance(result, Interval)
    assert (result.lb, result.ub) == (3.0, 3.0)


def test_interval_equality_and_comparisons():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 3.0)
    assert Interval(1.0, 1.0) == 1.0
    assert Interval(1.0, 2.0) != 1.0
    with pytest.raises(NotImplementedError):
        Interval(1.0, 2.0) == "abc"

    assert Interval(1.0, 2.0) < Interval(2.0, 3.0)
    assert Interval(2.0, 3.0) > Interval(1.0, 5.0)
    assert Interval(1.0, 2.0) <= Interval(1.0, 9.0)
    assert Interval(1.0, 2.0) >= 1.0
    assert Interval(1.0, 2.0) < 1.5


# --------------------------------------------------------------------- Box

def test_box_basic_properties_and_str():
    b = Box(Interval(1.0, 2.0), Interval(3.0, 5.0))
    assert len(b) == 2
    assert np.array_equal(b.lb, np.array([1.0, 3.0]))
    assert np.array_equal(b.ub, np.array([2.0, 5.0]))
    assert np.allclose(b.middle, [1.5, 4.0])
    assert str(b) == "[1.0; 2.0] x [3.0; 5.0]"


def test_box_width_returns_widest_indices():
    b = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), Interval(5.0, 8.0))
    indices, w = b.width
    assert indices == [1, 2]
    assert w == 3.0

    single = Box(Interval(0.0, 1.0), 2.0)
    assert single.width == ([0], 1.0)


def test_box_with_mixed_numbers_and_intervals():
    b = Box(Interval(1.0, 3.0), 2.0)
    assert np.allclose(b.lb, [1.0, 2.0])
    assert np.allclose(b.ub, [3.0, 2.0])
    assert np.allclose(b.middle, [2.0, 2.0])


def test_box_equality_and_inequality():
    b1 = Box(Interval(1.0, 2.0), Interval(3.0, 4.0))
    b2 = Box(Interval(1.0, 2.0), Interval(3.0, 4.0))
    b3 = Box(Interval(1.0, 2.0), Interval(3.0, 5.0))
    assert b1 == b2
    assert not (b1 != b2)
    assert b1 != b3
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    with pytest.raises(NotImplementedError):
        b1 == 5


def test_box_copy_and_try_to_reduce():
    b = Box(Interval(1.0, 2.0), Interval(3.0, 4.0))
    c = copy_module.copy(b)
    assert isinstance(c, Box)
    assert c == b
    c[0].ub = 10.0
    assert b[0].ub == 2.0

    not_reduced = b._try_to_reduce()
    assert isinstance(not_reduced, Box)

    degenerate = Box(Interval(1.0, 1.0), 2.0)
    reduced = degenerate._try_to_reduce()
    assert np.allclose(reduced, [1.0, 2.0])
    assert not isinstance(reduced, Box) or reduced.dtype != object


# -------------------------------------------------- RS_SubinclusionFunction

def test_rs_subinclusion_function_encloses_sampled_values():
    np.random.seed(0)
    f = RS_SubinclusionFunction(lambda x: x[0] + x[1], number_of_samples=200)
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    result = f(b)
    assert isinstance(result, Interval)
    assert 2.0 <= result.lb <= result.ub <= 4.0
    assert result.width > 0.5


def test_rs_subinclusion_function_calls_f_n_times_and_constant():
    calls = []

    def func(x):
        calls.append(1)
        return 5.0

    f = RS_SubinclusionFunction(func, number_of_samples=7)
    result = f(Box(Interval(0.0, 1.0), Interval(-1.0, 1.0)))
    assert len(calls) == 7
    assert result.lb == 5.0
    assert result.ub == 5.0


# --------------------------------------------------------- hausdorff_distance

def test_hausdorff_distance_intervals():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == 2.0
    assert hausdorff_distance(Interval(2.0, 3.0), Interval(0.0, 1.0)) == 2.0
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == 0.0
    assert hausdorff_distance(Interval(0.0, 4.0), Interval(1.0, 2.0)) == 2.0
    assert hausdorff_distance(Interval(1.0, 2.0), Interval(0.0, 4.0)) == 2.0


def test_hausdorff_distance_with_numbers():
    assert hausdorff_distance(1.0, 4.0) == 3.0
    assert hausdorff_distance(4.0, 1.0) == 3.0
    assert hausdorff_distance(2.0, 2.0) == 0.0
    assert hausdorff_distance(Interval(0.0, 2.0), 5.0) == 5.0
