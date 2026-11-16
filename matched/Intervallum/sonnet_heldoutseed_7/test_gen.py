import copy
import math

import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import hausdorff_distance


# ---------------------------------------------------------------- Interval

def test_interval_construction_and_properties():
    i = Interval(1.0, 4.0)
    assert i.lb == 1.0
    assert i.ub == 4.0
    assert i.middle == pytest.approx(2.5)
    assert i.width == pytest.approx(3.0)
    assert str(i) == "[1.0; 4.0]"
    assert repr(i) == str(i)


def test_interval_wrong_bounds_raises_and_fix_swaps():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    i = Interval(2.0, 1.0, fix=True)
    assert (i.lb, i.ub) == (1.0, 2.0)


def test_interval_from_point():
    i = Interval.from_point(3.0)
    assert i.lb == 3.0 and i.ub == 3.0
    assert i.width == 0.0


def test_interval_bound_setters_validate():
    i = Interval(0.0, 1.0)
    i.lb = 0.5
    i.ub = 2.0
    assert (i.lb, i.ub) == (0.5, 2.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.1


def test_interval_copy_is_independent():
    i = Interval(0.0, 1.0)
    c = copy.copy(i)
    c.ub = 5.0
    assert i.ub == 1.0
    assert c.ub == 5.0


def test_interval_addition_and_subtraction():
    r = Interval(1.0, 2.0) + Interval(10.0, 20.0)
    assert (r.lb, r.ub) == (11.0, 22.0)
    r = Interval(1.0, 2.0) + 1.0
    assert (r.lb, r.ub) == (2.0, 3.0)
    r = 1.0 + Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (2.0, 3.0)
    r = Interval(1.0, 3.0) - Interval(0.0, 1.0)
    assert (r.lb, r.ub) == (0.0, 3.0)
    r = 10.0 - Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (8.0, 9.0)


def test_interval_negation():
    r = -Interval(1.0, 3.0)
    assert (r.lb, r.ub) == (-3.0, -1.0)


def test_interval_multiplication():
    r = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (r.lb, r.ub) == (-4.0, 8.0)
    r = Interval(1.0, 2.0) * -2.0
    assert (r.lb, r.ub) == (-4.0, -2.0)
    r = 3.0 * Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (3.0, 6.0)


def test_interval_inverse_and_division():
    r = ~Interval(2.0, 4.0)
    assert r.lb == pytest.approx(0.25)
    assert r.ub == pytest.approx(0.5)

    r = ~Interval(0.0, 2.0)
    assert r.lb == pytest.approx(0.5)
    assert r.ub == math.inf

    r = ~Interval(-1.0, 1.0)
    assert r.lb == -math.inf and r.ub == math.inf

    r = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert r.lb == pytest.approx(0.25)
    assert r.ub == pytest.approx(1.0)

    r = Interval(2.0, 4.0) / 2.0
    assert (r.lb, r.ub) == (1.0, 2.0)


def test_interval_power():
    r = Interval(-2.0, 3.0) ** 2
    assert (r.lb, r.ub) == (0.0, 9.0)
    r = Interval(1.0, 2.0) ** 3
    assert (r.lb, r.ub) == (1.0, 8.0)
    r = Interval(-2.0, -1.0) ** 3
    assert (r.lb, r.ub) == (-8.0, -1.0)


def test_interval_narrow_result_reduced_to_number():
    r = Interval(0.0, 1e-6) + 1.0
    assert not isinstance(r, Interval)
    assert r == pytest.approx(1.0000005)


def test_interval_equality_and_comparisons():
    assert Interval(0.0, 1.0) == Interval(0.0, 1.0)
    assert Interval(0.0, 1.0) != Interval(0.0, 1.5)
    with pytest.raises(NotImplementedError):
        Interval(0.0, 1.0) == "a"
    assert Interval(0.0, 1.0) < Interval(1.0, 2.0)
    assert Interval(0.0, 1.0) <= Interval(0.0, 5.0)
    assert Interval(2.0, 3.0) > 1.0
    assert Interval(2.0, 3.0) >= 2.0
    assert not (Interval(2.0, 3.0) < 2.0)


# --------------------------------------------------------------------- Box

def test_box_creation_and_bounds():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))
    assert isinstance(b, Box)
    assert len(b) == 2
    np.testing.assert_allclose(b.lb, [0.0, 2.0])
    np.testing.assert_allclose(b.ub, [1.0, 4.0])
    np.testing.assert_allclose(b.middle, [0.5, 3.0])


def test_box_with_mixed_intervals_and_numbers():
    b = Box(Interval(0.0, 1.0), 2.0)
    np.testing.assert_allclose(b.lb, [0.0, 2.0])
    np.testing.assert_allclose(b.ub, [1.0, 2.0])
    np.testing.assert_allclose(b.middle, [0.5, 2.0])


def test_box_width_returns_indices_of_widest_components():
    b = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), 5.0, Interval(1.0, 4.0))
    indices, w = b.width
    assert indices == [1, 3]
    assert w == pytest.approx(3.0)


def test_box_str_repr():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    assert str(b) == "[0.0; 1.0] x [2.0; 3.0]"
    assert repr(b) == "[0.0; 1.0] x [2.0; 3.0]"


def test_box_equality():
    b1 = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    b2 = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    b3 = Box(Interval(0.0, 1.0), Interval(2.0, 3.5))
    assert b1 == b2
    assert not (b1 != b2)
    assert b1 != b3
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    with pytest.raises(NotImplementedError):
        b1 == 5


def test_box_copy_is_deep_for_intervals():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    c = copy.copy(b)
    assert isinstance(c, Box)
    c[0].ub = 10.0
    assert b[0].ub == 1.0


def test_box_try_to_reduce():
    degenerate = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    reduced = degenerate._try_to_reduce()
    np.testing.assert_allclose(reduced, [1.0, 2.0])

    proper = Box(Interval(0.0, 1.0), 2.0)
    kept = proper._try_to_reduce()
    assert isinstance(kept, Box)
    assert kept is not proper
    np.testing.assert_allclose(kept.lb, [0.0, 2.0])
    np.testing.assert_allclose(kept.ub, [1.0, 2.0])


# -------------------------------------------------- RS_SubinclusionFunction

def test_rs_subinclusion_encloses_samples_within_range():
    np.random.seed(0)
    f = RS_SubinclusionFunction(lambda p: p[0] + p[1], 200)
    res = f(Box(Interval(0.0, 1.0), Interval(0.0, 1.0)))
    assert isinstance(res, Interval)
    assert 0.0 <= res.lb <= res.ub <= 2.0
    assert res.width > 1.0  # 200 samples should cover most of the range


def test_rs_subinclusion_calls_function_requested_number_of_times_in_box():
    np.random.seed(1)
    calls = []

    def g(p):
        calls.append(np.array(p, dtype=float))
        return float(p[0] * p[1])

    f = RS_SubinclusionFunction(g, 50)
    f(Box(Interval(1.0, 2.0), 3.0))
    assert len(calls) == 50
    for p in calls:
        assert 1.0 <= p[0] <= 2.0
        assert p[1] == pytest.approx(3.0)


def test_rs_subinclusion_constant_function_gives_degenerate_interval():
    f = RS_SubinclusionFunction(lambda p: 7.0, 10)
    res = f(Box(Interval(0.0, 1.0)))
    assert res.lb == 7.0 and res.ub == 7.0


# ------------------------------------------------------ hausdorff_distance

def test_hausdorff_distance_intervals():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 3.0)) == pytest.approx(2.0)
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(5.0, 6.0)) == pytest.approx(5.0)
    assert hausdorff_distance(Interval(0.0, 2.0), Interval(0.0, 2.0)) == 0.0


def test_hausdorff_distance_symmetric_and_with_numbers():
    a, b = Interval(-1.0, 2.0), Interval(0.0, 5.0)
    assert hausdorff_distance(a, b) == pytest.approx(hausdorff_distance(b, a))
    assert hausdorff_distance(1.0, 4.0) == pytest.approx(3.0)
    assert hausdorff_distance(4.0, 1.0) == pytest.approx(3.0)
    assert hausdorff_distance(2.0, Interval(0.0, 5.0)) == pytest.approx(3.0)
