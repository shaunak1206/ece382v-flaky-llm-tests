import copy as copy_module
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
    fixed = Interval(2.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 2.0


def test_interval_from_point_and_copy():
    p = Interval.from_point(3.0)
    assert p.lb == 3.0 and p.ub == 3.0
    i = Interval(1.0, 2.0)
    c = copy_module.copy(i)
    assert c is not i
    assert c.lb == 1.0 and c.ub == 2.0


def test_interval_bound_setters_validate():
    i = Interval(1.0, 3.0)
    i.lb = 2.0
    i.ub = 5.0
    assert (i.lb, i.ub) == (2.0, 5.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 6.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 1.0
    assert (i.lb, i.ub) == (2.0, 5.0)


def test_interval_addition_and_subtraction():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 5.0)
    s = a + b
    assert (s.lb, s.ub) == (4.0, 7.0)
    s2 = 1.0 + a
    assert (s2.lb, s2.ub) == (2.0, 3.0)
    d = Interval(1.0, 3.0) - Interval(0.0, 1.0)
    assert (d.lb, d.ub) == (0.0, 3.0)
    r = 5.0 - Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (3.0, 4.0)


def test_interval_negation():
    n = -Interval(1.0, 3.0)
    assert (n.lb, n.ub) == (-3.0, -1.0)


def test_interval_multiplication():
    m = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (m.lb, m.ub) == (-4.0, 8.0)
    m2 = Interval(1.0, 2.0) * -1.0
    assert (m2.lb, m2.ub) == (-2.0, -1.0)
    m3 = 2.0 * Interval(1.0, 2.0)
    assert (m3.lb, m3.ub) == (2.0, 4.0)


def test_interval_inversion_cases():
    pos = ~Interval(2.0, 4.0)
    assert pos.lb == pytest.approx(0.25)
    assert pos.ub == pytest.approx(0.5)

    zero_lb = ~Interval(0.0, 2.0)
    assert zero_lb.lb == pytest.approx(0.5)
    assert zero_lb.ub == math.inf

    zero_ub = ~Interval(-2.0, 0.0)
    assert zero_ub.lb == -math.inf
    assert zero_ub.ub == pytest.approx(-0.5)

    both = ~Interval(-1.0, 1.0)
    assert both.lb == -math.inf and both.ub == math.inf


def test_interval_division():
    q = Interval(2.0, 4.0) / 2.0
    assert (q.lb, q.ub) == (1.0, 2.0)
    q2 = Interval(2.0, 4.0) / Interval(1.0, 2.0)
    assert q2.lb == pytest.approx(1.0)
    assert q2.ub == pytest.approx(4.0)
    q3 = 1.0 / Interval(2.0, 4.0)
    assert q3.lb == pytest.approx(0.25)
    assert q3.ub == pytest.approx(0.5)


def test_interval_power_even_and_odd():
    even = Interval(-2.0, 3.0) ** 2
    assert even.lb == pytest.approx(0.0)
    assert even.ub == pytest.approx(9.0)
    even_pos = Interval(1.0, 2.0) ** 2
    assert even_pos.lb == pytest.approx(1.0)
    assert even_pos.ub == pytest.approx(4.0)
    odd = Interval(-2.0, 3.0) ** 3
    assert odd.lb == pytest.approx(-8.0)
    assert odd.ub == pytest.approx(27.0)


def test_interval_narrow_results_reduce_to_numbers():
    narrow = Interval(1.0, 1.0 + 1e-8)
    result = narrow + 1.0
    assert not isinstance(result, Interval)
    assert result == pytest.approx(2.0)
    point_sum = Interval.from_point(1.0) + 2.0
    assert not isinstance(point_sum, Interval)
    assert point_sum == pytest.approx(3.0)


def test_interval_comparisons_use_lower_bound():
    a = Interval(1.0, 10.0)
    b = Interval(2.0, 3.0)
    assert a < b
    assert a <= b
    assert b > a
    assert b >= a
    assert a < 2.0
    assert a >= 1.0
    assert not (a > 1.5)


def test_interval_equality_and_inequality():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 3.0)
    assert Interval(1.0, 1.0) == 1.0
    assert Interval(1.0, 2.0) != 1.0
    with pytest.raises(NotImplementedError):
        Interval(1.0, 2.0) == "abc"


def test_interval_inner_subtraction_operator():
    r = Interval(1.0, 3.0) << Interval(0.0, 1.0)
    assert (r.lb, r.ub) == (1.0, 2.0)
    # Swapped bounds are fixed
    r2 = Interval(0.0, 1.0) << Interval(1.0, 3.0)
    assert (r2.lb, r2.ub) == (-1.0, 1.0 - 3.0) or (r2.lb, r2.ub) == (-2.0, -1.0)


# --------------------------------------------------------------------- Box

def test_box_bounds_middle_and_width():
    b = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), 2.0)
    assert np.allclose(b.lb, [0.0, 0.0, 2.0])
    assert np.allclose(b.ub, [1.0, 3.0, 2.0])
    assert np.allclose(b.middle, [0.5, 1.5, 2.0])
    indices, max_width = b.width
    assert indices == [1]
    assert max_width == pytest.approx(3.0)


def test_box_width_ties_and_all_points():
    b = Box(Interval(0.0, 2.0), Interval(1.0, 3.0))
    indices, max_width = b.width
    assert indices == [0, 1]
    assert max_width == pytest.approx(2.0)
    points = Box(1.0, 2.0)
    indices, max_width = points.width
    assert indices == [0, 1]
    assert max_width == 0


def test_box_str_and_equality():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    assert str(b) == "[0.0; 1.0] x [2.0; 3.0]"
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == [1.0, 2.0]


def test_box_copy_and_try_to_reduce():
    b = Box(Interval(0.0, 1.0), 2.0)
    c = copy_module.copy(b)
    assert isinstance(c, Box)
    assert c is not b
    assert c[0] is not b[0]
    assert c[0].lb == 0.0 and c[0].ub == 1.0

    reduced_same = b._try_to_reduce()
    assert isinstance(reduced_same, Box)

    degenerate = Box(Interval(1.0, 1.0), 2.0)
    reduced = degenerate._try_to_reduce()
    assert not isinstance(reduced, Box)
    assert np.allclose(reduced, [1.0, 2.0])


# --------------------------------------------------- RS_SubinclusionFunction

def test_rs_subinclusion_function_encloses_sampled_values():
    np.random.seed(0)
    f = RS_SubinclusionFunction(lambda p: p[0] + p[1], 200)
    box = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    result = f(box)
    assert isinstance(result, Interval)
    assert result.lb <= result.ub
    assert result.lb >= 2.0
    assert result.ub <= 4.0
    # With 200 samples the estimate should cover most of the range
    assert result.width > 1.0


def test_rs_subinclusion_function_with_degenerate_component_and_single_sample():
    np.random.seed(1)
    f = RS_SubinclusionFunction(lambda p: p[0] * 2.0 + p[1], 1)
    box = Box(Interval(1.0, 2.0), 5.0)
    result = f(box)
    assert result.lb == result.ub
    assert 7.0 <= result.lb <= 9.0

    const = RS_SubinclusionFunction(lambda p: 3.0, 10)
    r = const(Box(Interval(0.0, 1.0), Interval(0.0, 1.0)))
    assert r.lb == 3.0 and r.ub == 3.0


# ------------------------------------------------------ hausdorff_distance

def test_hausdorff_distance_intervals_and_numbers():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 4.0)) == pytest.approx(3.0)
    assert hausdorff_distance(Interval(2.0, 4.0), Interval(0.0, 1.0)) == pytest.approx(3.0)
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == 0.0
    assert hausdorff_distance(1.0, 4.0) == pytest.approx(3.0)
    assert hausdorff_distance(Interval(0.0, 1.0), 0.5) == pytest.approx(0.5)
    assert hausdorff_distance(0.5, Interval(0.0, 1.0)) == pytest.approx(0.5)
