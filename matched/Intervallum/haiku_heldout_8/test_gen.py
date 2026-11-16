import math
from copy import copy

import numpy as np
import pytest

from intervallum import interval_functions as ifn
from intervallum.box import Box
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.interval_functions import hausdorff_distance


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

def test_box_creation_from_floats():
    b = Box(1.0, 2.0, 3.0)
    assert isinstance(b, Box)
    assert isinstance(b, np.ndarray)
    assert len(b) == 3
    assert list(b) == [1.0, 2.0, 3.0]


def test_box_str_mixed_components():
    b = Box(Interval(1.0, 2.0), 3.0)
    assert str(b) == "[1.0; 2.0] x 3.0"


def test_box_lb_ub_middle_with_mixed_components():
    b = Box(Interval(0.0, 2.0), 5.0)
    np.testing.assert_array_equal(b.lb, [0.0, 5.0])
    np.testing.assert_array_equal(b.ub, [2.0, 5.0])
    np.testing.assert_array_equal(b.middle, [1.0, 5.0])


def test_box_width_reports_widest_components():
    b = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), 2.0)
    indices, max_width = b.width
    assert indices == [1]
    assert max_width == 3.0

    tie = Box(Interval(0.0, 2.0), Interval(5.0, 7.0))
    tie_indices, tie_width = tie.width
    assert tie_indices == [0, 1]
    assert tie_width == 2.0


def test_box_equality_and_inequality():
    a = Box(1.0, 2.0)
    assert a == Box(1.0, 2.0)
    assert not (a != Box(1.0, 2.0))
    assert a != Box(1.0, 3.0)
    with pytest.raises(NotImplementedError):
        a == [1.0, 2.0]


def test_box_copy_is_independent():
    b = Box(Interval(0.0, 1.0), 2.0)
    c = copy(b)
    assert isinstance(c, Box)
    assert c is not b
    assert str(c) == str(b)
    assert c[0] is not b[0]
    assert c[0].lb == 0.0


def test_box_try_to_reduce_both_branches():
    degenerate = Box(Interval(1.0, 1.0), 2.0)
    reduced = degenerate._try_to_reduce()
    np.testing.assert_array_equal(reduced, [1.0, 2.0])

    non_degenerate = Box(Interval(0.0, 1.0), 2.0)
    kept = non_degenerate._try_to_reduce()
    assert isinstance(kept, Box)
    assert str(kept) == str(non_degenerate)


# ---------------------------------------------------------------------------
# Interval: construction, bounds and properties
# ---------------------------------------------------------------------------

def test_interval_bounds_validation_and_fix():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)

    fixed = Interval(3.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 3.0


def test_interval_lb_ub_setters_validate():
    i = Interval(1.0, 2.0)
    i.lb = 1.5
    assert i.lb == 1.5
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.5


def test_interval_middle_width_and_from_point():
    i = Interval(1.0, 3.0)
    assert i.middle == 2.0
    assert i.width == 2.0

    p = Interval.from_point(4.0)
    assert p.lb == 4.0
    assert p.ub == 4.0
    assert p.width == 0.0


def test_interval_addition_negation_subtraction():
    s = Interval(1.0, 2.0) + Interval(3.0, 4.0)
    assert (s.lb, s.ub) == (4.0, 6.0)

    t = Interval(1.0, 2.0) + 1.0
    assert (t.lb, t.ub) == (2.0, 3.0)

    n = -Interval(1.0, 2.0)
    assert (n.lb, n.ub) == (-2.0, -1.0)

    d = Interval(1.0, 2.0) - Interval(0.5, 1.0)
    assert d.lb == pytest.approx(0.0)
    assert d.ub == pytest.approx(1.5)

    r = 10.0 - Interval(1.0, 2.0)
    assert r.lb == pytest.approx(8.0)
    assert r.ub == pytest.approx(9.0)


def test_interval_multiplication():
    p = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (p.lb, p.ub) == (-4.0, 8.0)

    q = 2.0 * Interval(1.0, 3.0)
    assert (q.lb, q.ub) == (2.0, 6.0)


def test_interval_degenerate_reduces_to_float_and_can_be_disabled(monkeypatch):
    reduced = Interval(1.0, 1.0) + 0.0
    assert isinstance(reduced, float)
    assert reduced == 1.0

    assert isinstance(Interval(1.0, 2.0) + 0.0, Interval)

    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    not_reduced = Interval(1.0, 1.0) + 0.0
    assert isinstance(not_reduced, Interval)


def test_interval_equality_tolerance_and_bad_type():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 3.0)
    assert Interval(1.0, 1.0) == 1.0
    assert Interval(1.0, 2.0) != Interval(1.0, 2.0 + 1e-3)
    with pytest.raises(NotImplementedError):
        Interval(1.0, 2.0) == "not a number"


def test_interval_inverse_and_division():
    inv = ~Interval(2.0, 4.0)
    assert inv.lb == pytest.approx(0.25)
    assert inv.ub == pytest.approx(0.5)

    unbounded = ~Interval(-1.0, 1.0)
    assert math.isinf(unbounded.lb) and unbounded.lb < 0
    assert math.isinf(unbounded.ub) and unbounded.ub > 0

    div = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert div.lb == pytest.approx(0.25)
    assert div.ub == pytest.approx(1.0)

    half = Interval(2.0, 4.0) / 2.0
    assert (half.lb, half.ub) == (1.0, 2.0)

    rdiv = 1.0 / Interval(2.0, 4.0)
    assert rdiv.lb == pytest.approx(0.25)
    assert rdiv.ub == pytest.approx(0.5)


def test_interval_power_even_and_odd():
    even = Interval(-2.0, 3.0) ** 2
    assert even.lb == pytest.approx(0.0)
    assert even.ub == pytest.approx(9.0)

    odd = Interval(-2.0, 3.0) ** 3
    assert odd.lb == pytest.approx(-8.0)
    assert odd.ub == pytest.approx(27.0)

    negative_odd = Interval(-2.0, -1.0) ** 3
    assert negative_odd.lb == pytest.approx(-8.0)
    assert negative_odd.ub == pytest.approx(-1.0)

    negative_even = Interval(-2.0, -1.0) ** 2
    assert negative_even.lb == pytest.approx(1.0)
    assert negative_even.ub == pytest.approx(4.0)


def test_interval_comparisons_use_lower_bound():
    assert Interval(1.0, 5.0) < Interval(2.0, 3.0)
    assert Interval(2.0, 3.0) <= Interval(2.0, 9.0)
    assert Interval(3.0, 4.0) > Interval(1.0, 9.0)
    assert Interval(3.0, 4.0) >= 3.0
    assert Interval(1.0, 5.0) < 2.0
    assert not (Interval(3.0, 4.0) < 2.0)


# ---------------------------------------------------------------------------
# RS_SubinclusionFunction
# ---------------------------------------------------------------------------

def test_rs_subinclusion_encloses_true_range():
    np.random.seed(0)
    rs = RS_SubinclusionFunction(lambda x: x[0] + x[1], 500)
    result = rs(Box(Interval(0.0, 1.0), Interval(0.0, 1.0)))
    assert isinstance(result, Interval)
    assert result.lb >= 0.0
    assert result.ub <= 2.0
    assert result.lb < result.ub
    assert result.ub > 1.0


def test_rs_subinclusion_degenerate_box():
    rs = RS_SubinclusionFunction(lambda x: x[0] * x[1], 10)
    result = rs(Box(2.0, 3.0))
    assert result.lb == pytest.approx(6.0)
    assert result.ub == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# interval_functions: elementary functions
# ---------------------------------------------------------------------------

def test_trigonometric_functions():
    s = ifn.sin(Interval(0.0, math.pi / 2))
    assert s.lb == pytest.approx(0.0, abs=1e-12)
    assert s.ub == pytest.approx(1.0)

    s_interior = ifn.sin(Interval(1.0, 2.0))
    assert s_interior.lb == pytest.approx(math.sin(1.0))
    assert s_interior.ub == pytest.approx(1.0)

    s_full = ifn.sin(Interval(0.0, 7.0))
    assert s_full.lb == -1.0
    assert s_full.ub == 1.0

    c = ifn.cos(Interval(0.0, math.pi))
    assert c.lb == pytest.approx(-1.0)
    assert c.ub == pytest.approx(1.0)

    point = ifn.sin(0.0)
    assert isinstance(point, float)
    assert point == pytest.approx(0.0, abs=1e-12)


def test_abs_exp_sqrt_log_on_valid_domains():
    a = ifn.abs(Interval(-2.0, 3.0))
    assert (a.lb, a.ub) == (0.0, 3.0)

    a_pos = ifn.abs(Interval(1.0, 2.0))
    assert (a_pos.lb, a_pos.ub) == (1.0, 2.0)

    assert ifn.abs(-3.0) == 3.0

    e = ifn.exp(Interval(0.0, 1.0))
    assert e.lb == pytest.approx(1.0)
    assert e.ub == pytest.approx(math.e)

    sq = ifn.sqrt(Interval(4.0, 9.0))
    assert sq.lb == pytest.approx(2.0)
    assert sq.ub == pytest.approx(3.0)

    sq_clipped = ifn.sqrt(Interval(-1.0, 4.0))
    assert sq_clipped.lb == pytest.approx(0.0)
    assert sq_clipped.ub == pytest.approx(2.0)

    lg = ifn.log(Interval(1.0, math.e))
    assert lg.lb == pytest.approx(0.0)
    assert lg.ub == pytest.approx(1.0)

    lg_clipped = ifn.log(Interval(-1.0, math.e))
    assert lg_clipped.lb == -math.inf
    assert lg_clipped.ub == pytest.approx(1.0)


def test_sqrt_and_log_undefined_domains_raise():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.sqrt(Interval(-3.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.log(Interval(-2.0, 0.0))


# ---------------------------------------------------------------------------
# interval_functions: split, bisect, constrain, shrink
# ---------------------------------------------------------------------------

def test_split_bisect_constrain_shrink():
    parts = ifn.split(Interval(1.0, 5.0), [1.0, 1.0])
    assert len(parts) == 2
    assert (parts[0].lb, parts[0].ub) == pytest.approx((1.0, 3.0))
    assert (parts[1].lb, parts[1].ub) == pytest.approx((3.0, 5.0))

    halves = ifn.bisect(Interval(0.0, 2.0))
    assert [(h.lb, h.ub) for h in halves] == pytest.approx([(0.0, 1.0), (1.0, 2.0)])
    assert ifn.bisect(5.0) == [5.0, 5.0]
    assert ifn.split(3.0, [1.0, 2.0, 3.0]) == [3.0, 3.0, 3.0]

    c = ifn.constrain(Interval(-1.0, 5.0), 0.0, 3.0)
    assert (c.lb, c.ub) == (0.0, 3.0)
    assert ifn.constrain(-2.0, 0.0, 3.0) == 0.0
    assert ifn.constrain(7.0, 0.0, 3.0) == 3.0

    s = ifn.shrink(Interval(0.0, 4.0), 0.5)
    assert (s.lb, s.ub) == pytest.approx((1.0, 3.0))
    assert ifn.shrink(2.0, 0.5) == 2.0


@pytest.mark.xfail(
    strict=True,
    reason="split() reuses the previous segment's lower bound when more than two ratios are given",
)
def test_split_three_ratios_known_bug():
    parts = ifn.split(Interval(0.0, 3.0), [1.0, 1.0, 1.0])
    assert [(p.lb, p.ub) for p in parts] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


# ---------------------------------------------------------------------------
# hausdorff_distance
# ---------------------------------------------------------------------------

def test_hausdorff_distance_cases():
    assert hausdorff_distance(1.0, 1.0) == 0.0
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == 0.0
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.5, 1.5)) == pytest.approx(0.5)
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 3.0)) == pytest.approx(2.0)
    assert hausdorff_distance(5.0, Interval(0.0, 1.0)) == pytest.approx(5.0)
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.5, 1.5)) == pytest.approx(
        hausdorff_distance(Interval(0.5, 1.5), Interval(0.0, 1.0))
    )
