import math

import numpy as np
import pytest
from copy import copy

from intervallum import interval_functions as ifn
from intervallum.box import Box
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions


@pytest.fixture
def no_reduction(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)


# ---------------------------------------------------------------- Box

def test_box_construction_and_str_of_floats():
    b = Box(1.0, 2.0)
    assert isinstance(b, Box)
    assert isinstance(b, np.ndarray)
    assert b.shape == (2,)
    assert str(b) == "1.0 x 2.0"


def test_box_str_with_interval_component():
    b = Box(Interval(0.0, 1.0), 2.0)
    assert str(b) == "[0.0; 1.0] x 2.0"


def test_box_equality_and_inequality():
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    assert not (Box(1.0, 2.0) != Box(1.0, 2.0))
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == 3.0


def test_box_bound_properties_with_mixed_components():
    b = Box(Interval(0.0, 2.0), 3.0)
    np.testing.assert_array_equal(b.lb, np.array([0.0, 3.0]))
    np.testing.assert_array_equal(b.ub, np.array([2.0, 3.0]))
    np.testing.assert_array_equal(b.middle, np.array([1.0, 3.0]))


def test_box_width_reports_widest_components():
    indices, max_width = Box(Interval(0.0, 1.0), Interval(0.0, 3.0)).width
    assert indices == [1]
    assert max_width == 3.0

    indices, max_width = Box(Interval(0.0, 1.0), 5.0).width
    assert indices == [0]
    assert max_width == 1.0


def test_box_try_to_reduce_degenerate_returns_middle_and_otherwise_box():
    degenerate = Box(Interval(1.0, 1.0), 2.0)
    reduced = degenerate._try_to_reduce()
    assert not isinstance(reduced, Box)
    np.testing.assert_array_equal(reduced, np.array([1.0, 2.0]))

    wide = Box(Interval(0.0, 1.0), Interval(0.0, 3.0))
    result = wide._try_to_reduce()
    assert isinstance(result, Box)
    assert str(result) == str(wide)


def test_box_copy_is_independent_for_intervals():
    b = Box(Interval(0.0, 1.0), 2.0)
    c = copy(b)
    assert isinstance(c, Box)
    assert c is not b
    assert c[0] is not b[0]
    c[0].ub = 5.0
    assert b[0].ub == 1.0


# ---------------------------------------------------------------- Interval

def test_interval_rejects_wrong_bounds_and_fix_swaps_them():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    fixed = Interval(2.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 2.0


def test_interval_properties_and_str():
    point = Interval.from_point(2.0)
    assert point.lb == point.ub == 2.0
    assert point.width == 0.0
    assert point.middle == 2.0

    i = Interval(1.0, 3.0)
    assert i.middle == 2.0
    assert i.width == 2.0
    assert str(i) == "[1.0; 3.0]"
    assert repr(i) == "[1.0; 3.0]"


def test_interval_bound_setters_validate():
    i = Interval(1.0, 2.0)
    i.lb = 1.5
    assert i.lb == 1.5
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0
    assert (i.lb, i.ub) == (1.5, 2.0)


def test_interval_try_to_reduce_behaviour():
    # Degenerate result collapses to a float by default.
    reduced = Interval(1.0, 1.0) + 2.0
    assert isinstance(reduced, float)
    assert reduced == 3.0

    # Wide result stays an Interval.
    wide = Interval(0.0, 1.0) + 1.0
    assert isinstance(wide, Interval)
    assert (wide.lb, wide.ub) == (1.0, 2.0)


def test_interval_subtraction_is_dependency_unaware_and_inner_subtraction_fixed(no_reduction):
    a = Interval(1.0, 2.0)
    d = a - a
    assert (d.lb, d.ub) == (-1.0, 1.0)

    # inner_subtraction with fix=True swaps bounds when they would invert.
    swapped = Interval.inner_subtraction(Interval(1.0, 2.0), Interval(3.0, 5.0))
    assert (swapped.lb, swapped.ub) == (-3.0, -2.0)

    plain = Interval(1.0, 2.0) << Interval(0.0, 1.0)
    assert (plain.lb, plain.ub) == (1.0, 1.0)


def test_interval_equality_tolerance_and_type_checks():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 3.0)
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0 + 1e-9)
    assert Interval(1.0, 1.0) == 1.0
    assert Interval(1.0, 2.0) != 1.0
    with pytest.raises(NotImplementedError):
        Interval(1.0, 1.0) == 1


def test_interval_scalar_arithmetic():
    i = Interval(1.0, 2.0)
    add = i + 1.0
    assert (add.lb, add.ub) == (2.0, 3.0)
    radd = 1.0 + i
    assert (radd.lb, radd.ub) == (2.0, 3.0)
    neg = -i
    assert (neg.lb, neg.ub) == (-2.0, -1.0)
    sub = i - 1.0
    assert (sub.lb, sub.ub) == (0.0, 1.0)
    rsub = 1.0 - i
    assert (rsub.lb, rsub.ub) == (-1.0, 0.0)
    mul = i * 2.0
    assert (mul.lb, mul.ub) == (2.0, 4.0)
    rmul = 2.0 * i
    assert (rmul.lb, rmul.ub) == (2.0, 4.0)


def test_interval_invert_and_division():
    inv = ~Interval(2.0, 4.0)
    assert (inv.lb, inv.ub) == (0.25, 0.5)

    unbounded = ~Interval(-1.0, 1.0)
    assert math.isinf(unbounded.lb) and unbounded.lb < 0
    assert math.isinf(unbounded.ub) and unbounded.ub > 0

    q = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert (q.lb, q.ub) == (0.25, 1.0)

    q2 = Interval(2.0, 4.0) / 2.0
    assert (q2.lb, q2.ub) == (1.0, 2.0)

    r = 1.0 / Interval(2.0, 4.0)
    assert (r.lb, r.ub) == (0.25, 0.5)


def test_interval_pow_even_and_odd():
    even_straddle = Interval(-2.0, 1.0) ** 2
    assert (even_straddle.lb, even_straddle.ub) == (0.0, 4.0)

    even_positive = Interval(2.0, 3.0) ** 2
    assert (even_positive.lb, even_positive.ub) == (4.0, 9.0)

    odd = Interval(-2.0, 1.0) ** 3
    assert (odd.lb, odd.ub) == (-8.0, 1.0)


def test_interval_comparisons_use_lower_bound():
    assert Interval(1.0, 5.0) < Interval(2.0, 3.0)
    assert Interval(1.0, 5.0) <= 1.0
    assert Interval(1.0, 5.0) > 0.5
    assert Interval(2.0, 5.0) >= Interval(2.0, 3.0)


# ---------------------------------------------------------------- interval_functions

def test_sin_and_cos_ranges(no_reduction):
    assert ifn.sin(math.pi / 2) == pytest.approx(1.0)

    s = ifn.sin(Interval(0.0, math.pi / 2))
    assert s.lb == pytest.approx(0.0, abs=1e-12)
    assert s.ub == pytest.approx(1.0)

    full = ifn.sin(Interval(0.0, 2 * math.pi))
    assert (full.lb, full.ub) == (-1.0, 1.0)

    c = ifn.cos(Interval(0.0, math.pi))
    assert c.lb == pytest.approx(-1.0)
    assert c.ub == pytest.approx(1.0)


def test_abs_and_exp(no_reduction):
    assert ifn.abs(-2.0) == 2.0

    a = ifn.abs(Interval(-2.0, 1.0))
    assert (a.lb, a.ub) == (0.0, 2.0)

    e = ifn.exp(Interval(0.0, 1.0))
    assert e.lb == pytest.approx(1.0)
    assert e.ub == pytest.approx(math.e)


def test_sqrt_and_log_on_valid_domains(no_reduction):
    r = ifn.sqrt(Interval(4.0, 9.0))
    assert (r.lb, r.ub) == (2.0, 3.0)

    clipped = ifn.sqrt(Interval(-1.0, 4.0))
    assert (clipped.lb, clipped.ub) == (0.0, 2.0)

    lg = ifn.log(Interval(1.0, math.e))
    assert lg.lb == pytest.approx(0.0)
    assert lg.ub == pytest.approx(1.0)


def test_sqrt_and_log_raise_outside_domain():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.sqrt(Interval(-2.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.log(Interval(-2.0, 0.0))


def test_split_and_bisect(no_reduction):
    parts = ifn.split(Interval(0.0, 4.0), [1.0, 1.0])
    assert [(p.lb, p.ub) for p in parts] == [(0.0, 2.0), (2.0, 4.0)]

    halves = ifn.bisect(Interval(0.0, 4.0))
    assert [(p.lb, p.ub) for p in halves] == [(0.0, 2.0), (2.0, 4.0)]

    assert ifn.split(1.0, [1.0, 2.0]) == [1.0, 1.0]


def test_constrain_and_shrink(no_reduction):
    c = ifn.constrain(Interval(-1.0, 5.0), 0.0, 3.0)
    assert (c.lb, c.ub) == (0.0, 3.0)
    assert ifn.constrain(-1.0, 0.0, 3.0) == 0.0

    s = ifn.shrink(Interval(0.0, 4.0), 0.5)
    assert (s.lb, s.ub) == (1.0, 3.0)
    assert ifn.shrink(2.0, 0.5) == 2.0


def test_hausdorff_distance():
    assert ifn.hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == 0.0
    assert ifn.hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 2.0)) == 1.0
    assert ifn.hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == 2.0
    assert ifn.hausdorff_distance(1.0, 3.0) == 2.0
    assert ifn.hausdorff_distance(1.0, 1.0) == 0.0


# ---------------------------------------------------------------- RS_SubinclusionFunction

def test_rs_subinclusion_encloses_sampled_values():
    np.random.seed(0)
    f = lambda x: x[0] + x[1]
    box = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    result = RS_SubinclusionFunction(f, 500)(box)

    assert isinstance(result, Interval)
    assert 2.0 <= result.lb <= result.ub <= 4.0
    assert result.lb < 2.5
    assert result.ub > 3.5
