import math
from copy import copy

import numpy as np
import pytest

from intervallum import interval_functions as IF
from intervallum.box import Box
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

def test_box_constructor_str_and_copy():
    b = Box(1.0, 2.0, 3.0)
    assert isinstance(b, Box)
    assert isinstance(b, np.ndarray)
    assert b.shape == (3,)
    assert np.allclose(np.asarray(b), [1.0, 2.0, 3.0])
    assert str(Box(1.0, 2.0)) == "1.0 x 2.0"

    mixed = Box(Interval(0.0, 1.0), 2.0)
    assert str(mixed) == "[0.0; 1.0] x 2.0"

    c = copy(mixed)
    assert isinstance(c, Box)
    assert c is not mixed
    assert str(c) == str(mixed)
    assert c[0] is not mixed[0]


def test_box_eq_compares_elementwise_and_rejects_non_arrays():
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 2.5)
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == [1.0, 2.0]


def test_box_lb_ub_middle_with_mixed_components():
    b = Box(Interval(0.0, 2.0), 5.0)
    assert np.allclose(b.lb, [0.0, 5.0])
    assert np.allclose(b.ub, [2.0, 5.0])
    assert np.allclose(b.middle, [1.0, 5.0])


def test_box_width_returns_indices_of_widest_components():
    indices, max_width = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), 2.0).width
    assert indices == [1]
    assert max_width == pytest.approx(3.0)

    # Plain numbers have zero width, so every index is widest.
    indices, max_width = Box(1.0, 2.0).width
    assert indices == [0, 1]
    assert max_width == 0


def test_box_try_to_reduce_collapses_degenerate_box_only():
    degenerate = Box(Interval(1.0, 1.0), 2.0)._try_to_reduce()
    assert not isinstance(degenerate, Box)
    assert np.allclose(degenerate, [1.0, 2.0])

    wide = Box(Interval(1.0, 3.0), 2.0)._try_to_reduce()
    assert isinstance(wide, Box)
    assert wide.width[1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Interval: construction, properties, equality, comparisons
# ---------------------------------------------------------------------------

def test_interval_init_rejects_reversed_bounds_and_fix_swaps():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)

    fixed = Interval(2.0, 1.0, fix=True)
    assert fixed.lb == pytest.approx(1.0)
    assert fixed.ub == pytest.approx(2.0)

    i = Interval(0.0, 1.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 2.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = -1.0
    i.lb = 0.5
    assert i.lb == pytest.approx(0.5)


def test_interval_str_middle_width_and_from_point():
    i = Interval(0.0, 2.0)
    assert str(i) == "[0.0; 2.0]"
    assert repr(i) == "[0.0; 2.0]"
    assert i.middle == pytest.approx(1.0)
    assert i.width == pytest.approx(2.0)

    p = Interval.from_point(3.0)
    assert p.lb == 3.0 and p.ub == 3.0


def test_interval_eq_uses_admissible_error_and_rejects_ints():
    assert Interval(1.0, 1.0) == 1.0
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 1.0 + 1e-9) == 1.0
    assert Interval(1.0, 2.0) != 1.0
    assert Interval(1.0, 1.0) != Interval(1.0, 1.5)
    with pytest.raises(NotImplementedError):
        Interval(1.0, 1.0) == 1


def test_interval_comparisons_use_lower_bound():
    assert Interval(0.0, 5.0) < Interval(1.0, 2.0)
    assert Interval(0.0, 5.0) < 1.0
    assert Interval(3.0, 4.0) > 2.0
    assert Interval(3.0, 4.0) >= Interval(3.0, 10.0)
    assert not (Interval(3.0, 4.0) <= 2.0)


# ---------------------------------------------------------------------------
# Interval: arithmetic
# ---------------------------------------------------------------------------

def test_interval_addition_subtraction_and_negation():
    s = Interval(1.0, 2.0) + Interval(3.0, 4.0)
    assert (s.lb, s.ub) == (pytest.approx(4.0), pytest.approx(6.0))

    s = Interval(1.0, 2.0) + 1.0
    assert (s.lb, s.ub) == (pytest.approx(2.0), pytest.approx(3.0))

    s = 1.0 + Interval(1.0, 2.0)
    assert (s.lb, s.ub) == (pytest.approx(2.0), pytest.approx(3.0))

    d = Interval(1.0, 2.0) - Interval(0.0, 1.0)
    assert (d.lb, d.ub) == (pytest.approx(0.0), pytest.approx(2.0))

    r = 5.0 - Interval(1.0, 2.0)
    assert (r.lb, r.ub) == (pytest.approx(3.0), pytest.approx(4.0))

    n = -Interval(1.0, 2.0)
    assert (n.lb, n.ub) == (pytest.approx(-2.0), pytest.approx(-1.0))


def test_interval_multiplication_takes_extrema_of_products():
    p = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert (p.lb, p.ub) == (pytest.approx(-4.0), pytest.approx(8.0))

    s = Interval(1.0, 2.0) * -1.0
    assert (s.lb, s.ub) == (pytest.approx(-2.0), pytest.approx(-1.0))

    s = 2.0 * Interval(1.0, 2.0)
    assert (s.lb, s.ub) == (pytest.approx(2.0), pytest.approx(4.0))


def test_interval_reciprocal_and_division():
    r = ~Interval(2.0, 4.0)
    assert (r.lb, r.ub) == (pytest.approx(0.25), pytest.approx(0.5))

    r = ~Interval(0.0, 2.0)
    assert r.lb == pytest.approx(0.5)
    assert math.isinf(r.ub) and r.ub > 0

    r = ~Interval(-1.0, 1.0)
    assert math.isinf(r.lb) and r.lb < 0
    assert math.isinf(r.ub) and r.ub > 0

    q = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert (q.lb, q.ub) == (pytest.approx(0.25), pytest.approx(1.0))

    q = Interval(2.0, 4.0) / 2.0
    assert (q.lb, q.ub) == (pytest.approx(1.0), pytest.approx(2.0))

    q = 1.0 / Interval(2.0, 4.0)
    assert (q.lb, q.ub) == (pytest.approx(0.25), pytest.approx(0.5))


def test_interval_power_even_odd_and_straddling_zero():
    p = Interval(-2.0, 3.0) ** 2
    assert p.lb == pytest.approx(0.0)
    assert p.ub == pytest.approx(9.0)

    p = Interval(2.0, 3.0) ** 2
    assert (p.lb, p.ub) == (pytest.approx(4.0), pytest.approx(9.0))

    p = Interval(-2.0, 3.0) ** 3
    assert (p.lb, p.ub) == (pytest.approx(-8.0), pytest.approx(27.0))


def test_interval_degenerate_result_reduces_to_float_by_default():
    result = Interval(1.0, 1.0) + 1.0
    assert isinstance(result, float)
    assert result == pytest.approx(2.0)

    result = Interval(1.0, 1.0) + Interval(1.0, 1.0)
    assert isinstance(result, float)
    assert result == pytest.approx(2.0)


def test_interval_reduction_can_be_disabled_with_monkeypatch(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    result = Interval(1.0, 1.0) + 1.0
    assert isinstance(result, Interval)
    assert result.lb == pytest.approx(2.0)
    assert result.ub == pytest.approx(2.0)


def test_inner_subtraction_fixes_reversed_result():
    r = Interval.inner_subtraction(Interval(0.0, 1.0), Interval(1.0, 3.0))
    assert isinstance(r, Interval)
    assert r.lb == pytest.approx(-2.0)
    assert r.ub == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# interval_functions
# ---------------------------------------------------------------------------

def test_sin_monotonic_pieces_and_full_period():
    s = IF.sin(Interval(0.0, math.pi / 2))
    assert s.lb == pytest.approx(0.0, abs=1e-12)
    assert s.ub == pytest.approx(1.0)

    full = IF.sin(Interval(0.0, 2 * math.pi))
    assert full.lb == pytest.approx(-1.0)
    assert full.ub == pytest.approx(1.0)

    assert IF.sin(math.pi / 2) == pytest.approx(1.0)


def test_cos_includes_extrema_inside_interval():
    c = IF.cos(Interval(0.0, math.pi))
    assert c.lb == pytest.approx(-1.0)
    assert c.ub == pytest.approx(1.0)

    c = IF.cos(Interval(0.0, 1.0))
    assert c.lb == pytest.approx(math.cos(1.0))
    assert c.ub == pytest.approx(1.0)


def test_abs_includes_zero_when_interval_straddles_zero():
    a = IF.abs(Interval(-2.0, 3.0))
    assert a.lb == pytest.approx(0.0)
    assert a.ub == pytest.approx(3.0)

    assert IF.abs(-1.5) == pytest.approx(1.5)


def test_sqrt_and_log_domain_handling():
    s = IF.sqrt(Interval(4.0, 9.0))
    assert (s.lb, s.ub) == (pytest.approx(2.0), pytest.approx(3.0))

    s = IF.sqrt(Interval(-1.0, 4.0))
    assert s.lb == pytest.approx(0.0)
    assert s.ub == pytest.approx(2.0)

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        IF.sqrt(Interval(-3.0, -1.0))

    lg = IF.log(Interval(1.0, math.e))
    assert lg.lb == pytest.approx(0.0, abs=1e-12)
    assert lg.ub == pytest.approx(1.0)

    lg = IF.log(Interval(-1.0, 1.0))
    assert math.isinf(lg.lb) and lg.lb < 0
    assert lg.ub == pytest.approx(0.0)

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        IF.log(Interval(-1.0, 0.0))


def test_split_and_bisect_partition_the_interval():
    parts = IF.split(Interval(0.0, 4.0), [1.0, 1.0, 2.0])
    bounds = [(p.lb, p.ub) for p in parts]
    expected = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)]
    assert len(bounds) == 3
    for (lb, ub), (elb, eub) in zip(bounds, expected):
        assert lb == pytest.approx(elb)
        assert ub == pytest.approx(eub)

    halves = IF.bisect(Interval(0.0, 2.0))
    assert len(halves) == 2
    assert halves[0].lb == pytest.approx(0.0) and halves[0].ub == pytest.approx(1.0)
    assert halves[1].lb == pytest.approx(1.0) and halves[1].ub == pytest.approx(2.0)

    assert IF.split(3.0, [1.0, 1.0]) == [3.0, 3.0]


def test_constrain_and_shrink():
    c = IF.constrain(Interval(-5.0, 5.0), 0.0, 3.0)
    assert (c.lb, c.ub) == (pytest.approx(0.0), pytest.approx(3.0))
    assert IF.constrain(7.0, 0.0, 3.0) == pytest.approx(3.0)

    s = IF.shrink(Interval(0.0, 4.0), 0.5)
    assert (s.lb, s.ub) == (pytest.approx(1.0), pytest.approx(3.0))
    assert IF.shrink(3.0, 0.5) == pytest.approx(3.0)


def test_hausdorff_distance_known_values_and_symmetry():
    assert IF.hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == pytest.approx(0.0)
    assert IF.hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == pytest.approx(2.0)
    assert IF.hausdorff_distance(Interval(0.0, 4.0), Interval(1.0, 2.0)) == pytest.approx(2.0)
    assert IF.hausdorff_distance(Interval(0.0, 1.0), Interval(0.5, 0.75)) == pytest.approx(0.5)
    assert IF.hausdorff_distance(1.0, 3.0) == pytest.approx(2.0)

    pairs = [
        (Interval(0.0, 1.0), Interval(2.0, 3.0)),
        (Interval(0.0, 4.0), Interval(1.0, 2.0)),
        (Interval(-1.0, 0.5), Interval(0.0, 0.25)),
    ]
    for a, b in pairs:
        assert IF.hausdorff_distance(a, b) == pytest.approx(IF.hausdorff_distance(b, a))


# ---------------------------------------------------------------------------
# RS_SubinclusionFunction (uses numpy.random)
# ---------------------------------------------------------------------------

def test_rs_subinclusion_degenerate_box_gives_exact_value():
    np.random.seed(0)
    f = lambda x: 2 * x[0] + x[1]
    rs = RS_SubinclusionFunction(f, 20)
    result = rs(Box(Interval(1.0, 1.0), Interval(3.0, 3.0)))
    assert isinstance(result, Interval)
    assert result.lb == pytest.approx(5.0)
    assert result.ub == pytest.approx(5.0)


def test_rs_subinclusion_stays_within_true_range_and_is_reproducible():
    np.random.seed(42)
    f = lambda x: x[0] + x[1]
    box = Box(Interval(1.0, 2.0), Interval(-1.0, 0.0))
    rs = RS_SubinclusionFunction(f, 50)

    # Every possible sample lies in [0, 2], so these hold for any random outcome.
    tol = 1e-9
    first = rs(box)
    assert first.lb <= first.ub
    assert first.lb >= 0.0 - tol
    assert first.ub <= 2.0 + tol

    # Same seed, same result.
    np.random.seed(42)
    second = rs(box)
    assert first.lb == pytest.approx(second.lb)
    assert first.ub == pytest.approx(second.ub)
