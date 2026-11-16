import math

import numpy as np
import pytest

from intervallum import interval_functions as ifn
from intervallum.box import Box
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------

def test_interval_stores_bounds():
    i = Interval(1.0, 3.0)
    assert i.lb == 1.0
    assert i.ub == 3.0
    assert i.width == 2.0
    assert i.middle == 2.0


def test_interval_rejects_wrong_bounds():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(3.0, 1.0)


def test_interval_fix_swaps_wrong_bounds():
    i = Interval(3.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 3.0


def test_from_point_has_zero_width():
    i = Interval.from_point(2.5)
    assert i.lb == 2.5
    assert i.ub == 2.5
    assert i.width == 0.0


def test_bound_setters_reject_crossing_values():
    i = Interval(0.0, 1.0)
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 2.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = -1.0
    i.lb = 0.5
    i.ub = 2.0
    assert (i.lb, i.ub) == (0.5, 2.0)


def test_add_and_subtract_intervals():
    s = Interval(1.0, 2.0) + Interval(3.0, 4.0)
    assert (s.lb, s.ub) == (4.0, 6.0)

    d = Interval(3.0, 5.0) - Interval(1.0, 2.0)
    assert (d.lb, d.ub) == (1.0, 4.0)


def test_add_scalar_and_negation():
    s = Interval(1.0, 2.0) + 1.0
    assert (s.lb, s.ub) == (2.0, 3.0)

    n = -Interval(1.0, 2.0)
    assert (n.lb, n.ub) == (-2.0, -1.0)


def test_multiplication_with_mixed_signs():
    p = Interval(-1.0, 2.0) * Interval(-3.0, 4.0)
    assert (p.lb, p.ub) == (-6.0, 8.0)


def test_division_by_interval():
    q = Interval(1.0, 2.0) / Interval(2.0, 4.0)
    assert q.lb == pytest.approx(0.25)
    assert q.ub == pytest.approx(1.0)


def test_even_power_containing_zero_has_zero_lower_bound():
    p = Interval(-2.0, 1.0) ** 2
    assert (p.lb, p.ub) == (0.0, 4.0)


def test_odd_power_keeps_sign():
    p = Interval(-2.0, 3.0) ** 3
    assert (p.lb, p.ub) == (-8.0, 27.0)


def test_equality_uses_tolerance_for_intervals_and_floats():
    assert Interval(0.0, 1.0) == Interval(0.0, 1.0)
    assert not (Interval(0.0, 1.0) == Interval(2.0, 3.0))
    assert not (Interval(1.0, 2.0) == 1.0)
    assert Interval(0.0, 1.0) != Interval(2.0, 3.0)


def test_equality_with_unsupported_type_raises():
    with pytest.raises(NotImplementedError):
        Interval(0.0, 1.0) == 1  # int is not float or Interval


def test_zero_width_result_reduces_to_float():
    r = Interval.from_point(1.0) + 0.0
    assert isinstance(r, float)
    assert r == 1.0


def test_reduction_can_be_disabled(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    r = Interval.from_point(1.0) + 0.0
    assert isinstance(r, Interval)
    assert (r.lb, r.ub) == (1.0, 1.0)


def test_comparisons_use_lower_bounds():
    a = Interval(1.0, 5.0)
    b = Interval(2.0, 3.0)
    assert a < b
    assert b > a
    assert a <= Interval(1.0, 0.0 + 1.0)
    assert b >= 2.0
    assert a < 1.5


def test_copy_is_independent_and_str_format():
    import copy as copy_module

    original = Interval(0.0, 2.0)
    clone = copy_module.copy(original)
    assert clone is not original
    assert (clone.lb, clone.ub) == (0.0, 2.0)
    assert str(original) == "[0.0; 2.0]"


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

def test_box_bounds_and_middle_with_mixed_entries():
    b = Box(Interval(0.0, 2.0), 4.0)
    assert np.array_equal(b.lb, np.array([0.0, 4.0]))
    assert np.array_equal(b.ub, np.array([2.0, 4.0]))
    assert np.array_equal(b.middle, np.array([1.0, 4.0]))


def test_box_width_reports_indices_of_maximum_width():
    widths, max_width = Box(Interval(0.0, 1.0), Interval(0.0, 3.0)).width
    assert widths == [1]
    assert max_width == 3.0


def test_box_width_of_degenerate_box_lists_all_indices():
    widths, max_width = Box(1.0, 2.0).width
    assert widths == [0, 1]
    assert max_width == 0


def test_box_str_and_repr_join_components():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    assert str(b) == "[0.0; 1.0] x [2.0; 3.0]"
    assert repr(b) == "[0.0; 1.0] x [2.0; 3.0]"


def test_box_equality_against_arrays():
    b = Box(1.0, 2.0)
    assert b == np.array([1.0, 2.0])
    assert b != Box(1.0, 3.0)
    assert not (b == np.array([1.0, 2.5]))
    with pytest.raises(NotImplementedError):
        b == [1.0, 2.0]


def test_box_try_to_reduce_degenerate_returns_middle():
    reduced = Box(1.0, 2.0)._try_to_reduce()
    assert np.array_equal(reduced, np.array([1.0, 2.0]))

    kept = Box(Interval(0.0, 1.0), 2.0)._try_to_reduce()
    assert isinstance(kept, Box)


# ---------------------------------------------------------------------------
# RS_SubinclusionFunction
# ---------------------------------------------------------------------------

def test_rs_result_is_enclosed_by_box_range():
    np.random.seed(0)
    rs = RS_SubinclusionFunction(lambda x: x[0] + x[1], number_of_samples=200)
    box = Box(Interval(0.0, 1.0), Interval(0.0, 1.0))
    result = rs(box)
    assert isinstance(result, Interval)
    assert result.lb <= result.ub
    assert result.lb >= 0.0
    assert result.ub <= 2.0


def test_rs_on_degenerate_box_gives_exact_value():
    rs = RS_SubinclusionFunction(lambda x: x[0] * x[1], number_of_samples=5)
    result = rs(Box(2.0, 3.0))
    assert result.lb == pytest.approx(6.0)
    assert result.ub == pytest.approx(6.0)


def test_rs_evaluates_function_number_of_samples_times():
    calls = []

    def f(x):
        calls.append(x)
        return float(x[0])

    rs = RS_SubinclusionFunction(f, number_of_samples=17)
    rs(Box(Interval(0.0, 1.0), Interval(0.0, 1.0)))
    assert len(calls) == 17


def test_rs_is_reproducible_with_same_seed():
    box = Box(Interval(0.0, 1.0), Interval(-1.0, 1.0))
    rs = RS_SubinclusionFunction(lambda x: x[0] - x[1], number_of_samples=50)

    np.random.seed(123)
    first = rs(box)
    np.random.seed(123)
    second = rs(box)
    assert first.lb == second.lb
    assert first.ub == second.ub


# ---------------------------------------------------------------------------
# hausdorff_distance
# ---------------------------------------------------------------------------

def test_hausdorff_identical_intervals_is_zero():
    assert ifn.hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == 0.0


def test_hausdorff_disjoint_intervals():
    assert ifn.hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == pytest.approx(2.0)


def test_hausdorff_is_symmetric():
    a = Interval(0.0, 4.0)
    b = Interval(1.0, 2.0)
    assert ifn.hausdorff_distance(a, b) == ifn.hausdorff_distance(b, a)
    assert ifn.hausdorff_distance(a, b) == pytest.approx(2.0)


def test_hausdorff_with_plain_numbers():
    assert ifn.hausdorff_distance(3.0, 5.0) == pytest.approx(2.0)
    assert ifn.hausdorff_distance(1.0, Interval(0.0, 3.0)) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# interval_functions (selected)
# ---------------------------------------------------------------------------

def test_abs_of_interval_crossing_zero():
    r = ifn.abs(Interval(-2.0, 1.0))
    assert (r.lb, r.ub) == (0.0, 2.0)


def test_sqrt_of_negative_domain_raises():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.sqrt(Interval(-2.0, -1.0))


def test_split_interval_into_equal_parts():
    parts = ifn.split(Interval(0.0, 4.0), [1.0, 1.0])
    assert len(parts) == 2
    assert (parts[0].lb, parts[0].ub) == (0.0, 2.0)
    assert (parts[1].lb, parts[1].ub) == (2.0, 4.0)


def test_constrain_and_shrink():
    c = ifn.constrain(Interval(-5.0, 5.0), -1.0, 2.0)
    assert (c.lb, c.ub) == (-1.0, 2.0)

    s = ifn.shrink(Interval(0.0, 4.0), 0.5)
    assert (s.lb, s.ub) == (1.0, 3.0)
