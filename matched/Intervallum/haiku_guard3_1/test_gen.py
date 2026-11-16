import math
from copy import copy

import numpy as np
import pytest

from intervallum import interval_functions as F
from intervallum.box import Box
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

def test_box_bounds_middle_and_width_with_intervals_and_floats():
    b = Box(Interval(0.0, 2.0), Interval(-1.0, 1.0))
    assert np.array_equal(b.lb, np.array([0.0, -1.0]))
    assert np.array_equal(b.ub, np.array([2.0, 1.0]))
    assert np.array_equal(b.middle, np.array([1.0, 0.0]))

    widest = Box(Interval(0.0, 2.0), Interval(-1.0, 0.5))
    indices, max_width = widest.width
    assert indices == [0]
    assert max_width == pytest.approx(2.0)

    mixed = Box(Interval(0.0, 2.0), 5.0)
    assert np.array_equal(mixed.lb, np.array([0.0, 5.0]))
    assert np.array_equal(mixed.ub, np.array([2.0, 5.0]))

    point_box = Box(1.0, 2.0)
    indices, max_width = point_box.width
    assert indices == [0, 1]
    assert max_width == 0


def test_box_width_reports_all_tied_indices():
    b = Box(Interval(0.0, 1.0), Interval(5.0, 6.0), 3.0)
    indices, max_width = b.width
    assert indices == [0, 1]
    assert max_width == pytest.approx(1.0)


def test_box_equality_and_inequality_and_type_check():
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    assert Box(1.0, 2.0) != Box(1.0, 2.5)
    assert not (Box(1.0, 2.0) == Box(1.0, 2.5))
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0).__eq__([1.0, 2.0])


def test_box_str_joins_component_strings():
    b = Box(Interval(0.0, 1.0), 2.0)
    assert str(b) == "[0.0; 1.0] x 2.0"


def test_box_copy_is_independent_of_original():
    original = Box(Interval(0.0, 1.0), 2.0)
    duplicate = copy(original)
    assert duplicate is not original
    assert isinstance(duplicate, Box)
    assert str(duplicate) == str(original)

    duplicate[0].lb = 0.5
    assert duplicate[0].lb == 0.5
    assert original[0].lb == 0.0


def test_box_try_to_reduce_collapses_degenerate_and_keeps_regular_boxes():
    degenerate = Box(Interval(1.0, 1.0), 2.0)
    reduced = degenerate._try_to_reduce()
    assert np.array_equal(reduced, np.array([1.0, 2.0]))

    regular = Box(Interval(0.0, 1.0), 2.0)
    kept = regular._try_to_reduce()
    assert isinstance(kept, Box)
    assert kept.width[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------

def test_interval_bounds_derived_values_from_point_and_wrong_bounds_fix():
    i = Interval(1.0, 3.0)
    assert i.lb == 1.0
    assert i.ub == 3.0
    assert i.middle == pytest.approx(2.0)
    assert i.width == pytest.approx(2.0)

    p = Interval.from_point(4.0)
    assert p.lb == 4.0
    assert p.ub == 4.0
    assert p.width == 0.0

    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(3.0, 1.0)

    fixed = Interval(3.0, 1.0, fix=True)
    assert fixed.lb == 1.0
    assert fixed.ub == 3.0


def test_interval_addition_with_interval_and_scalar():
    a = Interval(1.0, 2.0)
    s = a + Interval(3.0, 4.0)
    assert s.lb == pytest.approx(4.0)
    assert s.ub == pytest.approx(6.0)

    t = a + 1.0
    assert t.lb == pytest.approx(2.0)
    assert t.ub == pytest.approx(3.0)

    r = 1.0 + a
    assert r.lb == pytest.approx(2.0)
    assert r.ub == pytest.approx(3.0)


def test_interval_multiplication_with_mixed_signs():
    p = Interval(-1.0, 2.0) * Interval(3.0, 4.0)
    assert p.lb == pytest.approx(-4.0)
    assert p.ub == pytest.approx(8.0)

    q = Interval(-1.0, 2.0) * 2.0
    assert q.lb == pytest.approx(-2.0)
    assert q.ub == pytest.approx(4.0)


def test_interval_subtraction_negation_and_reflected_subtraction():
    d = Interval(1.0, 2.0) - Interval(0.0, 1.0)
    assert d.lb == pytest.approx(0.0)
    assert d.ub == pytest.approx(2.0)

    n = -Interval(1.0, 2.0)
    assert n.lb == pytest.approx(-2.0)
    assert n.ub == pytest.approx(-1.0)

    rs = 5.0 - Interval(1.0, 2.0)
    assert rs.lb == pytest.approx(3.0)
    assert rs.ub == pytest.approx(4.0)


def test_interval_inversion_including_zero_cases():
    inv = ~Interval(2.0, 4.0)
    assert inv.lb == pytest.approx(0.25)
    assert inv.ub == pytest.approx(0.5)

    from_zero = ~Interval(0.0, 2.0)
    assert from_zero.lb == pytest.approx(0.5)
    assert math.isinf(from_zero.ub) and from_zero.ub > 0

    across_zero = ~Interval(-1.0, 1.0)
    assert math.isinf(across_zero.lb) and across_zero.lb < 0
    assert math.isinf(across_zero.ub) and across_zero.ub > 0


def test_interval_even_and_odd_powers():
    even = Interval(-2.0, 3.0) ** 2
    assert even.lb == pytest.approx(0.0)
    assert even.ub == pytest.approx(9.0)

    odd = Interval(-2.0, 3.0) ** 3
    assert odd.lb == pytest.approx(-8.0)
    assert odd.ub == pytest.approx(27.0)

    positive = Interval(1.0, 2.0) ** 2
    assert positive.lb == pytest.approx(1.0)
    assert positive.ub == pytest.approx(4.0)


def test_interval_equality_uses_tolerance_and_rejects_int():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) == Interval(1.0 + 1e-9, 2.0 + 1e-9)
    assert not (Interval(1.0, 2.0) == Interval(1.5, 2.5))
    assert Interval(2.0, 2.0) == 2.0
    assert not (Interval(1.0, 2.0) == 1.0)
    with pytest.raises(NotImplementedError):
        Interval(1.0, 2.0).__eq__(2)


def test_interval_narrow_results_reduce_to_numbers_unless_disabled(monkeypatch):
    narrow = Interval(1.0, 1.0 + 1e-6) + 0.0
    assert isinstance(narrow, float)
    assert narrow == pytest.approx(1.0 + 5e-7, abs=1e-12)

    wide = Interval(1.0, 2.0) + 0.0
    assert isinstance(wide, Interval)

    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    kept = Interval(1.0, 1.0) + 0.0
    assert isinstance(kept, Interval)
    assert kept.lb == pytest.approx(1.0)
    assert kept.ub == pytest.approx(1.0)


def test_interval_bound_setters_reject_crossing_bounds():
    i = Interval(0.0, 1.0)
    i.lb = 0.5
    assert i.lb == 0.5
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 2.0
    assert i.lb == 0.5
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = -1.0
    assert i.ub == 1.0


def test_interval_comparisons_use_lower_bounds():
    a = Interval(1.0, 5.0)
    b = Interval(2.0, 3.0)
    assert a < b
    assert b > a
    assert a <= b
    assert b >= a
    assert a < 2.0
    assert a >= 1.0
    assert not (a > 1.0)


# ---------------------------------------------------------------------------
# RS_SubinclusionFunction (uses numpy's global RNG; seeded in each test)
# ---------------------------------------------------------------------------

def test_rs_subinclusion_constant_function_gives_degenerate_interval():
    state = np.random.get_state()
    try:
        np.random.seed(2024)
        rs = RS_SubinclusionFunction(lambda x: 3.0, number_of_samples=5)
        box = Box(Interval(0.0, 1.0), Interval(0.0, 1.0))
        result = rs(box)
        assert isinstance(result, Interval)
        assert result.lb == pytest.approx(3.0)
        assert result.ub == pytest.approx(3.0)
    finally:
        np.random.set_state(state)


def test_rs_subinclusion_result_is_within_true_range_for_any_sample():
    state = np.random.get_state()
    try:
        np.random.seed(7)
        rs = RS_SubinclusionFunction(lambda x: x[0], number_of_samples=200)
        box = Box(Interval(0.0, 1.0), Interval(-2.0, 2.0))
        result = rs(box)
        assert isinstance(result, Interval)
        # Holds for every possible random outcome: samples lie in [0, 1).
        assert 0.0 <= result.lb <= result.ub <= 1.0
    finally:
        np.random.set_state(state)


# ---------------------------------------------------------------------------
# interval_functions
# ---------------------------------------------------------------------------

def test_sin_on_float_and_on_intervals():
    assert F.sin(0.0) == pytest.approx(0.0, abs=1e-12)
    assert F.sin(math.pi / 2) == pytest.approx(1.0)

    r = F.sin(Interval(1.0, 2.0))
    assert isinstance(r, Interval)
    assert r.lb == pytest.approx(math.sin(1.0))
    assert r.ub == pytest.approx(1.0)

    rising = F.sin(Interval(0.0, math.pi / 2))
    assert rising.lb == pytest.approx(0.0, abs=1e-12)
    assert rising.ub == pytest.approx(1.0)

    wide = F.sin(Interval(0.0, 2 * math.pi))
    assert wide.lb == pytest.approx(-1.0)
    assert wide.ub == pytest.approx(1.0)


def test_abs_across_zero_and_on_one_side():
    across = F.abs(Interval(-2.0, 3.0))
    assert across.lb == pytest.approx(0.0)
    assert across.ub == pytest.approx(3.0)

    negative = F.abs(Interval(-3.0, -1.0))
    assert negative.lb == pytest.approx(1.0)
    assert negative.ub == pytest.approx(3.0)

    assert F.abs(-2.5) == pytest.approx(2.5)


def test_sqrt_clips_negative_part_and_rejects_fully_negative():
    r = F.sqrt(Interval(4.0, 9.0))
    assert r.lb == pytest.approx(2.0)
    assert r.ub == pytest.approx(3.0)

    clipped = F.sqrt(Interval(-1.0, 4.0))
    assert clipped.lb == pytest.approx(0.0)
    assert clipped.ub == pytest.approx(2.0)

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        F.sqrt(Interval(-2.0, -1.0))


def test_exp_and_log_including_undefined_and_zero_points():
    e_ = F.exp(Interval(0.0, 1.0))
    assert e_.lb == pytest.approx(1.0)
    assert e_.ub == pytest.approx(math.e)

    l = F.log(Interval(-1.0, 1.0))
    assert math.isinf(l.lb) and l.lb < 0
    assert l.ub == pytest.approx(0.0)

    positive = F.log(Interval(1.0, math.e))
    assert positive.lb == pytest.approx(0.0)
    assert positive.ub == pytest.approx(1.0)

    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        F.log(Interval(-2.0, 0.0))


def test_bisect_split_constrain_and_shrink():
    # Only two-part splits are exercised here.
    halves = F.bisect(Interval(0.0, 4.0))
    assert len(halves) == 2
    assert halves[0].lb == pytest.approx(0.0)
    assert halves[0].ub == pytest.approx(2.0)
    assert halves[1].lb == pytest.approx(2.0)
    assert halves[1].ub == pytest.approx(4.0)

    uneven = F.split(Interval(0.0, 4.0), [1.0, 3.0])
    assert uneven[0].lb == pytest.approx(0.0)
    assert uneven[0].ub == pytest.approx(1.0)
    assert uneven[1].lb == pytest.approx(1.0)
    assert uneven[1].ub == pytest.approx(4.0)

    assert F.split(2.0, [1.0, 1.0]) == [2.0, 2.0]

    clipped = F.constrain(Interval(-1.0, 5.0), 0.0, 3.0)
    assert clipped.lb == pytest.approx(0.0)
    assert clipped.ub == pytest.approx(3.0)
    assert F.constrain(-1.0, 0.0, 3.0) == 0.0

    shrunk = F.shrink(Interval(0.0, 4.0), 0.5)
    assert shrunk.lb == pytest.approx(1.0)
    assert shrunk.ub == pytest.approx(3.0)
    assert F.shrink(2.0, 0.5) == 2.0


def test_hausdorff_distance_between_points_and_intervals():
    assert F.hausdorff_distance(1.0, 3.0) == pytest.approx(2.0)
    assert F.hausdorff_distance(3.0, 1.0) == pytest.approx(2.0)

    overlapping = F.hausdorff_distance(Interval(0.0, 1.0), Interval(0.5, 2.0))
    assert overlapping == pytest.approx(1.0)
    assert F.hausdorff_distance(Interval(0.5, 2.0), Interval(0.0, 1.0)) == pytest.approx(1.0)

    nested = F.hausdorff_distance(Interval(0.0, 10.0), Interval(4.0, 6.0))
    assert nested == pytest.approx(4.0)

    assert F.hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == pytest.approx(0.0)
