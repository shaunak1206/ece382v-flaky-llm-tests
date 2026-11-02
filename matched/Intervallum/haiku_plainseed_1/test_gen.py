import math

import numpy as np
import pytest

from intervallum import interval_functions as F
from intervallum.box import Box
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.interval_functions import hausdorff_distance


def check_interval(value, lb, ub):
    assert isinstance(value, Interval)
    assert value.lb == pytest.approx(lb, abs=1e-12)
    assert value.ub == pytest.approx(ub, abs=1e-12)


def test_box_of_floats_exposes_bounds_and_middle():
    b = Box(1.0, 3.0)
    assert isinstance(b, Box)
    assert np.array_equal(b.lb, [1.0, 3.0])
    assert np.array_equal(b.ub, [1.0, 3.0])
    assert np.array_equal(b.middle, [1.0, 3.0])

    mixed = Box(Interval(0.0, 2.0), 5.0)
    assert np.array_equal(mixed.lb, [0.0, 5.0])
    assert np.array_equal(mixed.ub, [2.0, 5.0])
    assert np.array_equal(mixed.middle, [1.0, 5.0])


def test_box_str_joins_components_with_x():
    assert str(Box(1.0, 2.5)) == "1.0 x 2.5"
    assert str(Box(Interval(0.0, 1.0), 2.0)) == "[0.0; 1.0] x 2.0"


def test_box_eq_compares_values_and_rejects_other_types():
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == [1.0, 2.0]


def test_box_width_reports_widest_components():
    indices, widest = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), 5.0).width
    assert indices == [1]
    assert widest == 3.0

    indices, widest = Box(Interval(0.0, 2.0), Interval(1.0, 3.0)).width
    assert indices == [0, 1]
    assert widest == 2.0


def test_box_try_to_reduce_collapses_degenerate_box():
    collapsed = Box(Interval(1.0, 1.0), 2.0)._try_to_reduce()
    assert not isinstance(collapsed, Box)
    assert np.array_equal(collapsed, [1.0, 2.0])

    kept = Box(Interval(0.0, 1.0), 2.0)._try_to_reduce()
    assert isinstance(kept, Box)
    assert np.array_equal(kept.lb, [0.0, 2.0])
    assert np.array_equal(kept.ub, [1.0, 2.0])


def test_reversed_bounds_raise_unless_fixed():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)
    swapped = Interval(2.0, 1.0, fix=True)
    assert swapped.lb == 1.0
    assert swapped.ub == 2.0


def test_from_point_str_and_repr():
    p = Interval.from_point(2.0)
    assert p.lb == 2.0 and p.ub == 2.0
    assert str(p) == "[2.0; 2.0]"
    assert repr(p) == str(p)
    assert str(Interval(-1.5, 0.25)) == "[-1.5; 0.25]"


def test_middle_width_and_setters_enforce_order():
    i = Interval(0.0, 4.0)
    assert i.middle == 2.0
    assert i.width == 4.0

    i.lb = 1.0
    assert i.lb == 1.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 5.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.5
    assert i.lb == 1.0
    assert i.ub == 4.0


def test_add_neg_sub_and_inner_subtraction():
    a = Interval(1.0, 2.0)
    b = Interval(3.0, 4.0)
    check_interval(a + b, 4.0, 6.0)
    check_interval(a + 1.0, 2.0, 3.0)
    check_interval(1.0 + a, 2.0, 3.0)
    check_interval(-a, -2.0, -1.0)
    check_interval(Interval(3.0, 5.0) - a, 1.0, 4.0)
    check_interval(10.0 - a, 8.0, 9.0)
    check_interval(Interval(1.0, 3.0) << Interval(0.0, 1.0), 1.0, 2.0)
    # inner subtraction swaps the bounds when the result is inverted
    check_interval(Interval(0.0, 1.0) << Interval(-1.0, 5.0), -4.0, 1.0)


def test_multiplication_takes_extreme_products():
    check_interval(Interval(-1.0, 2.0) * Interval(3.0, 4.0), -4.0, 8.0)
    check_interval(Interval(1.0, 2.0) * 2.0, 2.0, 4.0)
    check_interval(2.0 * Interval(1.0, 2.0), 2.0, 4.0)
    check_interval(Interval(-2.0, -1.0) * Interval(-3.0, 1.0), -2.0, 6.0)


def test_inversion_cases():
    check_interval(~Interval(2.0, 4.0), 0.25, 0.5)
    check_interval(~Interval(-4.0, -2.0), -0.5, -0.25)

    inv = ~Interval(0.0, 2.0)
    assert inv.lb == pytest.approx(0.5)
    assert inv.ub == math.inf

    inv = ~Interval(-2.0, 0.0)
    assert inv.lb == -math.inf
    assert inv.ub == pytest.approx(-0.5)

    inv = ~Interval(-1.0, 1.0)
    assert inv.lb == -math.inf
    assert inv.ub == math.inf


def test_division_and_reverse_division():
    check_interval(Interval(2.0, 4.0) / Interval(2.0, 4.0), 0.5, 2.0)
    check_interval(Interval(1.0, 2.0) / 2.0, 0.5, 1.0)
    check_interval(1.0 / Interval(2.0, 4.0), 0.25, 0.5)


def test_power_even_and_odd():
    check_interval(Interval(-2.0, 3.0) ** 2, 0.0, 9.0)
    check_interval(Interval(-3.0, -1.0) ** 2, 1.0, 9.0)
    check_interval(Interval(-2.0, 3.0) ** 3, -8.0, 27.0)
    check_interval(Interval(2.0, 3.0) ** 3, 8.0, 27.0)


def test_degenerate_results_reduce_to_float(monkeypatch):
    collapsed = Interval(1.0, 1.0) + 1.0
    assert isinstance(collapsed, float)
    assert collapsed == 2.0
    assert isinstance(Interval(1.0, 2.0) + 1.0, Interval)

    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    kept = Interval(1.0, 1.0) + 1.0
    check_interval(kept, 2.0, 2.0)


def test_equality_uses_admissible_error():
    assert Interval(1.0, 1.0) == 1.0
    assert not (Interval(1.0, 2.0) == 1.0)
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0)
    assert Interval(1.0, 2.0) != Interval(1.0, 3.0)
    assert Interval(0.0, 1.0) == Interval(0.0, 1.0 + 1e-9)
    assert Interval(0.0, 1.0) != Interval(0.0, 1.5)


def test_ordering_compares_lower_bounds():
    assert Interval(1.0, 5.0) < Interval(2.0, 3.0)
    assert Interval(1.0, 5.0) < 2.0
    assert Interval(3.0, 4.0) >= 3.0
    assert Interval(3.0, 4.0) > Interval(2.0, 9.0)
    assert Interval(2.0, 9.0) <= Interval(2.0, 3.0)


def test_sin_and_cos_extrema():
    check_interval(F.sin(Interval(0.0, math.pi / 2)), 0.0, 1.0)
    check_interval(F.sin(Interval(0.0, 2 * math.pi)), -1.0, 1.0)
    check_interval(F.cos(Interval(0.0, math.pi)), -1.0, 1.0)
    check_interval(F.cos(Interval(0.0, 1.0)), math.cos(1.0), 1.0)
    assert isinstance(F.sin(0.0), float)
    assert F.sin(0.0) == pytest.approx(0.0, abs=1e-12)


def test_abs_exp_sqrt_log_ranges():
    check_interval(F.abs(Interval(-2.0, 3.0)), 0.0, 3.0)
    check_interval(F.abs(Interval(-3.0, -1.0)), 1.0, 3.0)
    check_interval(F.exp(Interval(0.0, 1.0)), 1.0, math.e)
    check_interval(F.sqrt(Interval(4.0, 9.0)), 2.0, 3.0)
    check_interval(F.sqrt(Interval(-1.0, 4.0)), 0.0, 2.0)
    check_interval(F.log(Interval(1.0, math.e)), 0.0, 1.0)


def test_sqrt_and_log_raise_outside_domain():
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        F.sqrt(Interval(-3.0, -1.0))
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        F.log(Interval(-2.0, -1.0))


def test_bisect_and_split_pieces():
    left, right = F.bisect(Interval(0.0, 4.0))
    check_interval(left, 0.0, 2.0)
    check_interval(right, 2.0, 4.0)

    # Two-piece splits only: split() rebinds its loop variable, so
    # three or more pieces are not exercised here.
    pieces = F.split(Interval(0.0, 2.0), [1.0, 3.0])
    assert len(pieces) == 2
    check_interval(pieces[0], 0.0, 0.5)
    check_interval(pieces[1], 0.5, 2.0)

    # Real numbers pass through unchanged, one copy per ratio.
    assert F.split(2.0, [1.0, 1.0, 1.0]) == [2.0, 2.0, 2.0]

    # Pieces narrower than the reduction width collapse to floats.
    tiny = F.split(Interval(0.0, 1e-6), [1.0, 1.0])
    assert all(isinstance(v, float) for v in tiny)
    assert tiny[0] == pytest.approx(2.5e-7, abs=1e-12)
    assert tiny[1] == pytest.approx(7.5e-7, abs=1e-12)


def test_constrain_and_shrink():
    check_interval(F.constrain(Interval(-1.0, 5.0), 0.0, 2.0), 0.0, 2.0)
    check_interval(F.constrain(Interval(-3.0, -2.0), 0.0, 2.0), 0.0, 0.0)
    assert F.constrain(7.0, 0.0, 2.0) == 2.0
    assert F.constrain(-1.0, 0.0, 2.0) == 0.0

    check_interval(F.shrink(Interval(0.0, 4.0), 0.5), 1.0, 3.0)
    check_interval(F.shrink(Interval(0.0, 4.0), 0.0), 2.0, 2.0)
    assert F.shrink(3.0, 0.5) == 3.0


def test_rs_subinclusion_degenerate_box_is_exact():
    f = RS_SubinclusionFunction(lambda p: p[0] + p[1], number_of_samples=5)
    check_interval(f(Box(1.0, 2.0)), 3.0, 3.0)


def test_rs_samples_stay_inside_box(monkeypatch):
    calls = []

    def midpoint_uniform(low, high):
        calls.append((np.array(low, copy=True), np.array(high, copy=True)))
        return 0.5 * (low + high)

    monkeypatch.setattr(np.random, "uniform", midpoint_uniform)
    box = Box(Interval(0.0, 1.0), Interval(2.0, 5.0))
    f = RS_SubinclusionFunction(lambda p: p[0] + p[1], number_of_samples=7)
    result = f(box)

    assert len(calls) == 7
    for low, high in calls:
        assert np.array_equal(low, [0.0, 2.0])
        assert np.array_equal(high, [1.0, 5.0])
    check_interval(result, 4.0, 4.0)

    # Real sampling on a box whose true range of p0 * p1 is [1, 2] at most.
    monkeypatch.undo()
    np.random.seed(2024)
    box = Box(Interval(0.0, 1.0), Interval(1.0, 2.0))
    f = RS_SubinclusionFunction(lambda p: p[0] * p[1], number_of_samples=2000)
    result = f(box)
    assert 0.0 <= result.lb < 0.05
    assert 1.5 < result.ub <= 2.0


def test_hausdorff_distance_cases():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == pytest.approx(0.0)
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 2.0)) == pytest.approx(1.0)
    assert hausdorff_distance(Interval(0.0, 2.0), Interval(0.0, 1.0)) == pytest.approx(1.0)
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == pytest.approx(2.0)
    assert hausdorff_distance(1.0, 3.0) == pytest.approx(2.0)
    assert hausdorff_distance(0.5, Interval(0.0, 1.0)) == pytest.approx(0.5)
    assert hausdorff_distance(Interval(0.0, 2.0), 1.0) == pytest.approx(1.0)
