import math
from copy import copy

import numpy as np
import pytest

import intervallum.interval_functions as ifn
from intervallum.box import Box
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.interval_functions import hausdorff_distance


def bounds(i):
    return i.lb, i.ub


@pytest.fixture
def no_reduction(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)


# ---------------------------------------------------------------- Interval

def test_interval_construction_and_derived_properties():
    i = Interval(1.0, 3.0)
    assert bounds(i) == (1.0, 3.0)
    assert i.middle == 2.0
    assert i.width == 2.0

    p = Interval.from_point(3.0)
    assert isinstance(p, Interval)
    assert bounds(p) == (3.0, 3.0)

    assert str(Interval(0.0, 1.0)) == "[0.0; 1.0]"
    assert repr(Interval(0.0, 1.0)) == "[0.0; 1.0]"


def test_interval_wrong_bounds_raise_and_fix_swaps():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)

    fixed = Interval(2.0, 1.0, fix=True)
    assert bounds(fixed) == (1.0, 2.0)


def test_interval_bound_setters_validate_order():
    i = Interval(1.0, 2.0)
    i.lb = 1.5
    assert i.lb == 1.5
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 1.0


def test_copy_returns_independent_interval():
    i = Interval(0.0, 1.0)
    j = copy(i)
    assert j is not i
    j.ub = 5.0
    assert bounds(i) == (0.0, 1.0)


def test_equality_with_interval_uses_admissible_error():
    assert Interval(1.0, 2.0) == Interval(1.0, 2.0 + 1e-9)
    assert Interval(0.0, 1.0) != Interval(0.0, 2.0)
    assert not (Interval(0.0, 1.0) == Interval(0.0, 2.0))


def test_equality_with_float_and_unsupported_type():
    assert Interval.from_point(2.0) == 2.0
    assert Interval(2.0, 3.0) != 2.0
    with pytest.raises(NotImplementedError):
        _ = Interval(0.0, 1.0) == 1


def test_addition_subtraction_and_reflected_operations():
    assert bounds(Interval(0.0, 1.0) + Interval(2.0, 3.0)) == (2.0, 4.0)
    assert bounds(Interval(1.0, 2.0) + 1.0) == (2.0, 3.0)
    assert bounds(1.0 + Interval(1.0, 2.0)) == (2.0, 3.0)

    assert bounds(Interval(1.0, 2.0) - Interval(0.0, 1.0)) == (0.0, 2.0)
    assert bounds(5.0 - Interval(1.0, 2.0)) == (3.0, 4.0)


def test_degenerate_result_reduces_to_float_and_reduction_can_be_disabled(no_reduction):
    # With reduction disabled, a degenerate result stays an Interval.
    result = Interval(1.0, 1.0) + Interval(2.0, 2.0)
    assert isinstance(result, Interval)
    assert bounds(result) == (3.0, 3.0)

    IntervalConstants._reduce_intervals_to_numbers = True
    try:
        reduced = Interval(1.0, 1.0) + Interval(2.0, 2.0)
        assert isinstance(reduced, float)
        assert reduced == 3.0
    finally:
        IntervalConstants._reduce_intervals_to_numbers = False


def test_multiplication_takes_extreme_products():
    assert bounds(Interval(-1.0, 2.0) * Interval(3.0, 4.0)) == (-4.0, 8.0)
    assert bounds(Interval(1.0, 2.0) * 2.0) == (2.0, 4.0)
    assert bounds(2.0 * Interval(1.0, 2.0)) == (2.0, 4.0)


def test_division_and_inversion():
    assert bounds(Interval(1.0, 2.0) / Interval(1.0, 2.0)) == (0.5, 2.0)
    assert bounds(Interval(1.0, 2.0) / 2.0) == (0.5, 1.0)
    assert bounds(1.0 / Interval(1.0, 2.0)) == (0.5, 1.0)

    assert bounds(~Interval(1.0, 2.0)) == (0.5, 1.0)
    assert bounds(~Interval(0.0, 2.0)) == (0.5, math.inf)
    assert bounds(~Interval(-2.0, 0.0)) == (-math.inf, -0.5)
    assert bounds(~Interval(-1.0, 1.0)) == (-math.inf, math.inf)


def test_power_even_and_odd_exponents():
    assert bounds(Interval(-2.0, 3.0) ** 2) == (0.0, 9.0)
    assert bounds(Interval(1.0, 2.0) ** 2) == (1.0, 4.0)
    assert bounds(Interval(-2.0, 3.0) ** 3) == (-8.0, 27.0)

    degenerate = Interval(2.0, 2.0) ** 2
    assert isinstance(degenerate, float)
    assert degenerate == 4.0


# ---------------------------------------------------------------- Box

def test_box_construction_and_bounds():
    plain = Box(1.0, 2.0, 3.0)
    assert isinstance(plain, np.ndarray)
    assert isinstance(plain, Box)
    np.testing.assert_array_equal(plain, [1.0, 2.0, 3.0])

    b = Box(Interval(0.0, 1.0), Interval(2.0, 4.0))
    np.testing.assert_allclose(b.lb, [0.0, 2.0])
    np.testing.assert_allclose(b.ub, [1.0, 4.0])
    np.testing.assert_allclose(b.middle, [0.5, 3.0])

    mixed = Box(Interval(0.0, 1.0), 5.0)
    np.testing.assert_allclose(mixed.lb, [0.0, 5.0])
    np.testing.assert_allclose(mixed.ub, [1.0, 5.0])


def test_box_width_reports_indices_of_max_width():
    b = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), Interval(0.0, 3.0))
    indices, max_width = b.width
    assert indices == [1, 2]
    assert max_width == 3.0

    mixed = Box(Interval(0.0, 1.0), 7.0)
    assert mixed.width == ([0], 1.0)


def test_box_str_and_equality():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    assert str(b) == "[0.0; 1.0] x [2.0; 3.0]"

    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    with pytest.raises(NotImplementedError):
        _ = Box(1.0, 2.0) == [1.0, 2.0]


def test_box_copy_is_deep_and_try_to_reduce():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    c = copy(b)
    assert isinstance(c, Box)
    assert c[0] is not b[0]
    assert c[1] is not b[1]
    assert str(c) == str(b)

    # Degenerate box reduces to its middle point.
    degenerate = Box(Interval(1.0, 1.0), 2.0)
    reduced = degenerate._try_to_reduce()
    np.testing.assert_allclose(reduced, [1.0, 2.0])

    # Non-degenerate box is returned as a copy.
    nondegenerate = Box(Interval(0.0, 1.0), 2.0)
    result = nondegenerate._try_to_reduce()
    assert isinstance(result, Box)
    np.testing.assert_allclose(result.lb, [0.0, 2.0])


# ---------------------------------------------------------------- RS_SubinclusionFunction

def test_rs_subinclusion_returns_sampled_bounds():
    np.random.seed(0)
    f = RS_SubinclusionFunction(lambda x: x[0] + x[1], number_of_samples=2000)
    result = f(Box(Interval(2.0, 3.0), Interval(-1.0, 0.0)))

    assert isinstance(result, Interval)
    assert 1.0 <= result.lb <= result.ub <= 3.0
    assert result.lb < 1.05
    assert result.ub > 2.95


def test_rs_subinclusion_passes_full_dimension_points():
    seen = []

    def f(x):
        seen.append(np.asarray(x).shape)
        return 0.0

    np.random.seed(1)
    RS_SubinclusionFunction(f, number_of_samples=5)(
        Box(Interval(0.0, 1.0), Interval(0.0, 1.0), 2.0)
    )
    assert seen == [(3,)] * 5


# ---------------------------------------------------------------- interval_functions

def test_sin_bounds_and_float_input():
    assert isinstance(ifn.sin(math.pi / 2), float)
    assert ifn.sin(math.pi / 2) == pytest.approx(1.0)

    r = ifn.sin(Interval(0.0, math.pi / 2))
    assert r.lb == pytest.approx(0.0)
    assert r.ub == pytest.approx(1.0)

    full = ifn.sin(Interval(0.0, 2 * math.pi))
    assert bounds(full) == (-1.0, 1.0)


def test_cos_bounds_including_extremum_at_zero():
    r = ifn.cos(Interval(-0.1, 0.1))
    assert r.lb == pytest.approx(math.cos(0.1))
    assert r.ub == pytest.approx(1.0)


def test_abs_and_exp_bounds():
    assert ifn.abs(-2.0) == 2.0
    assert bounds(ifn.abs(Interval(-2.0, 1.0))) == (0.0, 2.0)
    assert bounds(ifn.abs(Interval(1.0, 3.0))) == (1.0, 3.0)

    r = ifn.exp(Interval(0.0, 1.0))
    assert r.lb == pytest.approx(1.0)
    assert r.ub == pytest.approx(math.e)


def test_sqrt_and_log_bounds_and_domain_errors():
    assert bounds(ifn.sqrt(Interval(4.0, 9.0))) == (2.0, 3.0)
    assert bounds(ifn.sqrt(Interval(-1.0, 4.0))) == (0.0, 2.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.sqrt(Interval(-2.0, -1.0))

    r = ifn.log(Interval(1.0, math.e))
    assert r.lb == pytest.approx(0.0)
    assert r.ub == pytest.approx(1.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        ifn.log(Interval(-1.0, 0.0))


def test_split_and_bisect():
    parts = ifn.split(Interval(1.0, 5.0), [1.0, 1.0])
    assert len(parts) == 2
    assert bounds(parts[0]) == (1.0, 3.0)
    assert bounds(parts[1]) == (3.0, 5.0)

    halves = ifn.bisect(Interval(0.0, 2.0))
    assert bounds(halves[0]) == (0.0, 1.0)
    assert bounds(halves[1]) == (1.0, 2.0)

    assert ifn.split(3.0, [1.0, 2.0, 1.0]) == [3.0, 3.0, 3.0]


def test_constrain_and_shrink():
    assert bounds(ifn.constrain(Interval(-5.0, 5.0), -1.0, 1.0)) == (-1.0, 1.0)
    assert ifn.constrain(3.0, 0.0, 2.0) == 2.0

    assert bounds(ifn.shrink(Interval(0.0, 2.0), 0.5)) == (0.5, 1.5)
    assert ifn.shrink(3.0, 0.5) == 3.0


def test_hausdorff_distance():
    # Identical intervals.
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == 0.0

    # Same lower bound, different upper bound.
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 2.0)) == 1.0

    # Disjoint intervals.
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == 2.0

    # Plain numbers are treated as degenerate intervals.
    assert hausdorff_distance(1.0, 3.0) == 2.0
    assert hausdorff_distance(Interval(0.0, 1.0), 0.5) == 0.5

    # Symmetry.
    a, b = Interval(0.0, 1.0), Interval(0.5, 3.0)
    assert hausdorff_distance(a, b) == hausdorff_distance(b, a) == 2.0
