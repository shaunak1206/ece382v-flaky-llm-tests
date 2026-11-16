import numpy as np
import pytest
from copy import copy

from intervallum.box import Box
from intervallum.interval import Interval, IntervalConstants, IntervalExceptions
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import hausdorff_distance


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------

def test_box_construction_from_intervals_keeps_elements():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    assert isinstance(b, Box)
    assert len(b) == 2
    assert isinstance(b[0], Interval)
    assert b[0].lb == 0.0 and b[0].ub == 1.0
    assert b[1].lb == 2.0 and b[1].ub == 3.0


def test_box_lb_ub_middle_with_intervals():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    assert np.array_equal(b.lb, np.array([0.0, 2.0]))
    assert np.array_equal(b.ub, np.array([1.0, 3.0]))
    assert np.array_equal(b.middle, np.array([0.5, 2.5]))


def test_box_lb_ub_middle_with_plain_floats():
    b = Box(1.0, 2.0)
    assert np.array_equal(b.lb, np.array([1.0, 2.0]))
    assert np.array_equal(b.ub, np.array([1.0, 2.0]))
    assert np.array_equal(b.middle, np.array([1.0, 2.0]))


def test_box_width_returns_index_of_max_width():
    b = Box(Interval(0.0, 1.0), Interval(0.0, 3.0), Interval(1.0, 2.0))
    indices, max_width = b.width
    assert indices == [1]
    assert max_width == 3.0


def test_box_width_ties_return_all_indices():
    b = Box(Interval(0.0, 2.0), Interval(5.0, 7.0), Interval(0.0, 1.0))
    indices, max_width = b.width
    assert indices == [0, 1]
    assert max_width == 2.0


def test_box_width_treats_float_components_as_zero_width():
    b = Box(Interval(0.0, 1.0), 5.0)
    indices, max_width = b.width
    assert indices == [0]
    assert max_width == 1.0


def test_box_str_and_repr_format():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    expected = "[0.0; 1.0] x [2.0; 3.0]"
    assert str(b) == expected
    assert repr(b) == expected


def test_box_equality_and_inequality_with_box():
    assert Box(1.0, 2.0) == Box(1.0, 2.0)
    assert Box(1.0, 2.0) != Box(1.0, 3.0)
    assert not (Box(1.0, 2.0) == Box(1.0, 3.0))


def test_box_equality_with_plain_ndarray():
    assert Box(1.0, 2.0) == np.array([1.0, 2.0])


def test_box_equality_with_unsupported_type_raises():
    with pytest.raises(NotImplementedError):
        Box(1.0, 2.0) == [1.0, 2.0]


def test_box_copy_is_deep_for_intervals():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    c = copy(b)
    assert isinstance(c, Box)
    assert c is not b
    assert c[0] is not b[0]
    c[0].lb = 0.5
    assert b[0].lb == 0.0
    assert c[0].lb == 0.5


def test_box_try_to_reduce_degenerate_returns_middle_array():
    b = Box(Interval(1.0, 1.0), Interval(2.0, 2.0))
    reduced = b._try_to_reduce()
    assert isinstance(reduced, np.ndarray)
    assert np.array_equal(reduced, np.array([1.0, 2.0]))


def test_box_try_to_reduce_nondegenerate_returns_copy():
    b = Box(Interval(0.0, 1.0), Interval(2.0, 3.0))
    reduced = b._try_to_reduce()
    assert isinstance(reduced, Box)
    assert reduced is not b
    assert reduced == b


# ---------------------------------------------------------------------------
# Interval: construction, properties, validation
# ---------------------------------------------------------------------------

def test_interval_init_rejects_wrong_bounds():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)


def test_interval_init_fix_swaps_wrong_bounds():
    i = Interval(2.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 2.0


def test_interval_from_point_middle_and_width():
    p = Interval.from_point(3.0)
    assert p.lb == 3.0 and p.ub == 3.0 and p.width == 0.0

    i = Interval(1.0, 3.0)
    assert i.middle == 2.0
    assert i.width == 2.0


def test_interval_str_and_repr():
    i = Interval(1.0, 2.5)
    assert str(i) == "[1.0; 2.5]"
    assert repr(i) == "[1.0; 2.5]"


def test_interval_lb_and_ub_setters_validate_bounds():
    i = Interval(1.0, 2.0)
    i.lb = 0.5
    assert i.lb == 0.5
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.lb = 3.0
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        i.ub = 0.0


# ---------------------------------------------------------------------------
# Interval: arithmetic, comparison, equality
# ---------------------------------------------------------------------------

def test_interval_addition_and_negation():
    s = Interval(1.0, 2.0) + Interval(3.0, 4.0)
    assert isinstance(s, Interval)
    assert (s.lb, s.ub) == (4.0, 6.0)

    t = Interval(1.0, 2.0) + 1.0
    assert (t.lb, t.ub) == (2.0, 3.0)

    n = -Interval(1.0, 2.0)
    assert (n.lb, n.ub) == (-2.0, -1.0)


def test_interval_subtraction():
    d = Interval(1.0, 2.0) - Interval(0.0, 1.0)
    assert (d.lb, d.ub) == (0.0, 2.0)

    e = Interval(1.0, 2.0) - 0.5
    assert (e.lb, e.ub) == (0.5, 1.5)


def test_interval_multiplication_with_mixed_signs():
    p = Interval(-1.0, 2.0) * Interval(-3.0, 4.0)
    assert isinstance(p, Interval)
    assert (p.lb, p.ub) == (-6.0, 8.0)


def test_interval_inversion_and_division():
    inv = ~Interval(2.0, 4.0)
    assert (inv.lb, inv.ub) == pytest.approx((0.25, 0.5))

    unbounded = ~Interval(-1.0, 1.0)
    assert unbounded.lb == -float("inf")
    assert unbounded.ub == float("inf")

    q = Interval(1.0, 2.0) / 2.0
    assert (q.lb, q.ub) == pytest.approx((0.5, 1.0))


def test_interval_power_even_and_odd():
    even = Interval(-2.0, 3.0) ** 2
    assert isinstance(even, Interval)
    assert (even.lb, even.ub) == (0.0, 9.0)

    odd = Interval(-2.0, 3.0) ** 3
    assert (odd.lb, odd.ub) == (-8.0, 27.0)

    pos = Interval(1.0, 2.0) ** 2
    assert (pos.lb, pos.ub) == (1.0, 4.0)


def test_interval_comparisons_use_lower_bound():
    assert Interval(1.0, 5.0) < Interval(2.0, 3.0)
    assert Interval(1.0, 5.0) < 2.0
    assert Interval(3.0, 4.0) >= 3.0
    assert Interval(3.0, 4.0) > 2.5
    assert Interval(3.0, 4.0) <= Interval(3.0, 10.0)


def test_interval_equality_with_tolerance_and_type_check():
    assert Interval(1.0, 1.0) == 1.0
    assert Interval(0.0, 1.0) == Interval(0.0, 1.0)
    assert not (Interval(0.0, 1.0) == Interval(0.0, 2.0))
    assert Interval(0.0, 1.0) != Interval(0.0, 2.0)
    with pytest.raises(NotImplementedError):
        Interval(1.0, 1.0) == 1  # int is not float


# ---------------------------------------------------------------------------
# Interval: reduction behaviour
# ---------------------------------------------------------------------------

def test_reduction_collapses_point_interval_to_float_by_default():
    result = Interval(1.0, 1.0) + 1.0
    assert isinstance(result, float)
    assert result == 2.0


def test_reduction_disabled_returns_interval(monkeypatch):
    monkeypatch.setattr(IntervalConstants, "_reduce_intervals_to_numbers", False)
    result = Interval(1.0, 1.0) + 1.0
    assert isinstance(result, Interval)
    assert (result.lb, result.ub) == (2.0, 2.0)


def test_try_to_reduce_threshold_behaviour():
    tiny = Interval(0.0, 1e-6)._try_to_reduce()
    assert isinstance(tiny, float)
    assert tiny == pytest.approx(5e-7)

    wide = Interval(0.0, 1.0)._try_to_reduce()
    assert isinstance(wide, Interval)
    assert (wide.lb, wide.ub) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# hausdorff_distance
# ---------------------------------------------------------------------------

def test_hausdorff_identical_intervals_is_zero():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(0.0, 1.0)) == pytest.approx(0.0)


def test_hausdorff_disjoint_intervals():
    assert hausdorff_distance(Interval(0.0, 1.0), Interval(2.0, 3.0)) == pytest.approx(2.0)


def test_hausdorff_nested_intervals():
    i1 = Interval(0.0, 2.0)
    i2 = Interval(0.5, 1.5)
    assert hausdorff_distance(i1, i2) == pytest.approx(0.5)
    assert hausdorff_distance(i2, i1) == pytest.approx(0.5)


def test_hausdorff_numbers_and_mixed_arguments():
    assert hausdorff_distance(1.0, 3.0) == pytest.approx(2.0)
    assert hausdorff_distance(1.0, Interval(0.0, 2.0)) == pytest.approx(1.0)
    assert hausdorff_distance(Interval(0.0, 2.0), 1.0) == pytest.approx(1.0)


def test_hausdorff_is_symmetric():
    a = Interval(-1.0, 4.0)
    b = Interval(2.0, 6.0)
    assert hausdorff_distance(a, b) == pytest.approx(hausdorff_distance(b, a))


# ---------------------------------------------------------------------------
# RS_SubinclusionFunction
# ---------------------------------------------------------------------------

def test_rs_evaluates_function_number_of_samples_times():
    calls = []

    def f(v):
        calls.append(1)
        return float(v[0])

    rs = RS_SubinclusionFunction(f, number_of_samples=7)
    rs(Box(Interval(0.0, 1.0), Interval(0.0, 1.0)))
    assert len(calls) == 7


def test_rs_returns_interval_enclosing_sampled_values():
    np.random.seed(0)
    box = Box(Interval(0.0, 1.0), Interval(5.0, 6.0))
    rs = RS_SubinclusionFunction(lambda v: float(v[0] + v[1]), number_of_samples=500)
    result = rs(box)
    assert isinstance(result, Interval)
    assert result.lb >= 5.0
    assert result.ub <= 7.0
    assert result.lb <= result.ub


def test_rs_approximates_range_of_first_coordinate():
    np.random.seed(42)
    box = Box(Interval(0.0, 1.0), Interval(5.0, 6.0))
    rs = RS_SubinclusionFunction(lambda v: float(v[0]), number_of_samples=2000)
    result = rs(box)
    assert result.lb < 0.01
    assert result.ub > 0.99


def test_rs_degenerate_box_gives_point_interval():
    rs = RS_SubinclusionFunction(lambda v: float(v[0] * v[1]), number_of_samples=5)
    result = rs(Box(1.0, 2.0))
    assert isinstance(result, Interval)
    assert result.lb == pytest.approx(2.0)
    assert result.ub == pytest.approx(2.0)
