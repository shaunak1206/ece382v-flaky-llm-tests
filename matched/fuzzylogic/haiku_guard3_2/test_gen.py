"""Unit tests for fuzzylogic.classes (Domain, Set) and fuzzylogic.functions.

These tests are deterministic: the code under test uses no random numbers,
no clock, and no hash-dependent ordering. Floating point results are compared
with pytest.approx. Module attributes are replaced only via monkeypatch.
"""

import logging
import types

import pytest

from fuzzylogic import functions
from fuzzylogic import classes
from fuzzylogic.classes import Domain, Set, FuzzyWarning


# ---------------------------------------------------------------------------
# functions module
# ---------------------------------------------------------------------------

def test_inv_complements_value():
    f = functions.inv(functions.constant(0.3))
    assert f(0) == pytest.approx(0.7)
    assert f(123.4) == pytest.approx(0.7)


def test_constant_and_noop():
    assert functions.constant(0.25)(-5) == 0.25
    assert functions.noop()(0.6) == 0.6


def test_alpha_clips_values():
    f = functions.alpha(floor=0.2, ceiling=0.8, func=lambda x: x)
    assert f(0.1) == pytest.approx(0.2)   # below floor -> floor_clip
    assert f(0.5) == pytest.approx(0.5)   # in between -> unchanged
    assert f(0.9) == pytest.approx(0.8)   # above ceiling -> ceiling_clip


def test_normalize_scales_max_to_one():
    f = functions.normalize(0.5, lambda x: x * 0.5)
    assert f(1) == pytest.approx(1.0)
    assert f(0.5) == pytest.approx(0.5)


def test_moderate_fixed_points():
    f = functions.moderate(lambda x: x)
    assert f(0.5) == pytest.approx(0.5)
    assert f(0) == pytest.approx(0.0)
    assert f(1) == pytest.approx(1.0)


def test_singleton_spike():
    f = functions.singleton(2, no_m=0.1, c_m=0.9)
    assert f(2) == pytest.approx(0.9)
    assert f(1) == pytest.approx(0.1)
    assert f(3) == pytest.approx(0.1)


def test_linear_is_clipped():
    f = functions.linear(1, -1)
    assert f(-2) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(3) == 1


def test_bounded_linear_endpoints_and_middle():
    f = functions.bounded_linear(2, 3)
    assert f(1) == pytest.approx(0.0)
    assert f(2.5) == pytest.approx(0.5)
    assert f(3) == pytest.approx(1.0)
    assert f(4) == pytest.approx(1.0)


def test_bounded_linear_inverse_reverses_slope():
    f = functions.bounded_linear(0, 1, inverse=True)
    assert f(0) == pytest.approx(1.0)
    assert f(0.5) == pytest.approx(0.5)
    assert f(1) == pytest.approx(0.0)


def test_R_and_S_slopes():
    r = functions.R(0, 10)
    s = functions.S(0, 10)
    assert r(-1) == 0
    assert r(5) == pytest.approx(0.5)
    assert r(20) == 1
    assert s(-1) == 1
    assert s(5) == pytest.approx(0.5)
    assert s(20) == 0


def test_rectangular_core_and_outside():
    f = functions.rectangular(1, 2, c_m=0.8, no_m=0.1)
    assert f(0) == pytest.approx(0.1)
    assert f(1.5) == pytest.approx(0.8)
    assert f(3) == pytest.approx(0.1)


def test_triangular_peak_and_feet():
    f = functions.triangular(0, 4)  # default peak at 2
    assert f(0) == pytest.approx(0.0)
    assert f(1) == pytest.approx(0.5)
    assert f(2) == pytest.approx(1.0)
    assert f(3) == pytest.approx(0.5)
    assert f(4) == pytest.approx(0.0)


def test_trapezoid_plateau_and_slopes():
    f = functions.trapezoid(0, 2, 4, 6)
    assert f(0) == pytest.approx(0.0)
    assert f(1) == pytest.approx(0.5)
    assert f(3) == pytest.approx(1.0)
    assert f(5) == pytest.approx(0.5)
    assert f(7) == pytest.approx(0.0)


def test_sigmoid_midpoint_is_half_of_L_and_bounded_sigmoid_defining_points():
    assert functions.sigmoid(1, 2, 5)(5) == pytest.approx(0.5)
    g = functions.bounded_sigmoid(0, 1)
    assert g(0) == pytest.approx(0.1)
    assert g(1) == pytest.approx(0.9)


def test_bounded_exponential_starts_at_zero_and_approaches_limit():
    f = functions.bounded_exponential(k=1, limit=1)
    assert f(0) == pytest.approx(0.0)
    assert f(100) == pytest.approx(1.0)


def test_simple_sigmoid_midpoint_and_symmetry():
    f = functions.simple_sigmoid()
    assert f(0) == pytest.approx(0.5)
    assert f(20) + f(-20) == pytest.approx(1.0)


def test_gauss_peak_and_decay():
    f = functions.gauss(3, 1, c_m=0.5)
    assert f(3) == pytest.approx(0.5)
    g = functions.gauss(0, 1)
    assert g(1) == pytest.approx(2.718281828459045 ** -1)


def test_all_membership_functions_stay_in_unit_interval():
    funcs = [
        functions.linear(2, -0.5),
        functions.bounded_linear(0, 10),
        functions.R(0, 10),
        functions.S(0, 10),
        functions.triangular(0, 10),
        functions.trapezoid(0, 3, 6, 10),
        functions.sigmoid(1, 1, 5),
        functions.bounded_sigmoid(0, 10),
        functions.bounded_exponential(0.5),
        functions.simple_sigmoid(0.3),
        functions.gauss(5, 0.2),
        functions.singleton(3),
        functions.rectangular(2, 8),
    ]
    grid = [i * 0.5 for i in range(0, 41)]  # 0.0 .. 20.0, deterministic
    for f in funcs:
        for x in grid:
            v = f(x)
            assert 0 <= v <= 1, (f, x, v)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_rejects_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("bad", 5, 5)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 10, res=0)


def test_domain_range_includes_upper_bound():
    d = Domain("t", 0, 1, res=0.25)
    values = list(d.range())
    assert values == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_domain_attribute_assignment_wraps_function_and_registers():
    d = Domain("t", 0, 10)
    d.hot = lambda x: x / 10
    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d.hot(5) == pytest.approx(0.5)
    assert str(d) == "t"
    assert str(d.hot) == "t.hot"


def test_domain_call_returns_membership_per_set():
    d = Domain("t", 0, 10)
    d.hot = lambda x: x / 10
    d.cold = lambda x: 1 - x / 10
    result = d(5)
    assert set(result.keys()) == {"hot", "cold"}
    assert result["hot"] == pytest.approx(0.5)
    assert result["cold"] == pytest.approx(0.5)


def test_domain_call_outside_bounds_logs_warning_but_still_evaluates(caplog):
    d = Domain("t", 0, 10)
    d.hot = lambda x: 0.4
    with caplog.at_level(logging.WARNING):
        result = d(20)
    assert result == {"hot": pytest.approx(0.4)}
    assert "outside of domain" in caplog.text


def test_domain_delete_set_and_reject_unknown_attribute():
    d = Domain("t", 0, 10)
    d.hot = lambda x: 1
    del d.hot
    with pytest.raises(AttributeError):
        _ = d.hot
    with pytest.raises(FuzzyWarning):
        del d.nonexistent


def test_domain_min_and_max_over_sets():
    d = Domain("t", 0, 10)
    d.low = lambda x: 0.2
    d.high = lambda x: 0.9
    assert d.min(5) == pytest.approx(0.2)
    assert d.max(5) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

def test_set_init_requires_name_and_domain_together():
    d = Domain("t", 0, 10)
    s = Set(lambda x: 0.5, name="mid", domain=d)
    assert d.mid is s
    with pytest.raises(FuzzyWarning):
        Set(lambda x: 0.5, name="only_name")


def test_set_logical_operators_min_max_not():
    a = Set(functions.constant(0.3))
    b = Set(functions.constant(0.6))
    assert (a & b)(0) == pytest.approx(0.3)
    assert (a | b)(0) == pytest.approx(0.6)
    assert (~a)(0) == pytest.approx(0.7)
    assert (-a)(0) == pytest.approx(0.7)


def test_set_power_concentrated_dilated_multiplied():
    s = Set(functions.constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.concentrated()(0) == pytest.approx(0.25)
    assert s.dilated()(0) == pytest.approx(0.5 ** 0.5)
    assert s.multiplied(0.8)(0) == pytest.approx(0.4)


def test_set_array_matches_domain_range_and_cardinality():
    d = Domain("t", 0, 4)  # 5 points with res=1
    d.half = Set(functions.constant(0.5))
    assert len(d.half) == 5
    assert d.half.array() == pytest.approx([0.5] * 5)
    assert d.half.cardinality() == pytest.approx(2.5)
    assert d.half.relative_cardinality() == pytest.approx(0.5)


def test_set_comparisons_use_domain_values():
    d = Domain("t", 0, 2)
    d.low = Set(functions.constant(0.2))
    d.high = Set(functions.constant(0.8))
    assert d.low <= d.high
    assert d.low < d.high
    assert not (d.low >= d.high)
    assert not (d.low > d.high)
    assert d.low == Set(functions.constant(0.2))


def test_set_equality_without_domain_raises():
    a = Set(functions.constant(0.5))
    b = Set(functions.constant(0.5))
    with pytest.raises(FuzzyWarning):
        _ = a == b
    with pytest.raises(FuzzyWarning):
        _ = a.array()
    with pytest.raises(FuzzyWarning):
        len(a)


def test_set_normalized_has_max_one_over_domain():
    d = Domain("t", 0, 2)  # points 0, 1, 2
    d.ramp = Set(lambda x: 0.5 * x / 2)  # values 0, 0.25, 0.5
    norm = d.ramp.normalized()
    assert norm(2) == pytest.approx(1.0)
    assert norm(1) == pytest.approx(0.5)


def test_set_plot_uses_domain_range_with_patched_plt(monkeypatch):
    calls = []
    fake_plt = types.SimpleNamespace(
        plot=lambda xs, ys: calls.append((list(xs), list(ys))))
    monkeypatch.setattr(classes, "plt", fake_plt)

    d = Domain("t", 0, 2)
    d.s = Set(functions.constant(0.5))
    d.s.plot()

    xs, ys = calls[0]
    assert xs == pytest.approx([0.0, 1.0, 2.0])
    assert ys == pytest.approx([0.5, 0.5, 0.5])
