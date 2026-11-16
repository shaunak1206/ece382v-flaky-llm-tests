import logging
import math

import numpy as np
import pytest

from fuzzylogic import functions as fz
from fuzzylogic.classes import Domain, FuzzyWarning, Set


@pytest.fixture
def temp():
    """A domain with two sets: cold = S(0, 10), hot = R(0, 10)."""
    d = Domain("temperature", 0, 10)
    d.cold = Set(fz.S(0, 10))
    d.hot = Set(fz.R(0, 10))
    return d


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_rejects_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("bad", 5, 5)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 10, res=0)


def test_domain_range_is_inclusive_and_respects_resolution():
    d = Domain("t", 0, 10)
    rng = d.range()
    assert len(rng) == 11
    assert rng[-1] == 10
    fine = Domain("f", 0, 1, res=0.25)
    assert np.allclose(fine.range(), [0, 0.25, 0.5, 0.75, 1.0])


def test_domain_assignment_sets_name_domain_and_wraps_callables():
    d = Domain("t", 0, 10)
    d.warm = lambda x: 0.5
    assert isinstance(d.warm, Set)
    assert d.warm.name == "warm"
    assert d.warm.domain is d
    assert d.warm(3) == 0.5
    # Attribute names must be identifiers.
    with pytest.raises(AssertionError):
        setattr(d, "not valid", lambda x: 0)


def test_domain_call_returns_memberships_per_set(temp):
    result = temp(2)
    assert set(result) == {"cold", "hot"}
    assert result["cold"] == pytest.approx(0.8)
    assert result["hot"] == pytest.approx(0.2)


def test_domain_call_outside_range_warns(temp, caplog):
    # Domain.__call__ uses logging.warn, which is deprecated but still emits.
    with pytest.warns(DeprecationWarning), caplog.at_level(logging.WARNING):
        result = temp(20)
    assert "outside of domain" in caplog.text
    assert result["hot"] == 1


def test_domain_delete_and_missing_attributes(temp):
    del temp.cold
    with pytest.raises(AttributeError):
        temp.cold
    with pytest.raises(FuzzyWarning):
        del temp.nonexistent


def test_domain_min_and_max(temp):
    # At x=2: cold=0.8, hot=0.2
    assert temp.min(2) == pytest.approx(0.2)
    assert temp.max(2) == pytest.approx(0.8)


def test_domain_equality():
    assert Domain("a", 0, 10) == Domain("a", 0, 10)
    assert Domain("a", 0, 10) != Domain("b", 0, 10)
    assert Domain("a", 0, 10) != Domain("a", 0, 10, res=2)


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

def test_set_rejects_invalid_construction():
    with pytest.raises(AssertionError):
        Set(5)
    # A name without a domain (or vice versa) is an error.
    with pytest.raises(FuzzyWarning):
        Set(lambda x: 0.3, name="x")


def test_set_str_representation(temp):
    assert str(temp.hot) == "temperature.hot"
    assert str(Set(lambda x: 0)).startswith("dangling Set(")


def test_set_domain_dependent_operations_require_domain():
    s = Set(lambda x: 0.5)
    with pytest.raises(FuzzyWarning):
        s.array()
    with pytest.raises(FuzzyWarning):
        s.plot()
    with pytest.raises(FuzzyWarning):
        len(s)
    with pytest.raises(FuzzyWarning):
        s == Set(lambda x: 0.5)
    with pytest.raises(FuzzyWarning):
        s <= Set(lambda x: 0.5)


def test_set_invert_and_negation(temp):
    # hot(2) = 0.2, so the inverse is 0.8
    assert (~temp.hot)(2) == pytest.approx(0.8)
    assert (-temp.hot)(2) == pytest.approx(0.8)


def test_set_and_or_use_min_and_max(temp):
    # hot(2) = 0.2, cold(2) = 0.8
    assert (temp.hot & temp.cold)(2) == pytest.approx(0.2)
    assert (temp.hot | temp.cold)(2) == pytest.approx(0.8)


def test_set_pow_concentrated_and_multiplied(temp):
    # hot(5) = 0.5
    assert (temp.hot ** 2)(5) == pytest.approx(0.25)
    assert temp.hot.concentrated()(5) == pytest.approx(0.25)
    assert temp.hot.multiplied(0.5)(10) == pytest.approx(0.5)


def test_set_intensified_low_branch():
    # Only the x < 0.5 branch is exercised: 2 * 0.6**2 = 0.72
    s = Set(lambda x: 0.6)
    assert s.intensified()(0.2) == pytest.approx(0.72)


def test_set_normalized_peaks_at_one(temp):
    temp.half = Set(lambda x: 0.5 * x / 10)  # max over domain is 0.5
    n = temp.half.normalized()
    assert n(10) == pytest.approx(1.0)
    assert n(5) == pytest.approx(0.5)


def test_set_array_length_cardinality_and_relative_cardinality(temp):
    assert np.allclose(temp.hot.array(), np.arange(11) / 10)
    assert len(temp.hot) == 11
    assert temp.hot.cardinality() == pytest.approx(5.5)
    assert temp.hot.relative_cardinality() == pytest.approx(0.5)


def test_set_equality_with_domain(temp):
    assert temp.hot == temp.hot
    assert temp.hot != temp.cold


def test_set_subset_and_superset_comparisons(temp):
    temp.zero = Set(lambda x: 0.0)
    assert temp.zero <= temp.cold
    # cold(10) == 0, so the strict comparison fails elementwise
    assert not (temp.zero < temp.cold)
    assert temp.cold >= temp.zero
    assert temp.hot >= temp.zero
    assert not (temp.hot > temp.zero)


# ---------------------------------------------------------------------------
# functions module
# ---------------------------------------------------------------------------

def test_basic_transforms_inv_noop_constant_alpha_normalize_moderate():
    assert fz.inv(lambda x: 0.3)(0) == pytest.approx(0.7)
    assert fz.noop()(5) == 5
    assert fz.constant(0.4)(100) == 0.4

    clipped = fz.alpha(func=lambda x: x, floor=0.2, ceiling=0.8)
    assert clipped(0.1) == pytest.approx(0.2)
    assert clipped(0.9) == pytest.approx(0.8)
    assert clipped(0.5) == pytest.approx(0.5)

    assert fz.normalize(0.5, lambda x: 0.25)(0) == pytest.approx(0.5)

    assert fz.moderate(lambda x: 0.5)(0) == pytest.approx(0.5)
    assert fz.moderate(lambda x: 1)(0) == pytest.approx(1.0)
    assert fz.moderate(lambda x: 0)(0) == pytest.approx(0.0)


def test_singleton_linear_and_bounded_linear():
    s = fz.singleton(2, no_m=0.1, c_m=0.9)
    assert s(2) == 0.9
    assert s(3) == 0.1

    lin = fz.linear(1, -1)
    assert lin(-2) == 0
    assert lin(1.5) == pytest.approx(0.5)
    assert lin(3) == 1

    bl = fz.bounded_linear(2, 3)
    assert bl(1) == 0
    assert bl(2.5) == pytest.approx(0.5)
    assert bl(4) == 1

    inv_bl = fz.bounded_linear(2, 3, inverse=True)
    assert inv_bl(2) == pytest.approx(1.0)
    assert inv_bl(3) == pytest.approx(0.0)


def test_ramps_rectangular_triangular_trapezoid():
    r = fz.R(2, 4)
    assert r(1) == 0
    assert r(3) == pytest.approx(0.5)
    assert r(5) == 1

    s = fz.S(2, 4)
    assert s(1) == 1
    assert s(3) == pytest.approx(0.5)
    assert s(5) == 0

    rect = fz.rectangular(1, 3)
    assert rect(0) == 0
    assert rect(2) == 1
    assert rect(4) == 0

    tri = fz.triangular(0, 10)
    assert tri(0) == pytest.approx(0.0)
    assert tri(5) == pytest.approx(1.0)
    assert tri(2.5) == pytest.approx(0.5)
    assert tri(7.5) == pytest.approx(0.5)

    trap = fz.trapezoid(0, 2, 4, 6)
    assert trap(-1) == 0
    assert trap(1) == pytest.approx(0.5)
    assert trap(3) == 1
    assert trap(5) == pytest.approx(0.5)
    assert trap(7) == 0


def test_sigmoids_and_bounded_exponential():
    sg = fz.sigmoid(1, 1, 0)
    assert sg(0) == pytest.approx(0.5)
    assert sg(-1000) == pytest.approx(0.0, abs=1e-12)
    assert sg(1000) == pytest.approx(1.0)

    bs = fz.bounded_sigmoid(0, 1)
    assert bs(0) == pytest.approx(0.1)
    assert bs(1) == pytest.approx(0.9)
    assert bs(-100000) == pytest.approx(0.0)
    assert fz.bounded_sigmoid(0, 1, inverse=True)(0) == pytest.approx(0.9)

    be = fz.bounded_exponential(k=1, limit=1)
    assert be(0) == pytest.approx(0.0)
    assert be(1) == pytest.approx(1 - 1 / math.e)
    assert be(10000) == pytest.approx(1.0)

    ss = fz.simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(1000) == pytest.approx(1.0)
    assert ss(-1000) == pytest.approx(0.0, abs=1e-6)


def test_triangular_sigmoid_gauss_and_parameter_checks():
    g = fz.triangular_sigmoid(2, 4)
    assert g(2) == pytest.approx(0.1)
    assert g(3) == pytest.approx(0.9)

    ga = fz.gauss(0, 1)
    assert ga(0) == pytest.approx(1.0)
    assert ga(1) == pytest.approx(math.exp(-1))
    assert fz.gauss(0, 1, c_m=0.5)(0) == pytest.approx(0.5)

    with pytest.raises(AssertionError):
        fz.bounded_linear(3, 2)
    with pytest.raises(AssertionError):
        fz.singleton(1, no_m=0.5, c_m=0.2)
    with pytest.raises(AssertionError):
        fz.gauss(0, 0)
    with pytest.raises(AssertionError):
        fz.S(3, 1)
