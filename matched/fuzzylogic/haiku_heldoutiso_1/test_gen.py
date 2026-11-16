import logging

import pytest

from fuzzylogic import functions as F
from fuzzylogic.classes import Domain, Set, FuzzyWarning


@pytest.fixture
def temp():
    """A fresh domain 0..10 with resolution 1 (11 sample points)."""
    return Domain("temp", 0, 10)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_rejects_inverted_bounds():
    with pytest.raises(AssertionError):
        Domain("bad", 10, 0)


def test_domain_rejects_nonpositive_resolution():
    with pytest.raises(AssertionError):
        Domain("bad", 0, 10, res=0)


def test_domain_str_returns_name(temp):
    assert str(temp) == "temp"


def test_domain_range_includes_upper_bound(temp):
    r = temp.range()
    assert len(r) == 11
    assert r[0] == 0
    assert r[-1] == 10


def test_domain_assign_function_wraps_in_set_and_links(temp):
    temp.hot = lambda x: 0.5
    assert isinstance(temp.hot, Set)
    assert temp.hot.domain is temp
    assert temp.hot.name == "hot"
    assert temp.hot(3) == 0.5


def test_domain_assign_rejects_non_identifier_name(temp):
    with pytest.raises(AssertionError):
        setattr(temp, "not-valid", Set(F.constant(0.1)))


def test_domain_call_returns_membership_per_set(temp):
    temp.cold = Set(F.constant(0.2))
    temp.hot = Set(F.constant(0.8))
    result = temp(5)
    assert result == {"cold": 0.2, "hot": 0.8}


def test_domain_call_outside_bounds_logs_warning(temp, caplog):
    temp.cold = Set(F.constant(0.2))
    with caplog.at_level(logging.WARNING):
        temp(42)
    assert "outside of domain" in caplog.text


def test_domain_getattr_unknown_raises_attribute_error(temp):
    with pytest.raises(AttributeError):
        _ = temp.nonexistent


def test_domain_delete_set(temp):
    temp.warm = Set(F.constant(0.5))
    del temp.warm
    with pytest.raises(AttributeError):
        _ = temp.warm


def test_domain_delete_unknown_attr_raises_fuzzy_warning(temp):
    with pytest.raises(FuzzyWarning):
        del temp.missing


def test_domain_min_and_max_over_sets(temp):
    temp.low = Set(F.constant(0.3))
    temp.high = Set(F.constant(0.7))
    assert temp.min(0) == 0.3
    assert temp.max(0) == 0.7


# ---------------------------------------------------------------------------
# Set: construction and basic behaviour
# ---------------------------------------------------------------------------

def test_set_call_delegates_to_function():
    s = Set(lambda x: x * 2)
    assert s(3) == 6


def test_set_name_without_domain_raises_fuzzy_warning():
    with pytest.raises(FuzzyWarning):
        Set(F.constant(1), name="x")


def test_set_with_name_and_domain_registers_on_domain(temp):
    Set(F.constant(0.5), name="warm", domain=temp)
    assert temp.warm(0) == 0.5


def test_set_str_without_domain_is_dangling():
    s = Set(F.constant(1))
    assert str(s).startswith("dangling Set")


def test_set_str_with_domain_shows_qualified_name(temp):
    temp.hot = Set(F.constant(1))
    assert str(temp.hot) == "temp.hot"


# ---------------------------------------------------------------------------
# Set: logical operators and hedges
# ---------------------------------------------------------------------------

def test_set_invert_is_complement():
    s = ~Set(F.constant(0.25))
    assert s(0) == pytest.approx(0.75)


def test_set_and_or_use_min_max():
    a = Set(F.constant(0.2))
    b = Set(F.constant(0.7))
    assert (a & b)(0) == pytest.approx(0.2)
    assert (a | b)(0) == pytest.approx(0.7)


def test_set_pow_and_concentrated_square_membership():
    s = Set(F.constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.concentrated()(0) == pytest.approx(0.25)


def test_set_multiplied_scales_membership():
    s = Set(F.constant(0.4)).multiplied(0.5)
    assert s(0) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Set: domain-dependent operations
# ---------------------------------------------------------------------------

def test_set_array_length_matches_domain(temp):
    temp.s = Set(F.constant(0.5))
    assert len(temp.s) == 11
    assert temp.s.array().shape == (11,)


def test_set_cardinality_and_relative_cardinality(temp):
    temp.s = Set(F.constant(0.5))
    assert temp.s.cardinality() == pytest.approx(5.5)
    assert temp.s.relative_cardinality() == pytest.approx(0.5)


def test_set_equality_with_identical_values(temp):
    temp.a = Set(F.constant(0.5))
    temp.b = Set(lambda x: 0.5)
    assert temp.a == temp.b


def test_set_subset_and_proper_subset(temp):
    temp.low = Set(F.constant(0.2))
    temp.high = Set(F.constant(0.5))
    assert temp.low <= temp.high
    assert temp.low < temp.high
    assert not (temp.high <= temp.low)


def test_set_normalized_has_maximum_one(temp):
    temp.s = Set(F.constant(0.5))
    n = temp.s.normalized()
    assert n(0) == pytest.approx(1.0)


def test_set_array_without_domain_raises_fuzzy_warning():
    s = Set(F.constant(0.5))
    with pytest.raises(FuzzyWarning):
        s.array()


# ---------------------------------------------------------------------------
# functions module
# ---------------------------------------------------------------------------

def test_inv_and_constant_and_noop():
    assert F.inv(F.constant(0.3))(0) == pytest.approx(0.7)
    assert F.constant(0.6)(123) == 0.6
    assert F.noop()(4) == 4


def test_alpha_clips_values_above_ceiling():
    clipped = F.alpha(func=F.constant(0.95), ceiling=0.9, ceiling_clip=0.0)
    passed = F.alpha(func=F.constant(0.5), ceiling=0.9, ceiling_clip=0.0)
    assert clipped(0) == 0.0
    assert passed(0) == 0.5


def test_normalize_scales_and_rejects_bad_height():
    assert F.normalize(0.5, F.constant(0.25))(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        F.normalize(0, F.constant(0.25))


def test_moderate_fixed_points():
    assert F.moderate(F.constant(0.5))(0) == pytest.approx(0.5)
    assert F.moderate(F.constant(1))(0) == pytest.approx(1.0)


def test_singleton_spike_at_peak():
    f = F.singleton(2)
    assert f(1) == 0
    assert f(2) == 1


def test_linear_is_clipped_to_unit_interval():
    f = F.linear(1, -1)
    assert f(-2) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(3) == 1


def test_R_and_S_ramps():
    r = F.R(0, 10)
    s = F.S(0, 10)
    assert r(-1) == 0
    assert r(5) == pytest.approx(0.5)
    assert r(20) == 1
    assert s(-1) == 1
    assert s(5) == pytest.approx(0.5)
    assert s(20) == 0


def test_rectangular_core_and_shoulders():
    f = F.rectangular(2, 4)
    assert f(1) == 0
    assert f(3) == 1
    assert f(5) == 0


def test_triangular_peak_and_feet():
    f = F.triangular(0, 10)
    assert f(5) == pytest.approx(1.0)
    assert f(0) == pytest.approx(0.0)
    assert f(10) == pytest.approx(0.0)
    assert f(2.5) == pytest.approx(0.5)


def test_trapezoid_plateau_and_edges():
    f = F.trapezoid(0, 2, 4, 6)
    assert f(3) == pytest.approx(1.0)
    assert f(1) == pytest.approx(0.5)
    assert f(-1) == 0
    assert f(7) == 0


def test_sigmoid_midpoint_is_half():
    f = F.sigmoid(1, 1, 0)
    assert f(0) == pytest.approx(0.5)


def test_bounded_sigmoid_hits_ten_and_ninety_percent_points():
    f = F.bounded_sigmoid(0, 1)
    assert f(0) == pytest.approx(0.1)
    assert f(1) == pytest.approx(0.9)


def test_gauss_peak_and_falloff():
    f = F.gauss(0, 1)
    assert f(0) == pytest.approx(1.0)
    assert f(1) == pytest.approx(pytest.approx(0.36787944117144233))
