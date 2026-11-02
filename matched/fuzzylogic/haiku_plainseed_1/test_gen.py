"""Unit tests for fuzzylogic.classes (Domain, Set) and fuzzylogic.functions."""

import logging
import math

import pytest

from fuzzylogic import functions as F
from fuzzylogic.classes import Domain, Set, FuzzyWarning


# ----------------------------------------------------------------- Domain

def test_domain_rejects_bad_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("t", 5, 5)
    with pytest.raises(AssertionError):
        Domain("t", 5, 1)
    with pytest.raises(AssertionError):
        Domain("t", 0, 1, res=0)
    with pytest.raises(AssertionError):
        Domain("t", 0, 1, res=-1)


def test_domain_range_includes_upper_bound():
    d = Domain("t", 0, 1, res=0.25)
    assert list(d.range()) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    d10 = Domain("t", 0, 10)
    assert len(d10.range()) == 11
    assert d10.range()[-1] == 10


def test_domain_attribute_assignment_wraps_callables_and_call_returns_memberships():
    d = Domain("temp", 0, 100)
    d.hot = lambda x: 0.5
    existing = Set(F.constant(0.25))
    d.cold = existing

    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d.cold is existing
    assert d(50) == {"hot": 0.5, "cold": 0.25}


def test_domain_call_out_of_range_logs_warning(caplog):
    d = Domain("t", 0, 10)
    d.a = F.constant(1)

    with caplog.at_level(logging.WARNING):
        inside = d(5)
    assert inside == {"a": 1}
    assert not [r for r in caplog.records if "outside of domain" in r.getMessage()]

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        outside = d(42)
    assert outside == {"a": 1}
    assert any("outside of domain" in r.getMessage() for r in caplog.records)


def test_domain_unknown_attribute_and_set_deletion():
    d = Domain("t", 0, 10)
    d.a = F.constant(0.1)
    assert d.a(0) == 0.1

    del d.a
    with pytest.raises(AttributeError):
        _ = d.a
    with pytest.raises(AttributeError):
        _ = d.never_defined
    with pytest.raises(FuzzyWarning):
        del d.never_defined


def test_domain_rejects_non_identifier_set_names():
    d = Domain("t", 0, 10)
    with pytest.raises(AssertionError):
        setattr(d, "not an identifier", F.constant(0))
    with pytest.raises(AssertionError):
        setattr(d, "1abc", F.constant(0))


def test_domain_min_max_str_repr_and_equality():
    d = Domain("t", 0, 10)
    d.low = F.constant(0.2)
    d.high = F.constant(0.7)
    assert d.min(5) == pytest.approx(0.2)
    assert d.max(5) == pytest.approx(0.7)

    assert str(d) == "t"
    assert repr(d).startswith("Domain('t', 0, 10, res=1, sets=")

    assert Domain("t", 0, 10) == Domain("t", 0, 10)
    assert not (Domain("t", 0, 10) == Domain("u", 0, 10))


# -------------------------------------------------------------------- Set

def test_set_construction_rules():
    with pytest.raises(AssertionError):
        Set(5)

    # a name without a domain (or a domain without a name) is an error
    with pytest.raises(FuzzyWarning):
        Set(F.constant(0.5), name="hot")
    with pytest.raises(FuzzyWarning):
        Set(F.constant(0.5), domain=Domain("t", 0, 10))

    # with both, the set is registered on the domain
    d = Domain("t", 0, 10)
    s = Set(F.constant(0.5), name="hot", domain=d)
    assert d.hot is s
    assert s.name == "hot"
    assert str(s) == "t.hot"


def test_set_invert_and_negation_complement_membership():
    s = Set(F.constant(0.25))
    assert (~s)(0) == pytest.approx(0.75)
    assert (-s)(0) == pytest.approx(0.75)
    assert (~~s)(0) == pytest.approx(0.25)


def test_set_and_or_use_min_and_max():
    a = Set(F.constant(0.3))
    b = Set(F.constant(0.6))
    assert (a & b)(0) == pytest.approx(0.3)
    assert (a | b)(0) == pytest.approx(0.6)


def test_set_mul_add_xor_combinators():
    half = Set(F.constant(0.5))
    forty = Set(F.constant(0.4))
    assert (half * forty)(0) == pytest.approx(0.2)
    # bounded sum: x + y - x*y
    assert (half + half)(0) == pytest.approx(0.75)

    zero = Set(F.constant(0))
    one = Set(F.constant(1))
    assert (zero ^ one)(0) == pytest.approx(1)
    assert (zero ^ zero)(0) == pytest.approx(0)
    assert (one ^ one)(0) == pytest.approx(0)


def test_set_power_and_hedges_transform_membership():
    s = Set(F.constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.concentrated()(0) == pytest.approx(0.25)
    assert s.multiplied(3)(0) == pytest.approx(1.5)
    # intensified() low branch (x < 0.5): 2 * m**2
    assert s.intensified()(0) == pytest.approx(0.5)


def test_set_array_and_len_follow_domain_resolution():
    d = Domain("t", 0, 10)
    d.half = F.constant(0.5)
    arr = d.half.array()
    assert len(arr) == 11
    assert len(d.half) == 11
    assert list(arr) == pytest.approx([0.5] * 11)


def test_set_cardinality_relative_cardinality_and_normalized():
    d = Domain("t", 0, 10)
    d.half = F.constant(0.5)
    assert d.half.cardinality() == pytest.approx(5.5)
    assert d.half.relative_cardinality() == pytest.approx(0.5)

    n = d.half.normalized()
    assert n(3) == pytest.approx(1.0)
    assert str(n).startswith("dangling Set(")


def test_set_domain_dependent_operations_raise_without_domain():
    s = Set(F.constant(0.5))
    t = Set(F.constant(0.5))
    with pytest.raises(FuzzyWarning):
        s.array()
    with pytest.raises(FuzzyWarning):
        len(s)
    with pytest.raises(FuzzyWarning):
        s.cardinality()
    with pytest.raises(FuzzyWarning):
        s.relative_cardinality()
    with pytest.raises(FuzzyWarning):
        s.normalized()
    with pytest.raises(FuzzyWarning):
        _ = (s == t)
    with pytest.raises(FuzzyWarning):
        _ = (s <= t)
    with pytest.raises(FuzzyWarning):
        _ = (s < t)


def test_set_subset_and_superset_comparisons_on_domain():
    d = Domain("t", 0, 4)
    d.low = F.constant(0.2)
    d.same = F.constant(0.2)
    d.high = F.constant(0.5)
    low, same, high = d.low, d.same, d.high

    assert low <= high
    assert low < high
    assert high >= low
    assert high > low
    assert not high <= low

    assert low <= same and not (low < same)
    assert low >= same and not (low > same)


def test_set_equality_compares_membership_arrays():
    d = Domain("t", 0, 4)
    d.a = F.constant(0.2)
    d.b = F.constant(0.2)
    d.c = F.constant(0.3)
    assert d.a == d.b
    assert not (d.a == d.c)
    assert d.a != d.c


# -------------------------------------------------------------- functions

def test_inv_constant_singleton_and_linear_basics():
    assert F.inv(F.constant(0.3))(0) == pytest.approx(0.7)
    assert F.constant(0.4)(123) == 0.4

    spike = F.singleton(2)
    assert spike(2) == 1
    assert spike(1) == 0
    spike2 = F.singleton(2, no_m=0.2, c_m=0.9)
    assert spike2(3) == pytest.approx(0.2)
    assert spike2(2) == pytest.approx(0.9)

    lin = F.linear(1, -1)
    assert lin(-2) == 0
    assert lin(1.5) == pytest.approx(0.5)
    assert lin(3) == 1


def test_R_S_and_rectangular_ramps_and_plateaus():
    r = F.R(0, 2)
    assert r(-1) == 0
    assert r(1) == pytest.approx(0.5)
    assert r(3) == 1

    s = F.S(0, 2)
    assert s(-1) == 1
    assert s(1) == pytest.approx(0.5)
    assert s(2) == 0

    rect = F.rectangular(1, 3, c_m=0.8, no_m=0.1)
    assert rect(0) == pytest.approx(0.1)
    assert rect(1) == pytest.approx(0.8)
    assert rect(3) == pytest.approx(0.8)
    assert rect(4) == pytest.approx(0.1)


def test_triangular_trapezoid_and_bounded_linear_shapes():
    tri = F.triangular(0, 2)
    assert tri(0) == pytest.approx(0)
    assert tri(0.5) == pytest.approx(0.5)
    assert tri(1) == pytest.approx(1)
    assert tri(1.5) == pytest.approx(0.5)
    assert tri(2) == pytest.approx(0)

    trap = F.trapezoid(0, 1, 2, 3)
    assert trap(-1) == 0
    assert trap(0.5) == pytest.approx(0.5)
    assert trap(1.5) == 1
    assert trap(2.5) == pytest.approx(0.5)
    assert trap(3.5) == 0

    bl = F.bounded_linear(2, 3)
    assert bl(1) == 0
    assert bl(2.5) == pytest.approx(0.5)
    assert bl(4) == 1

    inv_bl = F.bounded_linear(2, 3, inverse=True)
    assert inv_bl(2) == pytest.approx(1)
    assert inv_bl(2.5) == pytest.approx(0.5)
    assert inv_bl(4) == 0


def test_sigmoid_exponential_and_gauss_values():
    bs = F.bounded_sigmoid(0, 1)
    assert bs(0) == pytest.approx(0.1)
    assert bs(1) == pytest.approx(0.9)
    assert bs(100000) == pytest.approx(1.0)
    assert bs(-100000) == pytest.approx(0.0)

    ss = F.simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(20) == pytest.approx(0.99, abs=0.01)
    assert ss(-1000) == pytest.approx(0.0, abs=1e-9)

    lg = F.sigmoid(1, 1, 0)
    assert lg(0) == pytest.approx(0.5)

    be = F.bounded_exponential(k=1, limit=1)
    assert be(0) == pytest.approx(0)
    assert be(1) == pytest.approx(1 - 1 / math.e)
    assert be(100) == pytest.approx(1)

    g = F.gauss(2, 1)
    assert g(2) == pytest.approx(1)
    assert g(3) == pytest.approx(math.exp(-1))
    assert F.gauss(0, 1, c_m=0.5)(0) == pytest.approx(0.5)


def test_alpha_moderate_and_normalize_transform_values():
    clipped_top = F.alpha(func=F.constant(0.9), ceiling=0.8, ceiling_clip=0.3)
    assert clipped_top(0) == pytest.approx(0.3)

    clipped_bottom = F.alpha(func=F.constant(0.1), floor=0.2)
    assert clipped_bottom(0) == pytest.approx(0.2)

    untouched = F.alpha(func=F.constant(0.5), floor=0.2, ceiling=0.8)
    assert untouched(0) == pytest.approx(0.5)

    assert F.moderate(F.constant(0.75))(0) == pytest.approx(0.5625)
    assert F.moderate(F.constant(0))(0) == pytest.approx(0)
    assert F.moderate(F.constant(1))(0) == pytest.approx(1)

    assert F.normalize(0.5, F.constant(0.25))(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        F.normalize(0, F.constant(0.25))
