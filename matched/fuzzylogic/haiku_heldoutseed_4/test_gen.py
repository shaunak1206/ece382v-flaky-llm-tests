import math

import pytest

from fuzzylogic import functions as fz
from fuzzylogic.classes import Domain, FuzzyWarning, Set


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_name_and_range_include_upper_bound():
    d = Domain("temperature", 0, 3)
    assert str(d) == "temperature"
    assert list(d.range()) == [0, 1, 2, 3]

    fine = Domain("x", 0, 2, res=0.5)
    assert len(fine.range()) == 5
    assert fine.range()[-1] == pytest.approx(2.0)


def test_domain_rejects_invalid_bounds_or_resolution():
    with pytest.raises(AssertionError):
        Domain("t", 10, 0)
    with pytest.raises(AssertionError):
        Domain("t", 0, 10, res=0)
    with pytest.raises(AssertionError):
        Domain("t", 0, 10, res=-1)


def test_domain_call_returns_membership_per_set():
    d = Domain("temp", 0, 10)
    d.cold = fz.constant(0.25)
    d.hot = fz.constant(0.75)
    assert d(5) == {"cold": 0.25, "hot": 0.75}


def test_domain_assignment_registers_set_name_and_domain():
    d = Domain("temp", 0, 10)
    s = Set(fz.constant(0.5))
    d.warm = s
    assert d.warm is s
    assert s.name == "warm"
    assert s.domain is d


def test_domain_unknown_and_delete_attributes():
    d = Domain("t", 0, 10)
    with pytest.raises(AttributeError):
        _ = d.missing

    d.hot = fz.constant(1)
    del d.hot
    assert not hasattr(d, "hot")

    with pytest.raises(FuzzyWarning):
        del d.never_defined


def test_domain_min_and_max_over_sets():
    d = Domain("t", 0, 10)
    d.low = fz.constant(0.2)
    d.high = fz.constant(0.9)
    assert d.min(5) == pytest.approx(0.2)
    assert d.max(5) == pytest.approx(0.9)


def test_domain_equality_depends_on_parameters():
    d1 = Domain("t", 0, 10)
    d2 = Domain("t", 0, 10)
    assert d1 == d2
    assert d1 != Domain("t", 0, 11)
    assert d1 != Domain("other", 0, 10)


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

def test_set_init_registers_with_domain_and_requires_both():
    d = Domain("t", 0, 10)
    s = Set(fz.constant(0.5), name="mid", domain=d)
    assert d.mid is s

    with pytest.raises(FuzzyWarning):
        Set(fz.constant(0.5), name="x")


def test_set_invert_neg_and_or():
    s = Set(fz.constant(0.3))
    assert (~s)(0) == pytest.approx(0.7)
    assert (-s)(0) == pytest.approx(0.7)

    a = Set(fz.constant(0.3))
    b = Set(fz.constant(0.7))
    assert (a & b)(0) == pytest.approx(0.3)
    assert (a | b)(0) == pytest.approx(0.7)


def test_set_pow_multiplied_concentrated():
    s = Set(fz.constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.concentrated()(0) == pytest.approx(0.25)
    assert Set(fz.constant(0.8)).multiplied(0.5)(0) == pytest.approx(0.4)


def test_set_len_cardinality_and_relative_cardinality():
    d = Domain("t", 0, 10)  # 11 values with res=1
    d.half = fz.constant(0.5)
    assert len(d.half) == 11
    assert d.half.cardinality() == pytest.approx(5.5)
    assert d.half.relative_cardinality() == pytest.approx(0.5)
    assert len(d.half.array()) == 11


def test_set_equality_compares_values_over_domain():
    d = Domain("t", 0, 5)
    d.a = fz.constant(0.5)
    d.b = Set(lambda x: 0.5)
    d.c = fz.constant(0.4)
    assert bool(d.a == d.b)
    assert not bool(d.a == d.c)

    with pytest.raises(FuzzyWarning):
        Set(fz.constant(0.5)) == Set(fz.constant(0.5))


def test_set_subset_and_superset_comparisons():
    d = Domain("t", 0, 5)
    d.low = fz.constant(0.2)
    d.high = fz.constant(0.8)
    assert d.low <= d.high
    assert d.low < d.high
    assert d.high >= d.low
    assert d.high > d.low
    assert not (d.high <= d.low)


def test_set_str_forms_and_normalized():
    s = Set(fz.constant(0.5))
    assert str(s).startswith("dangling Set")

    d = Domain("temp", 0, 10)
    d.hot = fz.constant(0.5)
    assert str(d.hot) == "temp.hot"

    n = d.hot.normalized()
    assert n(3) == pytest.approx(1.0)

    with pytest.raises(FuzzyWarning):
        Set(fz.constant(0.5)).normalized()


# ---------------------------------------------------------------------------
# functions
# ---------------------------------------------------------------------------

def test_inv_and_alpha():
    assert fz.inv(fz.constant(0.3))(0) == pytest.approx(0.7)

    f = fz.alpha(floor=0.2, ceiling=0.8, func=lambda x: x)
    assert f(0.1) == pytest.approx(0.2)
    assert f(0.5) == pytest.approx(0.5)
    assert f(0.95) == pytest.approx(0.8)


def test_normalize_rescales_and_rejects_bad_height():
    assert fz.normalize(0.5, fz.constant(0.25))(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        fz.normalize(0, fz.constant(0.5))
    with pytest.raises(AssertionError):
        fz.normalize(1.5, fz.constant(0.5))


def test_singleton_and_rectangular():
    s = fz.singleton(2)
    assert s(1) == 0
    assert s(2) == 1

    s2 = fz.singleton(3, no_m=0.1, c_m=0.9)
    assert s2(5) == pytest.approx(0.1)
    assert s2(3) == pytest.approx(0.9)

    r = fz.rectangular(2, 4, c_m=0.8, no_m=0.1)
    assert r(1) == pytest.approx(0.1)
    assert r(2) == pytest.approx(0.8)
    assert r(3) == pytest.approx(0.8)
    assert r(5) == pytest.approx(0.1)


def test_linear_and_bounded_linear():
    lin = fz.linear(1, -1)
    assert lin(-2) == 0
    assert lin(1.5) == pytest.approx(0.5)
    assert lin(3) == 1

    bl = fz.bounded_linear(2, 3)
    assert bl(1) == pytest.approx(0.0)
    assert bl(2.5) == pytest.approx(0.5)
    assert bl(4) == pytest.approx(1.0)

    inv_bl = fz.bounded_linear(2, 3, inverse=True)
    assert inv_bl(2) == pytest.approx(1.0)
    assert inv_bl(3) == pytest.approx(0.0)


def test_R_and_S_ramps():
    r = fz.R(0, 10)
    assert r(-1) == 0
    assert r(5) == pytest.approx(0.5)
    assert r(20) == 1

    s = fz.S(0, 10)
    assert s(-1) == 1
    assert s(5) == pytest.approx(0.5)
    assert s(10) == 0


def test_triangular_and_trapezoid():
    t = fz.triangular(0, 10)
    assert t(0) == pytest.approx(0.0)
    assert t(2.5) == pytest.approx(0.5)
    assert t(5) == pytest.approx(1.0)
    assert t(7.5) == pytest.approx(0.5)
    assert t(10) == pytest.approx(0.0)

    tr = fz.trapezoid(0, 2, 4, 6)
    assert tr(-1) == 0
    assert tr(1) == pytest.approx(0.5)
    assert tr(3) == pytest.approx(1.0)
    assert tr(5) == pytest.approx(0.5)
    assert tr(7) == 0


def test_sigmoid_and_simple_sigmoid():
    s = fz.sigmoid(1, 1, 0)
    assert s(0) == pytest.approx(0.5)
    assert s(50) == pytest.approx(1.0)
    assert s(-50) == pytest.approx(0.0, abs=1e-9)

    ss = fz.simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(20) == pytest.approx(0.99, abs=1e-3)
    assert ss(-1000) == pytest.approx(0.0, abs=1e-9)
    assert fz.simple_sigmoid(0)(float("inf")) == pytest.approx(0.5)


def test_bounded_sigmoid_and_triangular_sigmoid():
    bs = fz.bounded_sigmoid(0, 1)
    assert bs(0) == pytest.approx(0.1)
    assert bs(1) == pytest.approx(0.9)
    assert bs(100000) == pytest.approx(1.0)
    assert bs(-100000) == pytest.approx(0.0, abs=1e-9)

    inv_bs = fz.bounded_sigmoid(0, 1, inverse=True)
    assert inv_bs(0) == pytest.approx(0.9)

    g = fz.triangular_sigmoid(2, 4)
    assert g(2) == pytest.approx(0.1)
    assert g(3) == pytest.approx(0.9)
    assert g(4) == pytest.approx(0.1)


def test_bounded_exponential_gauss_and_moderate():
    be = fz.bounded_exponential(k=0.1, limit=1)
    assert be(0) == pytest.approx(0.0)
    assert be(10) == pytest.approx(1 - math.exp(-1))
    assert be(10000) == pytest.approx(1.0)

    g = fz.gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) == pytest.approx(math.exp(-1))
    assert fz.gauss(0, 1, c_m=0.5)(0) == pytest.approx(0.5)

    assert fz.moderate(fz.constant(0.5))(0) == pytest.approx(0.5)
    assert fz.moderate(fz.constant(1))(0) == pytest.approx(1.0)
    assert fz.moderate(fz.constant(0))(0) == pytest.approx(0.0)
