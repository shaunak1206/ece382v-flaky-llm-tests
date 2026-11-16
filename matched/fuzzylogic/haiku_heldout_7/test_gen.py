import math

import pytest

from fuzzylogic import classes
from fuzzylogic import functions as F
from fuzzylogic.classes import Domain, Set, FuzzyWarning


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_init_and_range():
    d = Domain("temp", 0, 100, res=2)
    assert str(d) == "temp"
    r = d.range()
    assert len(r) == 51  # 0, 2, ..., 100 with upper bound included
    assert r[0] == 0
    assert r[-1] == 100


def test_domain_rejects_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("x", 5, 5)
    with pytest.raises(AssertionError):
        Domain("x", 5, 1)
    with pytest.raises(AssertionError):
        Domain("x", 0, 1, res=0)


def test_assigning_function_creates_set_and_set_constructor_binds_name():
    d = Domain("t", 0, 10)
    d.hot = F.constant(0.7)
    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d.hot(3) == 0.7

    Set(F.constant(0.4), name="cold", domain=d)
    assert isinstance(d.cold, Set)
    assert d.cold.name == "cold"
    assert d.cold(0) == 0.4


def test_domain_call_returns_memberships_per_set():
    d = Domain("t", 0, 10)
    d.cold = F.constant(0.2)
    d.hot = F.constant(0.7)
    assert d(5) == {"cold": 0.2, "hot": 0.7}


def test_domain_call_outside_range_logs_warning(caplog):
    d = Domain("t", 0, 10)
    d.a = F.constant(0.5)
    result = d(20)
    assert result == {"a": 0.5}
    assert "outside of domain" in caplog.text


def test_domain_min_and_max_over_sets():
    d = Domain("t", 0, 10)
    d.a = F.constant(0.2)
    d.b = F.constant(0.9)
    assert d.min(5) == 0.2
    assert d.max(5) == 0.9


def test_delete_set_and_unknown_attribute_errors():
    d = Domain("t", 0, 10)
    d.a = F.constant(0.5)
    del d.a
    assert not hasattr(d, "a")

    with pytest.raises(AttributeError):
        d.missing
    with pytest.raises(FuzzyWarning):
        del d.nothing


def test_setattr_rejects_non_identifier_names():
    d = Domain("t", 0, 10)
    with pytest.raises(AssertionError):
        setattr(d, "not valid", F.constant(1))


def test_domain_equality():
    assert Domain("t", 0, 10) == Domain("t", 0, 10)
    assert not (Domain("t", 0, 10) != Domain("t", 0, 10))
    assert Domain("t", 0, 10) != Domain("t", 0, 11)
    assert Domain("t", 0, 10) != Domain("u", 0, 10)


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

def test_set_call_invert_and_neg():
    s = Set(F.constant(0.3))
    assert s(0) == 0.3
    assert (~s)(0) == pytest.approx(0.7)
    assert (-s)(0) == pytest.approx(0.7)


def test_set_and_or_use_min_and_max():
    a = Set(F.constant(0.3))
    b = Set(F.constant(0.8))
    assert (a & b)(0) == pytest.approx(0.3)
    assert (a | b)(0) == pytest.approx(0.8)


def test_set_pow_multiplied_concentrated():
    s = Set(F.constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.multiplied(3)(0) == pytest.approx(1.5)
    assert s.concentrated()(0) == pytest.approx(0.25)


def test_set_str_and_repr():
    dangling = Set(F.constant(1))
    assert str(dangling).startswith("dangling Set(")
    assert repr(dangling).startswith("Set(")

    d = Domain("temp", 0, 10)
    d.hot = F.constant(1)
    assert str(d.hot) == "temp.hot"


def test_set_name_and_domain_must_be_given_together():
    d = Domain("t", 0, 10)
    with pytest.raises(FuzzyWarning):
        Set(F.constant(1), name="x")
    with pytest.raises(FuzzyWarning):
        Set(F.constant(1), domain=d)


def test_array_len_cardinality_and_domain_requirement():
    d = Domain("t", 0, 10)
    d.a = Set(F.constant(0.5))
    assert len(d.a) == 11
    assert len(d.a.array()) == 11
    assert d.a.cardinality() == pytest.approx(5.5)
    assert d.a.relative_cardinality() == pytest.approx(0.5)

    no_domain = Set(F.constant(0.5))
    with pytest.raises(FuzzyWarning):
        no_domain.array()
    with pytest.raises(FuzzyWarning):
        len(no_domain)
    with pytest.raises(FuzzyWarning):
        no_domain.cardinality()


def test_subset_and_superset_comparisons():
    d = Domain("t", 0, 10)
    d.a = Set(F.constant(0.2))
    d.b = Set(F.constant(0.5))

    assert d.a <= d.b
    assert d.a < d.b
    assert d.b >= d.a
    assert d.b > d.a
    assert not (d.b <= d.a)
    assert not (d.a < d.a)
    assert d.a <= d.a


def test_equality_requires_domains_and_compares_values():
    with pytest.raises(FuzzyWarning):
        Set(F.constant(1)) == Set(F.constant(1))

    d = Domain("t", 0, 10)
    d.a = Set(F.constant(0.5))
    d.b = Set(F.constant(0.5))
    d.c = Set(F.constant(0.6))
    assert d.a == d.b
    assert not (d.a == d.c)


def test_plot_requires_domain_and_passes_values(monkeypatch):
    with pytest.raises(FuzzyWarning):
        Set(F.constant(0.5)).plot()

    calls = []

    def fake_plot(x, y, *args, **kwargs):
        calls.append((list(x), list(y)))

    monkeypatch.setattr(classes.plt, "plot", fake_plot)

    d = Domain("t", 0, 4)
    d.a = Set(F.constant(0.5))
    d.a.plot()
    assert len(calls) == 1
    xs, ys = calls[0]
    assert len(xs) == 5
    assert all(y == pytest.approx(0.5) for y in ys)


def test_normalized_scales_max_to_one():
    d = Domain("t", 0, 10)
    d.a = Set(F.constant(0.5))
    n = d.a.normalized()
    assert n(3) == pytest.approx(1.0)

    with pytest.raises(FuzzyWarning):
        Set(F.constant(0.5)).normalized()


# ---------------------------------------------------------------------------
# functions module
# ---------------------------------------------------------------------------

def test_inv_noop_constant_and_singleton():
    assert F.inv(F.constant(0.3))(0) == pytest.approx(0.7)
    assert F.noop()(5) == 5
    assert F.noop()(-2.5) == -2.5
    assert F.constant(0.4)(100) == 0.4

    assert F.singleton(2)(2) == 1
    assert F.singleton(2)(1) == 0
    f = F.singleton(2, no_m=0.1, c_m=0.9)
    assert f(2) == 0.9
    assert f(3) == 0.1
    with pytest.raises(AssertionError):
        F.singleton(2, no_m=0.9, c_m=0.1)


def test_alpha_normalize_and_moderate():
    clipped = F.alpha(func=lambda x: x, floor=0.2, ceiling=0.8)
    assert clipped(0.1) == pytest.approx(0.2)
    assert clipped(0.9) == pytest.approx(0.8)
    assert clipped(0.5) == pytest.approx(0.5)

    hard = F.alpha(func=lambda x: x, floor=0.2, ceiling=0.8,
                   floor_clip=0, ceiling_clip=1)
    assert hard(0.1) == 0
    assert hard(0.9) == 1

    assert F.normalize(0.5, F.constant(0.25))(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        F.normalize(0, F.constant(0.25))

    assert F.moderate(F.constant(1))(0) == pytest.approx(1.0)
    assert F.moderate(F.constant(0))(0) == pytest.approx(0.0)
    assert F.moderate(F.constant(0.5))(0) == pytest.approx(0.5)


def test_linear_and_bounded_linear():
    f = F.linear(1, -1)
    assert f(-2) == 0
    assert f(0) == 0
    assert f(1) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(2) == 1
    assert f(3) == 1

    g = F.bounded_linear(2, 3)
    assert g(1) == 0.0
    assert g(2.5) == pytest.approx(0.5)
    assert g(3) == pytest.approx(1.0)
    assert g(4) == pytest.approx(1.0)

    inv_g = F.bounded_linear(2, 3, inverse=True)
    assert inv_g(2) == pytest.approx(1.0)
    assert inv_g(2.5) == pytest.approx(0.5)
    assert inv_g(3) == pytest.approx(0.0)


def test_R_and_S_ramps():
    r = F.R(0, 10)
    assert r(-1) == 0
    assert r(5) == pytest.approx(0.5)
    assert r(10) == pytest.approx(1.0)
    assert r(11) == 1

    s = F.S(0, 10)
    assert s(-1) == 1
    assert s(0) == 1
    assert s(5) == pytest.approx(0.5)
    assert s(10) == 0


def test_rectangular_triangular_and_trapezoid_shapes():
    rect = F.rectangular(2, 4, c_m=0.8, no_m=0.1)
    assert rect(1) == 0.1
    assert rect(3) == 0.8
    assert rect(5) == 0.1

    tri = F.triangular(0, 10)
    assert tri(5) == pytest.approx(1.0)
    assert tri(0) == pytest.approx(0.0)
    assert tri(10) == pytest.approx(0.0)
    assert tri(2.5) == pytest.approx(0.5)
    assert tri(7.5) == pytest.approx(0.5)

    trap = F.trapezoid(0, 2, 4, 6)
    assert trap(3) == 1
    assert trap(1) == pytest.approx(0.5)
    assert trap(5) == pytest.approx(0.5)
    assert trap(7) == 0
    assert trap(-1) == 0


def test_sigmoids_bounded_exponential_and_gauss():
    s = F.sigmoid(1, 1, 0)
    assert s(0) == pytest.approx(0.5)
    assert s(-1000) == pytest.approx(0.0)
    assert s(1000) == pytest.approx(1.0)

    bs = F.bounded_sigmoid(0, 1)
    assert bs(0) == pytest.approx(0.1)
    assert bs(1) == pytest.approx(0.9)
    assert bs(100000) == pytest.approx(1.0)
    assert bs(-100000) == pytest.approx(0.0)

    be = F.bounded_exponential(k=1, limit=1)
    assert be(0) == pytest.approx(0.0)
    assert be(1) == pytest.approx(1 - math.exp(-1))
    assert be(10000) == 1

    ss = F.simple_sigmoid(k=1)
    assert ss(0) == pytest.approx(0.5)
    assert ss(1000) == pytest.approx(1.0)
    assert ss(-1000) == pytest.approx(0.0)
    assert F.simple_sigmoid(k=0)(float("inf")) == 0.5

    ts = F.triangular_sigmoid(2, 4)
    assert ts(2) == pytest.approx(0.1)
    assert ts(3) == pytest.approx(0.9)
    assert ts(4) == pytest.approx(0.1)

    g = F.gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) == pytest.approx(math.exp(-1))
    assert F.gauss(0, 1, c_m=0.5)(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        F.gauss(0, 0)
