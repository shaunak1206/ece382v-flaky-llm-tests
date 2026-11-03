import math

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv, noop, constant, alpha, normalize, moderate, singleton, linear,
    bounded_linear, R, S, rectangular, triangular, trapezoid, sigmoid,
    bounded_sigmoid, bounded_exponential, simple_sigmoid, gauss,
)


def make_domain():
    d = Domain("d", 0, 4)
    d.a = R(0, 4)
    d.b = S(0, 4)
    return d


# ---------------------------------------------------------------- Domain

def test_domain_str_and_set_assignment():
    d = Domain("temperature", 0, 100)
    d.hot = R(50, 100)
    assert str(d) == "temperature"
    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d.hot(75) == pytest.approx(0.5)


def test_domain_call_returns_memberships_per_set():
    d = make_domain()
    result = d(1)
    assert set(result.keys()) == {"a", "b"}
    assert result["a"] == pytest.approx(0.25)
    assert result["b"] == pytest.approx(0.75)


def test_domain_range_includes_upper_bound():
    d = Domain("d", 0, 4)
    assert list(d.range()) == [0, 1, 2, 3, 4]
    d2 = Domain("d2", 0, 1, res=0.25)
    assert len(d2.range()) == 5


def test_domain_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("bad", 5, 5)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_domain_missing_attribute_raises_attribute_error():
    d = make_domain()
    with pytest.raises(AttributeError):
        d.does_not_exist


def test_domain_delete_set_and_regular_attr():
    d = make_domain()
    del d.a
    assert "a" not in d(1)
    with pytest.raises(AttributeError):
        d.a
    with pytest.raises(FuzzyWarning):
        del d.nonexistent


def test_domain_min_and_max():
    d = make_domain()
    assert d.min(1) == pytest.approx(0.25)
    assert d.max(1) == pytest.approx(0.75)


def test_domain_set_name_must_be_identifier():
    d = Domain("d", 0, 4)
    with pytest.raises(AssertionError):
        setattr(d, "not valid", R(0, 4))


# ------------------------------------------------------------------- Set

def test_set_logical_operators():
    d = make_domain()
    assert (~d.a)(1) == pytest.approx(0.75)
    assert (-d.a)(1) == pytest.approx(0.75)
    assert (d.a & d.b)(1) == pytest.approx(0.25)
    assert (d.a | d.b)(1) == pytest.approx(0.75)
    assert (d.a * d.b)(1) == pytest.approx(0.25 * 0.75)
    assert (d.a + d.b)(1) == pytest.approx(0.25 + 0.75 - 0.25 * 0.75)


def test_set_array_len_and_cardinality():
    d = make_domain()
    arr = d.a.array()
    assert list(arr) == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
    assert len(d.a) == 5
    assert d.a.cardinality() == pytest.approx(2.5)
    assert d.a.relative_cardinality() == pytest.approx(0.5)


def test_set_without_domain_raises_fuzzywarning():
    s = Set(R(0, 4))
    with pytest.raises(FuzzyWarning):
        s.array()
    with pytest.raises(FuzzyWarning):
        len(s)
    with pytest.raises(FuzzyWarning):
        s.cardinality()
    with pytest.raises(FuzzyWarning):
        s.normalized()
    with pytest.raises(FuzzyWarning):
        s.plot()


def test_set_equality_requires_domains():
    d = make_domain()
    other = Domain("e", 0, 4)
    other.x = R(0, 4)
    assert d.a == other.x
    assert not (d.a == d.b)
    with pytest.raises(FuzzyWarning):
        Set(R(0, 4)) == Set(R(0, 4))


def test_set_subset_superset_comparisons():
    d = make_domain()
    d.half = d.a.multiplied(0.5)
    assert d.half <= d.a
    assert d.a >= d.half
    assert not (d.a <= d.half)
    with pytest.raises(FuzzyWarning):
        Set(R(0, 4)) <= d.a


def test_set_pow_concentrated_and_multiplied():
    d = make_domain()
    assert (d.a ** 2)(2) == pytest.approx(0.25)
    assert d.a.concentrated()(2) == pytest.approx(0.25)
    assert d.a.multiplied(0.5)(4) == pytest.approx(0.5)


def test_set_normalized_scales_max_to_one():
    d = make_domain()
    d.half = d.a.multiplied(0.5)
    n = d.half.normalized()
    assert n(4) == pytest.approx(1.0)
    assert n(2) == pytest.approx(0.5)


def test_set_constructor_name_and_domain():
    d = Domain("d", 0, 4)
    s = Set(R(0, 4), name="z", domain=d)
    assert d.z is s
    assert str(s) == "d.z"
    with pytest.raises(FuzzyWarning):
        Set(R(0, 4), name="z")
    assert str(Set(R(0, 4))).startswith("dangling Set(")


# ------------------------------------------------------------- functions

def test_special_functions_inv_noop_constant():
    assert inv(constant(0.3))(5) == pytest.approx(0.7)
    assert noop()(0.42) == 0.42
    assert constant(1)(-100) == 1


def test_alpha_normalize_and_moderate():
    f = alpha(floor=0.2, ceiling=0.8, func=constant(0.9))
    assert f(0) == pytest.approx(0.8)
    assert alpha(floor=0.2, ceiling=0.8, func=constant(0.1))(0) == pytest.approx(0.2)
    assert alpha(floor=0.2, ceiling=0.8, func=constant(0.5))(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        alpha(floor=0.9, ceiling=0.1, func=constant(0.5))
    assert normalize(0.5, constant(0.25))(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        normalize(0, constant(0.25))
    assert moderate(constant(0.5))(0) == pytest.approx(0.5)
    assert moderate(constant(1))(0) == pytest.approx(1.0)
    assert moderate(constant(0))(0) == pytest.approx(0.0)


def test_singleton_and_rectangular():
    s = singleton(2)
    assert s(2) == 1
    assert s(1) == 0
    s2 = singleton(2, c_m=0.9, no_m=0.1)
    assert s2(2) == 0.9
    assert s2(3) == 0.1
    r = rectangular(1, 3)
    assert [r(x) for x in (0, 1, 2, 3, 4)] == [0, 1, 1, 1, 0]
    r2 = rectangular(1, 3, c_m=0.8, no_m=0.1)
    assert r2(2) == 0.8
    assert r2(5) == 0.1


def test_linear_is_clipped():
    f = linear(1, -1)
    assert f(-2) == 0
    assert f(0) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(2) == 1
    assert f(3) == 1


def test_bounded_linear_and_inverse():
    f = bounded_linear(2, 3)
    assert f(1) == 0
    assert f(2.5) == pytest.approx(0.5)
    assert f(4) == 1
    g = bounded_linear(2, 3, inverse=True)
    assert g(2) == pytest.approx(1.0)
    assert g(2.5) == pytest.approx(0.5)
    assert g(4) == 0
    with pytest.raises(AssertionError):
        bounded_linear(3, 2)


def test_R_and_S_slopes():
    r = R(0, 10)
    assert r(-1) == 0
    assert r(5) == pytest.approx(0.5)
    assert r(11) == 1
    s = S(0, 10)
    assert s(-5) == 1
    assert s(5) == pytest.approx(0.5)
    assert s(20) == 0
    with pytest.raises(AssertionError):
        R(5, 1)
    with pytest.raises(AssertionError):
        S(5, 1)


def test_triangular_and_trapezoid_shapes():
    t = triangular(0, 10)
    assert t(-1) == pytest.approx(0)
    assert t(2.5) == pytest.approx(0.5)
    assert t(5) == pytest.approx(1)
    assert t(7.5) == pytest.approx(0.5)
    assert t(12) == pytest.approx(0)
    with pytest.raises(AssertionError):
        triangular(0, 10, c=10)
    z = trapezoid(0, 2, 4, 6)
    assert z(-1) == 0
    assert z(1) == pytest.approx(0.5)
    assert z(2) == 1
    assert z(3) == 1
    assert z(4) == 1
    assert z(5) == pytest.approx(0.5)
    assert z(7) == 0


def test_sigmoid_variants():
    s = sigmoid(1, 1, 0)
    assert s(0) == pytest.approx(0.5)
    assert s(100) == pytest.approx(1.0)
    assert s(-1000) == pytest.approx(0.0)
    assert sigmoid(0.8, 1, 0)(0) == pytest.approx(0.4)
    with pytest.raises(AssertionError):
        sigmoid(0, 1, 0)
    b = bounded_sigmoid(0, 1)
    assert b(0) == pytest.approx(0.1)
    assert b(1) == pytest.approx(0.9)
    bi = bounded_sigmoid(0, 1, inverse=True)
    assert bi(0) == pytest.approx(0.9)
    assert bi(1) == pytest.approx(0.1)
    ss = simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(1000) == pytest.approx(1.0)
    assert ss(-1000) == pytest.approx(0.0)


def test_gauss_and_bounded_exponential():
    g = gauss(5, 1)
    assert g(5) == pytest.approx(1.0)
    assert g(6) == pytest.approx(math.exp(-1))
    assert g(4) == pytest.approx(g(6))
    assert gauss(5, 1, c_m=0.5)(5) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        gauss(5, 0)
    e = bounded_exponential(k=1, limit=2)
    assert e(0) == pytest.approx(0.0)
    assert e(1) == pytest.approx(2 - 2 / math.exp(1))
    assert e(1000) == pytest.approx(2.0)
