import math

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv, noop, constant, alpha, normalize, moderate, singleton, linear,
    bounded_linear, R, S, rectangular, triangular, trapezoid, sigmoid,
    bounded_sigmoid, bounded_exponential, simple_sigmoid, gauss,
)


# ---------------------------------------------------------------- Domain

def test_domain_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("d", 5, 5)
    with pytest.raises(AssertionError):
        Domain("d", 0, 10, res=0)


def test_domain_range_includes_upper_bound():
    d = Domain("d", 0, 10)
    r = list(d.range())
    assert len(r) == 11
    assert r[0] == 0
    assert r[-1] == 10


def test_domain_set_assignment_and_call():
    d = Domain("d", 0, 10)
    d.low = constant(0.25)
    d.high = Set(constant(0.75))
    assert isinstance(d.low, Set)
    assert d.low.domain is d
    assert d.low.name == "low"
    assert d(5) == {"low": 0.25, "high": 0.75}
    assert str(d) == "d"
    assert str(d.high) == "d.high"


def test_domain_unknown_attribute_raises():
    d = Domain("d", 0, 10)
    with pytest.raises(AttributeError):
        d.nonexistent


def test_domain_delete_set():
    d = Domain("d", 0, 10)
    d.a = constant(0.5)
    del d.a
    assert d(1) == {}
    with pytest.raises(FuzzyWarning):
        del d.not_a_set


def test_domain_min_max():
    d = Domain("d", 0, 10)
    d.a = constant(0.2)
    d.b = constant(0.9)
    assert d.min(3) == 0.2
    assert d.max(3) == 0.9


def test_domain_invalid_set_name():
    d = Domain("d", 0, 10)
    with pytest.raises(AssertionError):
        setattr(d, "not valid", constant(1))


# ------------------------------------------------------------------- Set

def test_set_call_and_operators():
    a = Set(constant(0.3))
    b = Set(constant(0.6))
    assert a(1) == 0.3
    assert (~a)(1) == pytest.approx(0.7)
    assert (-a)(1) == pytest.approx(0.7)
    assert (a & b)(1) == 0.3
    assert (a | b)(1) == 0.6
    assert (a ** 2)(1) == pytest.approx(0.09)


def test_set_array_len_and_cardinality():
    d = Domain("d", 0, 4)
    d.s = Set(lambda x: x / 4)
    assert list(d.s.array()) == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
    assert len(d.s) == 5
    assert d.s.cardinality() == pytest.approx(2.5)
    assert d.s.relative_cardinality() == pytest.approx(0.5)


def test_set_without_domain_raises():
    s = Set(constant(0.5))
    with pytest.raises(FuzzyWarning):
        s.array()
    with pytest.raises(FuzzyWarning):
        len(s)
    with pytest.raises(FuzzyWarning):
        s.cardinality()
    with pytest.raises(FuzzyWarning):
        s.normalized()
    with pytest.raises(FuzzyWarning):
        s == Set(constant(0.5))
    assert str(s).startswith("dangling Set(")


def test_set_name_without_domain_raises():
    with pytest.raises(FuzzyWarning):
        Set(constant(1), name="x")


def test_set_init_with_name_and_domain():
    d = Domain("d", 0, 3)
    s = Set(constant(0.5), name="mid", domain=d)
    assert d.mid is s
    assert s.domain is d


def test_set_comparisons_within_domain():
    d = Domain("d", 0, 5)
    d.a = constant(0.2)
    d.b = constant(0.5)
    d.c = constant(0.2)
    assert d.a <= d.b
    assert d.a < d.b
    assert d.b >= d.a
    assert d.b > d.a
    assert not (d.a > d.b)
    assert d.a == d.c
    assert not (d.a == d.b)


def test_set_concentrated_and_multiplied():
    s = Set(constant(0.5))
    assert s.concentrated()(0) == pytest.approx(0.25)
    assert s.multiplied(0.5)(0) == pytest.approx(0.25)


def test_set_normalized():
    d = Domain("d", 0, 4)
    d.s = Set(lambda x: x / 8)  # max is 0.5
    n = d.s.normalized()
    assert n(4) == pytest.approx(1.0)
    assert n(2) == pytest.approx(0.5)


# ------------------------------------------------------------- functions

def test_special_functions():
    assert inv(constant(0.25))(0) == pytest.approx(0.75)
    assert noop()(0.42) == 0.42
    assert constant(7)(123) == 7
    f = alpha(floor=0.2, ceiling=0.8, func=noop())
    assert f(0.9) == 0.8
    assert f(0.1) == 0.2
    assert f(0.5) == 0.5


def test_normalize_and_moderate():
    assert normalize(0.5, noop())(0.25) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        normalize(0, noop())
    m = moderate(noop())
    assert m(0.5) == pytest.approx(0.5)
    assert m(1) == pytest.approx(1.0)
    assert m(0) == pytest.approx(0.0)


def test_singleton_linear_rectangular():
    s = singleton(2)
    assert s(2) == 1
    assert s(1) == 0
    f = linear(1, -1)
    assert f(-2) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(3) == 1
    r = rectangular(1, 3)
    assert r(0) == 0
    assert r(1) == 1
    assert r(3) == 1
    assert r(4) == 0


def test_bounded_linear_R_S():
    f = bounded_linear(2, 3)
    assert f(1) == 0
    assert f(2.5) == pytest.approx(0.5)
    assert f(4) == 1
    g = bounded_linear(2, 3, inverse=True)
    assert g(2) == pytest.approx(1)
    assert g(2.5) == pytest.approx(0.5)
    assert g(3) == pytest.approx(0)
    with pytest.raises(AssertionError):
        bounded_linear(3, 2)
    r = R(0, 4)
    assert (r(-1), r(2), r(5)) == (0, 0.5, 1)
    s = S(0, 4)
    assert (s(-1), s(2), s(5)) == (1, 0.5, 0)


def test_triangular_and_trapezoid():
    t = triangular(0, 4)
    assert t(0) == 0
    assert t(1) == pytest.approx(0.5)
    assert t(2) == 1
    assert t(3) == pytest.approx(0.5)
    assert t(5) == pytest.approx(0)
    z = trapezoid(0, 1, 3, 4)
    assert z(-1) == 0
    assert z(0.5) == pytest.approx(0.5)
    assert z(2) == 1
    assert z(3.5) == pytest.approx(0.5)
    assert z(5) == 0
    with pytest.raises(AssertionError):
        trapezoid(0, 3, 2, 4)


def test_sigmoid_family():
    assert sigmoid(1, 1, 0)(0) == pytest.approx(0.5)
    assert sigmoid(0.5, 1, 0)(0) == pytest.approx(0.25)
    assert sigmoid(1, 1, 0)(10000) == pytest.approx(1.0)
    assert sigmoid(1, 1, 0)(-10000) == pytest.approx(0.0)
    assert simple_sigmoid()(0) == pytest.approx(0.5)
    b = bounded_sigmoid(0, 1)
    assert b(0) == pytest.approx(0.1)
    assert b(1) == pytest.approx(0.9)
    assert b(-100000) == pytest.approx(0.0, abs=1e-6)
    assert b(100000) == pytest.approx(1.0)


def test_gauss_and_bounded_exponential():
    g = gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) == pytest.approx(math.exp(-1))
    assert g(-1) == pytest.approx(g(1))
    e = bounded_exponential(k=1, limit=1)
    assert e(0) == pytest.approx(0.0)
    assert e(100) == pytest.approx(1.0)
    assert e(100000) == pytest.approx(1.0)
