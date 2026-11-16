import math

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic import functions as fn


# ---------------------------------------------------------------- Domain

def test_domain_range_includes_upper_bound():
    d = Domain("d", 0, 10)
    r = list(d.range())
    assert r[0] == 0
    assert r[-1] == 10
    assert len(r) == 11


def test_domain_range_with_resolution():
    d = Domain("d", 0, 1, res=0.25)
    assert list(d.range()) == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])


def test_domain_invalid_arguments():
    with pytest.raises(AssertionError):
        Domain("bad", 5, 5)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_domain_set_assignment_and_call():
    d = Domain("temp", 0, 100)
    d.hot = Set(fn.constant(0.5))
    d.cold = fn.constant(0.25)  # plain function gets wrapped
    assert isinstance(d.cold, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d(10) == {"hot": 0.5, "cold": 0.25}
    assert str(d) == "temp"


def test_domain_unknown_attribute_raises():
    d = Domain("d", 0, 10)
    with pytest.raises(AttributeError):
        d.nothing_here


def test_domain_delete_set():
    d = Domain("d", 0, 10)
    d.s = Set(fn.constant(1))
    del d.s
    assert d(1) == {}
    with pytest.raises(FuzzyWarning):
        del d.unknown


def test_domain_min_max():
    d = Domain("d", 0, 10)
    d.a = fn.constant(0.2)
    d.b = fn.constant(0.7)
    assert d.min(5) == 0.2
    assert d.max(5) == 0.7


# ------------------------------------------------------------------- Set

def test_set_call_and_invert():
    s = Set(fn.constant(0.3))
    assert s(42) == 0.3
    assert (~s)(42) == pytest.approx(0.7)
    assert (-s)(42) == pytest.approx(0.7)


def test_set_and_or_product():
    a = Set(fn.constant(0.3))
    b = Set(fn.constant(0.6))
    assert (a & b)(0) == pytest.approx(0.3)
    assert (a | b)(0) == pytest.approx(0.6)
    assert (a * b)(0) == pytest.approx(0.18)


def test_set_power_concentrated_multiplied():
    s = Set(fn.constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.concentrated()(0) == pytest.approx(0.25)
    assert s.multiplied(2)(0) == pytest.approx(1.0)


def test_set_array_len_cardinality():
    d = Domain("d", 0, 4)
    d.s = Set(fn.linear(0.25, 0))
    assert list(d.s.array()) == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
    assert len(d.s) == 5
    assert d.s.cardinality() == pytest.approx(2.5)
    assert d.s.relative_cardinality() == pytest.approx(0.5)


def test_set_without_domain_raises():
    s = Set(fn.constant(1))
    with pytest.raises(FuzzyWarning):
        s.array()
    with pytest.raises(FuzzyWarning):
        len(s)
    with pytest.raises(FuzzyWarning):
        s.cardinality()
    with pytest.raises(FuzzyWarning):
        s == Set(fn.constant(1))
    with pytest.raises(FuzzyWarning):
        s.normalized()
    with pytest.raises(FuzzyWarning):
        s.plot()


def test_set_equality_and_ordering_with_domain():
    d = Domain("d", 0, 4)
    d.a = Set(fn.linear(0.25, 0))
    d.b = Set(fn.linear(0.25, 0))
    d.c = Set(fn.linear(0.125, 0))
    assert d.a == d.b
    assert not (d.a == d.c)
    assert d.c <= d.a
    assert d.a >= d.c
    assert d.a >= d.b
    assert d.a <= d.b
    # at x == 0 both are 0, so no *proper* subset relation
    assert not (d.c < d.a)
    assert not (d.a > d.c)


def test_set_constructor_name_without_domain():
    with pytest.raises(FuzzyWarning):
        Set(fn.constant(1), name="x")


def test_set_constructor_with_name_and_domain():
    d = Domain("d", 0, 4)
    s = Set(fn.constant(1), name="whole", domain=d)
    assert d.whole is s
    assert s.name == "whole"
    assert str(s) == "d.whole"


def test_set_str_dangling():
    assert str(Set(fn.constant(1))).startswith("dangling Set(")


def test_set_normalized():
    d = Domain("d", 0, 4)
    d.s = Set(fn.linear(0.125, 0))  # max membership 0.5 at x=4
    n = d.s.normalized()
    assert n(4) == pytest.approx(1.0)
    assert n(2) == pytest.approx(0.5)


# ------------------------------------------------------------- functions

def test_inv_constant_singleton():
    assert fn.inv(fn.constant(0.25))(3) == pytest.approx(0.75)
    f = fn.singleton(2)
    assert f(2) == 1
    assert f(1) == 0
    with pytest.raises(AssertionError):
        fn.singleton(1, no_m=1, c_m=0)


def test_linear_and_bounded_linear():
    f = fn.linear(1, -1)
    assert f(0) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(3) == 1
    g = fn.bounded_linear(2, 3)
    assert g(1) == 0
    assert g(2.5) == pytest.approx(0.5)
    assert g(4) == 1
    h = fn.bounded_linear(2, 3, inverse=True)
    assert h(2) == pytest.approx(1.0)
    assert h(3) == pytest.approx(0.0)
    with pytest.raises(AssertionError):
        fn.bounded_linear(3, 2)


@pytest.mark.parametrize("x, expected_r, expected_s", [
    (1, 0, 1),
    (2, 0, 1),
    (3, 0.5, 0.5),
    (4, 1, 0),
    (5, 1, 0),
])
def test_R_and_S(x, expected_r, expected_s):
    assert fn.R(2, 4)(x) == pytest.approx(expected_r)
    assert fn.S(2, 4)(x) == pytest.approx(expected_s)


def test_rectangular():
    f = fn.rectangular(1, 3)
    assert [f(x) for x in (0, 1, 2, 3, 4)] == [0, 1, 1, 1, 0]


def test_triangular_and_trapezoid():
    t = fn.triangular(0, 4)
    assert t(0) == pytest.approx(0)
    assert t(1) == pytest.approx(0.5)
    assert t(2) == pytest.approx(1)
    assert t(3) == pytest.approx(0.5)
    assert t(5) == pytest.approx(0)
    with pytest.raises(AssertionError):
        fn.triangular(3, 1)

    z = fn.trapezoid(0, 1, 3, 4)
    assert z(-1) == 0
    assert z(0.5) == pytest.approx(0.5)
    assert z(2) == 1
    assert z(3.5) == pytest.approx(0.5)
    assert z(5) == 0


def test_alpha_normalize_moderate():
    f = fn.alpha(floor=0.2, ceiling=0.8, func=fn.noop())
    assert f(0.9) == 0.8
    assert f(0.1) == 0.2
    assert f(0.5) == 0.5
    assert fn.normalize(0.5, fn.constant(0.25))(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        fn.normalize(0, fn.constant(0.25))
    assert fn.moderate(fn.constant(0.5))(0) == pytest.approx(0.5)
    assert fn.moderate(fn.constant(1))(0) == pytest.approx(1.0)
    assert fn.moderate(fn.constant(0))(0) == pytest.approx(0.0)


def test_sigmoids():
    s = fn.sigmoid(1, 1, 0)
    assert s(0) == pytest.approx(0.5)
    assert s(-10000) == pytest.approx(0.0)  # overflow handled
    assert s(10000) == pytest.approx(1.0)

    b = fn.bounded_sigmoid(0, 1)
    assert b(0) == pytest.approx(0.1)
    assert b(1) == pytest.approx(0.9)
    assert b(100000) == pytest.approx(1.0)
    assert b(-100000) == pytest.approx(0.0)

    ss = fn.simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(-1000) == pytest.approx(0.0)
    assert ss(1000) == pytest.approx(1.0)


def test_gauss_and_bounded_exponential():
    g = fn.gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) == pytest.approx(math.exp(-1))
    assert g(-1) == pytest.approx(g(1))
    with pytest.raises(AssertionError):
        fn.gauss(0, 0)

    e = fn.bounded_exponential(k=0.1, limit=1)
    assert e(0) == pytest.approx(0.0)
    assert e(10) == pytest.approx(1 - math.exp(-1))
    assert e(1e9) == pytest.approx(1.0)
