import math

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    alpha,
    bounded_exponential,
    bounded_linear,
    bounded_sigmoid,
    constant,
    gauss,
    inv,
    linear,
    moderate,
    noop,
    normalize,
    R,
    rectangular,
    S,
    sigmoid,
    simple_sigmoid,
    singleton,
    trapezoid,
    triangular,
)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_init_and_str():
    d = Domain("temperature", 0, 100, res=2)
    assert str(d) == "temperature"
    assert d._low == 0
    assert d._high == 100
    assert d._res == 2


@pytest.mark.parametrize("low, high, res", [
    (5, 5, 1),     # equal bounds
    (10, 0, 1),    # reversed bounds
    (0, 10, 0),    # zero resolution
    (0, 10, -1),   # negative resolution
])
def test_domain_invalid_arguments_raise(low, high, res):
    with pytest.raises(AssertionError):
        Domain("t", low, high, res=res)


def test_domain_assignment_and_deletion():
    d = Domain("t", 0, 10)
    d.cold = S(0, 10)
    # plain functions are wrapped into Set and registered
    assert isinstance(d.cold, Set)
    assert d.cold.name == "cold"
    assert d.cold.domain is d

    # non-identifier names are rejected
    with pytest.raises(AssertionError):
        setattr(d, "not valid", constant(0))

    # deleting a set removes it
    del d.cold
    with pytest.raises(AttributeError):
        d.cold

    # deleting an unknown name raises FuzzyWarning
    with pytest.raises(FuzzyWarning):
        del d.nothing_here


def test_domain_call_and_min_max():
    d = Domain("t", 0, 10)
    d.cold = S(0, 10)
    d.hot = R(0, 10)

    memberships = d(5)
    assert set(memberships.keys()) == {"cold", "hot"}
    assert memberships["cold"] == pytest.approx(0.5)
    assert memberships["hot"] == pytest.approx(0.5)

    assert d.min(8) == pytest.approx(0.2)
    assert d.max(8) == pytest.approx(0.8)


def test_domain_range_inclusive_and_resolution():
    d = Domain("t", 0, 10)
    values = list(d.range())
    assert len(values) == 11
    assert values[0] == 0
    assert values[-1] == 10  # upper bound is included

    fine = Domain("t", 0, 1, res=0.25)
    fine_values = list(fine.range())
    assert len(fine_values) == 5
    assert fine_values[-1] == pytest.approx(1.0)


def test_domain_eq_and_unknown_attribute():
    d1 = Domain("t", 0, 10)
    d2 = Domain("t", 0, 10)
    d3 = Domain("t", 0, 20)
    assert d1 == d2
    assert d1 != d3

    with pytest.raises(AttributeError):
        d1.missing_set


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

def test_set_call_and_invert():
    s = Set(constant(0.3))
    assert s(5) == pytest.approx(0.3)
    assert (~s)(5) == pytest.approx(0.7)
    assert (-s)(5) == pytest.approx(0.7)


def test_set_name_requires_domain():
    # name without domain is an error
    with pytest.raises(FuzzyWarning):
        Set(constant(0.1), name="x")

    # name together with domain registers the set in the domain
    d = Domain("t", 0, 4)
    Set(constant(0.3), name="warm", domain=d)
    assert d.warm(2) == pytest.approx(0.3)
    assert d.warm.name == "warm"
    assert d.warm.domain is d


def test_set_and_or_pow():
    a = Set(constant(0.3))
    b = Set(constant(0.6))
    assert (a & b)(0) == pytest.approx(0.3)
    assert (a | b)(0) == pytest.approx(0.6)
    assert (Set(constant(0.5)) ** 2)(0) == pytest.approx(0.25)


def test_set_concentrated_intensified_multiplied():
    assert Set(constant(0.5)).concentrated()(0) == pytest.approx(0.25)
    assert Set(constant(0.4)).multiplied(0.5)(0) == pytest.approx(0.2)
    # intensified only has a working branch for x < 0.5
    assert Set(constant(0.3)).intensified()(0.1) == pytest.approx(0.18)


def test_set_array_len_cardinality():
    d = Domain("t", 0, 4)
    d.ramp = Set(lambda x: x / 4)

    assert len(d.ramp) == 5
    assert list(d.ramp.array()) == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
    assert d.ramp.cardinality() == pytest.approx(2.5)
    assert d.ramp.relative_cardinality() == pytest.approx(0.5)


def test_set_comparisons_with_domain():
    d = Domain("t", 0, 4)
    d.a = constant(0.2)
    d.b = constant(0.5)
    d.c = constant(0.2)

    assert d.a < d.b
    assert d.a <= d.b
    assert d.b > d.a
    assert d.b >= d.a
    assert not (d.b < d.a)
    assert d.a == d.c
    assert not (d.a == d.b)


def test_set_ops_without_domain_raise():
    a = Set(constant(0.1))
    b = Set(constant(0.2))
    with pytest.raises(FuzzyWarning):
        a == b
    with pytest.raises(FuzzyWarning):
        a < b
    with pytest.raises(FuzzyWarning):
        len(a)
    with pytest.raises(FuzzyWarning):
        a.cardinality()
    with pytest.raises(FuzzyWarning):
        a.array()
    with pytest.raises(FuzzyWarning):
        a.normalized()


def test_set_normalized_and_str():
    d = Domain("t", 0, 4)
    d.a = Set(lambda x: x / 8)  # max over domain is 0.5
    n = d.a.normalized()
    assert n(4) == pytest.approx(1.0)
    assert n(2) == pytest.approx(0.5)

    assert str(d.a) == "t.a"
    assert str(Set(constant(0.1))).startswith("dangling Set(")


# ---------------------------------------------------------------------------
# functions module
# ---------------------------------------------------------------------------

def test_inv_noop_constant():
    assert inv(noop())(0.25) == pytest.approx(0.75)
    assert noop()(0.3) == pytest.approx(0.3)
    assert constant(0.7)(123) == pytest.approx(0.7)


def test_alpha_clips():
    f = alpha(func=lambda x: x, floor=0.2, ceiling=0.8)
    assert f(0.1) == pytest.approx(0.2)
    assert f(0.9) == pytest.approx(0.8)
    assert f(0.5) == pytest.approx(0.5)

    g = alpha(func=lambda x: x, floor=0.2, ceiling=0.8,
              floor_clip=0, ceiling_clip=1)
    assert g(0.1) == pytest.approx(0)
    assert g(0.9) == pytest.approx(1)


def test_normalize_and_moderate():
    f = normalize(0.5, lambda x: x)
    assert f(0.25) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        normalize(0, lambda x: x)

    m = moderate(lambda x: x)
    assert m(0.5) == pytest.approx(0.5)
    assert m(1.0) == pytest.approx(1.0)
    assert m(0.0) == pytest.approx(0.0)
    assert m(0.75) == pytest.approx(0.5625)


def test_singleton_rectangular_linear():
    s = singleton(2)
    assert s(2) == 1
    assert s(1) == 0

    s2 = singleton(2, no_m=0.1, c_m=0.9)
    assert s2(3) == pytest.approx(0.1)
    assert s2(2) == pytest.approx(0.9)

    r = rectangular(2, 4)
    assert r(3) == 1
    assert r(1) == 0
    assert r(5) == 0

    lin = linear(1, -1)
    assert lin(0) == 0
    assert lin(1.5) == pytest.approx(0.5)
    assert lin(3) == 1


def test_bounded_linear_and_inverse():
    f = bounded_linear(2, 3)
    assert f(1) == pytest.approx(0.0)
    assert f(2.5) == pytest.approx(0.5)
    assert f(4) == pytest.approx(1.0)

    g = bounded_linear(2, 3, inverse=True)
    assert g(2) == pytest.approx(1.0)
    assert g(2.5) == pytest.approx(0.5)
    assert g(3) == pytest.approx(0.0)


def test_R_and_S():
    r = R(0, 10)
    assert r(-1) == 0
    assert r(5) == pytest.approx(0.5)
    assert r(11) == 1

    s = S(0, 10)
    assert s(-1) == 1
    assert s(5) == pytest.approx(0.5)
    assert s(11) == 0


def test_triangular_and_trapezoid():
    t = triangular(0, 10)
    assert t(0) == pytest.approx(0.0)
    assert t(2.5) == pytest.approx(0.5)
    assert t(5) == pytest.approx(1.0)
    assert t(7.5) == pytest.approx(0.5)

    z = trapezoid(0, 2, 4, 6)
    assert z(3) == pytest.approx(1.0)
    assert z(-1) == pytest.approx(0.0)
    assert z(1) == pytest.approx(0.5)
    assert z(5) == pytest.approx(0.5)
    assert z(7) == pytest.approx(0.0)


def test_bounded_sigmoid_and_inverse():
    f = bounded_sigmoid(0, 1)
    assert f(0) == pytest.approx(0.1)
    assert f(1) == pytest.approx(0.9)
    assert f(100000) == pytest.approx(1.0)
    assert f(-100000) == pytest.approx(0.0)

    g = bounded_sigmoid(0, 1, inverse=True)
    assert g(0) == pytest.approx(0.9)


def test_sigmoid_and_simple_sigmoid():
    s = sigmoid(1, 1, 0)
    assert s(0) == pytest.approx(0.5)
    assert s(1000) == pytest.approx(1.0)
    assert s(-1000) == pytest.approx(0.0)

    ss = simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(1000) == pytest.approx(1.0, abs=1e-6)
    assert ss(-1000) == pytest.approx(0.0, abs=1e-6)


def test_gauss_and_bounded_exponential():
    g = gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) == pytest.approx(math.exp(-1))

    g_half = gauss(0, 1, c_m=0.5)
    assert g_half(0) == pytest.approx(0.5)

    e = bounded_exponential(k=1, limit=1)
    assert e(0) == pytest.approx(0.0)
    assert e(1) == pytest.approx(1 - math.exp(-1))
    # overflow path returns the limit
    assert e(10000) == pytest.approx(1.0)
