import math

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv,
    noop,
    constant,
    alpha,
    normalize,
    moderate,
    singleton,
    linear,
    bounded_linear,
    R,
    S,
    rectangular,
    triangular,
    trapezoid,
    sigmoid,
    bounded_sigmoid,
    bounded_exponential,
    simple_sigmoid,
    triangular_sigmoid,
    gauss,
)


def test_domain_basic_attribute_assignment_and_call():
    d = Domain("temperature", 0, 10)
    d.hot = constant(0.75)
    d.cold = singleton(0)

    assert d.hot(3) == 0.75
    assert d(0) == {"hot": 0.75, "cold": 1}
    assert d(2) == {"hot": 0.75, "cold": 0}


def test_domain_warns_outside_range_and_returns_memberships():
    d = Domain("temperature", 0, 10)
    d.hot = constant(1)
    with pytest.warns(UserWarning):
        result = d(99)
    assert result == {"hot": 1}


def test_domain_range_includes_high_bound():
    d = Domain("x", 0, 1, res=0.5)
    assert list(d.range()) == [0.0, 0.5, 1.0]


def test_domain_delete_set_and_attribute_error():
    d = Domain("x", 0, 1)
    d.a = constant(1)
    del d.a
    with pytest.raises(AttributeError):
        _ = d.a


def test_domain_rejects_non_identifier_names():
    d = Domain("x", 0, 1)
    with pytest.raises(AssertionError):
        setattr(d, "not valid", constant(1))


def test_set_invert_and_boolean_operators():
    a = Set(constant(0.2))
    b = Set(constant(0.7))

    assert (~a)(0) == pytest.approx(0.8)
    assert (-a)(0) == pytest.approx(0.8)
    assert (a & b)(0) == pytest.approx(0.2)
    assert (a | b)(0) == pytest.approx(0.7)
    assert (a * b)(0) == pytest.approx(0.14)
    assert (a + b)(0) == pytest.approx(0.9)
    assert (a ^ b)(0) == pytest.approx(0.5)


def test_set_power_multiplication_and_noop():
    s = Set(noop())
    squared = s ** 2
    assert squared(0.4) == pytest.approx(0.16)
    assert s(0.4) == pytest.approx(0.4)


def test_set_len_cardinality_and_relative_cardinality():
    d = Domain("x", 0, 2, res=1)
    d.a = singleton(1)
    s = d.a

    assert len(s) == 3
    assert s.cardinality() == pytest.approx(1)
    assert s.relative_cardinality() == pytest.approx(1 / 3)


def test_set_comparisons_with_domains():
    d = Domain("x", 0, 2, res=1)
    d.a = rectangular(0, 2)
    d.b = singleton(1)

    assert d.b <= d.a
    assert d.b < d.a
    assert d.a >= d.b
    assert d.a > d.b
    assert not (d.a == d.b)


def test_set_array_and_normalized():
    d = Domain("x", 0, 2, res=1)
    d.a = Set(lambda x: x)
    s = d.a

    assert list(s.array()) == [0.0, 1.0, 2.0]
    n = s.normalized()
    assert list(n.array()) == [0.0, 0.5, 1.0]


def test_set_without_domain_raises_fuzzywarning_for_array_and_len():
    s = Set(constant(1))
    with pytest.raises(FuzzyWarning):
        _ = len(s)
    with pytest.raises(FuzzyWarning):
        _ = s.array()
    with pytest.raises(FuzzyWarning):
        _ = s.normalized()


def test_inv_constant_and_normalize_and_alpha():
    f = constant(0.25)
    assert inv(f)(0) == pytest.approx(0.75)
    assert normalize(2, lambda x: 2)(3) == pytest.approx(1.0)

    clipped = alpha(floor=0.2, ceiling=0.8, func=constant(0.9), floor_clip=0.1, ceiling_clip=0.7)
    assert clipped(0) == pytest.approx(0.7)


def test_singleton_linear_and_bounded_linear():
    s = singleton(2)
    assert s(2) == 1
    assert s(3) == 0

    l = linear(1, -1)
    assert l(0) == 0
    assert l(1.5) == pytest.approx(0.5)
    assert l(3) == 1

    bl = bounded_linear(2, 3)
    assert bl(1) == pytest.approx(0.0)
    assert bl(2.5) == pytest.approx(0.5)
    assert bl(4) == pytest.approx(1.0)


def test_r_and_s_membership_functions():
    r = R(2, 4)
    s = S(2, 4)

    assert r(1) == 0
    assert r(2) == 0
    assert r(3) == pytest.approx(0.5)
    assert r(5) == 1

    assert s(1) == 1
    assert s(2) == 1
    assert s(3) == pytest.approx(0.5)
    assert s(4) == 0


def test_rectangular_triangular_and_trapezoid():
    rect = rectangular(1, 3)
    tri = triangular(0, 4)
    trap = trapezoid(0, 1, 3, 4)

    assert rect(0.5) == 0
    assert rect(2) == 1
    assert rect(4) == 0

    assert tri(0) == 0
    assert tri(2) == pytest.approx(1)
    assert tri(4) == 0

    assert trap(0) == 0
    assert trap(2) == 1
    assert trap(4) == 0


def test_sigmoid_variants_and_gauss():
    sig = sigmoid(1, 1, 0)
    assert sig(0) == pytest.approx(0.5)
    assert 0 < sig(-100) < sig(0) < sig(100) <= 1

    bs = bounded_sigmoid(0, 1)
    assert bs(0) == pytest.approx(0.1, rel=1e-7)
    assert round(bs(1), 2) == 0.9

    ss = simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(-1000) == pytest.approx(0.0, abs=1e-6)
    assert ss(1000) == pytest.approx(1.0, abs=1e-6)

    g = gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) < g(0)


def test_bounded_exponential_monotonic_and_limits():
    f = bounded_exponential(k=0.5, limit=2)
    assert f(0) == pytest.approx(0.0)
    assert f(10) > f(1)
    assert f(1000) == pytest.approx(2.0, rel=1e-7)


def test_moderate_biases_towards_half():
    f = moderate(constant(0.0))
    assert f(1) == pytest.approx(0.0)
    g = moderate(constant(1.0))
    assert g(1) == pytest.approx(1.0)
