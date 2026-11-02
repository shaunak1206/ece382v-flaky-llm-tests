import os

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv, noop, constant, alpha, normalize, moderate, singleton, linear,
    bounded_linear, R, S, rectangular, triangular, trapezoid, sigmoid,
    bounded_sigmoid, bounded_exponential, simple_sigmoid, triangular_sigmoid,
    gauss,
)


def make_domain():
    return Domain("t", 0, 4)


def test_domain_init_validation():
    with pytest.raises(AssertionError):
        Domain("bad", 5, 5)
    with pytest.raises(AssertionError):
        Domain("bad", 6, 5)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 5, res=0)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 5, res=-1)


def test_domain_range_includes_upper_bound():
    d = make_domain()
    assert list(d.range()) == [0, 1, 2, 3, 4]
    d2 = Domain("half", 0, 2, res=0.5)
    assert list(d2.range()) == [0, 0.5, 1.0, 1.5, 2.0]


def test_domain_assigning_function_wraps_in_set():
    d = make_domain()
    d.flat = constant(0.3)
    assert isinstance(d.flat, Set)
    assert d.flat.name == "flat"
    assert d.flat.domain is d
    assert d.flat(2) == 0.3
    assert str(d.flat) == "t.flat"


def test_domain_call_returns_membership_dict():
    d = make_domain()
    d.low = S(0, 4)
    d.high = R(0, 4)
    result = d(1)
    assert set(result.keys()) == {"low", "high"}
    assert result["high"] == pytest.approx(0.25)
    assert result["low"] == pytest.approx(0.75)


def test_domain_min_max_over_sets():
    d = make_domain()
    d.a = constant(0.2)
    d.b = constant(0.7)
    assert d.min(3) == pytest.approx(0.2)
    assert d.max(3) == pytest.approx(0.7)


def test_domain_attribute_errors_and_deletion():
    d = make_domain()
    with pytest.raises(AttributeError):
        d.missing
    with pytest.raises(AssertionError):
        setattr(d, "1bad", constant(1))
    d.a = constant(1)
    assert "a" in d(1)
    del d.a
    assert d(1) == {}
    with pytest.raises(FuzzyWarning):
        del d.not_a_set


def test_domain_str_repr_and_equality():
    d1 = Domain("t", 0, 10)
    d2 = Domain("t", 0, 10)
    d3 = Domain("t", 0, 11)
    d4 = Domain("other", 0, 10)
    assert str(d1) == "t"
    assert repr(d1).startswith("Domain('t', 0, 10, res=1")
    assert d1 == d2
    assert not (d1 == d3)
    assert not (d1 == d4)


def test_set_invert_and_neg():
    s = Set(R(0, 4))
    assert (~s)(1) == pytest.approx(0.75)
    assert (-s)(1) == pytest.approx(0.75)
    assert (~s)(0) == pytest.approx(1.0)
    assert (~s)(4) == pytest.approx(0.0)


def test_set_and_or_use_min_max():
    a = Set(R(0, 4))
    b = Set(S(0, 4))
    assert (a & b)(1) == pytest.approx(0.25)
    assert (a | b)(1) == pytest.approx(0.75)
    assert (a & b)(2) == pytest.approx(0.5)
    assert (a | b)(2) == pytest.approx(0.5)


def test_set_product_and_bounded_sum():
    a = Set(R(0, 4))
    b = Set(S(0, 4))
    assert float((a * b)(1)) == pytest.approx(0.25 * 0.75)
    assert float((a + b)(1)) == pytest.approx(0.25 + 0.75 - 0.25 * 0.75)


def test_set_pow_concentrated_and_multiplied():
    a = Set(R(0, 4))
    assert (a ** 2)(2) == pytest.approx(0.25)
    assert a.concentrated()(2) == pytest.approx(0.25)
    assert a.multiplied(2)(1) == pytest.approx(0.5)


def test_set_equality_requires_domain():
    a = Set(R(0, 4))
    b = Set(R(0, 4))
    with pytest.raises(FuzzyWarning):
        a == b
    d = make_domain()
    d.x = R(0, 4)
    d.y = R(0, 4)
    d.z = S(0, 4)
    assert d.x == d.y
    assert not (d.x == d.z)


def test_set_subset_superset_comparisons():
    d = make_domain()
    d.small = constant(0.2)
    d.big = constant(0.5)
    assert d.small <= d.big
    assert d.small < d.big
    assert d.big >= d.small
    assert d.big > d.small
    assert not (d.small >= d.big)
    assert d.small <= d.small
    assert not (d.small < d.small)
    with pytest.raises(FuzzyWarning):
        Set(constant(1)) <= d.big


def test_set_len_cardinality_and_relative_cardinality():
    d = make_domain()
    d.ramp = R(0, 4)
    assert len(d.ramp) == 5
    assert d.ramp.cardinality() == pytest.approx(2.5)
    assert d.ramp.relative_cardinality() == pytest.approx(0.5)
    dangling = Set(R(0, 4))
    with pytest.raises(FuzzyWarning):
        len(dangling)
    with pytest.raises(FuzzyWarning):
        dangling.cardinality()
    with pytest.raises(FuzzyWarning):
        dangling.array()


def test_set_array_and_normalized():
    d = make_domain()
    d.ramp = R(0, 4)
    arr = d.ramp.array()
    assert [round(float(v), 4) for v in arr] == [0.0, 0.25, 0.5, 0.75, 1.0]
    d.half = d.ramp.multiplied(0.5)
    assert max(d.half.array()) == pytest.approx(0.5)
    d.full = d.half.normalized()
    assert max(d.full.array()) == pytest.approx(1.0)
    with pytest.raises(FuzzyWarning):
        Set(constant(1)).normalized()


def test_set_constructor_name_domain_and_str():
    d = make_domain()
    with pytest.raises(FuzzyWarning):
        Set(constant(1), name="only_name")
    s = Set(constant(0.4), name="named", domain=d)
    assert d.named is s
    assert s.domain is d
    assert str(s) == "t.named"
    dangling = Set(constant(1))
    assert str(dangling).startswith("dangling Set(")
    assert repr(dangling).startswith("Set(")


def test_triangular_and_trapezoid_shapes():
    tri = triangular(0, 10)
    assert tri(0) == pytest.approx(0.0)
    assert tri(2.5) == pytest.approx(0.5)
    assert tri(5) == pytest.approx(1.0)
    assert tri(7.5) == pytest.approx(0.5)
    assert tri(10) == pytest.approx(0.0)
    assert tri(-3) == pytest.approx(0.0)
    assert tri(20) == pytest.approx(0.0)
    trap = trapezoid(0, 2, 4, 6)
    assert trap(-1) == 0
    assert trap(1) == pytest.approx(0.5)
    assert trap(3) == 1
    assert trap(5) == pytest.approx(0.5)
    assert trap(7) == 0


def test_rectangular_and_singleton():
    rect = rectangular(1, 3)
    assert rect(0) == 0
    assert rect(1) == 1
    assert rect(2) == 1
    assert rect(3) == 1
    assert rect(4) == 0
    sing = singleton(2)
    assert sing(2) == 1
    assert sing(1.9) == 0
    assert sing(3) == 0


def test_linear_and_bounded_linear():
    f = linear(1, -1)
    assert f(-2) == 0
    assert f(1) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(3) == 1
    g = bounded_linear(2, 3)
    assert g(1) == 0
    assert g(2.5) == pytest.approx(0.5)
    assert g(4) == 1
    h = bounded_linear(2, 3, inverse=True)
    assert h(2) == pytest.approx(1.0)
    assert h(2.5) == pytest.approx(0.5)
    assert h(3) == pytest.approx(0.0)


def test_R_and_S_are_opposite_ramps():
    r = R(0, 4)
    s = S(0, 4)
    assert r(-1) == 0
    assert r(2) == pytest.approx(0.5)
    assert r(5) == 1
    assert s(-1) == 1
    assert s(2) == pytest.approx(0.5)
    assert s(5) == 0
    for x in (0, 1, 2, 3, 4):
        assert r(x) + s(x) == pytest.approx(1.0)


def test_sigmoid_family():
    sg = sigmoid(1, 1, 0)
    assert sg(0) == pytest.approx(0.5)
    assert sg(50) > 0.99
    assert sg(-50) < 0.01
    assert sigmoid(0.5, 1, 0)(0) == pytest.approx(0.25)
    bs = bounded_sigmoid(0, 1)
    assert bs(0) == pytest.approx(0.1)
    assert bs(1) == pytest.approx(0.9)
    inv_bs = bounded_sigmoid(0, 1, inverse=True)
    assert inv_bs(0) == pytest.approx(0.9)
    assert inv_bs(1) == pytest.approx(0.1)
    ss = simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(1000) == pytest.approx(1.0)
    assert ss(-1000) == pytest.approx(0.0)
    ts = triangular_sigmoid(2, 4)
    assert ts(2) == pytest.approx(0.1)
    assert ts(3) == pytest.approx(0.9)
    assert ts(4) == pytest.approx(0.1)


def test_gauss_and_bounded_exponential():
    g = gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) == pytest.approx(g(-1))
    assert g(1) < g(0)
    assert g(100) == pytest.approx(0.0)
    assert gauss(2, 1, c_m=0.5)(2) == pytest.approx(0.5)
    e = bounded_exponential(k=1, limit=1)
    assert e(0) == pytest.approx(0.0)
    assert e(1) < e(2) < 1
    assert e(10000) == pytest.approx(1.0)


def test_helper_functions_inv_noop_constant_alpha_normalize_moderate():
    assert noop()(0.3) == 0.3
    assert constant(0.7)(123) == 0.7
    assert inv(constant(0.25))(5) == pytest.approx(0.75)
    a = alpha(floor=0.2, ceiling=0.8, func=noop())
    assert a(0.1) == pytest.approx(0.2)
    assert a(0.5) == pytest.approx(0.5)
    assert a(0.9) == pytest.approx(0.8)
    assert normalize(0.5, constant(0.25))(1) == pytest.approx(0.5)
    m = moderate(noop())
    assert m(0) == pytest.approx(0.0)
    assert m(0.5) == pytest.approx(0.5)
    assert m(1) == pytest.approx(1.0)


def test_function_argument_validation():
    with pytest.raises(AssertionError):
        bounded_linear(3, 2)
    with pytest.raises(AssertionError):
        triangular(0, 10, c=10)
    with pytest.raises(AssertionError):
        trapezoid(0, 5, 4, 6)
    with pytest.raises(AssertionError):
        normalize(0, constant(1))
    with pytest.raises(AssertionError):
        gauss(0, 0)
    with pytest.raises(AssertionError):
        sigmoid(0, 1, 0)
    with pytest.raises(AssertionError):
        R(2, 2)
    with pytest.raises(AssertionError):
        S(3, 1)
