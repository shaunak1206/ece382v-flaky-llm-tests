"""Unit tests for fuzzylogic.classes (Domain, Set) and fuzzylogic.functions."""

import math

import pytest

from fuzzylogic.classes import Domain, FuzzyWarning, Set
from fuzzylogic.functions import (
    R,
    S,
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
    rectangular,
    sigmoid,
    simple_sigmoid,
    singleton,
    trapezoid,
    triangular,
    triangular_sigmoid,
)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_init_sets_name_and_range():
    d = Domain("temperature", 0, 100, res=5)
    assert str(d) == "temperature"
    values = list(d.range())
    assert values[0] == 0
    assert values[-1] == 100  # upper bound is included
    assert len(values) == 21


def test_domain_init_rejects_invalid_arguments():
    with pytest.raises(AssertionError):
        Domain("bad", 10, 0)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 10, res=0)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 10, res=-1)


def test_domain_setattr_wraps_plain_function_in_set():
    d = Domain("t", 0, 10)
    d.hot = lambda x: 0.5
    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d.hot(3) == 0.5


def test_domain_setattr_rejects_non_identifier_name():
    d = Domain("t", 0, 10)
    with pytest.raises(AssertionError):
        setattr(d, "not valid", constant(0.5))


def test_domain_call_returns_membership_dict():
    d = Domain("t", 0, 10)
    d.hot = constant(0.8)
    d.cold = constant(0.2)
    assert d(5) == {"hot": 0.8, "cold": 0.2}


def test_domain_getattr_unknown_raises_and_delattr_removes_set():
    d = Domain("t", 0, 10)
    d.hot = constant(0.8)
    with pytest.raises(AttributeError):
        _ = d.missing

    del d.hot
    with pytest.raises(AttributeError):
        _ = d.hot
    with pytest.raises(FuzzyWarning):
        del d.not_there


def test_domain_min_and_max_over_sets():
    d = Domain("t", 0, 10)
    d.a = constant(0.2)
    d.b = constant(0.9)
    assert d.min(5) == 0.2
    assert d.max(5) == 0.9


# ---------------------------------------------------------------------------
# Set: construction and basic behaviour
# ---------------------------------------------------------------------------

def test_set_rejects_non_callable_function():
    with pytest.raises(AssertionError):
        Set(5)


def test_set_call_invert_and_negation():
    s = Set(constant(0.3))
    assert s(10) == 0.3
    assert (~s)(10) == pytest.approx(0.7)
    assert (-s)(10) == pytest.approx(0.7)


def test_set_and_or_use_min_and_max():
    a = Set(constant(0.3))
    b = Set(constant(0.7))
    assert (a & b)(0) == 0.3
    assert (a | b)(0) == 0.7


def test_set_pow_concentrated_and_multiplied():
    s = Set(constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.concentrated()(0) == pytest.approx(0.25)
    assert Set(constant(0.8)).multiplied(0.5)(0) == pytest.approx(0.4)


def test_set_init_with_name_and_domain_registers_set():
    d = Domain("t", 0, 4)
    Set(constant(0.1), name="hot", domain=d)
    assert d.hot(0) == 0.1
    assert d.hot.name == "hot"


def test_set_init_name_or_domain_alone_raises():
    d = Domain("t", 0, 4)
    with pytest.raises(FuzzyWarning):
        Set(constant(0.1), name="hot")
    with pytest.raises(FuzzyWarning):
        Set(constant(0.1), domain=d)


def test_set_str_for_named_and_dangling_sets():
    d = Domain("temp", 0, 4)
    d.hot = constant(0.1)
    assert str(d.hot) == "temp.hot"
    assert str(Set(constant(0.1))).startswith("dangling Set(")


# ---------------------------------------------------------------------------
# Set: operations that need a domain
# ---------------------------------------------------------------------------

def test_set_array_len_cardinality_and_relative_cardinality():
    d = Domain("t", 0, 4)  # values 0..4 -> 5 points
    d.s = constant(0.5)
    assert len(d.s) == 5
    assert list(d.s.array()) == pytest.approx([0.5] * 5)
    assert d.s.cardinality() == pytest.approx(2.5)
    assert d.s.relative_cardinality() == pytest.approx(0.5)


def test_set_comparisons_reflect_subset_relations():
    d = Domain("t", 0, 4)
    d.low = constant(0.2)
    d.high = constant(0.5)
    assert d.low < d.high
    assert d.low <= d.high
    assert not (d.low > d.high)
    assert not (d.low >= d.high)
    assert d.low <= d.low
    assert not (d.low < d.low)


def test_set_equality_compares_values_on_domain():
    d = Domain("t", 0, 4)
    d.a = constant(0.5)
    d.b = constant(0.5)
    d.c = constant(0.4)
    assert d.a == d.b
    assert not (d.a == d.c)


def test_set_operations_without_domain_raise():
    s1 = Set(constant(0.5))
    s2 = Set(constant(0.5))
    with pytest.raises(FuzzyWarning):
        len(s1)
    with pytest.raises(FuzzyWarning):
        s1.cardinality()
    with pytest.raises(FuzzyWarning):
        s1.array()
    with pytest.raises(FuzzyWarning):
        s1 == s2
    with pytest.raises(FuzzyWarning):
        s1 <= s2


def test_set_normalized_scales_max_to_one():
    d = Domain("t", 0, 4)
    d.ramp = lambda x: x / 10  # max over domain is 0.4
    n = d.ramp.normalized()
    assert n(4) == pytest.approx(1.0)
    assert n(2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# functions module
# ---------------------------------------------------------------------------

def test_inv_noop_and_constant():
    assert inv(lambda x: 0.2)(0) == pytest.approx(0.8)
    assert noop()(7) == 7
    assert constant(3)(None) == 3


def test_alpha_normalize_and_moderate():
    f = alpha(floor=0.2, ceiling=0.8, func=lambda x: x)
    assert f(0.9) == 0.8
    assert f(0.1) == 0.2
    assert f(0.5) == 0.5

    g = alpha(floor=0.2, ceiling=0.8, func=lambda x: x, ceiling_clip=0.9)
    assert g(0.95) == 0.9

    assert normalize(0.5, lambda x: 0.25)(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        normalize(0, lambda x: 0.25)

    assert moderate(lambda x: 0.5)(0) == pytest.approx(0.5)
    assert moderate(lambda x: 1)(0) == pytest.approx(1.0)
    assert moderate(lambda x: 0)(0) == pytest.approx(0.0)


def test_singleton_and_linear():
    s = singleton(2)
    assert s(2) == 1
    assert s(1) == 0
    assert s(3) == 0
    assert singleton(2, no_m=0.1, c_m=0.9)(5) == 0.1

    lin = linear(1, -1)
    assert lin(-2) == 0
    assert lin(0) == 0
    assert lin(1.5) == pytest.approx(0.5)
    assert lin(3) == 1


def test_bounded_linear_and_ramps():
    f = bounded_linear(2, 3)
    assert f(1) == 0.0
    assert f(2.5) == pytest.approx(0.5)
    assert f(4) == 1.0

    inverse = bounded_linear(2, 3, inverse=True)
    assert inverse(2) == pytest.approx(1.0)
    assert inverse(2.5) == pytest.approx(0.5)
    assert inverse(4) == 0

    r = R(0, 10)
    assert r(-1) == 0
    assert r(5) == pytest.approx(0.5)
    assert r(11) == 1

    s = S(0, 10)
    assert s(-1) == 1
    assert s(5) == pytest.approx(0.5)
    assert s(10) == 0


def test_rectangular_triangular_and_trapezoid():
    rect = rectangular(2, 4, c_m=0.7, no_m=0.1)
    assert rect(1) == pytest.approx(0.1)
    assert rect(3) == pytest.approx(0.7)
    assert rect(5) == pytest.approx(0.1)

    tri = triangular(0, 10)
    assert tri(5) == pytest.approx(1.0)
    assert tri(2.5) == pytest.approx(0.5)
    assert tri(7.5) == pytest.approx(0.5)
    assert tri(0) == pytest.approx(0.0)
    assert tri(10) == pytest.approx(0.0)

    trap = trapezoid(0, 2, 4, 6)
    assert trap(-1) == 0
    assert trap(1) == pytest.approx(0.5)
    assert trap(3) == 1
    assert trap(5) == pytest.approx(0.5)
    assert trap(7) == 0


def test_sigmoid_family():
    sig = sigmoid(1, 1, 0)
    assert sig(0) == pytest.approx(0.5)
    assert sig(50) == pytest.approx(1.0)

    bs = bounded_sigmoid(0, 1)
    assert bs(0) == pytest.approx(0.1)
    assert bs(1) == pytest.approx(0.9)

    ss = simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)

    ts = triangular_sigmoid(2, 4)
    assert ts(2) == pytest.approx(0.1)
    assert ts(3) == pytest.approx(0.9, abs=1e-2)


def test_bounded_exponential_and_gauss():
    be = bounded_exponential(k=1, limit=1)
    assert be(0) == pytest.approx(0.0)
    assert be(1) == pytest.approx(1 - 1 / math.e)
    assert bounded_exponential(k=1, limit=2)(1000) == pytest.approx(2.0)

    g = gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) == pytest.approx(math.exp(-1))
    assert gauss(0, 1, c_m=0.5)(0) == pytest.approx(0.5)
