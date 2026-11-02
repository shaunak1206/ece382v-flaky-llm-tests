import math
import warnings

import pytest

from fuzzylogic import functions as F
from fuzzylogic.classes import Domain, Set, FuzzyWarning


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_rejects_bad_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("x", 5, 1)
    with pytest.raises(AssertionError):
        Domain("x", 0, 1, res=0)


def test_domain_range_includes_upper_bound_and_respects_res():
    d = Domain("x", 0, 3)
    assert [float(v) for v in d.range()] == [0.0, 1.0, 2.0, 3.0]

    d2 = Domain("y", 0, 1, res=0.5)
    assert [float(v) for v in d2.range()] == [0.0, 0.5, 1.0]


def test_domain_assignment_wraps_function_and_names_set():
    d = Domain("temp", 0, 100)
    d.hot = lambda x: 0
    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert str(d.hot) == "temp.hot"
    assert str(d) == "temp"

    s = Set(lambda x: x / 100)
    d.cold = s
    assert d.cold is s


def test_domain_call_returns_membership_per_set_even_outside_bounds():
    d = Domain("temp", 0, 100)
    d.cold = lambda x: 1 - x / 100
    d.hot = lambda x: x / 100

    result = d(25)
    assert result == pytest.approx({"cold": 0.75, "hot": 0.25})

    # Values outside the domain still evaluate (a warning is logged, not raised).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outside = d(150)
    assert outside == pytest.approx({"cold": -0.5, "hot": 1.5})


def test_domain_unknown_attribute_and_invalid_set_name():
    d = Domain("x", 0, 1)
    with pytest.raises(AttributeError):
        d.missing
    with pytest.raises(AssertionError):
        setattr(d, "not an identifier", lambda x: 0)


def test_domain_del_removes_set_and_rejects_unknown_names():
    d = Domain("x", 0, 1)
    d.a = lambda x: 1
    del d.a
    with pytest.raises(AttributeError):
        d.a
    with pytest.raises(FuzzyWarning):
        del d.nope


def test_domain_min_max_and_equality():
    d = Domain("x", 0, 1)
    d.a = lambda x: x
    d.b = lambda x: 1 - x
    assert d.min(0.3) == pytest.approx(0.3)
    assert d.max(0.3) == pytest.approx(0.7)

    assert Domain("t", 0, 10) == Domain("t", 0, 10)
    assert Domain("t", 0, 10) != Domain("t", 0, 20)
    assert Domain("t", 0, 10) != Domain("u", 0, 10)


# ---------------------------------------------------------------------------
# Set: construction and attachment
# ---------------------------------------------------------------------------

def test_set_init_validation_and_domain_attachment():
    with pytest.raises(AssertionError):
        Set(5)
    with pytest.raises(FuzzyWarning):
        Set(lambda x: x, name="x")

    d = Domain("temp", 0, 10)
    s = Set(lambda x: 1, name="warm", domain=d)
    assert d.warm is s
    assert s.name == "warm"
    assert s.domain is d


# ---------------------------------------------------------------------------
# Set: logic operators
# ---------------------------------------------------------------------------

def test_set_invert_and_negate():
    s = Set(lambda x: x)
    assert (~s)(0.25) == pytest.approx(0.75)
    assert (-s)(0.25) == pytest.approx(0.75)


def test_set_and_or_use_min_max():
    a = Set(lambda x: x)
    b = Set(lambda x: 1 - x)
    assert (a & b)(0.2) == pytest.approx(0.2)
    assert (a | b)(0.2) == pytest.approx(0.8)


def test_set_mul_and_add_use_product_and_bounded_sum():
    a = Set(lambda x: 0.5)
    b = Set(lambda x: 0.4)
    assert (a * b)(0) == pytest.approx(0.2)
    assert (a + b)(0) == pytest.approx(0.5 + 0.4 - 0.5 * 0.4)


def test_set_xor_uses_simple_disjoint_sum():
    a = Set(lambda x: 0.2)
    b = Set(lambda x: 0.5)
    assert (a ^ b)(0) == pytest.approx(0.5)
    assert (b ^ a)(0) == pytest.approx(0.5)


def test_set_pow_multiplied_concentrated():
    s = Set(lambda x: 0.5)
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.multiplied(3)(0) == pytest.approx(1.5)
    assert s.concentrated()(0) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Set: domain-dependent operations
# ---------------------------------------------------------------------------

def test_set_array_len_and_cardinality_within_domain():
    d = Domain("x", 0, 4)
    d.a = lambda x: x / 4
    assert list(d.a.array()) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert len(d.a) == 5
    assert d.a.cardinality() == pytest.approx(2.5)
    assert d.a.relative_cardinality() == pytest.approx(0.5)


def test_set_domain_dependent_ops_need_domain():
    s = Set(lambda x: x)
    with pytest.raises(FuzzyWarning):
        s.array()
    with pytest.raises(FuzzyWarning):
        len(s)
    with pytest.raises(FuzzyWarning):
        s.cardinality()
    with pytest.raises(FuzzyWarning):
        s.relative_cardinality()


def test_set_comparisons_subset_and_equality():
    d = Domain("x", 0, 2)
    d.p = lambda x: 0.2
    d.q = lambda x: 0.5
    d.r = lambda x: 0.2

    assert d.p <= d.q
    assert d.p < d.q
    assert d.q >= d.p
    assert d.q > d.p
    assert d.p == d.r
    assert not (d.p < d.r)
    assert d.p != d.q


def test_set_normalized_scales_max_to_one():
    d = Domain("x", 0, 4)
    d.a = lambda x: x / 8  # max over the domain is 0.5
    n = d.a.normalized()
    assert n(4) == pytest.approx(1.0)
    assert n(2) == pytest.approx(0.5)

    with pytest.raises(FuzzyWarning):
        Set(lambda x: x).normalized()


# ---------------------------------------------------------------------------
# functions module
# ---------------------------------------------------------------------------

def test_functions_inv_noop_constant():
    assert F.inv(F.constant(0.3))(0) == pytest.approx(0.7)
    assert F.noop()(0.4) == pytest.approx(0.4)
    assert F.constant(0.6)(123) == pytest.approx(0.6)


def test_functions_alpha_clips_and_rejects_inverted_bounds():
    f = F.alpha(floor=0.2, ceiling=0.8, func=lambda x: x)
    assert f(0.1) == pytest.approx(0.2)
    assert f(0.9) == pytest.approx(0.8)
    assert f(0.5) == pytest.approx(0.5)

    g = F.alpha(floor=0.2, floor_clip=0.0, func=lambda x: x)
    assert g(0.1) == pytest.approx(0.0)

    with pytest.raises(AssertionError):
        F.alpha(floor=0.9, ceiling=0.1, func=lambda x: x)


def test_functions_linear_and_bounded_linear():
    lin = F.linear(1, -1)
    assert lin(-2) == 0
    assert lin(0) == 0
    assert lin(1.5) == pytest.approx(0.5)
    assert lin(2) == 1
    assert lin(3) == 1

    bl = F.bounded_linear(2, 3)
    assert bl(1) == 0
    assert bl(2.5) == pytest.approx(0.5)
    assert bl(3) == 1
    assert bl(4) == 1

    inv_bl = F.bounded_linear(2, 3, inverse=True)
    assert inv_bl(2.5) == pytest.approx(0.5)
    assert inv_bl(3) == pytest.approx(0.0)

    with pytest.raises(AssertionError):
        F.bounded_linear(3, 2)


def test_functions_R_and_S_ramps():
    r = F.R(0, 10)
    assert r(-1) == 0
    assert r(5) == pytest.approx(0.5)
    assert r(10) == 1
    assert r(20) == 1

    s = F.S(0, 10)
    assert s(-1) == 1
    assert s(0) == 1
    assert s(5) == pytest.approx(0.5)
    assert s(10) == 0

    with pytest.raises(AssertionError):
        F.R(5, 5)


def test_functions_rectangular_and_trapezoid():
    rect = F.rectangular(1, 3, c_m=0.8, no_m=0.1)
    assert rect(0) == pytest.approx(0.1)
    assert rect(1) == pytest.approx(0.8)
    assert rect(2) == pytest.approx(0.8)
    assert rect(3) == pytest.approx(0.8)
    assert rect(4) == pytest.approx(0.1)

    trap = F.trapezoid(0, 4, 6, 10)
    assert trap(-1) == 0
    assert trap(2) == pytest.approx(0.5)
    assert trap(5) == 1
    assert trap(8) == pytest.approx(0.5)
    assert trap(11) == 0


def test_functions_triangular_peak_and_flanks():
    tri = F.triangular(0, 10)
    assert tri(0) == pytest.approx(0.0)
    assert tri(2.5) == pytest.approx(0.5)
    assert tri(5) == pytest.approx(1.0)
    assert tri(7.5) == pytest.approx(0.5)
    assert tri(10) == pytest.approx(0.0)

    with pytest.raises(AssertionError):
        F.triangular(0, 10, c=12)


def test_functions_singleton_gauss_moderate():
    sing = F.singleton(2)
    assert sing(1) == 0
    assert sing(2) == 1
    sing2 = F.singleton(2, no_m=0.1, c_m=0.9)
    assert sing2(3) == pytest.approx(0.1)
    with pytest.raises(AssertionError):
        F.singleton(0, no_m=0.5, c_m=0.5)

    g = F.gauss(2, 1)
    assert g(2) == pytest.approx(1.0)
    assert g(1) == pytest.approx(math.exp(-1))
    assert g(3) == pytest.approx(math.exp(-1))
    assert F.gauss(0, 1, c_m=0.5)(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        F.gauss(0, 0)

    assert F.moderate(lambda x: 0.5)(0) == pytest.approx(0.5)
    assert F.moderate(lambda x: 1)(0) == pytest.approx(1.0)
    assert F.moderate(lambda x: 0)(0) == pytest.approx(0.0)
    assert F.moderate(lambda x: 0.75)(0) == pytest.approx(0.5625)


def test_functions_sigmoids_and_exponential():
    assert F.sigmoid(1, 1, 0)(0) == pytest.approx(0.5)

    bs = F.bounded_sigmoid(0, 1)
    assert bs(0) == pytest.approx(0.1)
    assert bs(1) == pytest.approx(0.9)
    assert bs(100000) == pytest.approx(1.0)
    assert bs(-100000) == pytest.approx(0.0)
    with pytest.raises(AssertionError):
        F.bounded_sigmoid(1, 0)

    ss = F.simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(1000) == pytest.approx(1.0)
    assert ss(-1000) == pytest.approx(0.0)

    ts = F.triangular_sigmoid(2, 4)
    assert ts(2) == pytest.approx(0.1)
    assert ts(3) == pytest.approx(0.9)

    be = F.bounded_exponential(k=0.1, limit=1)
    assert be(0) == pytest.approx(0.0)
    assert be(10) == pytest.approx(1 - math.exp(-1))
    assert be(10000) == pytest.approx(1.0)
    with pytest.raises(AssertionError):
        F.bounded_exponential(k=0.1, limit=0)
