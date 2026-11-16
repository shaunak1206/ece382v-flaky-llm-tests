import math

import numpy as np
import pytest

from fuzzylogic import classes
from fuzzylogic import functions as F
from fuzzylogic.classes import Domain, FuzzyWarning, Set


def approx(v):
    return pytest.approx(v, abs=1e-9)


# ---------------------------------------------------------------- functions

def test_inv_noop_constant():
    np.random.seed(0)
    assert F.inv(F.constant(0.25))(3) == approx(0.75)
    assert F.noop()(7) == 7
    assert F.constant(0.4)(-100) == 0.4


def test_singleton():
    np.random.seed(0)
    f = F.singleton(2)
    assert f(2) == 1
    assert f(1) == 0
    g = F.singleton(2, no_m=0.1, c_m=0.9)
    assert g(2) == 0.9
    assert g(3) == 0.1
    with pytest.raises(AssertionError):
        F.singleton(2, no_m=1, c_m=1)


def test_linear_clipped():
    np.random.seed(0)
    f = F.linear(1, -1)
    assert f(-2) == 0
    assert f(1) == 0
    assert f(1.5) == approx(0.5)
    assert f(2) == 1
    assert f(3) == 1


def test_bounded_linear_values_and_inverse():
    np.random.seed(0)
    f = F.bounded_linear(2, 3)
    assert f(1) == approx(0.0)
    assert f(2.5) == approx(0.5)
    assert f(3) == approx(1.0)
    assert f(4) == approx(1.0)
    g = F.bounded_linear(2, 3, inverse=True)
    assert g(2) == approx(1.0)
    assert g(2.5) == approx(0.5)
    assert g(3) == approx(0.0)
    with pytest.raises(AssertionError):
        F.bounded_linear(3, 2)


def test_R_and_S_are_mirror_ramps():
    np.random.seed(0)
    r = F.R(0, 10)
    s = F.S(0, 10)
    assert r(-5) == 0
    assert r(5) == approx(0.5)
    assert r(20) == 1
    assert s(-5) == 1
    assert s(5) == approx(0.5)
    assert s(20) == 0
    for x in (0, 2, 4, 6, 8, 10):
        assert r(x) + s(x) == approx(1.0)


def test_rectangular():
    np.random.seed(0)
    f = F.rectangular(1, 3)
    assert f(0.5) == 0
    assert f(1) == 1
    assert f(2) == 1
    assert f(3) == 1
    assert f(3.5) == 0
    g = F.rectangular(1, 3, c_m=0.7, no_m=0.1)
    assert g(2) == 0.7
    assert g(10) == 0.1


def test_triangular_shape():
    np.random.seed(0)
    f = F.triangular(0, 10)
    assert f(0) == approx(0.0)
    assert f(2.5) == approx(0.5)
    assert f(5) == approx(1.0)
    assert f(7.5) == approx(0.5)
    assert f(10) == approx(0.0)
    assert f(-3) == approx(0.0)
    assert f(15) == approx(0.0)
    with pytest.raises(AssertionError):
        F.triangular(0, 10, c=11)


def test_trapezoid_shape():
    np.random.seed(0)
    f = F.trapezoid(0, 2, 4, 6)
    assert f(-1) == 0
    assert f(1) == approx(0.5)
    assert f(3) == 1
    assert f(5) == approx(0.5)
    assert f(7) == 0
    with pytest.raises(AssertionError):
        F.trapezoid(0, 5, 4, 6)


def test_sigmoid_midpoint_and_limits():
    np.random.seed(0)
    f = F.sigmoid(1, 1, 0)
    assert f(0) == approx(0.5)
    assert f(100) == pytest.approx(1.0, abs=1e-9)
    g = F.sigmoid(0.8, 2, 3)
    assert g(3) == approx(0.4)
    # overflow in exp must be handled gracefully
    assert g(-1e6) == approx(0.0)
    assert g(1e6) == approx(0.8)
    with pytest.raises(AssertionError):
        F.sigmoid(1.5, 1, 0)


def test_bounded_sigmoid_defining_points():
    np.random.seed(0)
    f = F.bounded_sigmoid(0, 1)
    assert f(0) == approx(0.1)
    assert f(1) == approx(0.9)
    assert f(100000) == approx(1.0)
    assert f(-100000) == approx(0.0)
    g = F.bounded_sigmoid(0, 1, inverse=True)
    assert g(0) == approx(0.9)
    assert g(1) == approx(0.1)


def test_simple_sigmoid_and_triangular_sigmoid():
    np.random.seed(0)
    f = F.simple_sigmoid()
    assert f(0) == approx(0.5)
    assert round(f(20), 2) == 0.99
    assert round(f(-20), 2) == 0.01
    assert f(-1e9) == approx(0.0)
    t = F.triangular_sigmoid(2, 4)
    assert t(2) == approx(0.1)
    assert t(3) == approx(0.9)
    assert t(4) == approx(0.1)


def test_gauss_and_bounded_exponential():
    np.random.seed(0)
    g = F.gauss(0, 1)
    assert g(0) == approx(1.0)
    assert g(1) == approx(math.exp(-1))
    assert g(-2) == approx(g(2))
    assert F.gauss(5, 2, c_m=0.5)(5) == approx(0.5)
    e = F.bounded_exponential(0.1, 1)
    assert e(0) == approx(0.0)
    assert e(10) == approx(1 - math.exp(-1))
    assert e(1e9) == approx(1.0)


def test_alpha_normalize_moderate():
    np.random.seed(0)
    a = F.alpha(floor=0.2, ceiling=0.8, func=F.noop())
    assert a(0.1) == approx(0.2)
    assert a(0.5) == approx(0.5)
    assert a(0.9) == approx(0.8)
    n = F.normalize(0.5, F.noop())
    assert n(0.25) == approx(0.5)
    with pytest.raises(AssertionError):
        F.normalize(0, F.noop())
    m = F.moderate(F.noop())
    assert m(0.5) == approx(0.5)
    assert m(1) == approx(1.0)
    assert m(0) == approx(0.0)


# ------------------------------------------------------------------ Domain

def test_domain_validation_and_range():
    np.random.seed(0)
    with pytest.raises(AssertionError):
        Domain("d", 5, 5)
    with pytest.raises(AssertionError):
        Domain("d", 0, 5, res=0)
    d = Domain("d", 0, 10)
    assert list(d.range()) == list(range(0, 11))
    d2 = Domain("d2", 0, 10, res=5)
    assert list(d2.range()) == [0, 5, 10]
    assert str(d) == "d"


def test_domain_set_assignment_and_lookup():
    np.random.seed(0)
    d = Domain("temp", 0, 100)
    d.hot = lambda x: x / 100  # plain function is wrapped
    assert isinstance(d.hot, Set)
    assert d.hot.domain is d
    assert d.hot.name == "hot"
    assert str(d.hot) == "temp.hot"
    assert d.hot(50) == approx(0.5)
    with pytest.raises(AttributeError):
        d.missing
    with pytest.raises(AssertionError):
        setattr(d, "not valid", lambda x: 0)


def test_domain_call_min_max_and_delete():
    np.random.seed(0)
    d = Domain("temp", 0, 10)
    d.hot = Set(lambda x: x / 10)
    d.cold = ~d.hot
    res = d(4)
    assert set(res.keys()) == {"hot", "cold"}
    assert res["hot"] == approx(0.4)
    assert res["cold"] == approx(0.6)
    assert d.min(4) == approx(0.4)
    assert d.max(4) == approx(0.6)
    del d.cold
    assert set(d(4).keys()) == {"hot"}
    with pytest.raises(FuzzyWarning):
        del d.nonexistent


def test_domain_warns_outside_bounds(monkeypatch):
    np.random.seed(0)
    messages = []
    monkeypatch.setattr(classes, "warn", lambda msg, *a, **k: messages.append(msg))
    d = Domain("d", 0, 10)
    d.a = lambda x: 0.5
    d(5)
    assert messages == []
    assert d(11) == {"a": 0.5}
    assert len(messages) == 1


# --------------------------------------------------------------------- Set

def test_set_operators():
    np.random.seed(0)
    a = Set(lambda x: 0.3)
    b = Set(lambda x: 0.8)
    assert (~a)(0) == approx(0.7)
    assert (-a)(0) == approx(0.7)
    assert (a & b)(0) == approx(0.3)
    assert (a | b)(0) == approx(0.8)
    assert (a ** 2)(0) == approx(0.09)
    assert a.concentrated()(0) == approx(0.09)
    assert a.multiplied(2)(0) == approx(0.6)
    assert isinstance(a & b, Set)


def test_set_array_len_and_cardinality():
    np.random.seed(0)
    d = Domain("d", 0, 4)
    d.s = Set(lambda x: x / 4)
    arr = d.s.array()
    assert np.allclose(arr, [0, 0.25, 0.5, 0.75, 1.0])
    assert len(d.s) == 5
    assert d.s.cardinality() == approx(2.5)
    assert d.s.relative_cardinality() == approx(0.5)


def test_set_without_domain_raises():
    np.random.seed(0)
    s = Set(lambda x: 0.5)
    assert str(s).startswith("dangling Set")
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
    with pytest.raises(FuzzyWarning):
        s == Set(lambda x: 0.5)
    with pytest.raises(FuzzyWarning):
        s <= s


def test_set_comparisons_with_domain():
    np.random.seed(0)
    d = Domain("d", 0, 4)
    d.low = Set(lambda x: x / 8)
    d.high = Set(lambda x: x / 4)
    d.same = Set(lambda x: x / 8)
    assert d.low == d.same
    assert not (d.low == d.high)
    assert d.low <= d.high
    assert d.high >= d.low
    assert d.low <= d.same
    assert not (d.low < d.same)
    assert not (d.high <= d.low)
    # strict comparisons fail at x == 0 where both are 0
    assert not (d.low < d.high)
    d.big = Set(lambda x: 0.5 + x / 10)
    assert d.low < d.big
    assert d.big > d.low


def test_set_normalized():
    np.random.seed(0)
    d = Domain("d", 0, 4)
    d.s = Set(lambda x: x / 8)  # max is 0.5
    n = d.s.normalized()
    assert n(4) == approx(1.0)
    assert n(2) == approx(0.5)
    assert max(n(x) for x in d.range()) == approx(1.0)
