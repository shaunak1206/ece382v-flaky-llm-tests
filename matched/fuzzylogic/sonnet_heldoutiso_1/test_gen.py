import math

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic import functions as fn


# ---------------------------------------------------------------------------
# functions module
# ---------------------------------------------------------------------------

def test_inv_constant_noop():
    assert fn.inv(fn.constant(0.25))(123) == pytest.approx(0.75)
    assert fn.constant(0.4)(-5) == 0.4
    assert fn.noop()(7) == 7


def test_singleton():
    f = fn.singleton(2)
    assert f(2) == 1
    assert f(1) == 0
    assert f(3) == 0
    g = fn.singleton(2, no_m=0.1, c_m=0.9)
    assert g(2) == 0.9
    assert g(0) == 0.1


def test_linear_is_clipped():
    f = fn.linear(1, -1)
    assert f(-2) == 0
    assert f(1) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(2) == 1
    assert f(3) == 1


def test_bounded_linear_and_inverse():
    f = fn.bounded_linear(2, 3)
    assert f(1) == 0
    assert f(2.5) == pytest.approx(0.5)
    assert f(4) == 1
    g = fn.bounded_linear(2, 3, inverse=True)
    assert g(1) == pytest.approx(1)
    assert g(2.5) == pytest.approx(0.5)
    assert g(4) == pytest.approx(0)


def test_bounded_linear_invalid_bounds():
    with pytest.raises(AssertionError):
        fn.bounded_linear(2, 1)
    with pytest.raises(AssertionError):
        fn.bounded_linear(1, 2, c_m=0, no_m=0)


def test_R_and_S_are_mirror_ramps():
    r = fn.R(0, 10)
    s = fn.S(0, 10)
    assert r(-1) == 0
    assert r(5) == pytest.approx(0.5)
    assert r(11) == 1
    assert s(-1) == 1
    assert s(5) == pytest.approx(0.5)
    assert s(11) == 0
    for x in (0, 2, 5, 8, 10):
        assert r(x) + s(x) == pytest.approx(1)


def test_rectangular():
    f = fn.rectangular(1, 3)
    assert f(0) == 0
    assert f(1) == 1
    assert f(2) == 1
    assert f(3) == 1
    assert f(4) == 0
    g = fn.rectangular(1, 3, c_m=0.8, no_m=0.1)
    assert g(2) == 0.8
    assert g(10) == 0.1


def test_triangular():
    f = fn.triangular(0, 10)
    assert f(0) == pytest.approx(0)
    assert f(2.5) == pytest.approx(0.5)
    assert f(5) == pytest.approx(1)
    assert f(7.5) == pytest.approx(0.5)
    assert f(10) == pytest.approx(0)
    assert f(-3) == pytest.approx(0)
    assert f(13) == pytest.approx(0)


def test_triangular_invalid_peak():
    with pytest.raises(AssertionError):
        fn.triangular(0, 10, c=10)
    with pytest.raises(AssertionError):
        fn.triangular(5, 5)


def test_trapezoid():
    f = fn.trapezoid(0, 2, 4, 6)
    assert f(-1) == 0
    assert f(1) == pytest.approx(0.5)
    assert f(2) == pytest.approx(1)
    assert f(3) == pytest.approx(1)
    assert f(5) == pytest.approx(0.5)
    assert f(7) == 0


def test_sigmoid_midpoint_and_limits():
    f = fn.sigmoid(1, 1, 0)
    assert f(0) == pytest.approx(0.5)
    assert f(100) == pytest.approx(1.0)
    assert f(-100) == pytest.approx(0.0, abs=1e-9)
    g = fn.sigmoid(0.6, 2, 3)
    assert g(3) == pytest.approx(0.3)
    with pytest.raises(AssertionError):
        fn.sigmoid(0, 1, 0)


def test_bounded_sigmoid_quantiles():
    f = fn.bounded_sigmoid(0, 1)
    assert f(0) == pytest.approx(0.1)
    assert f(1) == pytest.approx(0.9)
    assert f(1e6) == pytest.approx(1.0, abs=1e-6)
    assert f(-1e6) == pytest.approx(0.0, abs=1e-6)
    g = fn.bounded_sigmoid(0, 1, inverse=True)
    assert g(0) == pytest.approx(0.9)
    assert g(1) == pytest.approx(0.1)


def test_simple_sigmoid_and_bounded_exponential():
    f = fn.simple_sigmoid()
    assert f(0) == pytest.approx(0.5)
    assert f(20) == pytest.approx(0.99, abs=0.005)
    assert f(-20) == pytest.approx(0.01, abs=0.005)
    e = fn.bounded_exponential(k=0.5, limit=0.8)
    assert e(0) == pytest.approx(0)
    assert e(1000) == pytest.approx(0.8)
    assert e(1) < e(2) < 0.8


def test_gauss():
    f = fn.gauss(0, 1)
    assert f(0) == pytest.approx(1)
    assert f(1) == pytest.approx(math.exp(-1))
    assert f(-2) == pytest.approx(f(2))
    g = fn.gauss(2, 1, c_m=0.5)
    assert g(2) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        fn.gauss(0, 0)


def test_alpha_normalize_moderate():
    hi = fn.alpha(floor=0.2, ceiling=0.8, func=fn.constant(0.9))
    lo = fn.alpha(floor=0.2, ceiling=0.8, func=fn.constant(0.1))
    mid = fn.alpha(floor=0.2, ceiling=0.8, func=fn.constant(0.5))
    assert hi(0) == 0.8
    assert lo(0) == 0.2
    assert mid(0) == 0.5

    assert fn.normalize(0.5, fn.constant(0.25))(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        fn.normalize(0, fn.constant(0.25))

    m = fn.moderate
    assert m(fn.constant(0.5))(0) == pytest.approx(0.5)
    assert m(fn.constant(1))(0) == pytest.approx(1.0)
    assert m(fn.constant(0))(0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_validation_and_range():
    with pytest.raises(AssertionError):
        Domain("bad", 5, 5)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 5, res=0)
    d = Domain("d", 0, 10)
    assert str(d) == "d"
    r = list(d.range())
    assert r[0] == 0 and r[-1] == 10 and len(r) == 11
    d2 = Domain("d2", 0, 1, res=0.5)
    assert list(d2.range()) == [0, 0.5, 1.0]


def test_domain_set_assignment_call_and_delete():
    d = Domain("temp", 0, 100)
    d.hot = fn.constant(0.7)          # plain function is wrapped
    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert str(d.hot) == "temp.hot"
    d.cold = ~d.hot
    assert d(3) == {"hot": pytest.approx(0.7), "cold": pytest.approx(0.3)}
    del d.cold
    assert "cold" not in d(3)
    with pytest.raises(AttributeError):
        d.cold
    with pytest.raises(FuzzyWarning):
        del d.nonexistent


def test_domain_min_max_and_invalid_name():
    d = Domain("d", 0, 10)
    d.a = fn.constant(0.2)
    d.b = fn.constant(0.7)
    assert d.min(1) == pytest.approx(0.2)
    assert d.max(1) == pytest.approx(0.7)
    with pytest.raises(AssertionError):
        setattr(d, "not valid", fn.constant(1))


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

def test_set_call_and_operators():
    a = Set(fn.constant(0.3))
    b = Set(fn.constant(0.6))
    assert a(5) == pytest.approx(0.3)
    assert (~a)(5) == pytest.approx(0.7)
    assert (-a)(5) == pytest.approx(0.7)
    assert (a & b)(5) == pytest.approx(0.3)
    assert (a | b)(5) == pytest.approx(0.6)
    assert (a * b)(5) == pytest.approx(0.18)
    assert (b ** 2)(5) == pytest.approx(0.36)
    assert a.concentrated()(5) == pytest.approx(0.09)
    assert a.multiplied(2)(5) == pytest.approx(0.6)


def test_set_without_domain_raises():
    a = Set(fn.constant(0.3))
    b = Set(fn.constant(0.3))
    assert "dangling" in str(a)
    with pytest.raises(FuzzyWarning):
        a == b
    with pytest.raises(FuzzyWarning):
        a <= b
    with pytest.raises(FuzzyWarning):
        a.array()
    with pytest.raises(FuzzyWarning):
        len(a)
    with pytest.raises(FuzzyWarning):
        a.cardinality()
    with pytest.raises(FuzzyWarning):
        a.normalized()
    with pytest.raises(FuzzyWarning):
        a.plot()
    with pytest.raises(FuzzyWarning):
        Set(fn.constant(1), name="x")


def test_set_array_len_and_cardinality():
    d = Domain("d", 0, 10)
    d.half = fn.constant(0.5)
    arr = d.half.array()
    assert len(arr) == 11
    assert all(v == 0.5 for v in arr)
    assert len(d.half) == 11
    assert d.half.cardinality() == pytest.approx(5.5)
    assert d.half.relative_cardinality() == pytest.approx(0.5)


def test_set_comparisons_in_domain():
    d = Domain("d", 0, 10)
    d.low = fn.constant(0.2)
    d.high = fn.constant(0.5)
    d.other_high = fn.constant(0.5)
    assert d.low <= d.high
    assert d.low < d.high
    assert d.high >= d.low
    assert d.high > d.low
    assert not (d.high <= d.low)
    assert d.high == d.other_high
    assert not (d.low == d.high)
    assert d.high <= d.other_high
    assert not (d.high < d.other_high)


def test_set_normalized():
    d = Domain("d", 0, 10)
    d.s = fn.constant(0.5)
    n = d.s.normalized()
    assert n(3) == pytest.approx(1.0)
    d.tri = fn.triangular(0, 10, c_m=0.5)
    nt = d.tri.normalized()
    assert max(nt(x) for x in d.range()) == pytest.approx(1.0)
