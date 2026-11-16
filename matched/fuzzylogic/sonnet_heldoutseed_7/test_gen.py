import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic import functions as fn


# ---------------------------------------------------------------------------
# functions module
# ---------------------------------------------------------------------------

def test_inv_constant_and_noop():
    assert fn.inv(fn.constant(0.25))(123) == pytest.approx(0.75)
    assert fn.inv(fn.noop())(0.3) == pytest.approx(0.7)
    assert fn.noop()(42) == 42
    assert fn.constant(7)(-1) == 7


def test_singleton():
    f = fn.singleton(2)
    assert f(2) == 1
    assert f(1) == 0
    assert f(2.0001) == 0
    g = fn.singleton(2, no_m=0.1, c_m=0.9)
    assert g(2) == 0.9
    assert g(3) == 0.1
    with pytest.raises(AssertionError):
        fn.singleton(1, no_m=1, c_m=0.5)


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
    assert f(2) == 0
    assert f(2.5) == pytest.approx(0.5)
    assert f(3) == 1
    assert f(10) == 1
    g = fn.bounded_linear(2, 3, inverse=True)
    assert g(1) == 1
    assert g(2.5) == pytest.approx(0.5)
    assert g(4) == 0
    with pytest.raises(AssertionError):
        fn.bounded_linear(3, 2)


def test_R_and_S_are_mirrored_ramps():
    r = fn.R(0, 10)
    s = fn.S(0, 10)
    assert r(-5) == 0
    assert r(0) == 0
    assert r(5) == pytest.approx(0.5)
    assert r(10) == 1
    assert r(20) == 1
    assert s(-5) == 1
    assert s(0) == 1
    assert s(5) == pytest.approx(0.5)
    assert s(10) == 0
    assert s(20) == 0
    for x in (1, 3, 7, 9):
        assert r(x) + s(x) == pytest.approx(1)
    with pytest.raises(AssertionError):
        fn.R(5, 5)
    with pytest.raises(AssertionError):
        fn.S(5, 1)


def test_rectangular():
    f = fn.rectangular(1, 3)
    assert f(0.5) == 0
    assert f(1) == 1
    assert f(2) == 1
    assert f(3) == 1
    assert f(3.5) == 0
    with pytest.raises(AssertionError):
        fn.rectangular(3, 1)


def test_triangular_default_and_custom_peak():
    f = fn.triangular(0, 10)
    assert f(-1) == 0
    assert f(0) == 0
    assert f(2.5) == pytest.approx(0.5)
    assert f(5) == pytest.approx(1)
    assert f(7.5) == pytest.approx(0.5)
    assert f(10) == pytest.approx(0)
    assert f(11) == pytest.approx(0)
    g = fn.triangular(0, 10, c=2)
    assert g(2) == pytest.approx(1)
    assert g(1) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        fn.triangular(0, 10, c=10)


def test_trapezoid():
    f = fn.trapezoid(0, 2, 4, 6)
    assert f(-1) == 0
    assert f(1) == pytest.approx(0.5)
    assert f(2) == pytest.approx(1)
    assert f(3) == 1
    assert f(4) == pytest.approx(1)
    assert f(5) == pytest.approx(0.5)
    assert f(7) == 0
    with pytest.raises(AssertionError):
        fn.trapezoid(0, 5, 4, 6)


def test_sigmoid_and_simple_sigmoid():
    f = fn.sigmoid(1, 1, 0)
    assert f(0) == pytest.approx(0.5)
    assert f(100) == pytest.approx(1.0)
    assert f(-100) == pytest.approx(0.0, abs=1e-12)
    assert fn.sigmoid(0.8, 1, 0)(0) == pytest.approx(0.4)
    with pytest.raises(AssertionError):
        fn.sigmoid(0, 1, 0)
    g = fn.simple_sigmoid()
    assert g(0) == pytest.approx(0.5)
    assert g(1000) == pytest.approx(1.0)
    assert g(-1000) == pytest.approx(0.0)
    assert g(20) == pytest.approx(0.99, abs=0.005)


def test_bounded_sigmoid_hits_ten_and_ninety_percent():
    f = fn.bounded_sigmoid(0, 1)
    assert f(0) == pytest.approx(0.1)
    assert f(1) == pytest.approx(0.9)
    assert f(100000) == pytest.approx(1.0)
    assert f(-100000) == pytest.approx(0.0)
    g = fn.bounded_sigmoid(0, 1, inverse=True)
    assert g(0) == pytest.approx(0.9)
    assert g(1) == pytest.approx(0.1)


def test_gauss_and_bounded_exponential():
    g = fn.gauss(5, 0.5)
    assert g(5) == pytest.approx(1)
    assert g(4) == pytest.approx(g(6))
    assert 0 < g(8) < g(6) < 1
    assert fn.gauss(0, 1, c_m=0.5)(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        fn.gauss(0, 0)
    e = fn.bounded_exponential(k=0.5, limit=1)
    assert e(0) == pytest.approx(0)
    assert e(100) == pytest.approx(1)
    assert e(1) < e(2) < 1


def test_alpha_normalize_moderate():
    f = fn.alpha(floor=0.2, ceiling=0.8, func=fn.noop())
    assert f(0.1) == 0.2
    assert f(0.9) == 0.8
    assert f(0.5) == 0.5
    n = fn.normalize(0.5, fn.constant(0.25))
    assert n(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        fn.normalize(0, fn.noop())
    m = fn.moderate(fn.noop())
    assert m(0.5) == pytest.approx(0.5)
    assert m(1) == pytest.approx(1.0)
    assert m(0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def make_domain():
    d = Domain("temp", 0, 10)
    d.low = Set(fn.constant(0.2))
    d.high = Set(fn.constant(0.7))
    return d


def test_domain_validation_and_range():
    with pytest.raises(AssertionError):
        Domain("x", 5, 5)
    with pytest.raises(AssertionError):
        Domain("x", 0, 5, res=0)
    d = Domain("x", 0, 10)
    assert list(d.range()) == list(range(0, 11))
    d2 = Domain("x", 0, 10, res=5)
    assert list(d2.range()) == [0, 5, 10]
    assert str(d) == "x"


def test_domain_set_assignment_and_lookup():
    d = make_domain()
    assert isinstance(d.low, Set)
    assert d.low.domain is d
    assert d.low.name == "low"
    assert str(d.low) == "temp.low"
    # plain functions are wrapped into Sets
    d.mid = fn.constant(0.5)
    assert isinstance(d.mid, Set)
    assert d.mid(3) == 0.5
    with pytest.raises(AttributeError):
        d.nonexistent


def test_domain_call_min_max_and_delete():
    d = make_domain()
    assert d(3) == {"low": 0.2, "high": 0.7}
    assert d.min(3) == 0.2
    assert d.max(3) == 0.7
    del d.low
    assert d(3) == {"high": 0.7}
    with pytest.raises(AttributeError):
        d.low
    with pytest.raises(FuzzyWarning):
        del d.something_else


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

def test_set_call_and_construction_errors():
    s = Set(fn.constant(0.4))
    assert s(10) == 0.4
    assert str(s).startswith("dangling Set")
    with pytest.raises(FuzzyWarning):
        Set(fn.constant(1), name="x")
    d = Domain("d", 0, 5)
    s2 = Set(fn.constant(0.3), name="foo", domain=d)
    assert d.foo is s2


def test_set_logic_operators():
    a = Set(fn.constant(0.3))
    b = Set(fn.constant(0.8))
    assert (~a)(0) == pytest.approx(0.7)
    assert (-a)(0) == pytest.approx(0.7)
    assert (a & b)(0) == pytest.approx(0.3)
    assert (a | b)(0) == pytest.approx(0.8)
    assert (a * b)(0) == pytest.approx(0.24)
    assert (a ** 2)(0) == pytest.approx(0.09)
    assert a.concentrated()(0) == pytest.approx(0.09)
    assert a.multiplied(2)(0) == pytest.approx(0.6)
    assert isinstance(a & b, Set)


def test_set_array_len_and_cardinality():
    d = Domain("d", 0, 10)
    d.half = Set(fn.constant(0.5))
    assert len(d.half) == 11
    assert list(d.half.array()) == [0.5] * 11
    assert d.half.cardinality() == pytest.approx(5.5)
    assert d.half.relative_cardinality() == pytest.approx(0.5)


def test_set_without_domain_raises():
    s = Set(fn.constant(0.5))
    other = Set(fn.constant(0.5))
    for call in (s.array, s.cardinality, s.relative_cardinality,
                 s.normalized, s.plot, lambda: len(s)):
        with pytest.raises(FuzzyWarning):
            call()
    with pytest.raises(FuzzyWarning):
        s == other
    with pytest.raises(FuzzyWarning):
        s <= other


def test_set_comparisons_with_domain():
    d = Domain("d", 0, 10)
    d.small = Set(fn.constant(0.2))
    d.big = Set(fn.constant(0.6))
    d.big_copy = Set(fn.constant(0.6))
    assert d.small <= d.big
    assert d.small < d.big
    assert d.big >= d.small
    assert d.big > d.small
    assert not (d.big <= d.small)
    assert d.big == d.big_copy
    assert d.big >= d.big_copy
    assert not (d.big > d.big_copy)
    assert not (d.small == d.big)
    # derived set assigned to the domain becomes comparable
    d.not_big = ~d.big
    assert d.not_big.array()[0] == pytest.approx(0.4)
    assert d.not_big > d.small


def test_set_normalized():
    d = Domain("d", 0, 10)
    d.tri = Set(fn.triangular(0, 10, c=5, c_m=0.5))
    n = d.tri.normalized()
    assert n(5) == pytest.approx(1.0)
    assert n(0) == pytest.approx(0.0)
    assert n(2.5) == pytest.approx(0.5)
