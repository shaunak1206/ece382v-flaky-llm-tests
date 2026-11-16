import pytest
from numpy import arange

from fuzzylogic import functions as fn
from fuzzylogic.classes import Domain, Set, FuzzyWarning


# ---------------------------------------------------------------- functions

def test_inv_constant_and_noop():
    assert fn.inv(fn.constant(0.3))(5) == pytest.approx(0.7)
    assert fn.constant(4)(123) == 4
    assert fn.noop()(7) == 7


def test_singleton():
    f = fn.singleton(2)
    assert f(2) == 1
    assert f(1) == 0
    g = fn.singleton(2, no_m=0.1, c_m=0.9)
    assert g(2) == 0.9
    assert g(3) == 0.1


def test_linear_is_clipped():
    f = fn.linear(1, -1)
    assert f(-2) == 0
    assert f(1) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(3) == 1


def test_bounded_linear_rising_and_inverse():
    f = fn.bounded_linear(2, 3)
    assert f(1) == 0
    assert f(2.5) == pytest.approx(0.5)
    assert f(4) == 1
    g = fn.bounded_linear(0, 10, inverse=True)
    assert g(0) == pytest.approx(1)
    assert g(5) == pytest.approx(0.5)
    assert g(10) == pytest.approx(0)


def test_bounded_linear_invalid_bounds():
    with pytest.raises(AssertionError):
        fn.bounded_linear(3, 2)


def test_R_and_S_are_mirror_slopes():
    r = fn.R(0, 10)
    s = fn.S(0, 10)
    assert r(-1) == 0 and r(11) == 1
    assert r(5) == pytest.approx(0.5)
    assert s(-1) == 1 and s(11) == 0
    assert s(5) == pytest.approx(0.5)
    assert s(2) == pytest.approx(0.8)


def test_rectangular():
    f = fn.rectangular(2, 4)
    assert f(1) == 0
    assert f(2) == 1
    assert f(3) == 1
    assert f(4) == 1
    assert f(5) == 0


def test_triangular_shape():
    f = fn.triangular(0, 10)
    assert f(0) == pytest.approx(0)
    assert f(2.5) == pytest.approx(0.5)
    assert f(5) == pytest.approx(1)
    assert f(7.5) == pytest.approx(0.5)
    assert f(10) == pytest.approx(0)


def test_triangular_invalid_peak():
    with pytest.raises(AssertionError):
        fn.triangular(0, 10, c=10)


def test_trapezoid_shape():
    f = fn.trapezoid(0, 2, 4, 6)
    assert f(-1) == 0
    assert f(1) == pytest.approx(0.5)
    assert f(3) == 1
    assert f(5) == pytest.approx(0.5)
    assert f(7) == 0


def test_sigmoid_midpoint_and_limits():
    f = fn.sigmoid(1, 1, 0)
    assert f(0) == pytest.approx(0.5)
    assert f(100) == pytest.approx(1.0)
    assert f(-100) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(AssertionError):
        fn.sigmoid(2, 1, 0)


def test_bounded_sigmoid_reference_points():
    f = fn.bounded_sigmoid(0, 1)
    assert f(0) == pytest.approx(0.1)
    assert f(1) == pytest.approx(0.9)
    assert f(100000) == pytest.approx(1.0)
    assert f(-100000) == pytest.approx(0.0)


def test_simple_sigmoid_and_bounded_exponential():
    s = fn.simple_sigmoid()
    assert s(0) == 0.5
    assert s(20) == pytest.approx(0.99, abs=0.01)
    e = fn.bounded_exponential(k=1, limit=1)
    assert e(0) == 0
    assert e(100) == pytest.approx(1.0)


def test_gauss():
    g = fn.gauss(0, 1)
    assert g(0) == pytest.approx(1)
    assert g(1) == pytest.approx(0.36787944117)
    assert g(2) == pytest.approx(g(-2))
    with pytest.raises(AssertionError):
        fn.gauss(0, 0)


def test_alpha_normalize_moderate():
    assert fn.alpha(func=fn.constant(0.9), ceiling=0.5)(0) == 0.5
    assert fn.alpha(func=fn.constant(0.1), floor=0.2)(0) == 0.2
    assert fn.alpha(func=fn.constant(0.3), floor=0.2, ceiling=0.5)(0) == 0.3
    assert fn.normalize(0.5, fn.constant(0.25))(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        fn.normalize(0, fn.constant(0.25))
    assert fn.moderate(fn.constant(0.5))(0) == pytest.approx(0.5)
    assert fn.moderate(fn.constant(1))(0) == pytest.approx(1.0)
    assert fn.moderate(fn.constant(0))(0) == pytest.approx(0.0)


# ------------------------------------------------------------------ Domain

def test_domain_validation_and_str():
    with pytest.raises(AssertionError):
        Domain("bad", 10, 0)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 10, res=0)
    assert str(Domain("temperature", 0, 100)) == "temperature"


def test_domain_range_includes_upper_bound():
    d = Domain("d", 0, 5)
    assert list(d.range()) == [0, 1, 2, 3, 4, 5]
    d2 = Domain("d2", 0, 2, res=0.5)
    assert list(d2.range()) == list(arange(0, 2.5, 0.5))


def test_domain_set_assignment_and_call():
    d = Domain("temp", 0, 10)
    d.low = fn.constant(0.25)  # plain function gets wrapped
    d.high = Set(fn.constant(0.75))
    assert isinstance(d.low, Set)
    assert d.low.name == "low"
    assert d.low.domain is d
    assert d(3) == {"low": 0.25, "high": 0.75}
    assert str(d.high) == "temp.high"


def test_domain_attribute_errors_and_deletion():
    d = Domain("temp", 0, 10)
    d.hot = fn.constant(1)
    with pytest.raises(AttributeError):
        d.missing
    with pytest.raises(AssertionError):
        setattr(d, "not valid", fn.constant(1))
    del d.hot
    assert d(1) == {}
    with pytest.raises(FuzzyWarning):
        del d.hot


def test_domain_min_max():
    d = Domain("temp", 0, 10)
    d.a = fn.constant(0.2)
    d.b = fn.constant(0.8)
    assert d.min(5) == 0.2
    assert d.max(5) == 0.8


# --------------------------------------------------------------------- Set

def test_set_logic_operators():
    a = Set(fn.constant(0.3))
    b = Set(fn.constant(0.7))
    assert (~a)(0) == pytest.approx(0.7)
    assert (-a)(0) == pytest.approx(0.7)
    assert (a & b)(0) == pytest.approx(0.3)
    assert (a | b)(0) == pytest.approx(0.7)
    assert (a * b)(0) == pytest.approx(0.21)
    assert isinstance(a & b, Set)


def test_set_hedges_and_multiplied():
    s = Set(fn.constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.concentrated()(0) == pytest.approx(0.25)
    assert Set(fn.constant(0.3)).multiplied(2)(0) == pytest.approx(0.6)


def test_set_array_len_and_cardinality():
    d = Domain("d", 0, 4)
    d.s = Set(lambda x: x / 4)
    assert list(d.s.array()) == pytest.approx([0, 0.25, 0.5, 0.75, 1])
    assert len(d.s) == 5
    assert d.s.cardinality() == pytest.approx(2.5)
    assert d.s.relative_cardinality() == pytest.approx(0.5)


def test_set_comparisons_with_domain():
    d = Domain("d", 0, 4)
    d.small = Set(lambda x: x / 8)
    d.big = Set(lambda x: x / 4)
    d.big2 = Set(lambda x: x / 4)
    assert d.small <= d.big
    assert not (d.small < d.big)  # equal at x == 0
    assert d.big >= d.small
    assert not (d.big > d.small)
    assert d.big == d.big2
    assert not (d.small == d.big)


def test_set_without_domain_raises():
    a = Set(fn.constant(0.5))
    b = Set(fn.constant(0.5))
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
    assert str(a).startswith("dangling Set")


def test_set_normalized():
    d = Domain("d", 0, 4)
    d.s = Set(lambda x: x / 8)
    n = d.s.normalized()
    values = [n(x) for x in d.range()]
    assert max(values) == pytest.approx(1.0)
    assert values[0] == pytest.approx(0.0)
    assert values[2] == pytest.approx(0.5)
