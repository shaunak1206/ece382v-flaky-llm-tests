import numpy as np
import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic import functions as fz


def test_domain_construction_and_validation():
    np.random.seed(0)
    d = Domain("temp", 0, 100)
    assert str(d) == "temp"
    assert d == Domain("temp", 0, 100)
    assert d != Domain("temp", 0, 50)
    with pytest.raises(AssertionError):
        Domain("bad", 10, 10)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 10, res=0)


def test_domain_range_includes_upper_bound():
    np.random.seed(0)
    d = Domain("d", 0, 10, res=2)
    assert list(d.range()) == [0, 2, 4, 6, 8, 10]


def test_domain_set_assignment_wraps_functions_and_sets_metadata():
    np.random.seed(0)
    d = Domain("d", 0, 10)
    d.f = fz.constant(0.4)
    assert isinstance(d.f, Set)
    assert d.f.name == "f"
    assert d.f.domain is d
    s = Set(fz.constant(0.9))
    d.g = s
    assert d.g is s
    assert s.domain is d
    assert s.name == "g"


def test_domain_missing_attribute_and_delete():
    np.random.seed(0)
    d = Domain("d", 0, 10)
    d.a = Set(fz.constant(1))
    with pytest.raises(AttributeError):
        d.missing
    del d.a
    with pytest.raises(AttributeError):
        d.a
    with pytest.raises(FuzzyWarning):
        del d.not_a_set


def test_domain_call_min_max():
    np.random.seed(0)
    d = Domain("d", 0, 10)
    d.low = Set(fz.constant(0.25))
    d.high = Set(fz.constant(0.75))
    d.inverse = ~d.low
    res = d(5)
    assert res == {"low": pytest.approx(0.25),
                   "high": pytest.approx(0.75),
                   "inverse": pytest.approx(0.75)}
    assert d.min(5) == pytest.approx(0.25)
    assert d.max(5) == pytest.approx(0.75)


def test_set_init_name_domain_consistency():
    np.random.seed(0)
    d = Domain("d", 0, 10)
    with pytest.raises(FuzzyWarning):
        Set(fz.constant(1), name="x")
    s = Set(fz.constant(1), name="x", domain=d)
    assert d.x is s
    assert str(s) == "d.x"


def test_set_str_dangling():
    np.random.seed(0)
    s = Set(fz.constant(1))
    assert str(s).startswith("dangling Set(")


def test_set_logical_operators():
    np.random.seed(0)
    a = Set(fz.constant(0.3))
    b = Set(fz.constant(0.7))
    assert (a & b)(1) == pytest.approx(0.3)
    assert (a | b)(1) == pytest.approx(0.7)
    assert (~a)(1) == pytest.approx(0.7)
    assert (-b)(1) == pytest.approx(0.3)
    assert a(42) == pytest.approx(0.3)


def test_set_pow_concentrated_multiplied():
    np.random.seed(0)
    a = Set(fz.constant(0.5))
    assert (a ** 2)(0) == pytest.approx(0.25)
    assert a.concentrated()(0) == pytest.approx(0.25)
    assert a.multiplied(0.5)(0) == pytest.approx(0.25)


def test_set_array_len_and_cardinality():
    np.random.seed(0)
    d = Domain("d", 0, 10)
    d.half = Set(fz.constant(0.5))
    arr = d.half.array()
    assert len(arr) == 11
    assert len(d.half) == 11
    assert np.allclose(arr, 0.5)
    assert d.half.cardinality() == pytest.approx(5.5)
    assert d.half.relative_cardinality() == pytest.approx(0.5)


def test_set_array_follows_function_values():
    np.random.seed(0)
    d = Domain("d", 0, 10, res=2)
    d.ramp = Set(fz.R(0, 10))
    assert np.allclose(d.ramp.array(), [0, 0.2, 0.4, 0.6, 0.8, 1.0])


def test_set_comparisons_with_domain():
    np.random.seed(0)
    d = Domain("d", 0, 10)
    d.small = Set(fz.constant(0.2))
    d.big = Set(fz.constant(0.5))
    assert d.small <= d.big
    assert d.small < d.big
    assert d.big >= d.small
    assert d.big > d.small
    assert not (d.big <= d.small)
    assert d.small == d.small
    assert not (d.small == d.big)


def test_set_comparisons_without_domain_raise():
    np.random.seed(0)
    a = Set(fz.constant(0.2))
    b = Set(fz.constant(0.5))
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


def test_set_normalized_reaches_one():
    np.random.seed(0)
    d = Domain("d", 0, 10)
    d.half = Set(fz.constant(0.5))
    n = d.half.normalized()
    assert n(3) == pytest.approx(1.0)
    d.tri = Set(fz.triangular(0, 10, c_m=0.5))
    nt = d.tri.normalized()
    assert nt(5) == pytest.approx(1.0)
    assert nt(0) == pytest.approx(0.0)


def test_special_functions():
    np.random.seed(0)
    assert fz.inv(fz.constant(0.3))(5) == pytest.approx(0.7)
    assert fz.noop()(7) == 7
    assert fz.constant(3)("anything") == 3
    f = fz.alpha(floor=0.2, ceiling=0.8, func=fz.noop())
    assert f(0.1) == pytest.approx(0.2)
    assert f(0.9) == pytest.approx(0.8)
    assert f(0.5) == pytest.approx(0.5)
    n = fz.normalize(0.5, fz.constant(0.25))
    assert n(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        fz.normalize(0, fz.noop())
    m = fz.moderate(fz.noop())
    assert m(0) == pytest.approx(0.0)
    assert m(0.5) == pytest.approx(0.5)
    assert m(1) == pytest.approx(1.0)


def test_singleton_and_rectangular():
    np.random.seed(0)
    s = fz.singleton(2)
    assert s(2) == 1
    assert s(3) == 0
    r = fz.rectangular(1, 3)
    assert [r(x) for x in (0, 1, 2, 3, 4)] == [0, 1, 1, 1, 0]
    with pytest.raises(AssertionError):
        fz.rectangular(3, 1)


def test_linear_and_bounded_linear():
    np.random.seed(0)
    f = fz.linear(1, -1)
    assert f(0) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(3) == 1
    g = fz.bounded_linear(2, 3)
    assert g(1) == pytest.approx(0.0)
    assert g(2.5) == pytest.approx(0.5)
    assert g(4) == pytest.approx(1.0)
    h = fz.bounded_linear(2, 3, inverse=True)
    assert h(2) == pytest.approx(1.0)
    assert h(2.5) == pytest.approx(0.5)
    assert h(3) == pytest.approx(0.0)
    with pytest.raises(AssertionError):
        fz.bounded_linear(3, 2)


def test_R_and_S_are_mirrored_ramps():
    np.random.seed(0)
    r = fz.R(0, 10)
    s = fz.S(0, 10)
    assert r(-1) == 0 and r(11) == 1
    assert s(-1) == 1 and s(11) == 0
    for x in (0, 2.5, 5, 7.5, 10):
        assert r(x) == pytest.approx(x / 10)
        assert s(x) == pytest.approx(1 - x / 10)


def test_triangular_and_trapezoid():
    np.random.seed(0)
    t = fz.triangular(0, 10)
    assert t(0) == pytest.approx(0.0)
    assert t(2.5) == pytest.approx(0.5)
    assert t(5) == pytest.approx(1.0)
    assert t(7.5) == pytest.approx(0.5)
    assert t(10) == pytest.approx(0.0)
    assert t(15) == pytest.approx(0.0)
    with pytest.raises(AssertionError):
        fz.triangular(0, 10, c=20)
    z = fz.trapezoid(0, 2, 4, 6)
    assert z(-1) == pytest.approx(0.0)
    assert z(1) == pytest.approx(0.5)
    assert z(3) == pytest.approx(1.0)
    assert z(5) == pytest.approx(0.5)
    assert z(7) == pytest.approx(0.0)


def test_sigmoid_and_simple_sigmoid():
    np.random.seed(0)
    s = fz.sigmoid(1, 1, 0)
    assert s(0) == pytest.approx(0.5)
    assert s(1000) == pytest.approx(1.0)
    assert s(-1e6) == pytest.approx(0.0)
    assert fz.sigmoid(0.8, 1, 0)(0) == pytest.approx(0.4)
    with pytest.raises(AssertionError):
        fz.sigmoid(0, 1, 0)
    ss = fz.simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(-1e6) == pytest.approx(0.0)
    assert ss(1e6) == pytest.approx(1.0)
    for x in (1, 5, 20):
        assert ss(x) + ss(-x) == pytest.approx(1.0)


def test_bounded_sigmoid_and_triangular_sigmoid():
    np.random.seed(0)
    f = fz.bounded_sigmoid(0, 1)
    assert f(0) == pytest.approx(0.1)
    assert f(1) == pytest.approx(0.9)
    assert f(1e6) == pytest.approx(1.0)
    assert f(-1e6) == pytest.approx(0.0)
    g = fz.bounded_sigmoid(0, 1, inverse=True)
    assert g(0) == pytest.approx(0.9)
    assert g(1) == pytest.approx(0.1)
    t = fz.triangular_sigmoid(2, 4)
    assert t(2) == pytest.approx(0.1)
    assert t(3) == pytest.approx(0.9)
    assert t(4) == pytest.approx(0.1)


def test_bounded_exponential_and_gauss():
    np.random.seed(0)
    e = fz.bounded_exponential(k=0.1, limit=1)
    assert e(0) == pytest.approx(0.0)
    assert e(10) == pytest.approx(1 - np.exp(-1))
    assert e(1e6) == pytest.approx(1.0)
    g = fz.gauss(5, 1)
    assert g(5) == pytest.approx(1.0)
    assert g(6) == pytest.approx(np.exp(-1))
    assert g(4) == pytest.approx(g(6))
    assert fz.gauss(5, 1, c_m=0.5)(5) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        fz.gauss(5, 0)
