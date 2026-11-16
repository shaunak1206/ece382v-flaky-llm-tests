import pytest

from fuzzylogic import functions as fn
from fuzzylogic.classes import Domain, Set, FuzzyWarning


# ---------------------------------------------------------------------------
# functions module
# ---------------------------------------------------------------------------

def test_inv_constant_and_noop():
    assert fn.inv(fn.constant(0.25))(3) == pytest.approx(0.75)
    assert fn.constant(0.4)(100) == 0.4
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


def test_bounded_linear_values_and_inverse():
    f = fn.bounded_linear(2, 3)
    assert f(1) == 0
    assert f(2) == 0
    assert f(2.5) == pytest.approx(0.5)
    assert f(3) == 1
    assert f(4) == 1
    g = fn.bounded_linear(2, 3, inverse=True)
    assert g(1) == 1
    assert g(2.5) == pytest.approx(0.5)
    assert g(4) == 0


def test_bounded_linear_invalid_bounds():
    with pytest.raises(AssertionError):
        fn.bounded_linear(3, 2)


def test_R_and_S_are_mirror_slopes():
    r = fn.R(0, 10)
    s = fn.S(0, 10)
    assert r(-1) == 0
    assert r(5) == pytest.approx(0.5)
    assert r(11) == 1
    assert s(-1) == 1
    assert s(5) == pytest.approx(0.5)
    assert s(11) == 0
    with pytest.raises(AssertionError):
        fn.R(5, 5)
    with pytest.raises(AssertionError):
        fn.S(5, 1)


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
    f = fn.triangular(0, 4)
    assert f(-1) == 0
    assert f(0) == 0
    assert f(1) == pytest.approx(0.5)
    assert f(2) == pytest.approx(1)
    assert f(3) == pytest.approx(0.5)
    assert f(4) == pytest.approx(0)
    assert f(5) == pytest.approx(0)


def test_triangular_invalid_peak():
    with pytest.raises(AssertionError):
        fn.triangular(0, 4, c=5)
    with pytest.raises(AssertionError):
        fn.triangular(4, 0)


def test_trapezoid():
    f = fn.trapezoid(0, 1, 3, 4)
    assert f(-1) == 0
    assert f(0.5) == pytest.approx(0.5)
    assert f(1) == pytest.approx(1)
    assert f(2) == 1
    assert f(3) == pytest.approx(1)
    assert f(3.5) == pytest.approx(0.5)
    assert f(5) == 0


def test_sigmoid_and_simple_sigmoid_midpoints():
    assert fn.sigmoid(1, 1, 0)(0) == pytest.approx(0.5)
    assert fn.sigmoid(0.8, 2, 3)(3) == pytest.approx(0.4)
    assert fn.sigmoid(1, 1, 0)(1000) == pytest.approx(1.0)
    f = fn.simple_sigmoid()
    assert f(0) == pytest.approx(0.5)
    assert f(1000) == pytest.approx(1.0)
    assert f(-1000) == pytest.approx(0.0)
    with pytest.raises(AssertionError):
        fn.sigmoid(0, 1, 0)


def test_bounded_sigmoid_hits_10_and_90_percent():
    f = fn.bounded_sigmoid(0, 1)
    assert f(0) == pytest.approx(0.1)
    assert f(1) == pytest.approx(0.9)
    assert f(100000) == pytest.approx(1.0)
    assert f(-100000) == pytest.approx(0.0)
    g = fn.bounded_sigmoid(0, 1, inverse=True)
    assert g(0) == pytest.approx(0.9)
    assert g(1) == pytest.approx(0.1)


def test_gauss_and_bounded_exponential():
    import math
    g = fn.gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) == pytest.approx(math.exp(-1))
    assert g(2) == pytest.approx(g(-2))
    with pytest.raises(AssertionError):
        fn.gauss(0, 0)
    e = fn.bounded_exponential(k=0.1, limit=1)
    assert e(0) == pytest.approx(0)
    assert e(1e6) == pytest.approx(1)


def test_alpha_normalize_moderate():
    a = fn.alpha(floor=0.2, ceiling=0.8, func=fn.noop())
    assert a(0.9) == 0.8
    assert a(0.1) == 0.2
    assert a(0.5) == 0.5
    n = fn.normalize(0.5, fn.noop())
    assert n(0.25) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        fn.normalize(0, fn.noop())
    m = fn.moderate(fn.noop())
    assert m(0.5) == pytest.approx(0.5)
    assert m(1) == pytest.approx(1.0)
    assert m(0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_invalid_arguments():
    with pytest.raises(AssertionError):
        Domain("x", 5, 1)
    with pytest.raises(AssertionError):
        Domain("x", 0, 5, res=0)


def test_domain_range_includes_upper_bound():
    d = Domain("x", 0, 5)
    assert list(d.range()) == [0, 1, 2, 3, 4, 5]
    d2 = Domain("y", 0, 10, res=2)
    assert list(d2.range()) == [0, 2, 4, 6, 8, 10]


def test_domain_set_assignment_and_call():
    d = Domain("temp", 0, 100)
    d.hot = fn.constant(1)          # plain function gets wrapped into a Set
    d.cold = ~d.hot
    assert isinstance(d.hot, Set)
    assert d.hot.domain is d
    assert d.hot.name == "hot"
    assert d.cold.name == "cold"
    assert d(5) == {"hot": 1, "cold": 0}
    assert str(d) == "temp"
    assert str(d.hot) == "temp.hot"


def test_domain_getattr_delattr_errors():
    d = Domain("temp", 0, 100)
    d.hot = Set(fn.constant(1))
    with pytest.raises(AttributeError):
        d.missing
    del d.hot
    assert d({}.get("x", 5)) == {}
    with pytest.raises(AttributeError):
        d.hot
    with pytest.raises(FuzzyWarning):
        del d.nonexistent


def test_domain_rejects_non_identifier_set_name():
    d = Domain("temp", 0, 100)
    with pytest.raises(AssertionError):
        setattr(d, "not valid", fn.constant(1))


def test_domain_min_max():
    d = Domain("x", 0, 10)
    d.a = fn.constant(0.2)
    d.b = fn.constant(0.7)
    assert d.min(3) == 0.2
    assert d.max(3) == 0.7


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

def test_set_call_and_operators():
    a = Set(fn.constant(0.3))
    b = Set(fn.constant(0.6))
    assert a(1) == 0.3
    assert (~a)(1) == pytest.approx(0.7)
    assert (-a)(1) == pytest.approx(0.7)
    assert (a & b)(1) == 0.3
    assert (a | b)(1) == 0.6
    assert (a * b)(1) == pytest.approx(0.18)
    assert (b ** 2)(1) == pytest.approx(0.36)


def test_set_derived_helpers():
    s = Set(fn.constant(0.5))
    assert s.concentrated()(0) == pytest.approx(0.25)
    assert s.multiplied(1.5)(0) == pytest.approx(0.75)


def test_set_name_without_domain_raises():
    with pytest.raises(FuzzyWarning):
        Set(fn.constant(1), name="a")


def test_set_constructor_with_name_and_domain():
    d = Domain("x", 0, 4)
    s = Set(fn.constant(1), name="a", domain=d)
    assert d.a is s
    assert s.domain is d
    assert s.name == "a"


def test_set_without_domain_raises_on_domain_dependent_ops():
    s = Set(fn.constant(1))
    other = Set(fn.constant(1))
    assert str(s).startswith("dangling Set")
    with pytest.raises(FuzzyWarning):
        len(s)
    with pytest.raises(FuzzyWarning):
        s.array()
    with pytest.raises(FuzzyWarning):
        s.cardinality()
    with pytest.raises(FuzzyWarning):
        s.normalized()
    with pytest.raises(FuzzyWarning):
        s.plot()
    with pytest.raises(FuzzyWarning):
        s == other
    with pytest.raises(FuzzyWarning):
        s <= other


def test_set_array_len_cardinality():
    d = Domain("x", 0, 4)
    d.a = fn.R(0, 4)
    assert list(d.a.array()) == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
    assert len(d.a) == 5
    assert d.a.cardinality() == pytest.approx(2.5)
    assert d.a.relative_cardinality() == pytest.approx(0.5)


def test_set_comparisons_and_equality():
    d = Domain("x", 0, 4)
    d.a = fn.R(0, 4)
    d.b = fn.constant(1)
    d.c = fn.R(0, 4)
    assert d.a <= d.b
    assert d.b >= d.a
    assert not (d.a < d.b)      # equal at x == 4
    assert not (d.a > d.b)
    assert d.a == d.c
    assert not (d.a == d.b)


def test_set_normalized():
    d = Domain("x", 0, 4)
    d.a = lambda x: 0.5 * x / 4
    n = d.a.normalized()
    assert n(4) == pytest.approx(1.0)
    assert n(2) == pytest.approx(0.5)
