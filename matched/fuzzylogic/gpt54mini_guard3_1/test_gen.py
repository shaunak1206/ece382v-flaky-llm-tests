import math
import warnings

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv,
    constant,
    alpha,
    normalize,
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


def test_domain_init_and_basic_properties():
    d = Domain("temperature", 0, 10, res=2)
    assert str(d) == "temperature"
    assert "Domain('temperature', 0, 10" in repr(d)
    assert list(d.range()) == [0, 2, 4, 6, 8, 10]


def test_domain_init_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("bad", 1, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 2, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_domain_set_assignment_from_callable_and_access():
    d = Domain("temp", 0, 10)
    d.hot = lambda x: 1 if x >= 7 else 0
    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d(8) == {"hot": 1}
    assert d.hot(6) == 0


def test_domain_set_assignment_requires_identifier():
    d = Domain("temp", 0, 10)
    with pytest.raises(AssertionError):
        setattr(d, "not valid", lambda x: 1)


def test_domain_delattr_and_warning_for_regular_attr():
    d = Domain("temp", 0, 10)
    d.cold = lambda x: 1 if x <= 3 else 0
    del d.cold
    assert "cold" not in d._sets
    with pytest.raises(FuzzyWarning):
        del d._name


def test_domain_call_warns_outside_range_and_returns_memberships():
    d = Domain("temp", 0, 10)
    d.a = lambda x: x
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = d(20)
    assert result == {"a": 20}
    assert any("outside of domain" in str(w.message) for w in rec)


def test_domain_min_and_max_over_sets():
    d = Domain("temp", 0, 10)
    d.a = lambda x: 0.2
    d.b = lambda x: 0.8
    assert d.min(5) == 0.2
    assert d.max(5) == 0.8


def test_set_invert_and_basic_boolean_combinators():
    a = Set(constant(0.25))
    b = Set(constant(0.75))

    assert a(0) == 0.25
    assert (~a)(0) == pytest.approx(0.75)
    assert (-a)(0) == pytest.approx(0.75)
    assert (a & b)(0) == pytest.approx(0.25)
    assert (a | b)(0) == pytest.approx(0.75)


def test_set_product_sum_and_xor():
    a = Set(constant(0.2))
    b = Set(constant(0.3))
    assert (a * b)(0) == pytest.approx(0.06)
    assert (a + b)(0) == pytest.approx(0.5)
    assert (a ^ b)(0) == pytest.approx(0.5)


def test_set_power_multiply_and_call():
    a = Set(constant(0.5))
    assert (a ** 2)(0) == pytest.approx(0.25)
    assert a.multiplied(3)(0) == pytest.approx(1.5)
    assert a(123) == 0.5


def test_set_comparisons_and_equality_with_domains():
    d1 = Domain("d1", 0, 4, res=1)
    d2 = Domain("d2", 0, 4, res=1)
    d1.a = lambda x: 1 if x >= 2 else 0
    d2.a = lambda x: 1 if x >= 2 else 0
    d1.b = lambda x: 1 if x >= 3 else 0
    d2.b = lambda x: 1 if x >= 3 else 0

    assert d1.a == d2.a
    assert d1.b <= d2.a
    assert d2.a >= d1.b
    assert d1.b < d1.a
    assert d1.a > d1.b


def test_set_comparison_without_domain_raises():
    a = Set(constant(0.1))
    b = Set(constant(0.2))
    with pytest.raises(FuzzyWarning):
        _ = a == b
    with pytest.raises(FuzzyWarning):
        _ = a <= b
    with pytest.raises(FuzzyWarning):
        _ = a < b
    with pytest.raises(FuzzyWarning):
        _ = a >= b
    with pytest.raises(FuzzyWarning):
        _ = a > b


def test_set_len_cardinality_relative_cardinality_and_array():
    d = Domain("d", 0, 2, res=1)
    d.a = lambda x: x / 2
    assert len(d.a) == 3
    assert d.a.cardinality() == pytest.approx(1.5)
    assert d.a.relative_cardinality() == pytest.approx(0.5)
    assert list(d.a.array()) == pytest.approx([0.0, 0.5, 1.0])


def test_set_normalized_and_plot(monkeypatch):
    d = Domain("d", 0, 2, res=1)
    d.a = lambda x: x / 2

    normalized = d.a.normalized()
    assert normalized(0) == pytest.approx(0.0)
    assert normalized(1) == pytest.approx(0.5)
    assert normalized(2) == pytest.approx(1.0)

    calls = {}

    def fake_plot(r, v):
        calls["r"] = list(r)
        calls["v"] = list(v)

    monkeypatch.setattr("fuzzylogic.classes.plt.plot", fake_plot)
    d.a.plot()
    assert calls["r"] == [0, 1, 2]
    assert calls["v"] == pytest.approx([0.0, 0.5, 1.0])


def test_singleton_linear_and_bounded_linear():
    f = singleton(2)
    assert f(1) == 0
    assert f(2) == 1

    g = linear(1, -1)
    assert g(0) == 0
    assert g(1.5) == pytest.approx(0.5)
    assert g(3) == 1

    h = bounded_linear(2, 3)
    assert h(1) == pytest.approx(0.0)
    assert h(2.5) == pytest.approx(0.5)
    assert h(4) == pytest.approx(1.0)


def test_rectangular_triangular_and_trapezoid():
    r = rectangular(1, 3)
    assert r(0) == 0
    assert r(2) == 1
    assert r(4) == 0

    t = triangular(0, 4)
    assert t(0) == pytest.approx(0.0)
    assert t(2) == pytest.approx(1.0)
    assert t(4) == pytest.approx(0.0)

    z = trapezoid(0, 1, 3, 4)
    assert z(0) == pytest.approx(0.0)
    assert z(1.5) == pytest.approx(1.0)
    assert z(3.5) == pytest.approx(0.5)


def test_sigmoid_variants_and_inverse_helpers():
    f = sigmoid(1, 1, 0)
    assert f(0) == pytest.approx(0.5)

    b = bounded_sigmoid(0, 1)
    assert b(0) == pytest.approx(0.1, rel=1e-7, abs=1e-7)
    assert b(1) == pytest.approx(0.9, rel=1e-7, abs=1e-7)
    assert b(100000) == pytest.approx(1.0, abs=1e-12)
    assert b(-100000) == pytest.approx(0.0, abs=1e-12)

    inv_b = inv(b)
    assert inv_b(0) == pytest.approx(0.9, rel=1e-7, abs=1e-7)
    assert inv_b(1) == pytest.approx(0.1, rel=1e-7, abs=1e-7)


def test_bounded_exponential_simple_sigmoid_triangular_sigmoid_and_gauss():
    e = bounded_exponential(k=1, limit=2)
    assert e(0) == pytest.approx(0.0)
    assert e(1) == pytest.approx(2 - 2 / math.e)

    s = simple_sigmoid()
    assert s(0) == pytest.approx(0.5)
    assert s(1000) == pytest.approx(1.0, abs=1e-12)
    assert s(-1000) == pytest.approx(0.0, abs=1e-12)

    ts = triangular_sigmoid(2, 4)
    assert ts(2) == pytest.approx(0.1, rel=1e-7, abs=1e-7)
    assert ts(3) == pytest.approx(0.9, rel=1e-7, abs=1e-7)

    g = gauss(c=0, b=1, c_m=1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) == pytest.approx(math.exp(-1))
    assert g(100) == pytest.approx(0.0, abs=1e-12)
