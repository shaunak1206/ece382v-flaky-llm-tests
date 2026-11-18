import math
import warnings

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    constant,
    inv,
    normalize,
    linear,
    bounded_linear,
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


def test_domain_init_asserts_on_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("bad", 1, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 2, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_domain_set_assignment_and_attribute_access():
    d = Domain("temperature", 0, 10)
    d.hot = constant(1)
    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d(5) == {"hot": 1}


def test_domain_call_warns_when_value_outside_domain():
    d = Domain("temperature", 0, 10)
    d.hot = constant(1)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = d(11)
    assert result == {"hot": 1}
    assert any("outside of domain" in str(w.message) for w in rec)


def test_domain_range_includes_upper_bound():
    d = Domain("n", 0, 3, res=1)
    assert list(d.range()) == [0, 1, 2, 3]


def test_domain_min_and_max_use_membership_functions():
    d = Domain("x", 0, 4)
    d.low = rectangular(0, 1, c_m=0.2, no_m=0.1)
    d.high = rectangular(3, 4, c_m=0.9, no_m=0.1)
    assert d.min(0.5) == 0.1
    assert d.max(0.5) == 0.2
    assert d.min(3.5) == 0.1
    assert d.max(3.5) == 0.9


def test_domain_delete_set_and_regular_attr_raises_fuzzywarning():
    d = Domain("temperature", 0, 10)
    d.hot = constant(1)
    del d.hot
    assert "hot" not in d._sets
    with pytest.raises(FuzzyWarning):
        del d._name


def test_set_without_domain_string_and_call_and_repr():
    s = Set(constant(0.25))
    assert "dangling Set(" in str(s)
    assert s(123) == 0.25
    assert "Set(" in repr(s)


def test_set_invert_and_basic_combinators_work():
    a = Set(constant(0.2))
    b = Set(constant(0.6))

    assert (~a)(0) == pytest.approx(0.8)
    assert (-a)(0) == pytest.approx(0.8)
    assert (a & b)(0) == pytest.approx(0.2)
    assert (a | b)(0) == pytest.approx(0.6)
    assert (a * b)(0) == pytest.approx(0.12)
    assert (a + b)(0) == pytest.approx(0.8)
    assert (a ^ b)(0) == pytest.approx(0.8)


def test_set_pow_and_multiplied_apply_expected_transformations():
    s = Set(constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.multiplied(3)(0) == pytest.approx(1.5)


def test_set_len_cardinality_and_relative_cardinality():
    d = Domain("x", 0, 2, res=1)
    d.half = constant(0.5)
    assert len(d.half) == 3
    assert d.half.cardinality() == pytest.approx(1.5)
    assert d.half.relative_cardinality() == pytest.approx(0.5)


def test_set_comparisons_and_equality_require_domains():
    d = Domain("x", 0, 2)
    d.a = constant(0.2)
    d.b = constant(0.7)

    assert d.a == d.a
    assert d.a <= d.b
    assert d.a < d.b
    assert d.b >= d.a
    assert d.b > d.a

    loose = Set(constant(0.2))
    with pytest.raises(FuzzyWarning):
        _ = loose == d.a
    with pytest.raises(FuzzyWarning):
        _ = loose <= d.a


def test_set_normalized_returns_max_one_over_domain():
    d = Domain("x", 0, 2)
    d.s = constant(0.25)
    n = d.s.normalized()
    assert n.domain is None
    assert n(0) == pytest.approx(1.0)


def test_set_array_and_plot_on_assigned_domain(monkeypatch):
    d = Domain("x", 0, 2, res=1)
    d.s = linear(0.5, 0)
    arr = d.s.array()
    assert arr.tolist() == [0.0, 0.5, 1.0]

    called = {}

    def fake_plot(x, y):
        called["x"] = list(x)
        called["y"] = list(y)

    monkeypatch.setattr("fuzzylogic.classes.plt.plot", fake_plot)
    d.s.plot()
    assert called["x"] == [0, 1, 2]
    assert called["y"] == [0.0, 0.5, 1.0]


def test_functions_constant_inv_and_normalize():
    f = constant(3)
    assert f("anything") == 3
    g = inv(constant(0.25))
    assert g(0) == pytest.approx(0.75)
    h = normalize(0.5, constant(0.25))
    assert h(0) == pytest.approx(0.5)


def test_membership_functions_basic_shapes():
    assert linear(1, -1)(0) == 0
    assert linear(1, -1)(2) == 1

    r = rectangular(2, 4)
    assert r(1) == 0
    assert r(3) == 1
    assert r(5) == 0

    t = triangular(0, 4)
    assert t(0) == 0
    assert t(2) == 1
    assert t(4) == 0

    z = trapezoid(0, 1, 3, 4)
    assert z(0) == 0
    assert z(2) == 1
    assert z(4) == 0


def test_sigmoid_and_bounded_variants_behave_reasonably():
    s = sigmoid(1, 1, 0)
    assert 0 < s(0) < 1
    assert s(0) == pytest.approx(0.5)

    bs = bounded_sigmoid(0, 1)
    assert bs(0) == pytest.approx(0.1, abs=1e-12)
    assert bs(1) == pytest.approx(0.9, abs=1e-12)

    be = bounded_exponential(k=1, limit=2)
    assert be(0) == pytest.approx(0.0)
    assert be(1) > 0
    assert be(1000) == pytest.approx(2.0)

    ss = simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)

    ts = triangular_sigmoid(2, 4)
    assert ts(2) == pytest.approx(0.1, abs=1e-12)
    assert round(ts(3), 2) == 0.9

    g = gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) < 1.0


def test_set_intensified_has_buggy_branch_that_raises_typeerror():
    s = Set(constant(0.6))
    intensified = s.intensified()
    with pytest.raises(TypeError):
        intensified(1)
