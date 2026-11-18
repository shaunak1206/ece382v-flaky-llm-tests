import math

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    alpha,
    bounded_linear,
    bounded_sigmoid,
    constant,
    gauss,
    inv,
    linear,
    normalize,
    rectangular,
    simple_sigmoid,
    triangular,
)


def test_domain_init_asserts_on_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("bad", 1, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 2, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_domain_set_assignment_and_access_by_attribute():
    d = Domain("temp", 0, 10)
    d.cold = constant(0.25)

    assert isinstance(d.cold, Set)
    assert d.cold.name == "cold"
    assert d.cold.domain is d
    assert d(3) == {"cold": 0.25}


def test_domain_set_assignment_wraps_plain_callable_into_set():
    d = Domain("temp", 0, 10)

    d.hot = lambda x: 1 if x >= 5 else 0

    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d(4) == {"hot": 0}
    assert d(5) == {"hot": 1}


def test_domain_range_includes_high_bound():
    d = Domain("temp", 0, 1, res=0.5)

    values = list(d.range())

    assert values == [0.0, 0.5, 1.0]


def test_domain_delattr_removes_set_and_preserves_regular_attrs():
    d = Domain("temp", 0, 10)
    d.cold = constant(0)
    assert "cold" in d._sets

    del d.cold

    assert "cold" not in d._sets
    with pytest.raises(AttributeError):
        _ = d.cold

    with pytest.raises(FuzzyWarning):
        del d._name


def test_domain_getattr_raises_for_unknown_name():
    d = Domain("temp", 0, 10)

    with pytest.raises(AttributeError):
        _ = d.not_a_set


def test_domain_equality_compares_structure_and_sets():
    d1 = Domain("temp", 0, 10)
    d2 = Domain("temp", 0, 10)
    d1.cold = constant(0)
    d2.cold = constant(0)

    assert d1 == d2

    d2.hot = constant(1)
    assert d1 != d2


def test_set_basic_boolean_operators_on_domain():
    d = Domain("temp", 0, 10, res=1)
    d.low = lambda x: 1 if x <= 3 else 0
    d.high = lambda x: 1 if x >= 7 else 0

    s_and = d.low & d.high
    s_or = d.low | d.high
    s_not = ~d.low
    s_xor = d.low ^ d.high

    assert s_and(0) == 0
    assert s_and(8) == 0
    assert s_or(0) == 1
    assert s_or(5) == 0
    assert s_not(0) == 0
    assert s_not(5) == 1
    assert s_xor(0) == 1
    assert s_xor(5) == 0
    assert s_xor(8) == 1


def test_set_arithmetic_like_ops_and_power():
    d = Domain("temp", 0, 10)
    d.a = constant(0.2)
    d.b = constant(0.3)

    multiplied = d.a * d.b
    summed = d.a + d.b
    powered = d.a ** 2

    assert pytest.approx(multiplied(0), rel=1e-9) == 0.06
    assert pytest.approx(summed(0), rel=1e-9) == 0.5
    assert pytest.approx(powered(0), rel=1e-9) == 0.04


def test_set_equality_and_subset_comparisons_use_domain_arrays():
    d = Domain("temp", 0, 2, res=1)
    d.a = constant(0)
    d.b = constant(0)
    d.c = constant(1)

    assert d.a == d.b
    assert d.a <= d.b
    assert d.a < d.c
    assert d.c >= d.a
    assert d.c > d.a


def test_set_cardinality_and_relative_cardinality():
    d = Domain("temp", 0, 2, res=1)
    d.a = constant(1)

    assert len(d.a) == 3
    assert d.a.cardinality() == 3
    assert d.a.relative_cardinality() == 1


def test_set_normalized_scales_maximum_to_one():
    d = Domain("temp", 0, 2, res=1)
    d.a = lambda x: 0.25 if x == 0 else (0.5 if x == 1 else 1.0)

    n = d.a.normalized()

    assert n(0) == pytest.approx(0.25)
    assert n(1) == pytest.approx(0.5)
    assert n(2) == pytest.approx(1.0)


def test_functions_constant_inv_alpha_and_normalize():
    f = constant(0.2)
    inv_f = inv(f)
    clipped = alpha(floor=0.1, ceiling=0.3, func=lambda x: x)
    norm = normalize(2, lambda x: x)

    assert f(99) == 0.2
    assert inv_f(99) == pytest.approx(0.8)
    assert clipped(0.05) == 0.1
    assert clipped(0.2) == 0.2
    assert clipped(0.5) == 0.3
    assert norm(1) == pytest.approx(0.5)


def test_functions_linear_and_bounded_linear_behave_as_expected():
    f = linear(1, -1)
    g = bounded_linear(2, 3)

    assert f(-2) == 0
    assert f(1) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(3) == 1

    assert g(1) == 0.0
    assert g(2) == 0.0
    assert g(2.5) == pytest.approx(0.5)
    assert g(3) == 1.0
    assert g(4) == 1.0


def test_functions_rectangular_and_triangular():
    rect = rectangular(2, 4)
    tri = triangular(2, 4)

    assert rect(1) == 0
    assert rect(3) == 1
    assert rect(5) == 0

    assert tri(2) == 0
    assert tri(3) == 1
    assert tri(4) == 0


def test_functions_sigmoids_and_gauss():
    bsig = bounded_sigmoid(0, 1)
    ssig = simple_sigmoid()
    g = gauss(0, 1)

    assert bsig(0) == pytest.approx(0.1, rel=1e-6)
    assert round(bsig(1), 2) == 0.9
    assert ssig(0) == pytest.approx(0.5, rel=1e-12)
    assert g(0) == pytest.approx(1.0, rel=1e-12)
    assert g(10) < 1e-20


def test_set_dangling_string_and_plot_requires_domain(monkeypatch):
    s = Set(constant(1))
    assert "dangling Set" in str(s)

    with pytest.raises(FuzzyWarning):
        s.plot()

    called = {}

    def fake_plot(x, y):
        called["x"] = list(x)
        called["y"] = list(y)

    monkeypatch.setattr("fuzzylogic.classes.plt.plot", fake_plot)

    d = Domain("temp", 0, 1, res=1)
    d.a = constant(1)

    d.a.plot()

    assert called["x"] == [0, 1]
    assert called["y"] == [1, 1]
