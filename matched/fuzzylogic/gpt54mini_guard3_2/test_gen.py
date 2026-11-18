import math
import pickle
import warnings

import numpy as np
import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic import functions as fn


def test_domain_init_rejects_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("t", 1, 1)
    with pytest.raises(AssertionError):
        Domain("t", 0, 1, res=0)
    with pytest.raises(AssertionError):
        Domain("t", 0, 1, res=-1)


def test_domain_set_assignment_access_and_call():
    d = Domain("temp", 0, 10, res=1)
    d.hot = fn.R(5, 10)
    d.cold = fn.S(2, 6)

    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d

    m = d(5)
    assert set(m) == {"hot", "cold"}
    assert m["hot"] == pytest.approx(0.0)
    assert m["cold"] == pytest.approx(0.0)


def test_domain_call_warns_outside_range_but_still_evaluates(monkeypatch):
    d = Domain("temp", 0, 10)
    d.hot = fn.constant(0.25)

    with pytest.warns(UserWarning):
        result = d(11)

    assert result == {"hot": 0.25}


def test_domain_range_includes_high_bound():
    d = Domain("x", 0, 2, res=0.5)
    values = list(d.range())
    assert values == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])


def test_domain_min_and_max_over_sets():
    d = Domain("x", 0, 10)
    d.a = fn.constant(0.2)
    d.b = fn.constant(0.7)
    d.c = fn.constant(0.5)

    assert d.min(3) == pytest.approx(0.2)
    assert d.max(3) == pytest.approx(0.7)


def test_domain_delete_set_and_delete_regular_attribute_raises():
    d = Domain("x", 0, 1)
    d.a = fn.constant(1)

    del d.a
    assert "a" not in d._sets

    with pytest.raises(FuzzyWarning):
        del d._name


def test_domain_equality_depends_on_all_fields():
    d1 = Domain("x", 0, 1, res=0.5)
    d2 = Domain("x", 0, 1, res=0.5)
    assert d1 == d2

    d1.a = fn.constant(0)
    assert d1 != d2

    d2.a = fn.constant(0)
    assert d1 == d2


def test_set_equality_and_ordering_on_same_domain():
    d = Domain("x", 0, 2, res=1)
    s1 = Set(fn.constant(0))
    s2 = Set(fn.constant(0))
    d.a = s1
    d.b = s2

    assert s1 == s2
    assert s1 <= s2
    assert not (s1 < s2)
    assert s1 >= s2
    assert not (s1 > s2)


def test_set_comparison_requires_domain():
    s1 = Set(fn.constant(0))
    s2 = Set(fn.constant(0))
    with pytest.raises(FuzzyWarning):
        _ = s1 == s2
    with pytest.raises(FuzzyWarning):
        _ = s1 <= s2
    with pytest.raises(FuzzyWarning):
        _ = s1 < s2
    with pytest.raises(FuzzyWarning):
        _ = s1 >= s2
    with pytest.raises(FuzzyWarning):
        _ = s1 > s2


def test_set_boolean_and_arithmetic_combinators():
    a = Set(lambda x: 0.2 if x == 0 else 0.8)
    b = Set(lambda x: 0.6 if x == 0 else 0.1)

    assert (a & b)(0) == pytest.approx(min(0.2, 0.6))
    assert (a | b)(1) == pytest.approx(max(0.8, 0.1))
    assert (a * b)(0) == pytest.approx(0.12)
    assert (a + b)(0) == pytest.approx(min(1.0, 0.8))
    assert (a ^ b)(0) == pytest.approx(abs(0.2 - 0.6))


def test_set_invert_neg_power_and_mul():
    s = Set(fn.constant(0.25))

    assert (~s)(0) == pytest.approx(0.75)
    assert (-s)(0) == pytest.approx(0.75)
    assert (s ** 2)(0) == pytest.approx(0.0625)
    assert s.multiplied(3)(0) == pytest.approx(0.75)


def test_set_len_cardinality_relative_cardinality_and_array():
    d = Domain("x", 0, 2, res=1)
    d.s = fn.constant(0.5)

    assert len(d.s) == 3
    assert d.s.cardinality() == pytest.approx(1.5)
    assert d.s.relative_cardinality() == pytest.approx(0.5)
    assert np.allclose(d.s.array(), np.array([0.5, 0.5, 0.5]))


def test_set_concentrated_and_normalized():
    d = Domain("x", 0, 2, res=1)
    d.s = lambda x: 0.5 if x == 0 else 1.0

    concentrated = d.s.concentrated()
    normalized = d.s.normalized()

    assert concentrated(0) == pytest.approx(0.25)
    assert concentrated(1) == pytest.approx(1.0)
    assert normalized(0) == pytest.approx(0.5)
    assert normalized(1) == pytest.approx(1.0)


def test_set_str_and_repr_for_dangling_and_attached_sets():
    s = Set(fn.constant(0.1))
    assert "dangling Set(" in str(s)
    assert repr(s).startswith("Set(")

    d = Domain("temp", 0, 1)
    d.hot = fn.constant(1)
    assert str(d.hot) == "temp.hot"


def test_set_pickle_roundtrip_via_string_constructor():
    original = fn.constant(0.33)
    dumped = pickle.dumps(original)
    restored = Set(dumped).func
    assert restored(123) == pytest.approx(0.33)


def test_functions_basic_closures_and_clipping():
    assert fn.noop()(3.5) == 3.5
    assert fn.constant(7)(-1) == 7
    clipped = fn.alpha(floor=0.2, ceiling=0.8, func=fn.constant(0.1))
    assert clipped(0) == pytest.approx(0.2)
    clipped = fn.alpha(floor=0.2, ceiling=0.8, func=fn.constant(0.9))
    assert clipped(0) == pytest.approx(0.8)


def test_functions_linear_and_bounded_linear():
    lin = fn.linear(1, -1)
    assert lin(0) == pytest.approx(0.0)
    assert lin(1.5) == pytest.approx(0.5)
    assert lin(3) == pytest.approx(1.0)

    b = fn.bounded_linear(2, 3)
    assert b(1) == pytest.approx(0.0)
    assert b(2.5) == pytest.approx(0.5)
    assert b(4) == pytest.approx(1.0)


def test_functions_rectangular_triangular_and_trapezoid():
    rect = fn.rectangular(2, 4)
    tri = fn.triangular(2, 4)
    trap = fn.trapezoid(2, 3, 5, 6)

    assert rect(1) == 0
    assert rect(3) == 1
    assert tri(2) == pytest.approx(0.0)
    assert tri(3) == pytest.approx(1.0)
    assert tri(4) == pytest.approx(0.0)
    assert trap(2.5) == pytest.approx(0.5)
    assert trap(4) == pytest.approx(1.0)
    assert trap(5.5) == pytest.approx(0.5)


def test_functions_sigmoids_and_gauss_properties():
    s = fn.simple_sigmoid()
    assert s(0) == pytest.approx(0.5, abs=1e-12)
    assert s(-1000) == pytest.approx(0.0, abs=1e-12)
    assert s(1000) == pytest.approx(1.0, abs=1e-12)

    b = fn.bounded_sigmoid(0, 1)
    assert b(0) == pytest.approx(0.1, rel=1e-12, abs=1e-12)
    assert b(1) == pytest.approx(0.9, rel=1e-12, abs=1e-12)

    g = fn.gauss(0, 1)
    assert g(0) == pytest.approx(1.0, abs=1e-12)
    assert g(1) < g(0)
    assert g(10) == pytest.approx(math.exp(-100), rel=1e-12, abs=1e-12)


def test_plot_calls_matplotlib_plot(monkeypatch):
    d = Domain("x", 0, 2, res=1)
    d.s = fn.constant(0.5)

    calls = []

    def fake_plot(x, y):
        calls.append((list(x), list(y)))

    monkeypatch.setattr("fuzzylogic.classes.plt.plot", fake_plot)
    d.s.plot()

    assert len(calls) == 1
    xs, ys = calls[0]
    assert xs == [0, 1, 2]
    assert ys == [0.5, 0.5, 0.5]
