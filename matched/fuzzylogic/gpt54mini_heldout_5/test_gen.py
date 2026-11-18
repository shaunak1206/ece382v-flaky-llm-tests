import math
import pickle

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    constant,
    inv,
    normalize,
    noop,
    alpha,
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


def test_domain_init_valid_and_invalid_bounds():
    d = Domain("temperature", 0, 100)
    assert str(d) == "temperature"
    assert d._low == 0
    assert d._high == 100
    assert d._res == 1

    with pytest.raises(AssertionError):
        Domain("bad", 10, 10)
    with pytest.raises(AssertionError):
        Domain("bad", 10, 5)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_domain_set_assignment_and_attribute_access():
    d = Domain("temperature", 0, 10)
    d.hot = constant(1)
    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d(3) == {"hot": 1}


def test_domain_assignment_of_non_set_wraps_callable():
    d = Domain("temperature", 0, 10)
    d.cold = lambda x: 0 if x > 5 else 1
    assert isinstance(d.cold, Set)
    assert d.cold(2) == 1
    assert d.cold(8) == 0


def test_domain_rejects_invalid_attribute_name_and_deletes_sets():
    d = Domain("temperature", 0, 10)
    with pytest.raises(AssertionError):
        setattr(d, "not valid", constant(1))

    d.hot = constant(1)
    del d.hot
    assert "hot" not in d._sets
    with pytest.raises(FuzzyWarning):
        del d._low


def test_domain_call_warns_outside_range_and_evaluates_all_sets():
    d = Domain("temperature", 0, 10)
    d.low = constant(0.25)
    d.high = constant(0.75)

    with pytest.warns(UserWarning):
        out = d(20)

    assert out == {"low": 0.25, "high": 0.75}


def test_domain_range_min_max_and_repr_eq():
    d1 = Domain("temperature", 0, 2, res=1)
    d1.a = constant(0)
    d1.b = constant(1)

    assert list(d1.range()) == [0, 1, 2]
    assert d1.min(0) == 0
    assert d1.max(0) == 1

    d2 = Domain("temperature", 0, 2, res=1)
    d2.a = constant(0)
    d2.b = constant(1)

    assert d1 == d2
    assert "Domain('temperature', 0, 2, res=1" in repr(d1)


def test_set_call_and_str_repr_and_dangling_str():
    s = Set(constant(0.5))
    assert s(123) == 0.5
    assert "dangling Set(" in str(s)
    assert "Set(" in repr(s)


def test_set_invert_and_negate():
    s = Set(constant(0.2))
    not_s = ~s
    neg_s = -s
    assert not_s(0) == pytest.approx(0.8)
    assert neg_s(0) == pytest.approx(0.8)


def test_set_logical_and_arithmetic_operators():
    a = Set(constant(0.2))
    b = Set(constant(0.7))

    assert (a & b)(0) == pytest.approx(0.2)
    assert (a | b)(0) == pytest.approx(0.7)
    assert (a * b)(0) == pytest.approx(0.14)
    assert (a + b)(0) == pytest.approx(0.9)
    assert (a ^ b)(0) == pytest.approx(0.7)  # simple disjoint sum on constants
    assert (a ** 2)(0) == pytest.approx(0.04)


def test_set_equality_and_order_comparisons_on_domain():
    d = Domain("x", 0, 2, res=1)
    d.a = constant(0.2)
    d.b = constant(0.2)
    d.c = constant(0.8)

    assert d.a == d.b
    assert d.a <= d.b
    assert not (d.a < d.b)
    assert d.c >= d.a
    assert d.c > d.a
    assert not (d.a >= d.c)

    e = Set(constant(0.2))
    with pytest.raises(FuzzyWarning):
        _ = d.a == e
    with pytest.raises(FuzzyWarning):
        _ = d.a <= e


def test_set_len_cardinality_and_relative_cardinality():
    d = Domain("x", 0, 2, res=1)
    d.half = constant(0.5)

    assert len(d.half) == 3
    assert d.half.cardinality() == pytest.approx(1.5)
    assert d.half.relative_cardinality() == pytest.approx(0.5)


def test_set_array_and_normalized():
    d = Domain("x", 0, 2, res=1)
    d.v = lambda x: x / 2
    arr = d.v.array()
    assert arr.tolist() == [0.0, 0.5, 1.0]

    norm = d.v.normalized()
    assert norm(0) == pytest.approx(0.0)
    assert norm(1) == pytest.approx(0.5)
    assert norm(2) == pytest.approx(1.0)


def test_set_plot_without_domain_raises_and_with_domain_calls_matplotlib(monkeypatch):
    s = Set(constant(1))
    with pytest.raises(FuzzyWarning):
        s.plot()

    calls = {}

    def fake_plot(x, y):
        calls["x"] = list(x)
        calls["y"] = list(y)

    monkeypatch.setattr("fuzzylogic.classes.plt.plot", fake_plot)

    d = Domain("x", 0, 2, res=1)
    d.s = constant(1)
    d.s.plot()
    assert calls["x"] == [0, 1, 2]
    assert calls["y"] == [1, 1, 1]


def test_set_pickle_string_constructor_and_name_domain_init():
    func = constant(0.3)
    dumped = pickle.dumps(func)

    s = Set(dumped)
    assert s(10) == pytest.approx(0.3)

    d = Domain("x", 0, 1)
    s2 = Set(constant(1), name="yes", domain=d)
    assert d.yes is s2
    assert s2.domain is d
    assert s2.name == "yes"


def test_functions_basic_helpers_constant_inv_noop_normalize_alpha():
    f = constant(4)
    assert f(0) == 4

    g = inv(constant(0.25))
    assert g(0) == pytest.approx(0.75)

    h = noop()
    assert h(3.5) == 3.5

    n = normalize(2, constant(1))
    assert n(0) == pytest.approx(0.5)

    clipped = alpha(floor=0.2, ceiling=0.8, func=lambda x: x, floor_clip=-1, ceiling_clip=2)
    assert clipped(0.1) == -1
    assert clipped(0.5) == pytest.approx(0.5)
    assert clipped(0.9) == 2


def test_functions_shape_and_boundary_membership_functions():
    bl = bounded_linear(2, 4)
    assert bl(1) == pytest.approx(0.0)
    assert bl(3) == pytest.approx(0.5)
    assert bl(5) == pytest.approx(1.0)

    r = R(2, 4)
    s = S(2, 4)
    assert r(1) == 0
    assert r(3) == pytest.approx(0.5)
    assert r(5) == 1
    assert s(1) == 1
    assert s(3) == pytest.approx(0.5)
    assert s(5) == 0

    rect = rectangular(2, 4, c_m=0.8, no_m=0.1)
    assert rect(1) == 0.1
    assert rect(3) == 0.8
    assert rect(5) == 0.1

    tri = triangular(0, 10, c=5)
    assert tri(0) == pytest.approx(0.0)
    assert tri(5) == pytest.approx(1.0)
    assert tri(10) == pytest.approx(0.0)

    trap = trapezoid(0, 2, 4, 6)
    assert trap(0) == pytest.approx(0.0)
    assert trap(3) == pytest.approx(1.0)
    assert trap(6) == pytest.approx(0.0)


def test_functions_sigmoids_and_gauss_behave_reasonably():
    sig = sigmoid(1, 1, 0)
    assert sig(0) == pytest.approx(0.5)
    assert 0 < sig(-10) < 0.5
    assert 0.5 < sig(10) < 1

    bs = bounded_sigmoid(0, 1)
    assert bs(0) == pytest.approx(0.1, rel=1e-6)
    assert bs(1) == pytest.approx(0.9, rel=1e-6)

    be = bounded_exponential(k=1, limit=2)
    assert be(0) == pytest.approx(0.0)
    assert be(10) == pytest.approx(2 - 2 / math.exp(10))

    ss = simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(-1000) == pytest.approx(0.0, abs=1e-3)
    assert ss(1000) == pytest.approx(1.0, abs=1e-3)

    ts = triangular_sigmoid(2, 4)
    assert ts(2) == pytest.approx(0.1, rel=1e-6)
    assert ts(3) == pytest.approx(0.9, rel=1e-6)

    g = gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) < g(0)
