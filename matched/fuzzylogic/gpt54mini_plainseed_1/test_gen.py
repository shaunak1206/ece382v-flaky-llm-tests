import math
import pickle

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv,
    constant,
    normalize,
    noop,
    alpha,
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


def test_domain_init_validation():
    with pytest.raises(AssertionError):
        Domain("bad", 1, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 2, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_domain_set_assignment_and_call():
    d = Domain("temperature", 0, 10)
    d.hot = constant(1)
    d.cold = constant(0)

    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d(5) == {"hot": 1, "cold": 0}


def test_domain_getattr_and_delete_set():
    d = Domain("temperature", 0, 10)
    d.hot = constant(1)

    assert d.hot(3) == 1
    del d.hot
    with pytest.raises(AttributeError):
        _ = d.hot


def test_domain_deleting_regular_attribute_raises_fuzzywarning():
    d = Domain("temperature", 0, 10)
    with pytest.raises(FuzzyWarning):
        del d._name


def test_domain_range_includes_high_bound():
    d = Domain("temperature", 0, 1, res=0.5)
    assert list(d.range()) == [0.0, 0.5, 1.0]


def test_domain_min_and_max():
    d = Domain("temperature", 0, 2)
    d.a = linear(0.5, 0)
    d.b = constant(0.25)

    assert d.min(0) == 0
    assert d.max(2) == 1


def test_set_call_invert_and_boolean_operators():
    a = Set(constant(0.25))
    b = Set(constant(0.75))

    assert (~a)(0) == pytest.approx(0.75)
    assert (-a)(0) == pytest.approx(0.75)
    assert (a & b)(0) == pytest.approx(0.25)
    assert (a | b)(0) == pytest.approx(0.75)
    assert (a * b)(0) == pytest.approx(0.1875)
    assert (a + b)(0) == pytest.approx(1.0)
    assert (a ^ b)(0) == pytest.approx(1.0 - 0.25 - 0.75)


def test_set_power_and_multiplied():
    s = Set(constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.multiplied(3)(0) == pytest.approx(1.5)


def test_set_equality_and_comparisons_with_domains():
    d = Domain("x", 0, 2)
    d.a = linear(0.5, 0)
    d.b = linear(0.5, 0)
    d.c = constant(0)

    assert d.a == d.b
    assert d.c <= d.a
    assert d.c < d.a
    assert d.a >= d.b
    assert not (d.a > d.b)


def test_set_equality_without_domain_raises():
    a = Set(constant(1))
    b = Set(constant(1))
    with pytest.raises(FuzzyWarning):
        _ = a == b
    with pytest.raises(FuzzyWarning):
        _ = a <= b


def test_set_len_cardinality_relative_cardinality():
    d = Domain("x", 0, 2, res=1)
    d.a = constant(0.5)

    assert len(d.a) == 3
    assert d.a.cardinality() == pytest.approx(1.5)
    assert d.a.relative_cardinality() == pytest.approx(0.5)


def test_set_array_and_normalized():
    d = Domain("x", 0, 2, res=1)
    d.a = linear(0.5, 0)  # values: 0, 0.5, 1.0

    arr = d.a.array()
    assert arr.tolist() == [0.0, 0.5, 1.0]

    n = d.a.normalized()
    assert n.domain is None
    assert n(0) == pytest.approx(0.0)
    assert n(2) == pytest.approx(1.0)


def test_set_without_domain_raises_on_array_len_plot_and_normalize():
    s = Set(constant(1))
    with pytest.raises(FuzzyWarning):
        _ = len(s)
    with pytest.raises(FuzzyWarning):
        _ = s.array()
    with pytest.raises(FuzzyWarning):
        _ = s.normalized()
    with pytest.raises(FuzzyWarning):
        s.plot()


def test_set_str_repr_and_dangling_str():
    s = Set(constant(1))
    assert "dangling Set(" in str(s)
    r = repr(s)
    assert r.startswith("Set(")

    d = Domain("temp", 0, 1)
    d.hot = constant(1)
    assert str(d.hot) == "temp.hot"


def test_set_construct_from_pickled_string():
    original = constant(0.33)
    dumped = pickle.dumps(original)
    s = Set(dumped)
    assert s(123) == pytest.approx(0.33)


def test_functions_basic_behavior():
    assert inv(constant(0.2))(0) == pytest.approx(0.8)
    assert noop()(42) == 42
    assert constant(7)(999) == 7
    assert normalize(0.5, constant(0.25))(0) == pytest.approx(0.5)
    assert alpha(floor=0.2, ceiling=0.8, func=constant(0.9))(0) == pytest.approx(0.8)
    assert singleton(2)(2) == 1
    assert singleton(2)(3) == 0


def test_membership_function_shapes():
    f = linear(1, -1)
    assert f(-1) == 0
    assert f(1) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(3) == 1

    g = bounded_linear(2, 3)
    assert g(1) == pytest.approx(0.0)
    assert g(2.5) == pytest.approx(0.5)
    assert g(4) == pytest.approx(1.0)

    h = rectangular(1, 2)
    assert h(0.5) == 0
    assert h(1.5) == 1
    assert h(3) == 0

    t = triangular(0, 2)
    assert t(0) == 0
    assert t(1) == 1
    assert t(2) == 0

    z = trapezoid(0, 1, 2, 3)
    assert z(0) == 0
    assert z(1.5) == 1
    assert z(3) == 0


def test_sigmoid_variants_and_gauss():
    f = sigmoid(1, 1, 0)
    assert f(0) == pytest.approx(0.5)

    b = bounded_sigmoid(0, 1)
    assert b(0) == pytest.approx(0.1, rel=1e-6)
    assert round(b(1), 1) == 0.9

    e = bounded_exponential(0.1, limit=2)
    assert e(0) == pytest.approx(0.0)
    assert e(100) == pytest.approx(2.0, rel=1e-6)

    s = simple_sigmoid()
    assert s(0) == pytest.approx(0.5)

    ts = triangular_sigmoid(2, 4)
    assert ts(2) == pytest.approx(0.1, rel=1e-6)
    assert round(ts(3), 1) == 0.9

    g = gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) < 1.0
