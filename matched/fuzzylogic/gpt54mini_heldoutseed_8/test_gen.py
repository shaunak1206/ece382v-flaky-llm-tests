import math
import pickle

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    constant,
    inv,
    normalize,
    noop,
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
    alpha,
    moderate,
)


def test_domain_init_and_range_includes_high():
    d = Domain("temperature", 0, 2, res=1)
    assert str(d) == "temperature"
    assert repr(d).startswith("Domain('temperature', 0, 2, res=1")
    assert list(d.range()) == [0, 1, 2]


def test_domain_set_assignment_access_and_call():
    d = Domain("temp", 0, 10)
    d.hot = constant(1)
    d.cold = constant(0)
    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d(3) == {"hot": 1, "cold": 0}


def test_domain_set_assignment_converts_callable_to_set():
    d = Domain("temp", 0, 10)
    d.mid = lambda x: x / 10
    assert isinstance(d.mid, Set)
    assert d.mid(5) == 0.5
    assert d.mid.name == "mid"
    assert d.mid.domain is d


def test_domain_setattr_requires_identifier():
    d = Domain("temp", 0, 10)
    with pytest.raises(AssertionError):
        setattr(d, "not an identifier", constant(1))


def test_domain_getattr_missing_raises_attributeerror():
    d = Domain("temp", 0, 10)
    with pytest.raises(AttributeError):
        _ = d.missing


def test_domain_delete_set_and_regular_attr_protected():
    d = Domain("temp", 0, 10)
    d.hot = constant(1)
    del d.hot
    with pytest.raises(AttributeError):
        _ = d.hot
    with pytest.raises(FuzzyWarning):
        del d._name


def test_set_call_invert_and_basic_boolean_ops():
    s1 = Set(lambda x: 0.2)
    s2 = Set(lambda x: 0.8)
    assert s1(5) == 0.2
    assert (~s1)(5) == 0.8
    assert (-s1)(5) == 0.8
    assert (s1 & s2)(5) == 0.2
    assert (s1 | s2)(5) == 0.8
    assert (s1 * s2)(5) == pytest.approx(0.16)
    assert (s1 + s2)(5) == 1
    assert (s1 ^ s2)(5) == pytest.approx(0.8)


def test_set_power_and_constant_multiplication():
    s = Set(lambda x: 0.5)
    assert (s ** 2)(0) == 0.25
    assert s.multiplied(3)(0) == 1.5


def test_set_len_cardinality_and_relative_cardinality():
    d = Domain("temp", 0, 2, res=1)
    d.a = constant(0)
    d.b = constant(1)
    assert len(d.b) == 3
    assert d.b.cardinality() == 3
    assert d.b.relative_cardinality() == 1
    assert d.a.relative_cardinality() == 0


def test_set_equality_and_ordering_with_domain():
    d = Domain("temp", 0, 2, res=1)
    d.a = constant(1)
    d.b = constant(1)
    d.c = constant(0)
    assert d.a == d.b
    assert d.c <= d.a
    assert d.c < d.a
    assert d.a >= d.c
    assert d.a > d.c


def test_set_equality_without_domain_raises():
    s1 = Set(lambda x: 1)
    s2 = Set(lambda x: 1)
    with pytest.raises(FuzzyWarning):
        _ = s1 == s2
    with pytest.raises(FuzzyWarning):
        _ = s1 <= s2


def test_set_array_and_normalized():
    d = Domain("temp", 0, 2, res=1)
    d.s = lambda x: x
    arr = d.s.array()
    assert arr.tolist() == [0.0, 1.0, 2.0]
    n = d.s.normalized()
    assert n(0) == 0.0
    assert n(1) == 0.5
    assert n(2) == 1.0


def test_set_str_repr_and_dangling_set():
    s = Set(lambda x: 1)
    assert "dangling Set(" in str(s)
    assert repr(s).startswith("Set(")
    d = Domain("temp", 0, 1)
    d.foo = s
    assert str(d.foo) == "temp.foo"


def test_set_pickle_roundtrip_via_string_constructor():
    func = constant(0.42)
    data = pickle.dumps(func)
    s = Set(pickle.dumps(func).__str__() if False else data.decode("latin1"))
    assert callable(s.func)


def test_basic_membership_functions():
    assert constant(3)(10) == 3
    assert noop()(7) == 7
    assert singleton(2)(2) == 1
    assert singleton(2)(1) == 0
    assert linear(1, -1)(1.5) == 0.5
    assert bounded_linear(2, 3)(2.5) == 0.5
    assert R(2, 4)(3) == 0.5
    assert S(2, 4)(3) == 0.5


def test_shape_functions_and_clipping_helpers():
    rect = rectangular(1, 3)
    tri = triangular(0, 4)
    trap = trapezoid(0, 1, 3, 4)
    assert rect(0) == 0
    assert rect(2) == 1
    assert tri(2) == 1
    assert tri(0) == 0
    assert trap(0) == 0
    assert trap(2) == 1
    clipped = alpha(floor=0.2, ceiling=0.8, func=constant(0.5))
    assert clipped(0) == 0.5
    assert moderate(constant(0.75))(0) == pytest.approx(0.78125)


def test_sigmoid_variants_and_gauss():
    s = sigmoid(1, 1, 0)
    bs = bounded_sigmoid(0, 1)
    simp = simple_sigmoid()
    ts = triangular_sigmoid(2, 4)
    g = gauss(0, 1)
    be = bounded_exponential(0.1, limit=2)
    assert 0 < s(0) < 1
    assert bs(0) == pytest.approx(0.1)
    assert round(bs(1), 2) == 0.9
    assert simp(0) == pytest.approx(0.5)
    assert ts(2) == pytest.approx(0.1)
    assert round(ts(3), 2) == 0.9
    assert g(0) == pytest.approx(1.0)
    assert be(0) == pytest.approx(0.0)
    assert be(1000) == pytest.approx(2.0)
