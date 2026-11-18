import math
import pickle

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv,
    normalize,
    constant,
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


def test_domain_init_and_range_includes_high():
    d = Domain("temp", 0, 2, res=1)
    assert str(d) == "temp"
    assert repr(d).startswith("Domain('temp'")
    assert list(d.range()) == [0, 1, 2]


def test_domain_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("bad", 2, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_domain_set_assignment_and_attribute_access():
    d = Domain("temp", 0, 10)
    d.hot = lambda x: 1 if x >= 7 else 0
    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d(8) == {"hot": 1}


def test_domain_delete_set_and_delete_regular_attr_warns():
    d = Domain("temp", 0, 10)
    d.hot = lambda x: 1
    del d.hot
    assert "hot" not in d._sets
    with pytest.raises(FuzzyWarning):
        del d._low


def test_domain_call_warns_outside_domain_and_evaluates_sets():
    d = Domain("temp", 0, 10)
    d.hot = lambda x: x
    with pytest.warns(UserWarning):
        memberships = d(11)
    assert memberships == {"hot": 11}


def test_domain_min_max_and_equality():
    d1 = Domain("temp", 0, 10)
    d2 = Domain("temp", 0, 10)
    d1.low = lambda x: 0
    d1.high = lambda x: 1
    d2.low = lambda x: 0
    d2.high = lambda x: 1
    assert d1 == d2
    assert d1.min(5) == 0
    assert d1.max(5) == 1


def test_set_call_and_basic_logical_operators():
    a = Set(lambda x: 0.2)
    b = Set(lambda x: 0.7)
    assert a(3) == 0.2
    assert (~a)(0) == 0.8
    assert (a & b)(0) == 0.2
    assert (a | b)(0) == 0.7
    assert (a * b)(0) == 0.14
    assert (a + b)(0) == 0.9
    assert (a ^ b)(0) == 0.5


def test_set_pow_and_multiplication_with_constants():
    a = Set(lambda x: 0.5)
    assert (a ** 2)(0) == 0.25
    assert a.multiplied(3)(0) == 1.5


def test_set_equality_and_comparisons_with_domain():
    d = Domain("d", 0, 2, res=1)
    d.a = lambda x: 0 if x < 1 else 1
    d.b = lambda x: 0 if x < 1 else 1
    d.c = lambda x: 0
    assert d.a == d.b
    assert d.c <= d.a
    assert d.c < d.a
    assert d.a >= d.c
    assert d.a > d.c


def test_set_comparison_without_domain_raises():
    a = Set(lambda x: 0)
    b = Set(lambda x: 1)
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
    d.a = lambda x: x
    assert len(d.a) == 3
    assert d.a.array().tolist() == [0.0, 1.0, 2.0]
    assert d.a.cardinality() == 3.0
    assert d.a.relative_cardinality() == 1.0


def test_set_normalized_and_concentrated():
    d = Domain("d", 0, 2, res=1)
    d.a = lambda x: x
    normalized = d.a.normalized()
    assert normalized(2) == 1
    assert normalized(1) == 0.5
    concentrated = d.a.concentrated()
    assert concentrated(2) == 4
    assert concentrated(1) == 1


def test_set_dilated_and_intensified_lower_branch():
    d = Domain("d", 0, 2, res=1)
    d.a = lambda x: 0.25
    assert pytest.approx(d.a.dilated()(0), rel=1e-12) == 0.5
    assert pytest.approx(d.a.intensified()(0), rel=1e-12) == 2 * (0.25 ** 2)


def test_set_repr_str_and_plot_no_domain_raises():
    s = Set(lambda x: 0.1)
    assert "dangling Set(" in str(s)
    assert repr(s).startswith("Set(")
    with pytest.raises(FuzzyWarning):
        s.plot()


def test_set_init_with_name_and_domain_assigns_to_domain():
    d = Domain("d", 0, 1)
    s = Set(lambda x: 1, name="always", domain=d)
    assert d.always is s
    assert s.name == "always"
    assert s.domain is d


def test_set_pickle_string_loads_or_leaves_callable():
    func = constant(3)
    data = pickle.dumps(func)
    s = Set(pickle.dumps(func), name=None, domain=None)
    assert callable(s.func)
    assert s(10) == 3


def test_functions_inv_constant_alpha_normalize():
    f = constant(2)
    inv_f = inv(f)
    assert inv_f(0) == -1
    clipped = alpha(floor=0.25, ceiling=0.75, func=lambda x: x)
    assert clipped(0.1) == 0.25
    assert clipped(0.9) == 0.75
    n = normalize(2, lambda x: x)
    assert n(1) == 0.5


def test_membership_functions_basic_shapes():
    assert singleton(2)(2) == 1
    assert singleton(2)(3) == 0
    assert linear(1, -1)(1.5) == 0.5
    assert bounded_linear(2, 3)(2.5) == 0.5
    assert R(2, 3)(2.5) == pytest.approx(0.5)
    assert S(2, 3)(2.5) == pytest.approx(0.5)
    assert rectangular(1, 2)(1.5) == 1
    tri = triangular(0, 2)
    assert tri(0) == 0
    assert tri(1) == 1
    trap = trapezoid(0, 1, 2, 3)
    assert trap(0.5) == pytest.approx(0.5)
    assert trap(1.5) == 1
    assert trap(2.5) == pytest.approx(0.5)


def test_sigmoid_variants_and_gauss():
    bs = bounded_sigmoid(0, 1)
    assert bs(0) == pytest.approx(0.1)
    assert round(bs(1), 2) == 0.9
    ss = simple_sigmoid()
    assert ss(0) == 0.5
    ts = triangular_sigmoid(2, 4)
    assert ts(2) == pytest.approx(0.1)
    assert round(ts(3), 1) == 0.9
    g = gauss(0, 1)
    assert g(0) == 1
    assert g(1) == pytest.approx(math.exp(-1))


def test_bounded_exponential():
    f = bounded_exponential(k=1, limit=2)
    assert f(0) == 0
    assert f(1) == pytest.approx(2 - 2 / math.e)
    assert f(1000) == 2
