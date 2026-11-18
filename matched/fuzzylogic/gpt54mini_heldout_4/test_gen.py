import math
import pickle

import numpy as np
import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic import functions as fn


def test_domain_init_validation():
    with pytest.raises(AssertionError):
        Domain("bad", 1, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_domain_set_assignment_and_access():
    d = Domain("temp", 0, 10)
    d.hot = fn.constant(1)

    assert "hot" in d._sets
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d(5) == {"hot": 1}
    assert str(d) == "temp"


def test_domain_call_warns_outside_range(monkeypatch):
    d = Domain("temp", 0, 10)
    d.hot = fn.constant(1)

    calls = []

    def fake_warn(msg):
        calls.append(msg)

    monkeypatch.setattr("fuzzylogic.classes.warn", fake_warn)
    result = d(-1)

    assert result == {"hot": 1}
    assert calls and "-1 is outside of domain!" in calls[0]


def test_domain_range_includes_high_bound():
    d = Domain("temp", 0, 1, res=0.5)
    arr = d.range()
    assert np.array_equal(arr, np.array([0.0, 0.5, 1.0]))


def test_domain_delete_set_and_regular_attr_error():
    d = Domain("temp", 0, 10)
    d.hot = fn.constant(1)

    del d.hot
    assert "hot" not in d._sets

    with pytest.raises(FuzzyWarning):
        del d._low


def test_domain_min_and_max_over_sets():
    d = Domain("temp", 0, 4, res=1)
    d.low = fn.constant(0.2)
    d.high = fn.constant(0.8)

    assert d.min(2) == 0.2
    assert d.max(2) == 0.8


def test_set_call_invert_and_negation():
    s = Set(fn.constant(0.25))
    inv_s = ~s
    neg_s = -s

    assert s(10) == 0.25
    assert inv_s(10) == 0.75
    assert neg_s(10) == 0.75


def test_set_boolean_and_arithmetic_operators():
    a = Set(fn.constant(0.2))
    b = Set(fn.constant(0.7))

    assert (a & b)(0) == 0.2
    assert (a | b)(0) == 0.7
    assert (a * b)(0) == pytest.approx(0.14)
    assert (a + b)(0) == pytest.approx(0.9)
    assert (a ^ b)(0) == pytest.approx(0.8)


def test_set_power_multiplied_and_concentrated():
    s = Set(fn.constant(0.5))
    assert (s ** 2)(0) == 0.25
    assert s.multiplied(3)(0) == 1.5
    assert s.concentrated()(0) == 0.25


def test_set_array_len_cardinality_and_relative_cardinality():
    d = Domain("temp", 0, 2, res=1)
    d.half = fn.constant(0.5)

    s = d.half
    assert len(s) == 3
    arr = s.array()
    assert np.array_equal(arr, np.array([0.5, 0.5, 0.5]))
    assert s.cardinality() == 1.5
    assert s.relative_cardinality() == pytest.approx(0.5)


def test_set_equality_and_subset_comparisons():
    d = Domain("temp", 0, 2, res=1)
    d.a = fn.constant(0.2)
    d.b = fn.constant(0.2)
    d.c = fn.constant(0.8)

    assert d.a == d.b
    assert d.a <= d.c
    assert d.a < d.c
    assert d.c >= d.a
    assert d.c > d.a


def test_set_comparison_without_domain_raises():
    a = Set(fn.constant(0.2))
    b = Set(fn.constant(0.8))

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


def test_set_normalized_scales_by_maximum():
    d = Domain("temp", 0, 2, res=1)
    d.s = fn.constant(0.25)

    n = d.s.normalized()
    assert n.domain is None
    assert n(0) == 1.0
    assert n(1) == 1.0


def test_set_repr_and_str_for_assigned_and_dangling_sets():
    d = Domain("temp", 0, 1)
    d.hot = fn.constant(1)
    dangling = Set(fn.constant(0.1))

    assert str(d.hot) == "temp.hot"
    assert "dangling Set(" in str(dangling)
    assert repr(d.hot).startswith("Set(")


def test_set_initialization_from_pickled_function_string():
    original = fn.constant(0.33)
    dumped = pickle.dumps(original)
    s = Set(dumped)

    assert s(123) == 0.33


def test_functions_constant_noop_and_inv():
    c = fn.constant(3)
    n = fn.noop()
    i = fn.inv(c)

    assert c(100) == 3
    assert n(7) == 7
    assert i(0) == -2
    assert i(1) == -2


def test_functions_linear_and_bounded_linear():
    lin = fn.linear(1, -1)
    bl = fn.bounded_linear(2, 3)

    assert lin(1) == 0
    assert lin(1.5) == 0.5
    assert lin(3) == 1

    assert bl(1) == 0.0
    assert bl(2.5) == 0.5
    assert bl(4) == 1.0


def test_functions_basic_memberships_and_sigmoids():
    rect = fn.rectangular(1, 2)
    tri = fn.triangular(0, 2)
    sig = fn.simple_sigmoid()
    gauss = fn.gauss(0, 1)

    assert rect(0) == 0
    assert rect(1.5) == 1
    assert tri(1) == 1
    assert round(sig(0), 6) == 0.5
    assert gauss(0) == pytest.approx(1.0)
    assert gauss(10) < 1e-10


def test_functions_bounded_sigmoid_and_normalize():
    bs = fn.bounded_sigmoid(0, 1)
    norm = fn.normalize(0.5, fn.constant(0.25))

    assert bs(0) == pytest.approx(0.1, rel=1e-6)
    assert round(bs(1), 2) == 0.9
    assert norm(10) == 0.5


def test_domain_repr_eval_roundtrip_basic():
    d = Domain("temp", 0, 10, res=2)
    d.hot = fn.constant(1)

    reconstructed = eval(repr(d), {"Domain": Domain, "Set": Set})
    assert reconstructed == d
