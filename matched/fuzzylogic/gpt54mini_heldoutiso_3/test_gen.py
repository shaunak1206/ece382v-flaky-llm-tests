import math
import pickle

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic import functions as fn


def test_domain_init_and_range_includes_high():
    d = Domain("temp", 0, 2, res=1)
    assert d.range().tolist() == [0, 1, 2]
    assert str(d) == "temp"
    assert "Domain('temp', 0, 2, res=1" in repr(d)


def test_domain_set_assignment_and_call_returns_memberships():
    d = Domain("temp", 0, 10)
    d.cold = fn.constant(0.2)
    d.hot = fn.constant(0.8)

    assert isinstance(d.cold, Set)
    assert d.cold.name == "cold"
    assert d.cold.domain is d
    assert d(5) == {"cold": 0.2, "hot": 0.8}


def test_domain_call_outside_range_warns_but_still_evaluates():
    d = Domain("temp", 0, 10)
    d.a = fn.constant(1)

    with pytest.warns(UserWarning):
        res = d(99)
    assert res == {"a": 1}


def test_domain_attribute_access_and_delete_set():
    d = Domain("temp", 0, 10)
    d.cold = fn.constant(0)

    assert d.cold(3) == 0
    del d.cold
    with pytest.raises(AttributeError):
        _ = d.cold


def test_domain_deleting_regular_attribute_raises_fuzzywarning():
    d = Domain("temp", 0, 10)
    with pytest.raises(FuzzyWarning):
        del d._name


def test_domain_rejects_invalid_set_name_and_invalid_bounds():
    with pytest.raises(AssertionError):
        Domain("bad", 10, 0)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 10, res=0)

    d = Domain("temp", 0, 10)
    with pytest.raises(AssertionError):
        setattr(d, "not an identifier", fn.constant(1))


def test_set_call_invert_and_basic_boolean_operators():
    a = Set(fn.constant(0.25))
    b = Set(fn.constant(0.75))

    assert a(1) == 0.25
    assert (~a)(1) == pytest.approx(0.75)
    assert (-a)(1) == pytest.approx(0.75)
    assert (a & b)(1) == pytest.approx(0.25)
    assert (a | b)(1) == pytest.approx(0.75)
    assert (a * b)(1) == pytest.approx(0.1875)
    assert (a + b)(1) == pytest.approx(1.0)
    assert (a ^ b)(1) == pytest.approx(abs(0.25 - 0.75))


def test_set_power_concentrated_and_multiplied_and_dilated():
    s = Set(fn.constant(0.5))

    assert (s ** 2)(1) == pytest.approx(0.25)
    assert s.concentrated()(1) == pytest.approx(0.25)
    assert s.multiplied(3)(1) == pytest.approx(1.5)
    assert s.dilated()(1) == pytest.approx(math.sqrt(0.5))


def test_set_str_and_repr_for_dangling_and_assigned_sets():
    dangling = Set(fn.constant(0.1))
    assert "dangling Set(" in str(dangling)
    assert "Set(" in repr(dangling)

    d = Domain("temp", 0, 1)
    d.low = fn.constant(0)
    assert str(d.low) == "temp.low"


def test_set_equality_and_ordering_use_domain_arrays():
    d = Domain("temp", 0, 2, res=1)
    d.a = fn.singleton(1)
    d.b = fn.singleton(1)
    d.c = fn.constant(0)
    d.d = fn.constant(1)

    assert d.a == d.b
    assert d.c <= d.d
    assert d.c < d.d
    assert d.d >= d.c
    assert d.d > d.c


def test_set_comparison_without_domain_raises_fuzzywarning():
    a = Set(fn.constant(0))
    b = Set(fn.constant(1))

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
    d = Domain("temp", 0, 2, res=1)
    d.a = fn.constant(0.5)

    assert len(d.a) == 3
    assert d.a.cardinality() == pytest.approx(1.5)
    assert d.a.relative_cardinality() == pytest.approx(0.5)
    assert d.a.array().tolist() == [0.5, 0.5, 0.5]


def test_set_normalized_scales_to_one():
    d = Domain("temp", 0, 2, res=1)
    d.a = fn.constant(0.2)
    n = d.a.normalized()

    assert n.domain is None
    assert n(5) == pytest.approx(1.0)


def test_domain_min_and_max_over_sets():
    d = Domain("temp", 0, 2, res=1)
    d.a = fn.constant(0.2)
    d.b = fn.constant(0.8)

    assert d.min(0) == pytest.approx(0.2)
    assert d.max(0) == pytest.approx(0.8)


def test_functions_constant_inv_normalize_and_singleton():
    c = fn.constant(3)
    assert c("x") == 3

    inv_c = fn.inv(fn.constant(0.25))
    assert inv_c(0) == pytest.approx(0.75)

    norm = fn.normalize(0.5, fn.constant(0.25))
    assert norm(10) == pytest.approx(0.5)

    s = fn.singleton(2)
    assert s(1) == 0
    assert s(2) == 1


def test_functions_basic_membership_shapes():
    lin = fn.linear(1, -1)
    assert lin(0) == 0
    assert lin(2) == 1
    assert lin(1.5) == pytest.approx(0.5)

    rect = fn.rectangular(2, 4)
    assert rect(1) == 0
    assert rect(3) == 1
    assert rect(5) == 0

    tri = fn.triangular(0, 2)
    assert tri(0) == 0
    assert tri(1) == pytest.approx(1)
    assert tri(2) == 0


def test_functions_sigmoid_and_gauss_behave_reasonably():
    sig = fn.simple_sigmoid()
    assert sig(0) == pytest.approx(0.5)
    assert sig(-1000) == pytest.approx(0.0, abs=1e-6)
    assert sig(1000) == pytest.approx(1.0, abs=1e-6)

    bsig = fn.bounded_sigmoid(0, 1)
    assert bsig(0) == pytest.approx(0.1)
    assert round(bsig(1), 2) == 0.9

    g = fn.gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) < g(0)
