import math

import pytest

import fuzzylogic.classes as classes
from fuzzylogic import functions as F
from fuzzylogic.classes import Domain, Set, FuzzyWarning


# ---------------------------------------------------------------------------
# Domain and Set
# ---------------------------------------------------------------------------

def test_domain_init_stores_attributes_and_rejects_bad_arguments():
    d = Domain("temperature", 0, 100, res=2)
    assert str(d) == "temperature"
    assert d._low == 0
    assert d._high == 100
    assert d._res == 2
    assert d._sets == {}

    with pytest.raises(AssertionError):
        Domain("bad", 5, 5)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_range_includes_upper_bound_and_uses_resolution():
    d = Domain("x", 0, 2, res=0.5)
    values = [float(v) for v in d.range()]
    assert values == pytest.approx([0, 0.5, 1, 1.5, 2])


def test_assigned_function_becomes_named_set_bound_to_domain():
    d = Domain("t", 0, 10)
    d.cold = lambda x: 1.0
    assert isinstance(d.cold, Set)
    assert d.cold.name == "cold"
    assert d.cold.domain is d
    assert str(d.cold) == "t.cold"

    s = Set(F.constant(0.3))
    d.warm = s
    assert d.warm is s


def test_calling_domain_returns_membership_per_set():
    d = Domain("t", 0, 10)
    d.cold = F.constant(0.2)
    d.hot = F.constant(0.7)
    result = d(5)
    assert result == pytest.approx({"cold": 0.2, "hot": 0.7})


def test_calling_domain_warns_only_outside_bounds(monkeypatch):
    messages = []
    monkeypatch.setattr(classes, "warn", messages.append)

    d = Domain("t", 0, 10)
    d.cold = F.constant(0.2)

    d(0)
    d(10)
    assert messages == []

    result = d(15)
    assert len(messages) == 1
    assert "15" in messages[0]
    assert result == pytest.approx({"cold": 0.2})


def test_missing_attribute_raises_and_invalid_set_name_is_rejected():
    d = Domain("t", 0, 10)
    with pytest.raises(AttributeError):
        d.nope
    with pytest.raises(AssertionError):
        setattr(d, "not an identifier", F.constant(0))


def test_deleting_set_removes_it_and_deleting_unknown_name_raises():
    d = Domain("t", 0, 10)
    d.cold = F.constant(0.2)
    del d.cold
    with pytest.raises(AttributeError):
        d.cold
    with pytest.raises(FuzzyWarning):
        del d.missing


def test_domain_min_and_max_over_all_sets():
    d = Domain("t", 0, 10)
    d.a = F.constant(0.2)
    d.b = F.constant(0.9)
    assert d.min(3) == pytest.approx(0.2)
    assert d.max(3) == pytest.approx(0.9)


def test_domain_equality_compares_name_bounds_and_resolution():
    assert Domain("a", 0, 1) == Domain("a", 0, 1)
    assert not (Domain("a", 0, 1) == Domain("b", 0, 1))
    assert not (Domain("a", 0, 1) == Domain("a", 0, 2))
    assert not (Domain("a", 0, 1, res=1) == Domain("a", 0, 1, res=0.5))


def test_set_init_binds_to_domain_and_rejects_name_without_domain():
    d = Domain("t", 0, 10)
    s = Set(F.constant(0.5), name="warm", domain=d)
    assert d.warm is s
    assert s.name == "warm"
    assert str(s) == "t.warm"

    with pytest.raises(FuzzyWarning):
        Set(F.constant(0.5), name="warm")

    assert str(Set(F.constant(0))).startswith("dangling Set(")


def test_invert_and_negation_give_complement():
    s = Set(F.constant(0.25))
    assert (~s)(0) == pytest.approx(0.75)
    assert (-s)(0) == pytest.approx(0.75)


def test_pow_concentrated_and_multiplied_transform_values():
    s = Set(F.constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.concentrated()(0) == pytest.approx(0.25)
    assert s.multiplied(0.5)(0) == pytest.approx(0.25)


def test_array_length_and_cardinality_over_domain():
    d = Domain("t", 0, 4)
    d.a = F.constant(0.5)
    assert len(d.a) == 5
    assert list(d.a.array()) == pytest.approx([0.5] * 5)
    assert d.a.cardinality() == pytest.approx(2.5)
    assert d.a.relative_cardinality() == pytest.approx(0.5)


def test_operations_requiring_domain_raise_without_domain():
    a = Set(F.constant(0.1))
    b = Set(F.constant(0.2))
    with pytest.raises(FuzzyWarning):
        a.array()
    with pytest.raises(FuzzyWarning):
        len(a)
    with pytest.raises(FuzzyWarning):
        a.cardinality()
    with pytest.raises(FuzzyWarning):
        a == b
    with pytest.raises(FuzzyWarning):
        a <= b


def test_pointwise_comparisons_between_sets_in_same_domain():
    d = Domain("t", 0, 4)
    d.low = F.constant(0.2)
    d.high = F.constant(0.5)
    d.same = F.constant(0.2)

    assert d.low <= d.high
    assert d.low < d.high
    assert d.high >= d.low
    assert d.high > d.low
    assert not (d.high <= d.low)
    assert d.low == d.same
    assert not (d.low == d.high)


def test_normalized_set_scales_peak_to_one():
    d = Domain("t", 0, 4)
    d.a = lambda x: 0.5 if x >= 2 else 0.25
    n = d.a.normalized()
    assert n(0) == pytest.approx(0.5)
    assert n(3) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# functions module
# ---------------------------------------------------------------------------

def test_inv_noop_and_constant():
    assert F.inv(F.constant(0.3))(7) == pytest.approx(0.7)
    assert F.noop()(4) == 4
    assert F.constant(1)(0) == 1


def test_alpha_clips_at_floor_and_ceiling():
    f = F.alpha(floor=0.2, ceiling=0.8, func=lambda x: x)
    assert f(0.1) == pytest.approx(0.2)
    assert f(0.5) == pytest.approx(0.5)
    assert f(0.9) == pytest.approx(0.8)


def test_moderate_pushes_towards_half_and_normalize_scales():
    assert F.moderate(F.constant(1))(0) == pytest.approx(1.0)
    assert F.moderate(F.constant(0))(0) == pytest.approx(0.0)
    assert F.moderate(F.constant(0.5))(0) == pytest.approx(0.5)
    assert F.moderate(F.constant(0.75))(0) == pytest.approx(0.5625)

    assert F.normalize(0.5, F.constant(0.25))(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        F.normalize(0, F.constant(1))


def test_singleton_and_linear():
    f = F.singleton(2, no_m=0.1, c_m=0.9)
    assert f(2) == pytest.approx(0.9)
    assert f(1) == pytest.approx(0.1)

    g = F.linear(1, -1)
    assert g(-2) == 0
    assert g(1.5) == pytest.approx(0.5)
    assert g(3) == 1


def test_bounded_linear_interpolates_and_inverse_flips():
    f = F.bounded_linear(2, 3)
    assert f(1) == pytest.approx(0.0)
    assert f(2) == pytest.approx(0.0)
    assert f(2.5) == pytest.approx(0.5)
    assert f(3) == pytest.approx(1.0)
    assert f(4) == pytest.approx(1.0)

    inv_f = F.bounded_linear(2, 3, inverse=True)
    assert inv_f(2) == pytest.approx(1.0)
    assert inv_f(2.5) == pytest.approx(0.5)
    assert inv_f(3) == pytest.approx(0.0)


def test_R_and_S_ramps_and_rectangular():
    r = F.R(0, 2)
    assert r(-1) == 0
    assert r(1) == pytest.approx(0.5)
    assert r(3) == 1

    s = F.S(0, 2)
    assert s(-1) == 1
    assert s(1) == pytest.approx(0.5)
    assert s(3) == 0

    rect = F.rectangular(1, 3, c_m=0.8, no_m=0.1)
    assert rect(0) == pytest.approx(0.1)
    assert rect(1) == pytest.approx(0.8)
    assert rect(2) == pytest.approx(0.8)
    assert rect(3) == pytest.approx(0.8)
    assert rect(4) == pytest.approx(0.1)


def test_triangular_and_trapezoid_shapes():
    tri = F.triangular(0, 4)
    assert tri(0) == pytest.approx(0.0)
    assert tri(1) == pytest.approx(0.5)
    assert tri(2) == pytest.approx(1.0)
    assert tri(3) == pytest.approx(0.5)

    trap = F.trapezoid(0, 1, 2, 3)
    assert trap(-1) == pytest.approx(0.0)
    assert trap(0.5) == pytest.approx(0.5)
    assert trap(1.5) == pytest.approx(1.0)
    assert trap(2.5) == pytest.approx(0.5)
    assert trap(3.5) == pytest.approx(0.0)


def test_sigmoid_family_midpoints_and_limits():
    assert F.sigmoid(1, 1, 0)(0) == pytest.approx(0.5)
    assert F.sigmoid(0.5, 1, 0)(0) == pytest.approx(0.25)

    b = F.bounded_sigmoid(0, 1)
    assert b(0) == pytest.approx(0.1)
    assert b(1) == pytest.approx(0.9)
    assert b(100000) == pytest.approx(1.0)
    assert b(-100000) == pytest.approx(0.0)

    e = F.bounded_exponential(k=1, limit=1)
    assert e(0) == pytest.approx(0.0)
    assert e(1) == pytest.approx(1 - 1 / math.e)
    assert e(1000) == pytest.approx(1.0)

    s = F.simple_sigmoid(k=1)
    assert s(0) == pytest.approx(0.5)
    assert s(-1000) == pytest.approx(0.0)
    assert s(1000) == pytest.approx(1.0)


def test_triangular_sigmoid_and_gauss():
    g = F.triangular_sigmoid(2, 4)
    assert g(2) == pytest.approx(0.1)
    assert g(3) == pytest.approx(0.9)

    gauss = F.gauss(0, 1)
    assert gauss(0) == pytest.approx(1.0)
    assert gauss(1) == pytest.approx(math.exp(-1))
    assert gauss(-1) == pytest.approx(math.exp(-1))
    assert F.gauss(0, 1, c_m=0.5)(0) == pytest.approx(0.5)
    # squaring overflows -> function returns 0 instead of raising
    assert gauss(1e200) == 0
