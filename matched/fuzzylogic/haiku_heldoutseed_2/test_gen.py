import pytest

from fuzzylogic import functions as fn
from fuzzylogic.classes import Domain, Set, FuzzyWarning


def test_domain_rejects_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("bad", 5, 5)
    with pytest.raises(AssertionError):
        Domain("bad", 10, 0)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_domain_range_includes_upper_bound_and_respects_res():
    d = Domain("t", 0, 10)
    r = d.range()
    assert len(r) == 11
    assert r[0] == 0
    assert r[-1] == 10

    coarse = Domain("t", 0, 10, res=2)
    r2 = coarse.range()
    assert len(r2) == 6
    assert r2[-1] == 10


def test_assigning_function_creates_named_set():
    d = Domain("t", 0, 10)
    d.hot = lambda x: 1 if x > 5 else 0
    assert isinstance(d.hot, Set)
    assert d.hot.name == "hot"
    assert d.hot.domain is d
    assert d.hot(7) == 1
    assert d.hot(2) == 0


def test_domain_call_returns_membership_dict():
    d = Domain("t", 0, 10)
    d.cold = fn.constant(0.2)
    d.hot = fn.constant(0.9)
    assert d(5) == {"cold": 0.2, "hot": 0.9}


def test_domain_min_and_max_over_sets():
    d = Domain("t", 0, 10)
    d.low = fn.constant(0.3)
    d.high = fn.constant(0.8)
    assert d.min(5) == pytest.approx(0.3)
    assert d.max(5) == pytest.approx(0.8)


def test_unknown_attribute_raises_attribute_error():
    d = Domain("t", 0, 10)
    with pytest.raises(AttributeError):
        _ = d.nothing


def test_delete_set_and_missing_delete_raises():
    d = Domain("t", 0, 10)
    d.hot = fn.constant(0.5)
    del d.hot
    with pytest.raises(AttributeError):
        _ = d.hot
    with pytest.raises(FuzzyWarning):
        del d.hot


def test_invalid_set_name_rejected():
    d = Domain("t", 0, 10)
    with pytest.raises(AssertionError):
        setattr(d, "not valid", fn.constant(0))


def test_set_construction_validation():
    d = Domain("t", 0, 10)
    # name and domain must be given together
    with pytest.raises(FuzzyWarning):
        Set(fn.constant(0), name="x")
    # a non-callable, non-string func is rejected
    with pytest.raises(AssertionError):
        Set(42)
    # name + domain registers the set in the domain
    s = Set(fn.constant(0.4), name="warm", domain=d)
    assert d.warm is s


def test_inverse_and_negation():
    s = Set(fn.constant(0.3))
    assert (~s)(0) == pytest.approx(0.7)
    assert (-s)(0) == pytest.approx(0.7)


def test_and_or_use_min_max():
    a = Set(fn.constant(0.2))
    b = Set(fn.constant(0.7))
    assert (a & b)(0) == pytest.approx(0.2)
    assert (a | b)(0) == pytest.approx(0.7)


def test_power_concentrated_multiplied():
    s = Set(fn.constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.concentrated()(0) == pytest.approx(0.25)
    assert s.multiplied(0.5)(0) == pytest.approx(0.25)


def test_array_len_cardinality_and_relative_cardinality():
    d = Domain("t", 0, 4)
    d.half = Set(fn.constant(0.5))
    assert list(d.half.array()) == pytest.approx([0.5] * 5)
    assert len(d.half) == 5
    assert d.half.cardinality() == pytest.approx(2.5)
    assert d.half.relative_cardinality() == pytest.approx(0.5)


def test_equality_on_same_domain():
    d = Domain("t", 0, 3)
    d.a = fn.constant(0.5)
    d.b = fn.constant(0.5)
    d.c = fn.constant(0.1)
    assert d.a == d.b
    assert not (d.a == d.c)


def test_domainless_set_operations_raise():
    s = Set(fn.constant(1))
    with pytest.raises(FuzzyWarning):
        _ = s == Set(fn.constant(1))
    with pytest.raises(FuzzyWarning):
        s.array()
    with pytest.raises(FuzzyWarning):
        len(s)
    with pytest.raises(FuzzyWarning):
        s.cardinality()


def test_ordering_relations():
    d = Domain("t", 0, 3)
    d.low = fn.constant(0.2)
    d.high = fn.constant(0.7)
    assert d.low < d.high
    assert d.high > d.low
    assert d.low <= d.high
    assert d.high >= d.low
    assert not (d.high <= d.low)
    assert not (d.low >= d.high)


def test_normalized_peaks_at_one():
    d = Domain("t", 0, 2)
    d.a = Set(lambda x: x / 4)
    n = d.a.normalized()
    assert n(2) == pytest.approx(1.0)
    assert n(1) == pytest.approx(0.5)
    assert n(0) == pytest.approx(0.0)


def test_str_representations():
    d = Domain("temperature", 0, 10)
    d.hot = fn.constant(0.5)
    assert str(d) == "temperature"
    assert str(d.hot) == "temperature.hot"
    assert str(Set(fn.constant(0))).startswith("dangling Set")


def test_inv_noop_constant():
    assert fn.inv(lambda x: 0.3)(0) == pytest.approx(0.7)
    assert fn.noop()(5) == 5
    assert fn.constant(1)(0) == 1
    assert fn.constant(0.4)(-100) == 0.4


def test_alpha_clips_outside_band():
    identity = lambda x: x
    f = fn.alpha(func=identity, floor=0.2, ceiling=0.8)
    assert f(0.1) == pytest.approx(0.2)
    assert f(0.9) == pytest.approx(0.8)
    assert f(0.5) == pytest.approx(0.5)

    g = fn.alpha(func=identity, floor=0.2, ceiling=0.8, floor_clip=0.0)
    assert g(0.1) == pytest.approx(0.0)


def test_moderate_and_normalize():
    assert fn.moderate(fn.constant(1))(0) == pytest.approx(1.0)
    assert fn.moderate(fn.constant(0))(0) == pytest.approx(0.0)
    assert fn.moderate(fn.constant(0.5))(0) == pytest.approx(0.5)
    assert fn.normalize(0.5, fn.constant(0.25))(0) == pytest.approx(0.5)


def test_singleton_and_rectangular():
    f = fn.singleton(2)
    assert f(2) == 1
    assert f(1) == 0
    g = fn.singleton(3, no_m=0.1, c_m=0.9)
    assert g(4) == pytest.approx(0.1)
    assert g(3) == pytest.approx(0.9)

    r = fn.rectangular(1, 3, c_m=0.8, no_m=0.1)
    assert r(0) == pytest.approx(0.1)
    assert r(2) == pytest.approx(0.8)
    assert r(1) == pytest.approx(0.8)
    assert r(4) == pytest.approx(0.1)


def test_R_and_S_ramps_and_linear():
    r = fn.R(0, 2)
    assert r(-1) == 0
    assert r(1) == pytest.approx(0.5)
    assert r(3) == 1

    s = fn.S(0, 2)
    assert s(0) == 1
    assert s(1) == pytest.approx(0.5)
    assert s(3) == 0

    lin = fn.linear(1, -1)
    assert lin(-2) == 0
    assert lin(1.5) == pytest.approx(0.5)
    assert lin(3) == 1


def test_triangular_and_trapezoid_shape():
    tri = fn.triangular(0, 4)
    assert tri(2) == pytest.approx(1.0)
    assert tri(0) == pytest.approx(0.0)
    assert tri(1) == pytest.approx(0.5)
    assert tri(3) == pytest.approx(0.5)
    assert tri(4) == pytest.approx(0.0)

    trap = fn.trapezoid(0, 1, 3, 4)
    assert trap(2) == pytest.approx(1.0)
    assert trap(-1) == pytest.approx(0.0)
    assert trap(0.5) == pytest.approx(0.5)
    assert trap(3.5) == pytest.approx(0.5)


def test_bounded_sigmoid_anchor_points():
    f = fn.bounded_sigmoid(0, 1)
    assert f(0) == pytest.approx(0.1)
    assert f(1) == pytest.approx(0.9)

    inv_f = fn.bounded_sigmoid(0, 1, inverse=True)
    assert inv_f(0) == pytest.approx(0.9)
    assert inv_f(1) == pytest.approx(0.1)
