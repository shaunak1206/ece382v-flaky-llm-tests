import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv, noop, constant, alpha, normalize, moderate, singleton, linear,
    bounded_linear, R, S, rectangular, triangular, trapezoid, sigmoid,
    bounded_sigmoid, bounded_exponential, simple_sigmoid, gauss,
)


# ---------------------------------------------------------------- Domain

def test_domain_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("d", 10, 0)
    with pytest.raises(AssertionError):
        Domain("d", 0, 10, res=0)


def test_domain_range_includes_upper_bound():
    d = Domain("d", 0, 5, res=1)
    assert list(d.range()) == [0, 1, 2, 3, 4, 5]
    d2 = Domain("d2", 0, 1, res=0.5)
    assert list(d2.range()) == [0, 0.5, 1.0]


def test_domain_set_assignment_and_lookup():
    d = Domain("temp", 0, 10)
    d.hot = Set(constant(0.7))
    assert isinstance(d.hot, Set)
    assert d.hot.domain is d
    assert d.hot.name == "hot"
    assert d.hot(3) == 0.7


def test_domain_assigning_plain_function_wraps_in_set():
    d = Domain("temp", 0, 10)
    d.cold = constant(0.1)
    assert isinstance(d.cold, Set)
    assert d.cold(5) == 0.1


def test_domain_call_returns_membership_dict():
    d = Domain("temp", 0, 10)
    d.a = Set(constant(0.25))
    d.b = Set(constant(0.75))
    assert d(5) == {"a": 0.25, "b": 0.75}


def test_domain_min_max():
    d = Domain("temp", 0, 10)
    d.a = Set(constant(0.25))
    d.b = Set(constant(0.75))
    assert d.min(3) == 0.25
    assert d.max(3) == 0.75


def test_domain_missing_attribute_raises_attribute_error():
    d = Domain("temp", 0, 10)
    with pytest.raises(AttributeError):
        d.nothing_here


def test_domain_delete_set():
    d = Domain("temp", 0, 10)
    d.a = Set(constant(0.5))
    del d.a
    assert d(1) == {}
    with pytest.raises(AttributeError):
        d.a
    with pytest.raises(FuzzyWarning):
        del d.a


def test_domain_str_is_name():
    assert str(Domain("temperature", 0, 100)) == "temperature"


# ------------------------------------------------------------------- Set

def test_set_call_and_logical_operators():
    a = Set(constant(0.3))
    b = Set(constant(0.6))
    assert a(0) == 0.3
    assert (~a)(0) == pytest.approx(0.7)
    assert (-a)(0) == pytest.approx(0.7)
    assert (a & b)(0) == pytest.approx(0.3)
    assert (a | b)(0) == pytest.approx(0.6)
    assert (a * b)(0) == pytest.approx(0.18)


def test_set_power_concentrated_and_multiplied():
    a = Set(constant(0.5))
    assert (a ** 2)(0) == pytest.approx(0.25)
    assert a.concentrated()(0) == pytest.approx(0.25)
    assert a.multiplied(0.5)(0) == pytest.approx(0.25)


def test_set_array_len_and_cardinality():
    d = Domain("d", 0, 4)
    d.s = Set(noop())
    assert list(d.s.array()) == [0, 1, 2, 3, 4]
    assert len(d.s) == 5
    assert d.s.cardinality() == 10
    assert d.s.relative_cardinality() == pytest.approx(2.0)


def test_set_without_domain_raises_fuzzy_warning():
    s = Set(constant(0.5))
    with pytest.raises(FuzzyWarning):
        s.array()
    with pytest.raises(FuzzyWarning):
        len(s)
    with pytest.raises(FuzzyWarning):
        s.cardinality()
    with pytest.raises(FuzzyWarning):
        s.normalized()
    with pytest.raises(FuzzyWarning):
        s.plot()
    with pytest.raises(FuzzyWarning):
        s == Set(constant(0.5))
    with pytest.raises(FuzzyWarning):
        s <= Set(constant(0.5))


def test_set_name_without_domain_raises():
    with pytest.raises(FuzzyWarning):
        Set(constant(0.5), name="x")


def test_set_construction_with_name_and_domain_registers_set():
    d = Domain("d", 0, 10)
    s = Set(constant(0.4), name="mid", domain=d)
    assert d.mid is s
    assert s.domain is d
    assert str(s) == "d.mid"


def test_set_str_dangling():
    assert "dangling" in str(Set(constant(0.5)))


def test_set_equality_and_comparisons_in_domain():
    d = Domain("d", 0, 5)
    d.low = Set(constant(0.2))
    d.high = Set(constant(0.5))
    d.low_copy = Set(constant(0.2))
    assert d.low == d.low_copy
    assert not (d.low == d.high)
    assert d.low <= d.high
    assert d.low < d.high
    assert d.high >= d.low
    assert d.high > d.low
    assert not (d.low >= d.high)
    assert d.low <= d.low_copy
    assert not (d.low < d.low_copy)


def test_set_normalized_has_max_one():
    d = Domain("d", 0, 5)
    d.s = Set(lambda x: x / 10)
    n = d.s.normalized()
    assert n(5) == pytest.approx(1.0)
    assert n(2) == pytest.approx(0.4)


# ------------------------------------------------------------- functions

def test_special_functions():
    assert inv(constant(0.25))(0) == pytest.approx(0.75)
    assert noop()(3.5) == 3.5
    assert constant(7)(123) == 7
    f = alpha(floor=0.2, ceiling=0.8, func=noop())
    assert f(0.1) == 0.2
    assert f(0.9) == 0.8
    assert f(0.5) == 0.5
    assert normalize(0.5, constant(0.25))(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        normalize(0, constant(0.25))
    assert moderate(constant(0.5))(0) == pytest.approx(0.5)
    assert moderate(constant(1))(0) == pytest.approx(1.0)
    assert moderate(constant(0))(0) == pytest.approx(0.0)


def test_singleton_and_rectangular():
    s = singleton(2)
    assert s(2) == 1
    assert s(1) == 0
    r = rectangular(1, 3)
    assert r(0) == 0
    assert r(1) == 1
    assert r(2) == 1
    assert r(3) == 1
    assert r(4) == 0
    with pytest.raises(AssertionError):
        rectangular(3, 1)


def test_linear_is_clipped():
    f = linear(1, -1)
    assert f(-2) == 0
    assert f(1) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(2) == 1
    assert f(3) == 1


def test_bounded_linear_and_inverse():
    f = bounded_linear(2, 3)
    assert f(1) == 0
    assert f(2.5) == pytest.approx(0.5)
    assert f(4) == 1
    g = bounded_linear(2, 3, inverse=True)
    assert g(1) == 1
    assert g(2.5) == pytest.approx(0.5)
    assert g(4) == 0
    with pytest.raises(AssertionError):
        bounded_linear(3, 2)


def test_R_and_S_are_mirror_ramps():
    r = R(2, 4)
    s = S(2, 4)
    assert (r(1), r(3), r(5)) == (0, pytest.approx(0.5), 1)
    assert (s(1), s(3), s(5)) == (1, pytest.approx(0.5), 0)
    with pytest.raises(AssertionError):
        R(4, 2)
    with pytest.raises(AssertionError):
        S(4, 2)


def test_triangular():
    f = triangular(0, 10)
    assert f(-1) == 0
    assert f(0) == 0
    assert f(2.5) == pytest.approx(0.5)
    assert f(5) == pytest.approx(1.0)
    assert f(7.5) == pytest.approx(0.5)
    assert f(10) == pytest.approx(0.0)
    assert f(11) == pytest.approx(0.0)
    with pytest.raises(AssertionError):
        triangular(0, 10, c=11)


def test_trapezoid():
    f = trapezoid(0, 2, 4, 6)
    assert f(-1) == 0
    assert f(1) == pytest.approx(0.5)
    assert f(2) == pytest.approx(1.0)
    assert f(3) == 1
    assert f(5) == pytest.approx(0.5)
    assert f(7) == 0
    with pytest.raises(AssertionError):
        trapezoid(0, 5, 4, 6)


def test_sigmoid_variants():
    assert sigmoid(1, 1, 0)(0) == pytest.approx(0.5)
    assert sigmoid(0.8, 1, 0)(1000) == pytest.approx(0.8)
    assert sigmoid(1, 1, 0)(-1e6) == pytest.approx(0.0)  # overflow handled

    b = bounded_sigmoid(0, 1)
    assert b(0) == pytest.approx(0.1)
    assert b(1) == pytest.approx(0.9)
    assert b(100000) == pytest.approx(1.0)
    assert b(-100000) == pytest.approx(0.0)
    bi = bounded_sigmoid(0, 1, inverse=True)
    assert bi(0) == pytest.approx(0.9)
    assert bi(1) == pytest.approx(0.1)

    s = simple_sigmoid()
    assert s(0) == pytest.approx(0.5)
    assert s(1000) == pytest.approx(1.0)
    assert s(-1000) == pytest.approx(0.0)


def test_gauss_and_bounded_exponential():
    g = gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) == pytest.approx(0.36787944117)
    assert g(-1) == pytest.approx(g(1))
    assert gauss(0, 1, c_m=0.5)(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        gauss(0, 0)

    e = bounded_exponential(0.1, 1)
    assert e(0) == pytest.approx(0.0)
    assert e(10000) == pytest.approx(1.0)
    assert 0 < e(5) < e(10) < 1
