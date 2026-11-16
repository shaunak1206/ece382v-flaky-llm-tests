import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic import functions as fn


# ---------------------------------------------------------------- Domain

def test_domain_range_includes_upper_bound():
    d = Domain("d", 0, 5)
    r = list(d.range())
    assert r == [0, 1, 2, 3, 4, 5]


def test_domain_range_with_resolution():
    d = Domain("d", 0, 1, res=0.5)
    assert list(d.range()) == pytest.approx([0, 0.5, 1.0])


def test_domain_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("bad", 5, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_domain_set_assignment_and_call():
    d = Domain("temp", 0, 100)
    d.low = fn.constant(0.2)
    d.high = fn.constant(0.7)
    assert isinstance(d.low, Set)
    assert d.low.domain is d
    assert d.low.name == "low"
    result = d(10)
    assert result == pytest.approx({"low": 0.2, "high": 0.7}) or (
        result["low"] == pytest.approx(0.2) and result["high"] == pytest.approx(0.7)
    )


def test_domain_missing_attribute_raises():
    d = Domain("d", 0, 10)
    with pytest.raises(AttributeError):
        d.nothing_here


def test_domain_delete_set_and_regular_attr():
    d = Domain("d", 0, 10)
    d.s = fn.constant(0.5)
    del d.s
    with pytest.raises(AttributeError):
        d.s
    with pytest.raises(FuzzyWarning):
        del d._name


def test_domain_min_max_over_sets():
    d = Domain("d", 0, 10)
    d.a = fn.constant(0.2)
    d.b = fn.constant(0.9)
    assert d.min(5) == pytest.approx(0.2)
    assert d.max(5) == pytest.approx(0.9)


def test_domain_str():
    assert str(Domain("temperature", 0, 10)) == "temperature"


# ------------------------------------------------------------------- Set

def test_set_call_and_invert():
    s = Set(fn.constant(0.3))
    assert s(0) == pytest.approx(0.3)
    assert (~s)(0) == pytest.approx(0.7)
    assert (-s)(0) == pytest.approx(0.7)


def test_set_and_or_use_min_max():
    a = Set(fn.constant(0.2))
    b = Set(fn.constant(0.7))
    assert (a & b)(1) == pytest.approx(0.2)
    assert (a | b)(1) == pytest.approx(0.7)


def test_set_pow_concentrated_multiplied():
    s = Set(fn.constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.concentrated()(0) == pytest.approx(0.25)
    assert Set(fn.constant(0.25)).multiplied(2)(0) == pytest.approx(0.5)


def test_set_name_without_domain_raises():
    with pytest.raises(FuzzyWarning):
        Set(fn.constant(1), name="x")


def test_set_with_name_and_domain_registers():
    d = Domain("d", 0, 4)
    s = Set(fn.constant(0.5), name="half", domain=d)
    assert d.half is s
    assert str(s) == "d.half"


def test_set_dangling_str_and_domainless_operations_raise():
    s = Set(fn.constant(0.5))
    assert str(s).startswith("dangling Set(")
    with pytest.raises(FuzzyWarning):
        s.array()
    with pytest.raises(FuzzyWarning):
        len(s)
    with pytest.raises(FuzzyWarning):
        s.cardinality()
    with pytest.raises(FuzzyWarning):
        s.normalized()
    with pytest.raises(FuzzyWarning):
        s == Set(fn.constant(0.5))


def test_set_array_len_cardinality():
    d = Domain("d", 0, 4)
    d.s = fn.R(0, 4)
    arr = d.s.array()
    assert list(arr) == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])
    assert len(d.s) == 5
    assert d.s.cardinality() == pytest.approx(2.5)
    assert d.s.relative_cardinality() == pytest.approx(0.5)


def test_set_comparisons_and_equality():
    d = Domain("d", 0, 4)
    d.a = fn.constant(0.2)
    d.b = fn.constant(0.5)
    d.c = fn.constant(0.2)
    assert d.a <= d.b
    assert d.a < d.b
    assert d.b >= d.a
    assert d.b > d.a
    assert not (d.a > d.b)
    assert d.a == d.c
    assert not (d.a == d.b)


def test_set_normalized():
    d = Domain("d", 0, 5)
    d.s = lambda x: x / 10  # max 0.5 on [0, 5]
    n = d.s.normalized()
    assert n(5) == pytest.approx(1.0)
    assert n(4) == pytest.approx(0.8)


# ------------------------------------------------------------- functions

def test_special_functions():
    assert fn.inv(fn.constant(0.25))(3) == pytest.approx(0.75)
    assert fn.noop()(7) == 7
    assert fn.constant(4)(123) == 4
    assert fn.normalize(0.5, fn.constant(0.25))(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        fn.normalize(0, fn.constant(0.25))


def test_alpha_and_moderate():
    f = fn.alpha(floor=0.2, ceiling=0.8, func=fn.noop())
    assert f(0.9) == pytest.approx(0.8)
    assert f(0.1) == pytest.approx(0.2)
    assert f(0.5) == pytest.approx(0.5)
    assert fn.moderate(fn.constant(0.5))(0) == pytest.approx(0.5)
    assert fn.moderate(fn.constant(1))(0) == pytest.approx(1.0)
    assert fn.moderate(fn.constant(0))(0) == pytest.approx(0.0)


def test_singleton_and_rectangular():
    s = fn.singleton(2)
    assert s(2) == 1
    assert s(1) == 0
    r = fn.rectangular(1, 3)
    assert [r(x) for x in (0, 1, 2, 3, 4)] == [0, 1, 1, 1, 0]


def test_linear_and_bounded_linear():
    f = fn.linear(1, -1)
    assert f(-2) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(3) == 1
    g = fn.bounded_linear(2, 3)
    assert g(1) == pytest.approx(0)
    assert g(2.5) == pytest.approx(0.5)
    assert g(4) == pytest.approx(1)
    h = fn.bounded_linear(2, 3, inverse=True)
    assert h(2) == pytest.approx(1)
    assert h(2.5) == pytest.approx(0.5)
    assert h(3) == pytest.approx(0)
    with pytest.raises(AssertionError):
        fn.bounded_linear(3, 2)


def test_R_and_S():
    r = fn.R(0, 10)
    assert r(-1) == 0
    assert r(5) == pytest.approx(0.5)
    assert r(11) == 1
    s = fn.S(0, 10)
    assert s(-1) == 1
    assert s(5) == pytest.approx(0.5)
    assert s(11) == 0


def test_triangular_and_trapezoid():
    t = fn.triangular(0, 4)
    assert t(0) == pytest.approx(0)
    assert t(1) == pytest.approx(0.5)
    assert t(2) == pytest.approx(1)
    assert t(3) == pytest.approx(0.5)
    assert t(5) == pytest.approx(0)
    z = fn.trapezoid(0, 1, 2, 3)
    assert z(-1) == pytest.approx(0)
    assert z(0.5) == pytest.approx(0.5)
    assert z(1.5) == pytest.approx(1)
    assert z(2.5) == pytest.approx(0.5)
    assert z(4) == pytest.approx(0)


def test_sigmoids_and_gauss():
    assert fn.sigmoid(1, 1, 0)(0) == pytest.approx(0.5)
    assert fn.sigmoid(0.8, 1, 0)(0) == pytest.approx(0.4)
    with pytest.raises(AssertionError):
        fn.sigmoid(0, 1, 0)
    b = fn.bounded_sigmoid(0, 1)
    assert b(0) == pytest.approx(0.1)
    assert b(1) == pytest.approx(0.9)
    assert fn.simple_sigmoid()(0) == pytest.approx(0.5)
    g = fn.gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) == pytest.approx(0.36787944117)


def test_bounded_exponential():
    f = fn.bounded_exponential(k=1, limit=1)
    assert f(0) == pytest.approx(0)
    assert f(100) == pytest.approx(1.0)
    assert f(1) == pytest.approx(1 - 0.36787944117)
