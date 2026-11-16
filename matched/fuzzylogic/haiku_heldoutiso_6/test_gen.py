import math

import pytest

from fuzzylogic import functions as F
from fuzzylogic import classes as classes_module
from fuzzylogic.classes import Domain, Set, FuzzyWarning


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_init_validates_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain('t', 5, 5)
    with pytest.raises(AssertionError):
        Domain('t', 0, 10, res=0)
    d = Domain('temp', 0, 10)
    assert str(d) == 'temp'


def test_domain_range_includes_upper_bound():
    d = Domain('t', 0, 10, res=2)
    assert list(d.range()) == [0, 2, 4, 6, 8, 10]


def test_domain_assignment_wraps_functions_and_sets_back_references():
    d = Domain('temp', 0, 10)
    d.cold = F.constant(0.25)
    assert isinstance(d.cold, Set)
    assert d.cold.name == 'cold'
    assert d.cold.domain is d
    assert d.cold(3) == 0.25

    existing = Set(F.constant(0.5))
    d.hot = existing
    assert d.hot is existing
    assert existing.domain is d
    assert existing.name == 'hot'


def test_domain_call_returns_membership_per_set():
    d = Domain('temp', 0, 10)
    d.cold = F.constant(0.25)
    d.hot = F.constant(0.75)
    assert d(5) == {'cold': 0.25, 'hot': 0.75}


def test_domain_min_and_max_over_sets():
    d = Domain('temp', 0, 10)
    d.a = F.constant(0.2)
    d.b = F.constant(0.7)
    assert d.min(5) == 0.2
    assert d.max(5) == 0.7


def test_domain_attribute_errors_and_invalid_names():
    d = Domain('t', 0, 10)
    with pytest.raises(AttributeError):
        _ = d.missing
    with pytest.raises(FuzzyWarning):
        del d.missing
    with pytest.raises(AssertionError):
        setattr(d, 'not an id', F.constant(0))


def test_domain_delete_removes_set():
    d = Domain('t', 0, 10)
    d.hot = F.constant(0.5)
    del d.hot
    with pytest.raises(AttributeError):
        _ = d.hot
    assert d(5) == {}


# ---------------------------------------------------------------------------
# Set
# ---------------------------------------------------------------------------

def test_set_init_validation_and_named_registration():
    with pytest.raises(AssertionError):
        Set(5)

    d = Domain('temp', 0, 10)
    with pytest.raises(FuzzyWarning):
        Set(F.constant(0.5), name='x')
    with pytest.raises(FuzzyWarning):
        Set(F.constant(0.5), domain=d)

    s = Set(F.constant(0.5), name='warm', domain=d)
    assert d.warm is s
    assert str(s) == 'temp.warm'
    assert str(Set(F.constant(0.5))).startswith('dangling Set(')


def test_set_invert_and_neg():
    s = Set(F.constant(0.3))
    assert (~s)(0) == pytest.approx(0.7)
    assert (-s)(0) == pytest.approx(0.7)


def test_set_and_or_use_min_max():
    a = Set(F.constant(0.3))
    b = Set(F.constant(0.6))
    assert (a & b)(0) == pytest.approx(0.3)
    assert (a | b)(0) == pytest.approx(0.6)


def test_set_hedge_like_modifiers():
    s = Set(F.constant(0.5))
    assert (s ** 2)(0) == pytest.approx(0.25)
    assert s.multiplied(0.5)(0) == pytest.approx(0.25)
    assert s.concentrated()(0) == pytest.approx(0.25)


def test_intensified_low_branch():
    s = Set(F.constant(0.4))
    # For x < 0.5 the function is 2 * m(x)**2
    assert s.intensified()(0) == pytest.approx(2 * 0.4 ** 2)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="dilated() computes m(x) ** 1. / 2. which is m(x) / 2, not sqrt")
def test_dilated_should_take_square_root():
    s = Set(F.constant(0.25))
    assert s.dilated()(0) == pytest.approx(0.5)


def test_cardinality_and_relative_cardinality():
    d = Domain('temp', 0, 10)  # 11 values with res=1
    d.s = F.constant(0.5)
    s = d.s
    assert len(s) == 11
    assert s.cardinality() == pytest.approx(5.5)
    assert s.relative_cardinality() == pytest.approx(0.5)


def test_eq_requires_domain_then_compares_values():
    a = Set(F.constant(0.5))
    b = Set(F.constant(0.5))
    with pytest.raises(FuzzyWarning):
        _ = a == b

    d = Domain('temp', 0, 10)
    d.a = F.constant(0.5)
    d.b = F.constant(0.5)
    d.c = F.constant(0.6)
    assert d.a == d.b
    assert d.a != d.c


def test_comparisons_subset_and_superset():
    d = Domain('temp', 0, 10)
    d.low = F.constant(0.2)
    d.high = F.constant(0.5)
    assert d.low <= d.high
    assert d.low < d.high
    assert d.high >= d.low
    assert d.high > d.low
    assert not (d.high <= d.low)

    x = Set(F.constant(0.1))
    y = Set(F.constant(0.2))
    with pytest.raises(FuzzyWarning):
        _ = x <= y


def test_plot_passes_domain_range_to_pyplot(monkeypatch):
    calls = []
    monkeypatch.setattr(classes_module.plt, 'plot',
                        lambda x, y: calls.append((list(x), list(y))))
    d = Domain('t', 0, 2)
    d.s = F.constant(0.5)
    d.s.plot()
    xs, ys = calls[0]
    assert xs == [0, 1, 2]
    assert ys == [0.5, 0.5, 0.5]

    with pytest.raises(FuzzyWarning):
        Set(F.constant(0.5)).plot()


def test_normalized_scales_max_to_one():
    d = Domain('t', 0, 10)
    d.half = F.constant(0.5)
    normalized = d.half.normalized()
    assert normalized(0) == pytest.approx(1.0)
    assert max(normalized.array()) == pytest.approx(1.0)

    with pytest.raises(FuzzyWarning):
        Set(F.constant(0.5)).normalized()


# ---------------------------------------------------------------------------
# functions module
# ---------------------------------------------------------------------------

def test_inv_noop_and_constant():
    assert F.inv(F.constant(0.3))(0) == pytest.approx(0.7)
    assert F.noop()(5) == 5
    assert F.constant(1)(0) == 1


def test_alpha_normalize_and_moderate():
    f = F.alpha(floor=0.2, ceiling=0.8, func=lambda x: x)
    assert f(0.1) == 0.2
    assert f(0.9) == 0.8
    assert f(0.5) == 0.5

    g = F.alpha(floor=0.2, ceiling=0.8, func=lambda x: x,
                floor_clip=0, ceiling_clip=1)
    assert g(0.1) == 0
    assert g(0.9) == 1

    with pytest.raises(AssertionError):
        F.alpha(floor=0.9, ceiling=0.1, func=lambda x: x)

    n = F.normalize(0.5, F.constant(0.25))
    assert n(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        F.normalize(0, F.constant(0.25))

    assert F.moderate(F.constant(1))(0) == pytest.approx(1.0)
    assert F.moderate(F.constant(0))(0) == pytest.approx(0.0)
    assert F.moderate(F.constant(0.5))(0) == pytest.approx(0.5)


def test_linear_and_bounded_linear():
    f = F.linear(1, -1)
    assert f(-2) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(3) == 1

    b = F.bounded_linear(2, 3)
    assert b(1) == 0
    assert b(2.5) == pytest.approx(0.5)
    assert b(4) == 1

    bi = F.bounded_linear(2, 3, inverse=True)
    assert bi(2) == pytest.approx(1.0)
    assert bi(3) == pytest.approx(0.0)
    assert bi(2.5) == pytest.approx(0.5)

    with pytest.raises(AssertionError):
        F.bounded_linear(3, 2)


def test_singleton_rectangular_R_and_S():
    s = F.singleton(2)
    assert s(2) == 1
    assert s(1) == 0

    r = F.rectangular(1, 3, c_m=0.8)
    assert r(2) == pytest.approx(0.8)
    assert r(0) == 0
    assert r(4) == 0

    rising = F.R(0, 2)
    assert rising(-1) == 0
    assert rising(1) == pytest.approx(0.5)
    assert rising(3) == 1

    falling = F.S(0, 2)
    assert falling(-1) == 1
    assert falling(1) == pytest.approx(0.5)
    assert falling(3) == 0


def test_triangular_trapezoid_and_gauss():
    t = F.triangular(0, 2)
    assert t(1) == pytest.approx(1.0)
    assert t(0.5) == pytest.approx(0.5)
    assert t(0) == 0
    assert t(2) == pytest.approx(0.0)

    tr = F.trapezoid(0, 1, 2, 3)
    assert tr(1.5) == 1
    assert tr(0.5) == pytest.approx(0.5)
    assert tr(2.5) == pytest.approx(0.5)
    assert tr(-1) == 0
    assert tr(4) == 0

    g = F.gauss(0, 1)
    assert g(0) == pytest.approx(1.0)
    assert g(1) == pytest.approx(math.exp(-1))
    assert F.gauss(0, 1, c_m=0.5)(0) == pytest.approx(0.5)
    with pytest.raises(AssertionError):
        F.gauss(0, 0)


def test_sigmoid_variants_and_bounded_exponential():
    assert F.sigmoid(1, 1, 0)(0) == pytest.approx(0.5)

    bs = F.bounded_sigmoid(0, 1)
    assert bs(0) == pytest.approx(0.1)
    assert bs(1) == pytest.approx(0.9)
    assert F.bounded_sigmoid(0, 1, inverse=True)(0) == pytest.approx(0.9)

    ss = F.simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(20) > 0.98

    tri = F.triangular_sigmoid(2, 4)
    assert tri(2) == pytest.approx(0.1)
    assert tri(3) == pytest.approx(0.9)

    be = F.bounded_exponential(0.1, 1)
    assert be(0) == pytest.approx(0.0)
    assert be(10) == pytest.approx(1 - math.exp(-1))
    assert be(10000) == 1  # exp overflow is caught and clipped to the limit
