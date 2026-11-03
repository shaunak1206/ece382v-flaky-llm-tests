import math
import pickle
import warnings

import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv,
    noop,
    constant,
    alpha,
    normalize,
    moderate,
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
    assert list(d.range()) == [0, 1, 2]
    assert "Domain('temp', 0, 0" not in repr(d)
    assert "Domain('temp', 0, 2" in repr(d)


def test_domain_rejects_invalid_bounds_and_resolution():
    with pytest.raises(AssertionError):
        Domain("bad", 1, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 2, 1)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 1, res=0)


def test_domain_set_assignment_and_call_returns_memberships():
    d = Domain("x", 0, 10, res=1)
    d.low = constant(0.2)
    d.high = constant(0.8)
    assert isinstance(d.low, Set)
    assert d.low.domain is d
    assert d.low.name == "low"
    assert d(5) == {"low": 0.2, "high": 0.8}


def test_domain_getattr_delete_and_warning_on_regular_attr_delete():
    d = Domain("x", 0, 1)
    d.a = constant(0.3)
    assert d.a(0) == 0.3
    del d.a
    with pytest.raises(AttributeError):
        _ = d.a
    with pytest.raises(FuzzyWarning):
        del d._name


def test_domain_warns_when_called_outside_bounds():
    d = Domain("x", 0, 1)
    d.a = constant(1)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert d(2) == {"a": 1}
    assert any("outside of domain" in str(w.message) for w in rec)


def test_domain_min_max_and_equality():
    d1 = Domain("x", 0, 2, res=1)
    d2 = Domain("x", 0, 2, res=1)
    d1.a = singleton(1)
    d1.b = constant(0.5)
    d2.a = singleton(1)
    d2.b = constant(0.5)
    assert d1 == d2
    assert d1.min(1) == 0.5
    assert d1.max(1) == 1


def test_set_basic_operations_and_call():
    a = Set(constant(0.25))
    b = Set(constant(0.75))
    assert a(0) == 0.25
    assert (~a)(0) == 0.75
    assert (-a)(0) == 0.75
    assert (a & b)(0) == 0.25
    assert (a | b)(0) == 0.75
    assert (a * b)(0) == 0.1875
    assert (a + b)(0) == 1.0
    assert (a ^ b)(0) == 1.0


def test_set_pow_concentrated_dilated_and_multiplied():
    s = Set(constant(0.5))
    assert (s ** 2)(0) == 0.25
    assert s.concentrated()(0) == 0.25
    assert s.multiplied(3)(0) == 1.5
    assert s.dilated()(0) == pytest.approx(math.sqrt(0.5))


def test_set_assignment_via_init_with_name_and_domain():
    d = Domain("x", 0, 1)
    s = Set(constant(1), name="hot", domain=d)
    assert d.hot is s
    assert s.domain is d
    assert s.name == "hot"


def test_set_len_cardinality_and_relative_cardinality():
    d = Domain("x", 0, 2, res=1)
    s = Set(constant(0.5), domain=d)
    assert len(s) == 3
    assert s.cardinality() == 1.5
    assert s.relative_cardinality() == pytest.approx(0.5)


def test_set_comparisons_and_equality_require_domains():
    d = Domain("x", 0, 2, res=1)
    a = Set(constant(0.5), domain=d)
    b = Set(constant(0.5), domain=d)
    c = Set(constant(0.25), domain=d)
    assert a == b
    assert c <= a
    assert c < a
    assert a >= c
    assert a > c
    with pytest.raises(FuzzyWarning):
        _ = Set(constant(1)) == Set(constant(1))
    with pytest.raises(FuzzyWarning):
        _ = Set(constant(1)) <= a


def test_set_array_plot_and_normalized(monkeypatch):
    d = Domain("x", 0, 2, res=1)
    s = Set(lambda x: x / 2, domain=d)
    assert list(s.array()) == [0.0, 0.5, 1.0]

    calls = []

    def fake_plot(r, v):
        calls.append((list(r), list(v)))

    monkeypatch.setattr("fuzzylogic.classes.plt.plot", fake_plot)
    s.plot()
    assert calls == [([0, 1, 2], [0.0, 0.5, 1.0])]

    n = s.normalized()
    assert n(0) == 0.0
    assert n(1) == 0.5
    assert n(2) == 1.0


def test_set_str_and_repr_for_dangling_and_assigned_sets():
    s = Set(constant(1))
    assert "dangling Set(" in str(s)
    assert "Set(" in repr(s)

    d = Domain("x", 0, 1)
    d.a = constant(1)
    assert str(d.a) == "x.a"


def test_functions_inv_noop_constant_normalize_and_moderate():
    f = constant(3)
    assert f(10) == 3
    assert noop()(7) == 7
    assert inv(constant(0.25))(0) == 0.75
    assert normalize(2, constant(1))(0) == 0.5
    assert moderate(constant(0))(0) == pytest.approx(0.0)
    assert moderate(constant(1))(0) == pytest.approx(1.0)


def test_functions_membership_shapes():
    assert singleton(2)(2) == 1
    assert singleton(2)(3) == 0

    lf = linear(1, -1)
    assert lf(0) == 0
    assert lf(2) == 1

    rf = R(2, 4)
    sf = S(2, 4)
    assert rf(1) == 0
    assert rf(3) == 0.5
    assert rf(5) == 1
    assert sf(1) == 1
    assert sf(3) == 0.5
    assert sf(5) == 0

    rect = rectangular(1, 3)
    tri = triangular(0, 2)
    trap = trapezoid(0, 1, 2, 3)
    assert rect(0) == 0
    assert rect(2) == 1
    assert tri(1) == 1
    assert tri(0) == 0
    assert trap(0.5) == pytest.approx(0.5)
    assert trap(1.5) == 1
    assert trap(2.5) == pytest.approx(0.5)


def test_functions_sigmoids_exponential_and_gauss():
    sig = sigmoid(1, 1, 0)
    assert sig(0) == pytest.approx(0.5)
    bsig = bounded_sigmoid(0, 1)
    assert bsig(0) == pytest.approx(0.1, rel=1e-7)
    assert round(bsig(1), 1) == 0.9

    bexp = bounded_exponential(k=1, limit=2)
    assert bexp(0) == 0
    assert bexp(1000) == 2

    ss = simple_sigmoid()
    assert ss(0) == pytest.approx(0.5)
    assert ss(1000) == pytest.approx(1.0, rel=1e-6)

    ts = triangular_sigmoid(2, 4)
    assert ts(2) == pytest.approx(0.1, rel=1e-7)
    assert round(ts(3), 2) == 0.9

    g = gauss(0, 1)
    assert g(0) == 1
    assert g(1) == pytest.approx(math.exp(-1))


def test_pickled_function_string_is_accepted_in_set_constructor():
    f = constant(0.4)
    dumped = pickle.dumps(f)
    s = Set(dumped)
    assert callable(s.func)
    assert s(123) == 0.4
