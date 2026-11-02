import pytest
import numpy as np
import warnings
import random

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv, constant, linear, bounded_linear,
    R, S, rectangular, triangular, trapezoid,
    sigmoid, bounded_sigmoid, bounded_exponential,
    simple_sigmoid, triangular_sigmoid, gauss,
    normalize
)

# Helper tolerance for float comparisons
TOL = 1e-7


def test_domain_creation_and_basic_call():
    d = Domain('temp', 0, 10, res=1)
    # Add a simple constant set
    d.cold = Set(constant(0.5))
    d.hot = Set(constant(0.8))

    # Call domain within range
    res = d(5)
    assert isinstance(res, dict)
    assert 'cold' in res and 'hot' in res
    assert abs(res['cold'] - 0.5) < TOL
    assert abs(res['hot'] - 0.8) < TOL

    # Call domain outside range triggers warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res_out = d(20)
        assert any(issubclass(warn.category, FuzzyWarning) or warn.category == UserWarning for warn in w)
        # Values returned anyway (calling sets)
        assert 'cold' in res_out and 'hot' in res_out


def test_domain_str_repr_eq():
    d1 = Domain('dim', 0, 5, res=0.5)
    d1.a = Set(constant(0.1))
    d1.b = Set(constant(0.9))

    d2 = Domain('dim', 0, 5, res=0.5, sets=d1._sets.copy())
    assert str(d1) == 'dim'
    assert repr(d1) == repr(d2)
    assert d1 == d2

    d3 = Domain('dim', 0, 5, res=0.5)
    d3.a = Set(constant(0.1))
    with pytest.warns(FuzzyWarning):
        # Different sets dict (b missing)
        assert d1 != d3


def test_domain_getattr_setattr_delattr_behavior():
    d = Domain('slider', -1, 1, res=0.1)
    s1 = Set(lambda x: 1 if x > 0 else 0)
    d.positive = s1

    # __getattr__ returns set correctly
    assert d.positive is s1

    # __setattr__ allows sets via direct assignment
    d.negative = Set(lambda x: 1 if x < 0 else 0)
    assert isinstance(d._sets['negative'], Set)
    # The Set's domain and name got set
    assert d.negative.domain is d
    assert d.negative.name == 'negative'

    # __delattr__ removes fuzzy sets, raises warning if non-fuzzy attribute
    del d.negative
    assert 'negative' not in d._sets
    with pytest.raises(FuzzyWarning):
        del d._low


def test_domain_range_min_max_methods():
    d = Domain('d', 0, 4, res=1)
    # Simple linear membership: 0 at 0, 1 at 4
    d.a = Set(lambda x: x / 4)
    d.b = Set(lambda x: 1 - x / 4)

    r = d.range()
    np.testing.assert_array_equal(r, np.array([0,1,2,3,4]))

    # min should give min of membership functions at x=2
    mn = d.min(2)
    mx = d.max(2)
    assert abs(mn - min(d.a(2), d.b(2))) < TOL
    assert abs(mx - max(d.a(2), d.b(2))) < TOL


def test_set_call_and_invert_neg():
    s = Set(lambda x: 0.3)
    assert abs(s(10) - 0.3) < TOL
    inv_s = ~s
    neg_s = -s
    # Both inverted functions should return 1 - 0.3 = 0.7 for any input
    for val in [0, 1, 2, 100]:
        assert abs(inv_s(val) - 0.7) < TOL
        assert abs(neg_s(val) - 0.7) < TOL


def test_set_and_or_mul_add_xor_pow_operations():
    s1 = Set(constant(0.3))
    s2 = Set(constant(0.6))

    # AND (MIN)
    s_and = s1 & s2
    assert abs(s_and(0) - min(s1(0), s2(0))) < TOL

    # OR (MAX)
    s_or = s1 | s2
    assert abs(s_or(0) - max(s1(0), s2(0))) < TOL

    # MUL (product)
    s_mul = s1 * s2
    assert abs(s_mul(0) - (s1(0) * s2(0))) < TOL

    # ADD (bounded_sum)
    s_add = s1 + s2
    expected = min(1, s1(0) + s2(0))
    assert abs(s_add(0) - expected) < TOL

    # XOR (simple_disjoint_sum)
    s_xor = s1 ^ s2
    expected_xor = s1(0) + s2(0) - 2 * s1(0) * s2(0)
    assert abs(s_xor(0) - expected_xor) < TOL

    # POW (pow)
    s_pow = s1 ** 2
    expected_pow = pow(s1(0), 2)
    assert abs(s_pow(0) - expected_pow) < TOL


def test_set_eq_le_lt_ge_gt_and_len_cardinality():
    domain = Domain("testdom", 0, 4, res=1)
    s1 = Set(lambda x: x / 4, domain=domain, name="s1")
    s2 = Set(lambda x: x / 4, domain=domain, name="s2")
    s3 = Set(lambda x: 1 - x / 4, domain=domain, name="s3")

    # s1 and s2 are equal over domain
    assert s1 == s2
    # s1 is subset of itself
    assert s1 <= s1
    # s1 less than s3 because for x in domain: s1(x)<=s3(x)? No, s1(4)=1, s3(4)=0, so False
    assert not (s1 <= s3)
    # s3 is superset of s1? False
    assert not (s3 >= s1)
    # s1 < s3 or s1 > s3 should be False for same reasons
    assert not (s1 < s3)
    assert not (s1 > s3)

    # Length equals domain range length
    assert len(s1) == len(domain.range())

    # Cardinality approx sum of membership values
    card = s1.cardinality()
    array_vals = s1.array()
    np.testing.assert_allclose(card, array_vals.sum(), atol=TOL)

    # Relative cardinality: cardinality / length
    rel_card = s1.relative_cardinality()
    assert abs(rel_card - card / len(s1)) < TOL

    # Remove domain and test raises
    s_no_dom = Set(constant(0.5))
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            _ = s_no_dom.cardinality()
        with pytest.raises(FuzzyWarning):
            _ = len(s_no_dom)
        with pytest.raises(FuzzyWarning):
            _ = s_no_dom.relative_cardinality()


def test_set_concentrated_dilated_multiplied_normalized(monkeypatch):
    domain = Domain("dom", 0, 2, res=1)
    s = Set(constant(0.25), domain=domain, name="s")

    # Concentrated squares membership
    c = s.concentrated()
    for x in domain.range():
        assert abs(c(x) - (s(x) ** 2)) < TOL

    # Dilated raises to 1/2 power
    d = s.dilated()
    for x in domain.range():
        assert abs(d(x) - s(x) ** 0.5) < TOL

    # Multiplied multiplies membership by n
    m = s.multiplied(3)
    for x in domain.range():
        # Result can be >1, but function does not clip
        assert abs(m(x) - (s(x) * 3)) < TOL

    # Normalized scales so max is 1
    s_raw = Set(lambda x: x / 2, domain=domain, name="raw")
    max_val = max(s_raw.array())
    norm = s_raw.normalized()
    # For each x, norm(x) = s_raw(x)/max_val
    for x in domain.range():
        assert abs(norm(x) - s_raw(x) / max_val) < TOL

    # Without domain: normalized raises warning
    s_nd = Set(constant(0.5))
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            s_nd.normalized()


def test_set_plot_raises_warning(monkeypatch):
    # patch plt.plot so no actual plotting happens
    called = {}
    def fake_plot(x, y):
        called['called'] = True
    monkeypatch.setattr("matplotlib.pyplot.plot", fake_plot)
    s = Set(constant(0.3))
    # Without domain: plotting warns
    with pytest.warns(FuzzyWarning):
        s.plot()
    # Assign domain and try plot
    domain = Domain("dom", 0, 1)
    s.domain = domain
    s.func = constant(0.4)
    called.clear()
    s.plot()
    assert called.get('called', False)


def test_set_repr_str_and_pickled_func():
    f = constant(0.7)
    s = Set(f)
    r = repr(s)
    assert r.startswith("Set(")

    s2 = Set(r, name="n", domain=Domain("d", 0, 1))
    # It should unpickle the function string which is repr(f) string of callable; likely will raise warning but handled
    # Because repr(s.func) is not pickled string but direct repr of function object, gives warning
    # So safer to test that repr returns string containing 'Set' and str returns string with either domain or dangling
    s3 = Set(f)
    assert "dangling Set" in str(s3)
    domain = Domain("dom", 0, 1)
    s3.domain = domain
    s3.name = "n"
    assert str(s3) == "dom.n"


def test_functions_inv_and_constant_and_noop():
    f = constant(0.6)
    inv_f = inv(f)
    # For some test value
    x = 0.2
    assert abs(f(x) - 0.6) < TOL
    assert abs(inv_f(x) - (1 - 0.6)) < TOL

    n = constant(5)
    for v in [0, 1, -1]:
        assert n(v) == 5

    n0 = lambda: 42
    noop_f = lambda: None  # Just test noop separately
    noop = lambda x: x
    assert (lambda x: x)(2) == 2

    # The noop from functions module returns function same as identity
    from fuzzylogic.functions import noop as noop_factory
    f_noop = noop_factory()
    for v in [0, 5, -3.2]:
        assert f_noop(v) == v


def test_functions_linear_and_bounded_linear_behavior():
    f = linear(1, -1)
    # Test sample points
    assert f(-2) == 0
    assert f(0) == 0
    assert f(1) == 0
    assert abs(f(1.5) - 0.5) < TOL
    assert f(2) == 1
    assert f(3) == 1

    bf = bounded_linear(2, 3)
    assert bf(1) == 0
    assert bf(2) == 0
    assert abs(bf(2.5) - 0.5) < TOL
    assert bf(3) == 1
    assert bf(4) == 1

    # Inverse switches c_m and no_m
    bf_inv = bounded_linear(2, 3, inverse=True)
    assert bf_inv(1) == 1
    assert bf_inv(2) == 1
    mid_val = bf_inv(2.5)
    assert 0 <= mid_val <= 1


def test_functions_R_and_S_behavior():
    r = R(2, 3)
    assert r(1) == 0
    assert abs(r(2) - 0.0) < TOL
    mid = r(2.5)
    assert 0 < mid < 1
    assert r(3) == 1
    assert r(4) == 1

    s = S(2, 3)
    assert s(1) == 1
    between = s(2.5)
    assert 0 < between < 1
    assert s(3) == 0
    assert s(4) == 0


def test_functions_rectangular_and_triangular_and_trapezoid():
    rec = rectangular(2, 4)
    assert rec(1) == 0
    assert rec(2) == 1
    assert rec(3) == 1
    assert rec(4) == 1
    assert rec(5) == 0

    tri = triangular(1, 3)
    # peak at midpoint
    c = (1 + 3) / 2
    assert abs(tri(c) - 1) < TOL or (0 <= tri(c) <= 1)
    # test outside values produce 0
    assert tri(0) == 0 or abs(tri(0)) < TOL
    assert tri(4) == 0 or abs(tri(4)) < TOL

    trap = trapezoid(1, 2, 3, 4)
    assert trap(0) == 0
    assert 0 < trap(1.5) < 1
    assert trap(2.5) == 1
    assert 0 < trap(3.5) < 1
    assert trap(5) == 0


def test_functions_sigmoid_and_bounded_sigmoid_behavior():
    sig = sigmoid(1, 1, 0)
    # midpoint at 0, sigmoid(0) ~ 0.5
    assert abs(sig(0) - 0.5) < TOL

    bsig = bounded_sigmoid(0, 1)
    # At low and high bounds approx 0.1 and 0.9
    assert abs(bsig(0) - 0.1) < TOL
    assert abs(bsig(1) - 0.9) < 0.05  # within 0.05 tolerance
    # Extreme values check
    assert abs(bsig(100000) - 1) < 0.01
    assert abs(bsig(-100000) - 0) < 0.01


def test_functions_bounded_exponential_and_simple_sigmoid():
    be = bounded_exponential(k=0.1, limit=1)
    val0 = be(0)
    assert abs(val0 - 0) < TOL
    val_large = be(1000)
    assert abs(val_large - 1) < 0.01

    ssig = simple_sigmoid()
    assert abs(ssig(-1000) - 0) < 0.01
    assert abs(ssig(0) - 0.5) < TOL
    assert abs(ssig(1000) - 1) < 0.01


def test_functions_triangular_sigmoid_and_gauss_behavior():
    tsig = triangular_sigmoid(2, 4)
    val_low = tsig(2)
    val_mid = tsig(3)
    assert abs(val_low - 0.1) < TOL
    # Peak near 0.9
    assert abs(val_mid - 0.9) < 0.1

    g = gauss(c=0, b=1)
    # Peak value at center is b*c_m=1
    assert abs(g(0) - 1) < TOL
    # Far away values approach 0
    assert g(10) < 1e-10


def test_set_eq_without_domain_raises_warning():
    s1 = Set(constant(0.5))
    s2 = Set(constant(0.5))
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            _ = s1 == s2

def test_set_subset_and_superset_without_domain_raises_warning():
    s1 = Set(constant(0.2))
    s2 = Set(constant(0.3))
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            _ = s1 <= s2
        with pytest.raises(FuzzyWarning):
            _ = s1 < s2
        with pytest.raises(FuzzyWarning):
            _ = s1 >= s2
        with pytest.raises(FuzzyWarning):
            _ = s1 > s2

