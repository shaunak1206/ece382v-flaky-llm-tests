import pytest
import numpy as np
import random

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv, normalize, noop, constant, alpha, linear, bounded_linear, R, S,
    rectangular, triangular, trapezoid, sigmoid, bounded_sigmoid,
    bounded_exponential, simple_sigmoid, triangular_sigmoid, gauss
)

def test_domain_init_and_call_basic():
    # Seed to ensure deterministic behavior if any randomness used internally
    random.seed(12345)
    dom = Domain('testdomain', 0, 10, res=1)
    # Add sets: constant functions
    dom.low = Set(constant(0))
    dom.high = Set(constant(1))
    # Check call returns correct membership dict with keys 'low' and 'high'
    ret = dom(5)
    assert isinstance(ret, dict)
    assert set(ret.keys()) == {'low', 'high'}
    assert abs(ret['low'] - 0) < 1e-9
    assert abs(ret['high'] - 1) < 1e-9

def test_domain_call_warns_out_of_range(monkeypatch):
    dom = Domain('test', 0, 1)
    dom.s = Set(constant(0.5))
    # monkeypatch warn to capture calls instead of printing
    called = {}
    def fake_warn(msg, *args, **kwargs):
        called['warned'] = msg
    monkeypatch.setattr("fuzzylogic.classes.warn", fake_warn)
    _ = dom(-1)  # out of range
    assert 'outside of domain' in called.get('warned', '')

def test_domain_eq_and_repr():
    dom1 = Domain('d', 0, 5, res=0.5)
    dom2 = Domain('d', 0, 5, res=0.5)
    # Add identical sets
    f = constant(0.3)
    dom1.a = Set(f)
    dom2.a = Set(f)
    assert dom1 == dom2
    r = repr(dom1)
    assert r.startswith("Domain(") and 'd' in r

def test_domain_getattr_and_setattr_errors():
    dom = Domain('d', 0, 1)
    dom.a = Set(constant(0.5))
    # existing set returns the Set instance
    assert isinstance(dom.a, Set)
    # accessing missing attribute raises AttributeError
    with pytest.raises(AttributeError):
        _ = dom.not_existing
    
    # Setting non-identifier name fails
    with pytest.raises(AssertionError):
        dom.__setattr__('not an id', constant(0))

def test_domain_delattr_and_warning(monkeypatch):
    dom = Domain('d', 0, 1)
    dom.a = Set(constant(0.3))
    assert 'a' in dom._sets
    del dom.a
    assert 'a' not in dom._sets
    # Deleting an attribute not in sets triggers FuzzyWarning
    with pytest.raises(FuzzyWarning):
        del dom.foo

def test_domain_range_includes_high():
    dom = Domain('d', 0, 1, res=0.5)
    r = dom.range()
    arr = np.array(list(r))
    # Must include high (1)
    assert np.isclose(arr[-1], 1)
    # Steps equal res
    np.testing.assert_allclose(np.diff(arr), 0.5)

def test_domain_min_max():
    # Build domain with multiple sets
    dom = Domain('d', 0, 10)
    dom.l = Set(constant(0))
    dom.h = Set(constant(1))
    dom.mid = Set(lambda x: 0.5)
    assert dom.min(5) == 0
    assert dom.max(5) == 1

def test_set_call_and_str_repr_and_name_domain_assignment():
    dom = Domain('d', 0, 10)
    f = constant(0.7)
    s = Set(f)
    # Initially no name or domain
    assert s.name is None
    assert s.domain is None
    # Assign to domain attribute
    dom.foo = s
    assert s.name == 'foo'
    assert s.domain == dom
    assert str(s) == 'd.foo'
    r = repr(s)
    assert r.startswith("Set(")
    assert abs(s(0) - 0.7) < 1e-9

def test_set_invert_and_neg_equivalence_and_and_or_operators():
    dom = Domain('d', 0, 1)
    dom.a = Set(constant(0.2))
    dom.b = Set(constant(0.6))
    # Invert returns new Set with function behaving as 1 - original
    not_a = ~dom.a
    neg_a = -dom.a
    for x in [0, 0.5, 1]:
        assert abs(not_a(x) - (1 - dom.a(x))) < 1e-9
        assert abs(neg_a(x) - (1 - dom.a(x))) < 1e-9
    # AND returns Set with min func
    c = dom.a & dom.b
    for x in [0, 0.5, 1]:
        assert abs(c(x) - min(dom.a(x), dom.b(x))) < 1e-9
    # OR returns Set with max func
    d = dom.a | dom.b
    for x in [0, 0.5, 1]:
        assert abs(d(x) - max(dom.a(x), dom.b(x))) < 1e-9
    # MULTIPLY returns Set with product func
    e = dom.a * dom.b
    for x in [0, 0.5, 1]:
        assert abs(e(x) - dom.a(x) * dom.b(x)) < 1e-9
    # ADD returns Set with bounded_sum
    f = dom.a + dom.b
    for x in [0, 0.5, 1]:
        val = min(dom.a(x) + dom.b(x), 1)
        assert abs(f(x) - val) < 1e-9
    # XOR returns Set with simple_disjoint_sum
    g = dom.a ^ dom.b
    for x in [0, 0.5, 1]:
        val = dom.a(x) + dom.b(x) - 2*dom.a(x)*dom.b(x)
        assert abs(g(x) - val) < 1e-9

def test_set_pow_concentrated_dilated_multiplied():
    dom = Domain('d', 0, 1)
    dom.s = Set(constant(0.5))
    # Power squares membership
    p = dom.s ** 2
    for x in [0, 0.5, 1]:
        assert abs(p(x) - 0.5**2) < 1e-9
    # concentrated returns square of func
    c = dom.s.concentrated()
    for x in [0, 0.5, 1]:
        assert abs(c(x) - (0.5**2)) < 1e-9
    # dilated returns sqrt of func
    d = dom.s.dilated()
    for x in [0, 0.5, 1]:
        assert abs(d(x) - (0.5 ** 0.5)) < 1e-9
    # multiplied scales values
    m = dom.s.multiplied(3)
    for x in [0, 0.5, 1]:
        assert abs(m(x) - 0.5*3) < 1e-9

def test_set_eq_with_and_without_domain_raises_warning():
    s1 = Set(constant(0.5))
    s2 = Set(constant(0.5))
    # Without domains set, equality raises warning
    with pytest.raises(FuzzyWarning):
        _ = s1 == s2
    # With domain, compares arrays
    dom = Domain('d', 0, 1, res=0.5)
    dom.a = Set(constant(0.5))
    dom.b = Set(constant(0.5))
    assert dom.a == dom.b

def test_set_comparison_ops_for_subsets_supersets():
    dom = Domain('d', 0, 2, res=1)
    # Define sets s1 <= s2, s1 < s2, s2 > s1, s2 >= s1 properly with simple funcs
    dom.s1 = Set(constant(0.2))
    dom.s2 = Set(constant(0.5))
    assert dom.s1 <= dom.s2
    assert dom.s1 < dom.s2
    assert dom.s2 >= dom.s1
    assert dom.s2 > dom.s1
    # Matches the reverse False case
    dom.s3 = Set(constant(0.5))
    assert not (dom.s3 < dom.s1)
    assert not (dom.s3 > dom.s2)

def test_set_len_cardinality_relative_cardinality_and_array():
    dom = Domain('d', 0, 2, res=1)
    dom.s = Set(constant(0.5))
    arr = dom.s.array()
    assert len(arr) == 3
    assert isinstance(arr, np.ndarray)
    # cardinality is sum of array values
    assert np.isclose(dom.s.cardinality(), sum(arr))
    # relative_cardinality equals cardinality / len
    assert np.isclose(dom.s.relative_cardinality(), dom.s.cardinality() / len(dom.s))
    # if domain is missing, methods raise warning
    s = Set(constant(0.5))
    with pytest.raises(FuzzyWarning):
        _ = len(s)
    with pytest.raises(FuzzyWarning):
        _ = s.cardinality()
    with pytest.raises(FuzzyWarning):
        _ = s.relative_cardinality()
    with pytest.raises(FuzzyWarning):
        _ = s.array()

def test_set_normalized_returns_expected():
    dom = Domain('d', 0, 2, res=1)
    # Create set with max 0.5 membership value function
    dom.s = Set(lambda x: 0.5)
    norm_s = dom.s.normalized()
    # normalized set membership is original divided by max (0.5)
    for x in [0, 1, 2]:
        assert abs(norm_s(x) - 1.0) < 1e-9

def test_functions_inv_and_noop_and_constant():
    f = constant(0.5)
    inv_f = inv(f)
    for x in [0, 0.5, 1]:
        assert abs(inv_f(x) - (1 - 0.5)) < 1e-9
    nop = noop()
    assert nop(42) == 42

def test_functions_linear_bounded_linear_R_S():
    lin = linear(1, 0)
    assert lin(0) == 0
    assert lin(0.5) == 0.5
    assert lin(2) == 1  # clipped
    
    blin = bounded_linear(0, 2)
    assert blin(0) == 0
    assert abs(blin(1) - 0.5) < 1e-9
    assert blin(2) == 1
    
    r = R(0, 1)
    assert r(-1) == 0
    assert abs(r(0.5) - 0.5) < 1e-9
    assert r(2) == 1
    
    s = S(0, 1)
    assert s(-1) == 1
    assert abs(s(0.5) - 0.5) < 1e-9
    assert s(2) == 0

def test_functions_rectangular_triangular_trapezoid():
    rect = rectangular(1, 3)
    assert rect(0) == 0
    assert rect(2) == 1
    assert rect(4) == 0

    tri = triangular(1, 3)
    peak = (1+3)/2
    assert tri(1) == 0
    assert tri(peak) == 1
    assert tri(3) == 0

    trap = trapezoid(1, 2, 3, 4)
    assert trap(0) == 0
    assert trap(1.5) > 0
    assert trap(2.5) == 1
    assert trap(4.5) == 0

def test_functions_sigmoid_bounded_sigmoid_simple_sigmoid():
    sig = sigmoid(1, 1, 0)
    val0 = sig(0)
    val5 = sig(5)
    # sig(0) is L/2 or close
    assert 0 < val0 < 1
    assert val5 > val0

    bsig = bounded_sigmoid(0, 1)
    assert abs(bsig(0) - 0.1) < 0.05
    assert abs(bsig(1) - 0.9) < 0.05
    # For large args, close to 1 and 0
    assert bsig(10000) > 0.99
    assert bsig(-10000) < 0.01

    ssig = simple_sigmoid()
    assert abs(ssig(0) - 0.5) < 1e-9
    assert 0 <= ssig(-1000) <= 0.01
    assert 0.99 <= ssig(1000) <= 1

def test_functions_triangular_sigmoid_and_gauss():
    tri_sig = triangular_sigmoid(1, 3)
    assert abs(tri_sig(1) - 0.1) < 0.05
    assert abs(tri_sig(2) - 0.9) < 0.1

    g = gauss(0, 5)
    assert g(0) == 0  # center is 0 but steepness 5 so likely small value
    assert g(0) >= 0 and g(0) <= 1
    # peak at c=0 with c_m=1
    assert g(0) <= 1

def test_alpha_normalize_moderate():
    base_f = constant(0.5)
    alf = alpha(floor=0.3, ceiling=0.7, func=base_f)
    # Base returns 0.5, so alpha returns same
    assert abs(alf(0) - 0.5) < 1e-9
    # Test clipping works
    alf2 = alpha(floor=0.6, ceiling=0.7, func=constant(0.5))
    assert abs(alf2(0) - 0.6) < 1e-9  # clipped up to floor_clip
    norm = normalize(0.5, base_f)
    assert abs(norm(0) - 1.0) < 1e-9
    mod = moderate(base_f)
    # moderate moves value toward 0.5; constant 0.5 remains 0.5
    assert abs(mod(0) - 0.5) < 1e-9

def test_bounded_exponential_behavior():
    bexp = bounded_exponential(k=0.1, limit=1)
    assert 0 <= bexp(0) <= 1
    # Large x approaches limit
    val_large = bexp(1000)
    assert abs(val_large - 1) < 1e-3
    # Small x near zero
    val_zero = bexp(0)
    assert 0 <= val_zero <= 1

# Test that Set.plot() does not raise with domain, and raises FuzzyWarning without domain
def test_set_plot(monkeypatch):
    dom = Domain('d', 0, 1)
    dom.s = Set(constant(0.5))
    # monkeypatch matplotlib.pyplot.plot to avoid real plotting
    called = {}
    def fake_plot(x, y):
        called['called'] = True
        assert len(x) == len(y)
    monkeypatch.setattr("matplotlib.pyplot.plot", fake_plot)
    dom.s.plot()
    assert called.get('called') is True

    s = Set(constant(0.5))
    with pytest.raises(FuzzyWarning):
        s.plot()
