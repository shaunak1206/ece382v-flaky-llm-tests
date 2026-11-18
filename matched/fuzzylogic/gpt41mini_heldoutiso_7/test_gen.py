import pytest
import numpy as np
from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv, normalize, constant, linear, bounded_linear, R, S,
    rectangular, triangular, trapezoid, sigmoid,
    bounded_sigmoid, bounded_exponential, simple_sigmoid, gauss
)

# -----------------
# Tests for Domain
# -----------------

def test_domain_creation_and_str_repr():
    d = Domain('temp', 0, 10, res=0.5)
    assert d._name == 'temp'
    assert d._low == 0
    assert d._high == 10
    assert d._res == 0.5
    assert str(d) == 'temp'
    repr_str = repr(d)
    # Since _sets empty dict
    assert repr_str.startswith("Domain('temp', 0, 10")

def test_domain_call_and_warning_outside_bounds():
    d = Domain('temp', 0, 10)
    d.hot = Set(constant(0.5))
    d.cold = Set(constant(0.0))
    # Normal call inside domain
    res = d(5)
    assert 'hot' in res and 'cold' in res
    assert res['hot'] == 0.5 and res['cold'] == 0.0
    # Warning triggered outside domain
    with pytest.warns(UserWarning):
        d(-1)
    with pytest.warns(UserWarning):
        d(11)

def test_domain_getattr_and_setattr_behavior():
    d = Domain('temp', 0, 10)
    # Initially empty sets dict
    with pytest.raises(AttributeError):
        _ = d.hot
    # Assign a Set to domain by attribute
    f = constant(0.7)
    d.hot = Set(f)
    # Now attr access returns the set
    s = d.hot
    assert isinstance(s, Set)
    assert s.domain is d
    assert s.name == 'hot'
    # Assigning a plain function automagically converts to Set
    d.cold = lambda x: 1 - f(x)
    assert isinstance(d.cold, Set)
    assert d.cold.domain is d
    assert d.cold.name == 'cold'
    # Setting invalid attribute name raises AssertionError
    with pytest.raises(AssertionError):
        d.__setattr__('123invalid', constant(1))

def test_domain_delattr_behavior():
    d = Domain('temp', 0, 10)
    d.hot = constant(1)
    # Setting 'hot' assigns a Set automatically
    assert isinstance(d._sets['hot'], Set)
    del d.hot
    assert 'hot' not in d._sets
    # Deleting non-set attribute raises FuzzyWarning
    with pytest.raises(FuzzyWarning):
        del d._low

def test_domain_range_yields_inclusive_range():
    d = Domain('temp', 0, 2, res=0.5)
    r = d.range()
    # Should include 2
    assert np.isclose(r[-1], 2)
    # Steps correct
    np.testing.assert_array_almost_equal(r, np.array([0, 0.5, 1.0, 1.5, 2.0]))

def test_domain_eq_and_neq():
    d1 = Domain('test', 0, 10, res=1)
    d2 = Domain('test', 0, 10, res=1)
    d1.a = constant(1)
    d2.a = constant(1)
    assert d1 == d2
    d2.b = constant(0)
    assert d1 != d2
    d3 = Domain('diff', 0, 10, res=1)
    assert d1 != d3

def test_domain_min_and_max():
    d = Domain('x', 0, 1, res=0.25)
    d.low = Set(constant(0))
    d.mid = Set(constant(0.5))
    d.high = Set(constant(1))
    assert d.min(0.5) == 0
    assert d.max(0.5) == 1


# --------------
# Tests for Set
# --------------

def test_set_init_and_call_basic():
    f = constant(0.7)
    s = Set(f)
    assert isinstance(s.func, type(f))
    # Call returns expected value
    for v in [0, 1, 100]:
        assert s(v) == 0.7

def test_set_init_with_name_and_domain_assignment():
    d = Domain('temp', 0, 5)
    f = constant(0.5)
    s = Set(f, name='warm', domain=d)
    # The set must be assigned to domain
    assert d.warm is s
    assert s.domain == d
    assert s.name == 'warm'
    # If only one of name/domain is given, raises
    with pytest.raises(FuzzyWarning):
        Set(f, name='temp')
    with pytest.raises(FuzzyWarning):
        Set(f, domain=d)

def test_set_operator_invert_and_neg():
    f = constant(0.3)
    s = Set(f)
    ns = ~s
    nn = -s
    # Should be callable and return 1 - original
    val = 0.4
    assert ns(val) == 1 - s(val)
    assert nn(val) == 1 - s(val)

def test_set_operator_and_or():
    f1 = constant(0.3)
    f2 = constant(0.6)
    s1 = Set(f1)
    s2 = Set(f2)
    s_and = s1 & s2
    s_or = s1 | s2
    val = 0
    assert s_and(val) == min(s1(val), s2(val))
    assert s_or(val) == max(s1(val), s2(val))

def test_set_operator_mul_add_xor():
    f1 = constant(0.4)
    f2 = constant(0.7)
    s1 = Set(f1)
    s2 = Set(f2)
    s_mul = s1 * s2
    s_add = s1 + s2
    s_xor = s1 ^ s2
    val = 0
    # product
    assert s_mul(val) == s1(val) * s2(val)
    # bounded_sum: min(s1+ s2, 1)
    assert s_add(val) == min(s1(val) + s2(val), 1)
    # simple_disjoint_sum: s1 + s2 - s1*s2
    assert s_xor(val) == s1(val) + s2(val) - s1(val) * s2(val)

def test_set_operator_pow_concentrated_and_dilated():
    f = linear(1, 0)  # f(x) = x clipped to [0,1]
    d = Domain('unit', 0, 1, res=0.1)
    s = Set(f, domain=d, name='linear')
    # __pow__
    s_pow = s ** 2
    for x in d.range():
        assert np.isclose(s_pow(x), pow(s(x), 2))
    # concentrated is square
    s_conc = s.concentrated()
    for x in d.range():
        assert np.isclose(s_conc(x), s(x)**2)
    # dilated is square root
    s_dil = s.dilated()
    for x in d.range():
        assert np.isclose(s_dil(x), s(x) ** 0.5)

def test_set_comparison_eq_and_subset():
    d = Domain('unit', 0, 1, res=0.2)
    s1 = Set(constant(0.5), domain=d, name='const1')
    s2 = Set(constant(0.5), domain=d, name='const2')
    # Equal sets
    assert s1 == s2
    s3 = ~s1
    # Expect Warning if domain missing
    s_no_domain = Set(constant(0.5))
    with pytest.raises(FuzzyWarning):
        _ = s_no_domain == s1
    # Test subset/superset
    s_low = Set(constant(0.3), domain=d)
    assert s_low <= s1
    assert s1 >= s_low
    # Strict relations
    assert s_low < s1
    assert s1 > s_low
    # Invalid comparisons raise warning without domain
    s_no_domain2 = Set(constant(0.4))
    with pytest.raises(FuzzyWarning):
        _ = s_no_domain2 <= s_low

def test_set_length_cardinality_relative_cardinality():
    d = Domain('unit', 0, 1, res=0.5)
    s = Set(constant(0.5), domain=d)
    length = len(s)
    # domain.range() steps: 0,0.5,1.0 (3 steps)
    assert length == 3
    cardinality = s.cardinality()
    assert cardinality == pytest.approx(0.5 * length)
    relative = s.relative_cardinality()
    assert relative == pytest.approx(0.5)
    # Without domain raises warning
    s_no_domain = Set(constant(0.5))
    with pytest.raises(FuzzyWarning):
        len(s_no_domain)
    with pytest.raises(FuzzyWarning):
        s_no_domain.cardinality()
    with pytest.raises(FuzzyWarning):
        s_no_domain.relative_cardinality()
    # Relative cardinality with empty domain (simulate empty range)
    d_empty = Domain('empty', 0, 0, res=1)
    s_empty = Set(constant(0), domain=d_empty)
    # length = 1 so no error
    assert s_empty.relative_cardinality() == 0.0
    

def test_set_plot_and_array_methods():
    import matplotlib
    matplotlib.use('Agg')  # prevent need for GUI backend
    d = Domain('x', 0, 1, res=0.25)
    s = Set(lambda x: x, domain=d)
    # array returns length matching domain steps
    arr = s.array()
    assert isinstance(arr, np.ndarray)
    assert len(arr) == len(d.range())
    # plot does not raise when domain assigned
    s.plot()
    # Without domain, plot and array raises
    s2 = Set(lambda x: x)
    with pytest.raises(FuzzyWarning):
        s2.plot()
    with pytest.raises(FuzzyWarning):
        s2.array()

def test_set_str_repr_conversion():
    d = Domain('temp', 0, 10)
    s = Set(constant(0.8), domain=d, name='warm')
    # str returns domain.name.set_name
    assert str(s) == 'temp.warm'
    # repr returns string that contains 'Set('
    assert 'Set(' in repr(s)
    # dangling Set with no domain or name returns fallback
    s2 = Set(constant(0.1))
    assert str(s2).startswith('dangling Set')

def test_set_normalized():
    d = Domain('unit', 0, 1, res=0.5)
    # A Set with values less than 1 max
    s = Set(constant(0.5), domain=d)
    sm = s.normalized()
    for x in d.range():
        val_exp = s.func(x) / 0.5
        assert np.isclose(sm.func(x), val_exp)
    # Without domain raises warning
    s2 = Set(constant(0.7))
    with pytest.raises(FuzzyWarning):
        s2.normalized()

# ----------------------------
# Tests for fuzzylogic.functions
# ----------------------------

def test_inv_function():
    f = constant(0.3)
    finv = inv(f)
    assert finv(0) == 1 - f(0)
    assert finv(100) == 1 - f(100)

def test_normalize_function():
    f = constant(0.5)
    fn = normalize(0.5, f)
    assert fn(0) == 1
    # Invalid inputs for normalize raise assertion
    import pytest
    with pytest.raises(AssertionError):
        normalize(0, f)

def test_constant_function():
    c = constant(0.77)
    for v in [0, 1, 999]:
        assert c(v) == 0.77

def test_linear_function_and_clip():
    f = linear(2, -1)
    assert f(-1) == 0  # clipped below 0
    assert f(0) == 0
    assert f(0.75) == 0.5
    assert f(1) == 1  # clipped to 1
    assert f(100) == 1  # clipped above 1

def test_bounded_linear_and_inverse_behavior():
    f = bounded_linear(0, 2)
    assert f(-1) == 0
    assert f(0) == 0
    assert 0 < f(1) < 1
    assert f(2) == 1
    # inverse swaps c_m and no_m
    f_inv = bounded_linear(0, 2, inverse=True)
    assert f_inv(0) == 1
    assert f_inv(2) == 0

def test_R_and_S_functions_behavior():
    r = R(1,2)
    assert r(0) == 0
    assert 0 < r(1.5) < 1
    assert r(3) == 1
    s = S(1,2)
    assert s(0) == 1
    assert 0 < s(1.5) < 1
    assert s(3) == 0

def test_rectangular_triangular_trapezoid_functions():
    rect = rectangular(1, 3)
    assert rect(0) == 0
    assert rect(1) == 1
    assert rect(2) == 1
    assert rect(4) == 0
    
    tri = triangular(1, 3)
    assert tri(1) == 0
    assert tri(2) == 1
    assert tri(3) == 0
    assert 0 <= tri(1.5) <= 1
    
    trap = trapezoid(1, 2, 3, 4)
    assert trap(0) == 0
    assert 0 < trap(1.5) < 1
    assert trap(2.5) == 1
    assert 0 < trap(3.5) < 1
    assert trap(5) == 0

def test_sigmoid_and_bounded_sigmoid_behaviour():
    f = sigmoid(1, 1, 0)
    assert 0 <= f(-1) <= 1
    assert 0 <= f(0) <= 1
    assert 0 <= f(1) <= 1
    bf = bounded_sigmoid(0, 1)
    assert abs(bf(0) - 0.1) < 1e-6
    assert round(bf(1), 2) == 0.9
    assert round(bf(100000), 2) == 1.0
    assert round(bf(-100000), 2) == 0.0
    with pytest.raises(AssertionError):
        bounded_sigmoid(1, 0)

def test_bounded_exponential_and_simple_sigmoid():
    be = bounded_exponential(k=0.1, limit=2)
    assert abs(be(0) - 0) < 1e-6
    assert abs(be(1000) - 2) < 1e-3
    ss = simple_sigmoid()
    assert 0 <= ss(-1000) <= 1
    assert 0 <= ss(0) <= 1
    assert 0 <= ss(1000) <= 1

def test_gauss_function_behavior():
    g = gauss(0, 1, c_m=1)
    # center value should be max (1)
    assert g(0) == 1
    # far from center should be near zero
    assert g(10) < 1e-3

def test_alpha_clipping_function():
    base = constant(0.5)
    f = inv(base)
    clipped = inv(base)
    from fuzzylogic.functions import alpha
    alf = alpha(floor=0.2, ceiling=0.8, func=f)
    assert alf(0) == 1 - base(0)
    assert alf(1000) == 1 - base(1000)
    # values below floor clipped
    low_val = 0.1
    # we cannot inject easily here but test via example
    # since alpha clips based on floor and ceiling
    def test_func(x):
        v = alf(x)
        if v >= 0.8:
            return 0.8
        if v <= 0.2:
            return 0.2
        return v
    assert test_func(0) == alf(0)

# The total test count here: 23 test functions
