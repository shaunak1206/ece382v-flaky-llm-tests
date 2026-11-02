import pytest
import math
from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import inv, normalize, constant, linear, bounded_linear, R, S, rectangular, triangular, trapezoid, sigmoid, bounded_sigmoid, bounded_exponential, simple_sigmoid, singleton, gauss

# Helpers for tests
def always_half(_):
    return 0.5

def always_one(_):
    return 1.0

def always_zero(_):
    return 0.0

def create_simple_domain():
    d = Domain('test', 0, 10, res=1)
    d.low_set = Set(always_zero)
    d.mid_set = Set(always_half)
    d.high_set = Set(always_one)
    return d

@pytest.fixture
def domain():
    return create_simple_domain()

@pytest.fixture
def set1(domain):
    # simple linear increasing membership set
    s = Set(lambda x: x/10, name='linear_inc', domain=domain)
    return s

@pytest.fixture
def set2(domain):
    # simple linear decreasing membership set
    s = Set(lambda x: 1 - x/10, name='linear_dec', domain=domain)
    return s

def test_domain_basic_attributes(domain):
    assert domain._name == 'test'
    assert domain._low == 0
    assert domain._high == 10
    assert domain._res == 1
    assert isinstance(domain._sets, dict)
    assert 'low_set' in domain._sets
    assert 'mid_set' in domain._sets
    assert 'high_set' in domain._sets

def test_domain_call_membership(domain):
    mem = domain(5)
    assert isinstance(mem, dict)
    # low_set always returns 0, mid_set 0.5, high_set 1
    assert mem['low_set'] == 0
    assert mem['mid_set'] == 0.5
    assert mem['high_set'] == 1

def test_domain_call_outside_warn(domain):
    with pytest.warns(UserWarning):
        mem = domain(-1)
    with pytest.warns(UserWarning):
        mem = domain(11)

def test_domain_setattr_and_getattr(domain):
    # Assign a new Set via function
    domain.new_set = lambda x: 0.7
    assert 'new_set' in domain._sets
    retrieved = getattr(domain, 'new_set')
    assert isinstance(retrieved, Set)
    assert abs(retrieved(0) - 0.7) < 1e-8

    # Accessing missing attribute raises AttributeError
    with pytest.raises(AttributeError):
        _ = domain.no_such_set

def test_domain_delattr(domain):
    domain.temp_set = lambda x: 0.3
    assert 'temp_set' in domain._sets
    del domain.temp_set
    assert 'temp_set' not in domain._sets
    # Deleting non-set attribute raises FuzzyWarning
    with pytest.raises(FuzzyWarning):
        del domain._name

def test_domain_range(domain):
    r = domain.range()
    import numpy as np
    np.testing.assert_array_equal(r, np.arange(0, 10+1, 1))

def test_domain_min_max(domain, set1, set2):
    # domain has set1 and set2 (linear_inc and linear_dec)
    # min at point 5 should be min(0.5, 0.5) = 0.5
    min_val = domain.min(5)
    max_val = domain.max(5)
    assert math.isclose(min_val, 0.5)
    assert math.isclose(max_val, 0.5)

    # Add a constant one set
    domain.one_set = Set(constant(1))
    assert domain.min(3) <= domain.max(3)
    assert domain.max(3) == 1

def test_set_call_and_invert(domain, set1):
    assert isinstance(set1(5), float)
    # Inverted should produce 1 - original
    inv_set = ~set1
    assert math.isclose(inv_set(5), 1 - set1(5))

    # Negation is alias for invert
    neg_set = -set1
    assert math.isclose(neg_set(5), 1 - set1(5))

def test_set_and_or(domain, set1, set2):
    s_and = set1 & set2
    s_or = set1 | set2
    x = 4
    assert math.isclose(s_and(x), min(set1(x), set2(x)))
    assert math.isclose(s_or(x), max(set1(x), set2(x)))

def test_set_mul_add_xor(domain, set1, set2):
    def product(f, g):
        return f*g
    def bounded_sum(f, g):
        return min(f + g, 1)
    def simple_disjoint_sum(f, g):
        s = f + g - f * g
        return s if s <= 1 else 1

    s_mul = set1 * set2
    s_add = set1 + set2
    s_xor = set1 ^ set2
    x = 3.0
    # test product
    assert math.isclose(s_mul(x), set1(x) * set2(x))
    # test bounded sum
    assert math.isclose(s_add(x), min(set1(x) + set2(x), 1))
    # test simple disjoint sum
    expected_xor = set1(x) + set2(x) - set1(x)*set2(x)
    assert math.isclose(s_xor(x), expected_xor)

def test_set_pow(domain, set1):
    power = 2
    s_pow = set1 ** power
    x = 2
    # s_pow should be set1(x)**2
    assert math.isclose(s_pow(x), set1(x)**power)

def test_set_equality(domain):
    s1 = Set(lambda x: 0.5, name='const_half', domain=domain)
    s2 = Set(constant(0.5), name='const_half2', domain=domain)
    # They should be equal since they produce same arrays
    assert s1 == s2

    # Without domain raises FuzzyWarning
    s3 = Set(constant(0.5))
    s4 = Set(constant(0.5))
    with pytest.raises(FuzzyWarning):
        _ = s3 == s4

def test_set_comparisons(domain):
    s_lower = Set(lambda x: 0.5, name='half', domain=domain)
    s_upper = Set(lambda x: 0.8, name='eight', domain=domain)
    # s_lower <= s_upper
    assert s_lower <= s_upper
    assert s_lower < s_upper
    assert s_upper >= s_lower
    assert s_upper > s_lower

    # Without domains raises FuzzyWarning
    s3 = Set(constant(0.5))
    s4 = Set(constant(0.8))
    with pytest.raises(FuzzyWarning):
        _ = s3 <= s4

def test_set_len_cardinality_relative(domain):
    s = Set(lambda x: 0.5, name='half', domain=domain)
    length = len(s)
    assert length == len(domain.range())
    card = s.cardinality()
    assert math.isclose(card, 0.5 * length)
    rel = s.relative_cardinality()
    assert math.isclose(rel, 0.5)

    # No domain raises
    s2 = Set(lambda x: 0.5)
    with pytest.raises(FuzzyWarning):
        len(s2)
    with pytest.raises(FuzzyWarning):
        s2.cardinality()
    with pytest.raises(FuzzyWarning):
        s2.relative_cardinality()

def test_set_relative_cardinality_empty_domain():
    d = Domain('empty', 0, 0, res=1)
    s = Set(lambda x: 0.5, name='half', domain=d)
    with pytest.raises(FuzzyWarning):
        s.relative_cardinality()

def test_set_derived_methods(domain):
    s = Set(lambda x: 0.5 if x < 5 else 1.0, name='step', domain=domain)

    # concentrated squares membership
    c = s.concentrated()
    assert math.isclose(c(3), s(3)**2)

    # dilated sqrt
    d = s.dilated()
    assert math.isclose(d(3), s(3)**0.5)

    # multiplied multiplies
    m = s.multiplied(2)
    assert math.isclose(m(3), s(3) * 2)

def test_set_intensified_functionality():
    # Intensified set testing behavior by manual calls
    s = Set(lambda x: x, name='linear', domain=Domain('d', 0, 1))
    intensified = s.intensified()
    # x < 0.5 branch
    assert math.isclose(intensified(0.4), 2 * (s(0.4)**2))
    # x >= 0.5 branch
    # Expected: 1 - 2(1 - (x^2)) = 1 - 2 + 2x^2 = 2x^2 -1
    val = intensified(0.7)
    expected = 1 - 2*(1 - s(0.7)**2)
    assert math.isclose(val, expected)

def test_set_array_and_repr(domain):
    s = Set(lambda x: x/10, name='linear_inc', domain=domain)
    arr = s.array()
    import numpy as np
    assert arr.shape[0] == len(domain.range())
    # repr returns string containing "Set"
    assert 'Set' in repr(s)

def test_set_str_print(domain):
    s = Set(lambda x: x/10)
    assert s.name is None and s.domain is None
    s_str = str(s)
    assert 'dangling Set' in s_str

    s2 = Set(lambda x: x/10, name='linear_inc', domain=domain)
    s2_str = str(s2)
    assert 'test.linear_inc' == s2_str

def test_set_normalized(domain):
    # create a non-normalized set function returning max 0.5
    s = Set(lambda x: 0.5, name='half', domain=domain)
    norm = s.normalized()
    # norm.func should scale outputs so max is 1
    values = [norm.func(x) for x in domain.range()]
    assert all(abs(v - 1.0) < 1e-8 for v in values)

def test_functions_inv_and_normalize():
    f = lambda x: x / 10
    invf = inv(f)
    assert math.isclose(invf(5), 1 - f(5))

    normf = normalize(0.5, f)
    assert math.isclose(normf(5), f(5)/0.5)

def test_functions_constant_and_noop():
    c = constant(0.7)
    for v in [0, 10, -5]:
        assert c(v) == 0.7

    n = inv(constant(0.7))
    assert all(0.3 <= n(v) <= 0.3 for v in [0,5])

    noop_f = lambda x: x
    # simple sanity test
    assert noop_f(5) == 5
    
def test_functions_linear_bounded_linear_R_S():
    # linear
    f = linear(1, -1)
    assert f(-2) == 0  # clipped
    assert f(2) == 1   # clipped
    assert 0 < f(1.5) < 1  # between 0 and 1

    # bounded_linear
    bl = bounded_linear(2, 3)
    assert bl(1) == 0
    assert math.isclose(bl(2.5), 0.5)
    assert bl(4) == 1

    # R function
    r = R(2,3)
    assert r(1) == 0
    assert math.isclose(r(2.5), 0.5)
    assert r(4) == 1

    # S function
    s = S(2,3)
    assert s(1) == 1
    assert math.isclose(s(2.5), 0.5)
    assert s(4) == 0

def test_functions_rectangular_triangular_trapezoid():
    rect = rectangular(1, 2)
    assert rect(0.5) == 0
    assert rect(1.5) == 1
    assert rect(2.5) == 0

    tri = triangular(1,3, c=2)
    assert tri(1) == 0
    assert tri(2) == 1
    assert tri(3) == 0

    trap = trapezoid(1, 2, 3, 4)
    assert trap(0.5) == 0
    assert trap(1.5) > 0 and trap(1.5) < 1
    assert trap(2.5) == 1
    assert trap(3.5) > 0 and trap(3.5) < 1
    assert trap(4.5) == 0

def test_functions_sigmoid_and_bounded_sigmoid():
    sig = sigmoid(1, 1, 0)
    v0 = sig(0)
    assert 0 < v0 < 1

    b_sig = bounded_sigmoid(0, 1)
    assert math.isclose(b_sig(0), 0.1, abs_tol=1e-2)
    assert math.isclose(b_sig(1), 0.9, abs_tol=1e-2)
    assert math.isclose(b_sig(100000), 1.0, abs_tol=1e-2)
    assert math.isclose(b_sig(-100000), 0.0, abs_tol=1e-2)

def test_functions_bounded_exponential_and_simple_sigmoid():
    b_exp = bounded_exponential(k=0.1, limit=1)
    assert 0 <= b_exp(0) < 1
    assert math.isclose(b_exp(1000), 1, rel_tol=1e-3)

    s_sig = simple_sigmoid()
    assert math.isclose(s_sig(0), 0.5)
    assert 0 <= s_sig(-1000) <= 0.01
    assert 0.99 <= s_sig(1000) <= 1.0

def test_functions_singleton_and_gauss():
    sing = singleton(2)
    assert sing(1) == 0
    assert sing(2) == 1

    g = gauss(0, 1, c_m=1)
    v_center = g(0)
    assert 0 < v_center <= 1
    v_far = g(1000)
    assert math.isclose(v_far, 0)

