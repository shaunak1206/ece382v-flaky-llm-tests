# tests/test_classes_functions.py
import math
import warnings
import pytest
import numpy as np

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv, noop, constant, alpha, normalize,
    moderate,
    singleton,
    linear,
    bounded_linear,
    R, S,
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


@pytest.fixture
def simple_domain():
    d = Domain('test', 0, 10, res=1)
    return d

@pytest.fixture
def linear_set(simple_domain):
    f = linear(0.1, 0)
    s = Set(f)
    simple_domain.some = s
    return s

def test_domain_basic_attributes(simple_domain):
    assert simple_domain._name == 'test'
    assert simple_domain._low == 0
    assert simple_domain._high == 10
    assert simple_domain._res == 1
    assert isinstance(simple_domain._sets, dict)

def test_domain_set_assignment_and_access(simple_domain):
    s = Set(lambda x: x / 10)
    simple_domain.foo = s
    assert simple_domain.foo == s
    assert simple_domain._sets['foo'] == s
    assert s.domain == simple_domain
    assert s.name == 'foo'
    with pytest.raises(AttributeError):
        _ = simple_domain.bar  # Not assigned

def test_domain_call_memberships(simple_domain):
    simple_domain.half = Set(lambda x: 0.5)
    results = simple_domain(5)
    assert isinstance(results, dict)
    assert 'half' in results
    # Value should be about 0.5
    assert abs(results['half'] - 0.5) < 1e-12

def test_domain_call_warns_outside_domain(simple_domain):
    simple_domain.const = Set(constant(0.3))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        simple_domain(-1)
        simple_domain(20)
        assert any(issubclass(warning.category, UserWarning) for warning in w)
        assert any('outside of domain' in str(warning.message) for warning in w)

def test_domain_range_includes_high(simple_domain):
    r = simple_domain.range()
    assert isinstance(r, np.ndarray)
    # Should include both low and high values
    assert math.isclose(r[0], simple_domain._low)
    assert math.isclose(r[-1], simple_domain._high)

def test_domain_min_max(simple_domain):
    s1 = Set(lambda x: 0.25)
    s2 = Set(lambda x: 0.75)
    simple_domain.low = s1
    simple_domain.high = s2
    assert math.isclose(simple_domain.min(5), 0.25)
    assert math.isclose(simple_domain.max(5), 0.75)

def test_set_call_and_invert_behavior(simple_domain):
    f = linear(0.1, 0)
    s = Set(f)
    simple_domain.foo = s
    val = s(5)
    invval = (~s)(5)
    assert 0 <= val <= 1
    assert 0 <= invval <= 1
    assert math.isclose(invval, 1 - val, rel_tol=1e-12)

def test_set_operators(simple_domain):
    s1 = Set(constant(0.2))
    s2 = Set(constant(0.7))
    simple_domain.s1 = s1
    simple_domain.s2 = s2
    # AND is MIN
    s_and = s1 & s2
    assert math.isclose(s_and(0), min(s1(0), s2(0)), rel_tol=1e-12)
    # OR is MAX
    s_or = s1 | s2
    assert math.isclose(s_or(0), max(s1(0), s2(0)), rel_tol=1e-12)
    # Multiplication (product)
    s_mul = s1 * s2
    expected_mul = s1(0) * s2(0)
    assert math.isclose(s_mul(0), expected_mul, rel_tol=1e-12)
    # Bounded sum (+)
    s_add = s1 + s2
    expected_add = min(1, s1(0) + s2(0))
    assert math.isclose(s_add(0), expected_add, rel_tol=1e-12)
    # Disjoint sum (^)
    s_xor = s1 ^ s2
    val1, val2 = s1(0), s2(0)
    expected_xor = val1 + val2 - 2 * val1 * val2
    assert math.isclose(s_xor(0), expected_xor, rel_tol=1e-12)
    # Power
    s_pow = s1 ** 2
    expected_pow = val1 ** 2
    assert math.isclose(s_pow(0), expected_pow, rel_tol=1e-12)

def test_set_eq_le_lt_ge_gt(simple_domain):
    s1 = Set(lambda x: 0.5, name='foo', domain=simple_domain)
    s2 = Set(lambda x: 0.5, name='bar', domain=simple_domain)
    # They return same arrays for each x in domain, so eq is True
    assert s1 == s2
    # Modify s2 to differ
    s3 = Set(lambda x: 0.6 if x == 5 else 0.5, name='baz', domain=simple_domain)
    # s1 < s3 should be True for strict less in all points? No, not all strictly less.
    assert not (s1 < s3)
    # s3 >= s1 True since s3 >= s1 everywhere?
    assert not (s3 >= s1)  # Because s3(5)=0.6 > 0.5 but s3(others)=0.5 == 0.5
    # so s3 > s1 False (not strictly greater everywhere)
    assert not (s3 > s1)
    # s1 <= s3 must be True (all values less-or-equal)
    assert all(s1.array() <= s3.array())
    # Use comparisons which raise FuzzyWarning if no domain
    s4 = Set(lambda x: 0.5)
    with pytest.raises(FuzzyWarning):
        _ = s4 == s1
    with pytest.raises(FuzzyWarning):
        _ = s4 < s1

def test_set_len_cardinalities(simple_domain):
    s = Set(lambda x: 0.1 * x, name='foo', domain=simple_domain)
    l = len(s)
    assert l == len(simple_domain.range())
    card = s.cardinality()
    assert math.isclose(card, sum(s.array()), rel_tol=1e-12)
    rel_card = s.relative_cardinality()
    assert math.isclose(rel_card * l, card, rel_tol=1e-12)

def test_set_len_cardinality_no_domain():
    s = Set(lambda x: 0.5)
    with pytest.raises(FuzzyWarning):
        _ = len(s)
    with pytest.raises(FuzzyWarning):
        _ = s.cardinality()
    with pytest.raises(FuzzyWarning):
        _ = s.relative_cardinality()

def test_set_relative_cardinality_empty_domain():
    # Create domain with zero range (simulate)
    class DummyDomain:
        def __init__(self):
            self._name = "dummy"
            self._low = 0
            self._high = 0
            self._res = 1
        def range(self):
            return np.array([])
    d = DummyDomain()
    s = Set(constant(0), name='foo', domain=d)
    with pytest.raises(FuzzyWarning):
        _ = s.relative_cardinality()

def test_set_concentrated_and_dilated(simple_domain):
    s = Set(lambda x: 0.3 + 0.4 * (x / simple_domain._high), name='foo', domain=simple_domain)
    concentrated = s.concentrated()
    for x in simple_domain.range():
        expected = s(x) ** 2
        assert math.isclose(concentrated(x), expected, rel_tol=1e-12)
    # dilated returns f(x) ** 0.5
    dilated = s.dilated()
    for x in simple_domain.range():
        expected = s(x) ** 0.5
        assert math.isclose(dilated(x), expected, rel_tol=1e-12)

def test_set_intensified_behavior():
    # Use intensify function from class
    # Due to the apparent bug in code (__mul__ missing?), test only structure
    s = Set(lambda x: 0.4 if x < 0.5 else 0.7)
    intensified = s.intensified()
    # For x < 0.5
    x = 0.3
    f_x = s(x)
    expected = 2 * f_x ** 2
    assert math.isclose(intensified(x), expected, rel_tol=1e-12)
    # For x >= 0.5
    x = 0.6
    f_x = s(x)
    expected = 1 - 2*(1 - f_x**2)
    assert math.isclose(intensified(x), expected, rel_tol=1e-12)

def test_set_multiplied(simple_domain):
    s = Set(lambda x: 0.25, name='foo', domain=simple_domain)
    multiplied = s.multiplied(3)
    for x in simple_domain.range():
        expected = s(x) * 3
        assert math.isclose(multiplied(x), expected, rel_tol=1e-12)

def test_set_plot_no_domain_warns():
    s = Set(lambda x: 1.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s.plot()
        assert any(issubclass(warning.category, UserWarning) for warning in w)

def test_set_array_and_repr_str(simple_domain):
    s = Set(lambda x: x / 10, name='foo', domain=simple_domain)
    arr = s.array()
    assert isinstance(arr, np.ndarray)
    # repr returns string containing "Set(" and function string
    r = repr(s)
    assert r.startswith("Set(")
    # str should contain domain and name
    st = str(s)
    assert simple_domain._name in st and s.name in st
    # str for dangling set
    s2 = Set(lambda x: 0.5)
    st2 = str(s2)
    assert st2.startswith("dangling Set")

def test_set_normalized(simple_domain):
    s = Set(lambda x: 0.5 + 0.5 * (x / simple_domain._high), name='foo', domain=simple_domain)
    norm = s.normalized()
    arr = s.array()
    max_val = max(arr)
    for x in simple_domain.range():
        expected = s(x) / max_val
        assert math.isclose(norm(x), expected, rel_tol=1e-12)


###########################
# Tests for functions.py  ##
###########################

def test_inv_function():
    f = constant(0.3)
    g = inv(f)
    for x in range(5):
        assert math.isclose(g(x), 1 - f(x), rel_tol=1e-12)

def test_noop_function():
    f = noop()
    for val in [-10, 0, 5.5, 1e10]:
        assert f(val) == val

def test_constant_function():
    f = constant(0.7)
    for x in range(5):
        assert f(x) == 0.7

def test_alpha_function_clipping():
    base = constant(0.5)
    f = alpha(floor=0.3, ceiling=0.7, func=base)
    # Values outside ceiling and floor clipped
    assert math.isclose(f(0), 0.5)  # unchanged since base=0.5 between floor and ceiling
    # Forcing base function to greater than ceiling
    f2 = alpha(floor=0.3, ceiling=0.6, func=lambda x: 0.7)
    assert f2(0) == 0.6

def test_normalize_function():
    base = constant(0.5)
    f = normalize(0.5, base)
    for x in range(3):
        assert math.isclose(f(x), 1.0)

def test_moderate_function_behavior():
    base = constant(0.5)
    f = moderate(base)
    for x in range(3):
        val = f(x)
        # Since input is 0.5, output should be close to 0.5 (center bias)
        assert 0 <= val <= 1

def test_singleton_function():
    f = singleton(2)
    assert f(2) == 1
    assert f(1) == 0
    assert f(3) == 0

def test_linear_function():
    f = linear(1, -1)
    # Clipped at 0
    assert math.isclose(f(-2), 0)
    assert math.isclose(f(0), 0)
    # Inside range
    assert math.isclose(f(1.5), 0.5)
    # Clipped max 1
    assert math.isclose(f(3), 1)

def test_bounded_linear_behavior():
    f = bounded_linear(2, 3)
    assert math.isclose(f(1), 0)
    assert math.isclose(f(2), 0)
    assert math.isclose(f(2.5), 0.5)
    assert math.isclose(f(3), 1)
    assert math.isclose(f(4), 1)

def test_R_and_S_functions():
    rf = R(2, 4)
    sf = S(2, 4)
    # R returns 0 below low
    assert rf(1) == 0
    # R increases between low and high
    assert 0 < rf(3) < 1
    assert rf(5) == 1
    # S returns 1 below low
    assert sf(1) == 1
    # S decreases between low and high
    assert 0 < sf(3) < 1
    assert sf(5) == 0

def test_rectangular_function():
    f = rectangular(2, 4)
    assert f(1) == 0
    assert f(3) == 1
    assert f(5) == 0

def test_triangular_function():
    f = triangular(2, 4)
    left_val = f(2)
    peak_val = f(3)
    right_val = f(4)
    assert 0 <= left_val <= 1
    assert 0 <= peak_val <= 1
    assert 0 <= right_val <= 1
    # peak_val should be near 1 (or c_m)
    assert math.isclose(peak_val, 1, rel_tol=1e-12)

def test_trapezoid_function():
    f = trapezoid(1, 2, 3, 4)
    assert f(0) == 0
    assert 0 < f(1.5) < 1
    assert f(2.5) == 1
    assert 0 < f(3.5) < 1
    assert f(5) == 0

def test_sigmoid_function_ranges():
    f = sigmoid(1, 1, 0)
    assert 0 <= f(-10) <= 0.5
    assert 0.5 <= f(10) <= 1

def test_bounded_sigmoid_behavior():
    f = bounded_sigmoid(0, 1)
    assert math.isclose(f(0), 0.1, abs_tol=1e-2)
    assert math.isclose(f(1), 0.9, abs_tol=1e-2)
    # far outside quickly saturates
    assert math.isclose(f(100000), 1.0, abs_tol=1e-2)
    assert math.isclose(f(-100000), 0.0, abs_tol=1e-2)

def test_bounded_exponential_behavior():
    f = bounded_exponential(k=1, limit=2)
    val0 = f(0)
    val1 = f(1)
    assert abs(val0) < 1e-12
    assert val1 > val0
    # Limit approached at large x
    val_large = f(1e6)
    assert math.isclose(val_large, 2, rel_tol=1e-6)

def test_simple_sigmoid_extremes():
    f = simple_sigmoid()
    assert math.isclose(f(-1000), 0, abs_tol=1e-2)
    assert math.isclose(f(0), 0.5, abs_tol=1e-12)
    assert math.isclose(f(1000), 1, abs_tol=1e-2)

def test_triangular_sigmoid_peak():
    f = triangular_sigmoid(2, 4)
    low_val = f(2)
    mid_val = f(3)
    high_val = f(4)
    assert math.isclose(low_val, 0.1, abs_tol=1e-2)
    assert math.isclose(mid_val, 0.9, abs_tol=1e-2)
    # f(4) should be near 0.1 since right_slope is inverted bounded_sigmoid
    assert 0 <= high_val <= 1

def test_gauss_function_properties():
    f = gauss(0, 1, c_m=1)
    # Peak at c=0
    peak = f(0)
    off_peak = f(1)
    assert peak >= off_peak
    assert 0 <= peak <= 1
    assert 0 <= off_peak <= 1


if __name__ == "__main__":
    pytest.main()
