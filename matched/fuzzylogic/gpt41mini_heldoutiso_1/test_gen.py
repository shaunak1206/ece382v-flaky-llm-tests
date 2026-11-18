import pytest
import numpy as np
import pickle
from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import inv, normalize, constant, linear, bounded_linear, R, S, rectangular, triangular, trapezoid, sigmoid, bounded_sigmoid, bounded_exponential, simple_sigmoid, gauss


#########################
# Domain class tests
#########################

def test_domain_init_valid():
    d = Domain("temp", 0, 100, res=1)
    assert d._name == "temp"
    assert d._low == 0
    assert d._high == 100
    assert d._res == 1
    assert isinstance(d._sets, dict)
    assert len(d._sets) == 0

def test_domain_init_invalid_bounds():
    with pytest.raises(AssertionError):
        Domain("fail", 1, 0)
    with pytest.raises(AssertionError):
        Domain("fail", 0, 1, res=0)
    with pytest.raises(AssertionError):
        Domain("fail", 0, 1, res=-1)

def test_domain_set_and_getattr_sets():
    d = Domain("temp", 0, 10)
    f = constant(0.5)
    s = Set(f)
    d.cool = s
    # The set is assigned to _sets dict and accessible as attribute
    assert d.cool is s
    assert "cool" in d._sets
    assert d._sets["cool"] is s

def test_domain_set_function_auto_wrap_in_set():
    d = Domain("temp", 0, 10)
    d.warm = lambda x: 0.7
    s = d._sets["warm"]
    assert isinstance(s, Set)
    assert s(5) == 0.7
    # domain and name are correctly set
    assert s.domain is d
    assert s.name == "warm"

def test_domain_delattr_set_and_warning():
    d = Domain("temp", 0, 10)
    d.hot = constant(1)
    assert "hot" in d._sets
    # deletion removes set
    del d.hot
    assert "hot" not in d._sets
    # deleting attribute that is not a set raises FuzzyWarning
    with pytest.raises(FuzzyWarning):
        del d._low

def test_domain_call_membership_warn_and_return():
    d = Domain("temp", 0, 10)
    d.cold = constant(1)
    # value in domain returns dict with membership values
    result = d(5)
    assert isinstance(result, dict)
    assert result.get("cold") == 1
    # value outside domain triggers warning and still returns dict
    with pytest.warns(UserWarning):
        res = d(20)
    assert "cold" in res

def test_domain_range_inclusive_and_resolution():
    d = Domain("temp", 0, 5, res=2)
    r = d.range()
    # arange includes upper bound with step res=2 => 0, 2, 4, 6 (but 6 > high 5), so stops at 4 + res = 6 but stops before 6 > high+res
    expected = np.arange(0, 5+2, 2)
    assert np.array_equal(r, expected)
    # all values are between low and high+res
    assert all((val >= 0 and val <= 7) for val in r)

def test_domain_min_max_membership():
    d = Domain("temp", 0, 10)
    d.low = constant(0.1)
    d.high = constant(0.9)
    assert d.min(5) == 0.1
    assert d.max(5) == 0.9

def test_domain_eq_and_repr():
    d1 = Domain("d", 0, 10, res=1)
    d2 = Domain("d", 0, 10, res=1)
    d1.hot = constant(1)
    d2.hot = constant(1)
    # equal domains should compare equal
    assert d1 == d2
    r = repr(d1)
    # eval repr should recreate similar domain (without sets equality check because functions are not serializable)
    d3 = eval(r)
    assert d3._name == "d"
    assert d3._low == 0
    assert d3._high == 10
    assert d3._res == 1

#########################
# Set class tests
#########################

def test_set_init_and_call():
    f = constant(0.4)
    s = Set(f)
    assert callable(s)
    assert s(0) == 0.4
    # Name or domain provided singly raises warning
    with pytest.warns(FuzzyWarning):
        Set(f, name="a")
    with pytest.warns(FuzzyWarning):
        Set(f, domain=Domain("x", 0, 1))

def test_set_invert_and_negation():
    f = constant(0.3)
    s = Set(f)
    s_inv = ~s
    s_neg = -s
    assert isinstance(s_inv, Set)
    assert isinstance(s_neg, Set)
    for v in [0, 0.5, 1]:
        assert abs(s_inv(v) - (1 - f(v))) < 1e-6
        assert abs(s_neg(v) - (1 - f(v))) < 1e-6

def test_set_and_or_mul_add_xor_pow():
    f1 = constant(0.2)
    f2 = constant(0.5)
    s1 = Set(f1)
    s2 = Set(f2)
    # AND: min
    s_and = s1 & s2
    for v in [0, 1]:
        assert s_and(v) == min(f1(v), f2(v))
    # OR: max
    s_or = s1 | s2
    for v in [0, 1]:
        assert s_or(v) == max(f1(v), f2(v))
    # Mul: product
    s_mul = s1 * s2
    for v in [0, 1]:
        assert s_mul(v) == f1(v) * f2(v)
    # Add: bounded sum (capped at 1)
    s_add = s1 + s2
    for v in [0, 1]:
        val = f1(v) + f2(v)
        if val > 1:
            val = 1
        assert abs(s_add(v) - val) < 1e-6
    # Xor: simple disjoint sum
    s_xor = s1 ^ s2
    for v in [0, 1]:
        expected = s1.func(v) + s2.func(v) - 2 * s1.func(v) * s2.func(v)
        assert abs(s_xor(v) - expected) < 1e-6
    # Pow: power
    s_pow = s1 ** 2
    for v in [0, 1]:
        assert abs(s_pow(v) - (f1(v)**2)) < 1e-6

def test_set_eq_and_comparisons():
    d = Domain("d", 0, 10)
    f1 = constant(0.5)
    f2 = constant(0.5)
    s1 = Set(f1)
    s2 = Set(f2)
    s1.domain = d
    s2.domain = d
    # Equal sets return True
    assert s1 == s2
    # Less or Equal (subset)
    f3 = constant(0.4)
    s3 = Set(f3)
    s3.domain = d
    assert (s3 <= s1) is True
    assert (s3 < s1) is True
    assert (s1 >= s3) is True
    assert (s1 > s3) is True
    # Comparison without domain raises warning
    s4 = Set(f1)
    s5 = Set(f1)
    with pytest.raises(FuzzyWarning):
        _ = s4 == s5
    with pytest.raises(FuzzyWarning):
        _ = s4 < s5
    with pytest.raises(FuzzyWarning):
        _ = s4 > s5

def test_set_len_cardinality_relative_cardinality():
    d = Domain("d", 0, 4, res=1)
    f = lambda x: 0.5
    s = Set(f)
    s.domain = d
    arr = s.array()
    assert len(s) == len(arr)
    # cardinality sums values (all 0.5 for 5 elements 0..4)
    assert np.isclose(s.cardinality(), 0.5 * len(arr))
    # relative cardinality is sum / length
    assert np.isclose(s.relative_cardinality(), 0.5)
    # Without domain raises warning
    s2 = Set(f)
    with pytest.raises(FuzzyWarning):
        len(s2)
    with pytest.raises(FuzzyWarning):
        s2.cardinality()
    with pytest.raises(FuzzyWarning):
        s2.relative_cardinality()
    # Domain with zero length causes warning on relative_cardinality
    d_empty = Domain("empty", 0, 0, res=1)
    s_empty = Set(constant(0))
    s_empty.domain = d_empty
    with pytest.raises(FuzzyWarning):
        s_empty.relative_cardinality()

def test_set_concentrated_and_dilated_multiplied():
    f = constant(0.5)
    s = Set(f)
    # concentrated squares membership
    s_conc = s.concentrated()
    for v in [0, 1]:
        assert abs(s_conc(v) - f(v)**2) < 1e-6
    # dilated applies power 1/2 (sqrt)
    s_dil = s.dilated()
    for v in [0, 1]:
        assert abs(s_dil(v) - f(v)**0.5) < 1e-6
    # multiplied scales membership values
    n = 3
    s_mult = s.multiplied(n)
    for v in [0, 1]:
        assert abs(s_mult(v) - f(v)*n) < 1e-6

def test_set_plot_and_array_raise_warning_without_domain(monkeypatch):
    s = Set(constant(0.5))
    # plot without domain raises warning
    with pytest.raises(FuzzyWarning):
        s.plot()
    # array without domain raises warning
    with pytest.raises(FuzzyWarning):
        s.array()

def test_set_repr_and_str():
    f = constant(0.5)
    s = Set(f)
    # Without name and domain, str returns "dangling Set"
    s.name = None
    s.domain = None
    assert s.__str__().startswith("dangling Set")
    # With domain/name, str returns with domain and name
    d = Domain("d", 0, 1)
    s.domain = d
    s.name = "myset"
    assert s.__str__() == "d.myset"
    # repr returns string containing "Set("
    r = repr(s)
    assert r.startswith("Set(")

def test_set_normalized_behavior():
    d = Domain("dn", 0, 4, res=1)
    f = lambda x: 0.5 * x
    s = Set(f)
    s.domain = d
    arr = s.array()
    height = max(arr)
    norm = s.normalized()
    # normalized max is 1
    norm_arr = norm.array()
    np.testing.assert_almost_equal(max(norm_arr), 1.0, decimal=6)

def test_set_intensified_specific_behavior():
    # This test covers the intensified method's internal function behavior
    f = lambda x: 0 if x < 0.5 else 1
    s = Set(f)
    intensified = s.intensified()
    # At x=0.4, func(x)**2=0, so expected return is 2*0=0
    assert intensified(0.4) == 0
    # At x=0.6, func(x)**2=1, formula is 1 - 2*(1-1)=1
    assert intensified(0.6) == 1


#########################
# functions.py tests
#########################

def test_inv_function():
    f = constant(0.3)
    inv_f = inv(f)
    for x in [0, 0.5, 1]:
        assert inv_f(x) == pytest.approx(1 - f(x))

def test_constant_function():
    cval = 0.77
    f = constant(cval)
    for v in [-100, 0, 100]:
        assert f(v) == cval

def test_linear_function():
    f = linear(1, -1)
    # some fixed points from docstring
    assert f(-2) == 0
    assert f(0) == 0
    assert f(1) == 0
    assert f(1.5) == 0.5
    assert f(2) == 1
    assert f(3) == 1

def test_bounded_linear_function_normal_and_inverse():
    f = bounded_linear(2, 3)
    assert f(1) == 0
    assert f(2) == 0
    assert pytest.approx(f(2.5), 0.001) == 0.5
    assert f(3) == 1
    assert f(4) == 1
    f_inv = bounded_linear(2, 3, inverse=True)
    # inverse flips c_m and no_m so behavior reverses
    assert f_inv(1) == 1
    assert f_inv(3) == 0

def test_R_and_S_functions_behavior():
    r = R(2, 3)
    assert r(1) == 0
    assert pytest.approx(r(2.5)) == 0.5
    assert r(4) == 1
    sfunc = S(2, 3)
    assert sfunc(1) == 1
    assert pytest.approx(sfunc(2.5)) == 0.5
    assert sfunc(4) == 0

def test_rectangular_function_behavior():
    f = rectangular(1, 3, c_m=0.8, no_m=0.1)
    assert f(0) == 0.1
    assert f(1) == 0.8
    assert f(2) == 0.8
    assert f(3) == 0.8
    assert f(4) == 0.1

def test_triangular_function_properties():
    f = triangular(1, 3, c=2)
    assert f(1) == pytest.approx(0)
    assert f(2) == 1
    assert f(3) == pytest.approx(0)

def test_trapezoid_function_behavior():
    f = trapezoid(1, 2, 3, 4)
    # below low returns no_m=0
    assert f(0) == 0
    # between low and c_low, linear increase
    assert 0 < f(1.5) < 1
    # between c_low and c_high returns core c_m (1 default)
    assert f(2.5) == 1
    # between c_high and high, decreasing linear
    assert 0 < f(3.5) < 1
    # above high returns no_m=0
    assert f(5) == 0

def test_sigmoid_function_properties():
    f = sigmoid(1, 1, 0)
    assert 0 <= f(-1000) <= 1
    assert 0 <= f(1000) <= 1
    mid = f(0)
    assert pytest.approx(mid, 0.01) == 0.5

def test_bounded_sigmoid_edge_and_inverse():
    f = bounded_sigmoid(0, 1)
    assert pytest.approx(f(0), 0.1) == 0.1
    assert pytest.approx(f(1), 0.1) == 0.9
    assert pytest.approx(f(100000), 0.01) == 1.0
    assert pytest.approx(f(-100000), 0.01) == 0.0
    f_inv = bounded_sigmoid(0, 1, inverse=True)
    assert pytest.approx(f_inv(0), 0.1) != f(0)
    assert 0 <= f_inv(0) <= 1

def test_bounded_exponential_behavior():
    f = bounded_exponential(k=0.1, limit=1)
    assert 0 <= f(0) <= 1
    assert pytest.approx(f(1000), 0.001) == 1.0

def test_simple_sigmoid_properties():
    f = simple_sigmoid()
    assert pytest.approx(f(-1000), 0.01) == 0.0
    assert f(0) == 0.5
    assert pytest.approx(f(1000), 0.01) == 1.0
    assert 0 < f(-20) < 0.05
    assert 0.95 < f(20) < 1

def test_gauss_behavior():
    f = gauss(c=0, b=1, c_m=1)
    assert f(0) == 1
    val = f(1)
    assert 0 < val < 1
    # Very large values return near 0
    val_large = f(1e6)
    assert pytest.approx(val_large, abs=1e-5) == 0

