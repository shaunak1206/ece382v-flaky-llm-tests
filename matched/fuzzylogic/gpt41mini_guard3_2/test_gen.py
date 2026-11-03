import pytest
import numpy as np
import random

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import inv, normalize, constant, linear, bounded_linear, R, S, rectangular, triangular, trapezoid, sigmoid, bounded_sigmoid, bounded_exponential, simple_sigmoid, triangular_sigmoid, gauss

# Domain and Set tests
def test_domain_init_and_str_repr_eq():
    d1 = Domain('temp', 0, 10, res=2)
    assert d1._name == 'temp'
    assert d1._low == 0
    assert d1._high == 10
    assert d1._res == 2
    assert str(d1) == 'temp'
    rep = repr(d1)
    # Eval repr should reconstruct something equal-ish (sets empty)
    d2 = eval(rep)
    assert d1 == d2

def test_domain_call_and_attribute_access():
    d = Domain('humidity', 0, 100)
    d.dry = constant(0.1)
    d.wet = constant(0.9)
    result = d(50)
    assert isinstance(result, dict)
    assert result['dry'] == 0.1
    assert result['wet'] == 0.9
    # access through attribute returns Set instance
    assert isinstance(d.dry, Set)
    # accessing unknown attribute raises AttributeError
    with pytest.raises(AttributeError):
        _ = d.nonexistent

def test_domain_set_deletion(monkeypatch):
    d = Domain('speed', 0, 100)
    d.fast = constant(0.8)
    assert 'fast' in d._sets
    del d.fast
    assert 'fast' not in d._sets
    # deleting non-set attribute raises FuzzyWarning
    with pytest.raises(FuzzyWarning):
        del d._low

def test_domain_range_and_min_max():
    d = Domain('pressure', 1, 5, res=1)
    d.low = constant(0.2)
    d.high = constant(0.8)
    rng = d.range()
    arr = np.array(list(rng))
    assert arr[0] == 1
    assert arr[-1] == 5
    # min returns lowest membership value at x=3
    assert d.min(3) == 0.2
    # max returns highest membership value at x=3
    assert d.max(3) == 0.8

def test_set_call_and_invert_and_and_or():
    # Setup domain and sets
    d = Domain('test', 0, 10)
    s1 = Set(lambda x: x/10, name='s1', domain=d)  # linear increasing
    s2 = Set(lambda x: 1 - x/10, name='s2', domain=d)  # linear decreasing
    # __call__ returns correct value
    val = s1(5)
    assert pytest.approx(val, 1e-7) == 0.5
    # __invert__ returns a set with inverted membership values
    s3 = ~s1
    assert callable(s3.func)
    assert pytest.approx(s3(0), 1e-7) == 1
    assert pytest.approx(s3(10), 1e-7) == 0
    # __and__ min operator
    s_and = s1 & s2
    for x in [0, 5, 10]:
        expected = min(s1(x), s2(x))
        assert pytest.approx(s_and(x), 1e-7) == expected
    # __or__ max operator
    s_or = s1 | s2
    for x in [0, 5, 10]:
        expected = max(s1(x), s2(x))
        assert pytest.approx(s_or(x), 1e-7) == expected

def test_set_mul_add_xor_pow():
    d = Domain('d', 0, 10)
    s1 = Set(lambda x: 0.5, name='s1', domain=d)
    s2 = Set(lambda x: 0.3, name='s2', domain=d)
    s_mul = s1 * s2
    s_add = s1 + s2
    s_xor = s1 ^ s2
    s_pow = s1 ** 2
    for x in d.range():
        xval = float(x)
        # product: multiply values
        expected_mul = 0.5 * 0.3
        assert pytest.approx(s_mul(xval), 1e-7) == expected_mul
        # bounded_sum: min(1, s1+s2)
        expected_add = min(1, 0.5 + 0.3)
        assert pytest.approx(s_add(xval), 1e-7) == expected_add
        # simple_disjoint_sum: s1 + s2 - s1*s2
        expected_xor = 0.5 + 0.3 - (0.5 * 0.3)
        assert pytest.approx(s_xor(xval), 1e-7) == expected_xor
        # pow: value squared
        expected_pow = 0.5 ** 2
        assert pytest.approx(s_pow(xval), 1e-7) == expected_pow

def test_set_eq_le_lt_ge_gt_len_cardinality_relative():
    d = Domain('test', 0, 4, res=1)
    s1 = Set(lambda x: x/4, name='s1', domain=d)
    s2 = Set(lambda x: x/4, name='s2', domain=d)
    s3 = Set(lambda x: (x/4) ** 2, name='s3', domain=d)
    # eq returns True for identical function values on domain
    assert s1 == s2
    assert not (s1 == s3)
    # le - subset
    assert s3 <= s1
    # lt - proper subset
    # s3 values always less than s1 except 0, so < should be False because all must be less (strict)
    assert not (s3 < s1)
    # ge - superset
    assert s1 >= s3
    # gt - proper superset
    assert not (s1 > s3)  # Because some values are equal not strictly greater
    # len returns size of array
    assert len(s1) == len(d.range())
    # cardinality is sum of values
    card = s1.cardinality()
    expected_card = sum([x/4 for x in d.range()])
    assert pytest.approx(card, 1e-7) == expected_card
    # relative_cardinality is average (sum / length)
    rel_card = s1.relative_cardinality()
    assert pytest.approx(rel_card, 1e-7) == card / len(d.range())

def test_set_cardinality_errors():
    s = Set(lambda x: 0.5)
    with pytest.raises(FuzzyWarning):
        s.cardinality()
    with pytest.raises(FuzzyWarning):
        s.relative_cardinality()

def test_set_concentrated_dilated_multiplied_normalized(monkeypatch):
    d = Domain('dom', 0, 10)
    s = Set(lambda x: x / 10, name='s', domain=d)
    # concentrated squares membership values
    sq_set = s.concentrated()
    for x in d.range():
        expected = (x/10) ** 2
        assert pytest.approx(sq_set(x), 1e-7) == expected
    # dilated raises to power 0.5
    dil_set = s.dilated()
    for x in d.range():
        expected = (x/10) ** 0.5
        assert pytest.approx(dil_set(x), 1e-7) == expected
    # multiplied multiplies membership
    mult_set = s.multiplied(3)
    for x in d.range():
        expected = (x/10) * 3
        assert pytest.approx(mult_set(x), 1e-7) == expected
    # normalized scales so max array value is 1
    norm_set = s.normalized()
    max_val = max(s.array())
    for x in d.range():
        expected = (x/10) / max_val
        assert pytest.approx(norm_set(x), 1e-7) == expected

def test_set_plot_no_domain(monkeypatch):
    s = Set(constant(1))
    with pytest.raises(FuzzyWarning):
        s.plot()

def test_set_array_and_string_repr(monkeypatch):
    d = Domain('dom', 0, 2)
    s = Set(lambda x: x / 2, name='name', domain=d)
    arr = s.array()
    expected = np.array([0.0, 0.5, 1.0])
    assert np.allclose(arr, expected)
    # __str__ returns domain_name.set_name
    assert str(s) == "dom.name"
    # repr returns something including Set and func
    r = repr(s)
    assert r.startswith("Set(")
    # dangling set (no name or domain) string representation
    s2 = Set(constant(0.5))
    s2.name = None
    s2.domain = None
    assert "dangling Set" in str(s2)

def test_domain_set_attribute_assignment_and_errors():
    d = Domain('dim', 0, 1)
    # Assigning a function turns it into Set
    d.member = lambda x: x
    assert isinstance(d.member, Set)
    # Assigning a non-Set and non-callable raises Assertion
    with pytest.raises(AssertionError):
        d.bad = 123
    # Assigning with invalid identifier name raises Assertion
    with pytest.raises(AssertionError):
        setattr(d, "bad name", constant(1))
    # Provide name or domain alone raises FuzzyWarning
    with pytest.raises(FuzzyWarning):
        Set(constant(1), name="alone")
    with pytest.raises(FuzzyWarning):
        Set(constant(1), domain=d)

def test_functions_inv_and_normalize_behavior():
    # Setup a simple base function f(x) = x*0.1 clipped to 0..1
    base_func = linear(m=0.1, b=0)
    inv_func = inv(base_func)
    # test inversion
    assert pytest.approx(inv_func(0), 1e-7) == 1.0
    assert pytest.approx(inv_func(5), 1e-7) == 1 - base_func(5)
    # normalize scales function such that max value becomes 1
    norm_func = normalize(0.5, base_func)  # divide all outputs by 0.5
    assert pytest.approx(norm_func(0), 1e-7) == base_func(0) / 0.5
    # values clipped to <=1
    assert norm_func(15) <= 1

@pytest.mark.parametrize("func_constructor,x,expected", [
    (constant(0.7), 100, 0.7),
    (linear(0.5, 0), 1, 0.5),
    (bounded_linear(0, 1), 0.5, 0.5),
    (R(0, 1), 1, 1.0),
    (S(0, 1), 0, 1.0),
    (rectangular(0, 1), 0.5, 1.0),
    (triangular(0, 2), 1, 1.0),
    (trapezoid(0, 1, 2, 3), 1, 1.0),
    (sigmoid(1, 5, 0), 0, 0.5),  # midpoint at x0=0
    (bounded_sigmoid(0, 1), 0, pytest.approx(0.1, abs=1e-2)),
    (bounded_exponential(k=1, limit=1), 0, 0),
    (simple_sigmoid(), 0, 0.5),
    (triangular_sigmoid(0, 2), 1, pytest.approx(0.9, abs=1e-2)),
    (gauss(0, 1), 0, 1.0),
])
def test_functions_memberships(func_constructor, x, expected):
    # Seed random just in case, but functions do not use randomness
    random.seed(123)
    f = func_constructor
    if callable(f):
        res = f(x)
    else:
        res = f
    if isinstance(expected, float):
        assert pytest.approx(res, abs=1e-2) == expected
    else:
        # expected can be pytest.approx itself (e.g. for bounded_sigmoid)
        assert pytest.approx(res, abs=1e-2) == expected

def test_bounded_sigmoid_inverse_symmetry():
    f = bounded_sigmoid(1, 0, inverse=True)
    assert pytest.approx(f(0), abs=1e-2) == 0.9
    assert pytest.approx(f(1), abs=1e-2) == 0.1
    # Increasing function with inverse=True but reversed parameters

def test_bounded_linear_gradient_cases():
    # gradient = 0 case (low == high for c_m == no_m)
    f = bounded_linear(0, 1, c_m=0.5, no_m=0.5)
    for x in [-10, 0, 10]:
        assert pytest.approx(f(x)) == 0.5
    # infinite gradient case is not triggered because low < high always
    # Just check function callable and returns float in domain range
    f = bounded_linear(0, 1, c_m=1, no_m=0)
    for x in [0, 0.5, 1]:
        val = f(x)
        assert 0 <= val <= 1

def test_set_eq_warning_for_no_domain():
    s1 = Set(lambda x: 0.5)
    s2 = Set(lambda x: 0.5)
    with pytest.raises(FuzzyWarning):
        _ = (s1 == s2)
    with pytest.raises(FuzzyWarning):
        _ = (s1 <= s2)

def test_set_len_warning_no_domain():
    s = Set(lambda x: 0.5)
    with pytest.raises(FuzzyWarning):
        len(s)

def test_set_call_with_pickled_func():
    import pickle
    f = lambda x: x / 10
    pickled = pickle.dumps(f)
    s = Set(pickled)
    assert abs(s(5) - 0.5) < 1e-7
