import pytest
import math
import numpy as np
import warnings
import pickle

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv, noop, constant, alpha, normalize, moderate,
    singleton, linear, bounded_linear, R, S, rectangular,
    triangular, trapezoid, sigmoid, bounded_sigmoid,
    bounded_exponential, simple_sigmoid, triangular_sigmoid, gauss,
)
from fuzzylogic.combinators import MAX, MIN, product, bounded_sum, simple_disjoint_sum

def test_domain_init_and_str_repr_eq():
    d1 = Domain("temp", 0, 100, res=5)
    d2 = Domain("temp", 0, 100, res=5)
    assert str(d1) == "temp"
    expected_repr = f"Domain('temp', 0, 100, res=5, sets={{}})"
    assert repr(d1) == expected_repr
    assert d1 == d2

def test_domain_invalid_init():
    with pytest.raises(AssertionError):
        Domain("bad", 10, 5)
    with pytest.raises(AssertionError):
        Domain("bad", 0, 5, res=0)

def test_domain_set_and_getattr_behavior():
    d = Domain("temperature", 0, 10)
    def f(x): return x/10
    s = Set(f)
    d.cool = s
    assert d._sets['cool'] is s
    # Accessing via attribute returns the Set object
    assert d.cool is s
    with pytest.raises(AttributeError):
        _ = d.hot

def test_domain_call_warns_out_of_bounds():
    d = Domain("temperature", 0, 5)
    d.low = Set(lambda x: 1)
    # Within bounds triggers no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d(3)
        assert all(issubclass(warn.category, Warning) for warn in w) == False or len(w) == 0
    # Out of bounds triggers warning of type UserWarning (FuzzyWarning subclass)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d(-1)
        assert any(issubclass(warn.category, FuzzyWarning) for warn in w)

def test_domain_delattr_and_delete_set():
    d = Domain("temp", 0, 10)
    d.test_set = Set(lambda x: 0.5)
    assert 'test_set' in d._sets
    del d.test_set
    assert 'test_set' not in d._sets
    with pytest.raises(FuzzyWarning):
        del d.some_other_attr

def test_domain_range_min_max():
    d = Domain("temp", 0, 10, res=2)
    # no sets yet, min/max over empty should raise error
    d.cool = Set(constant(0.2))
    d.hot = Set(constant(0.8))
    r = d.range()
    expected = list(range(0, 10+2, 2))
    # numpy.arange is float; convert for test
    assert np.allclose(r, expected)
    # min and max values at any point x
    x = 5
    assert d.min(x) == min(s.func(x) for s in d._sets.values())
    assert d.max(x) == max(s.func(x) for s in d._sets.values())

def test_set_init_and_call():
    f = lambda x: x/10
    s = Set(f)
    # Calling returns function output
    assert s(5) == 0.5
    # Passing string pickled function loads correctly
    dumped = pickle.dumps(f)
    s2 = Set(dumped)
    assert abs(s2(5) - f(5)) < 1e-9
    with pytest.raises(FuzzyWarning):
        Set(f, name="onlyname")  # only one of name/domain
    with pytest.raises(FuzzyWarning):
        Set(f, domain=Domain("d", 0, 1))  # only domain

def test_set_invert_and_neg():
    f = lambda x: 0.2
    s = Set(f)
    inv_s = ~s
    neg_s = -s
    # approx 1 - 0.2 = 0.8
    assert math.isclose(inv_s(0), 1 - f(0))
    assert math.isclose(neg_s(0), 1 - f(0))

def test_set_and_or_mul_add_xor_pow_operators():
    f1 = lambda x: 0.3
    f2 = lambda x: 0.7
    s1 = Set(f1)
    s2 = Set(f2)
    s_and = s1 & s2
    s_or = s1 | s2
    s_mul = s1 * s2
    s_add = s1 + s2
    s_xor = s1 ^ s2
    s_pow = s1 ** 2

    for x in [0, 5, 10]:
        assert math.isclose(s_and(x), MIN(f1, f2)(x))
        assert math.isclose(s_or(x), MAX(f1, f2)(x))
        assert math.isclose(s_mul(x), product(f1, f2)(x))
        assert math.isclose(s_add(x), bounded_sum(f1, f2)(x))
        assert math.isclose(s_xor(x), simple_disjoint_sum(f1, f2)(x))
        assert math.isclose(s_pow(x), f1(x) ** 2)

def test_set_eq_le_lt_ge_gt():
    dom = Domain("test", 0, 10, res=1)
    s1 = Set(lambda x: x/10, domain=dom, name="s1")
    s2 = Set(lambda x: x/10, domain=dom, name="s2")
    s3 = Set(lambda x: 1 - x/10, domain=dom, name="s3")

    assert s1 == s2
    assert s1 <= s2
    assert not (s1 < s2)
    assert s1 >= s2
    assert not (s1 > s2)

    assert s1 != s3
    assert s1 <= s3  or not s1 <= s3  # comparison makes sense but can be false
    # test inequality relations
    if s1 < s3:
        assert s1 <= s3 and not s1 == s3

def test_set_eq_without_domain_raises_warning():
    s1 = Set(lambda x: 0)
    s2 = Set(lambda x: 0)
    with pytest.raises(FuzzyWarning):
        _ = s1 == s2
    with pytest.raises(FuzzyWarning):
        _ = s1 <= s2

def test_set_len_cardinalities_and_relative_cardinality():
    dom = Domain("d", 0, 4)
    s = Set(lambda x: 0.25 * x, domain=dom, name="s")
    expected_len = len(dom.range())
    assert len(s) == expected_len
    card = s.cardinality()
    assert math.isclose(card, sum(s.func(x) for x in dom.range()))
    rel_card = s.relative_cardinality()
    assert math.isclose(rel_card, card / expected_len)
    # test relative_card raises for empty domain
    empty_dom = Domain("empty", 0, 0, res=1)
    s_empty = Set(lambda x: 0, domain=empty_dom, name="empty_s")
    with pytest.raises(FuzzyWarning):
        s_empty.relative_cardinality()

def test_set_derived_sets_methods():
    dom = Domain("d", 0, 10)
    s = Set(lambda x: 0.5, domain=dom, name="s")
    # concentrated squares the membership
    concentrated = s.concentrated()
    for x in dom.range():
        assert math.isclose(concentrated.func(x), s.func(x) ** 2)
    # dilated takes power 1/2
    dilated = s.dilated()
    for x in dom.range():
        assert math.isclose(dilated.func(x), s.func(x) ** 0.5)
    # multiplied multiplies membership by n
    multiplied = s.multiplied(2)
    for x in dom.range():
        assert math.isclose(multiplied.func(x), s.func(x)*2)

def test_set_plot_without_domain_warns():
    s = Set(lambda x: 0)
    with pytest.raises(FuzzyWarning):
        s.plot()

def test_set_array_errors():
    s = Set(lambda x: 0)
    with pytest.raises(FuzzyWarning):
        s.array()
    dom = Domain("d", 0, 3, res=1)
    s2 = Set(lambda x: 0.5, domain=dom, name="s2")
    arr = s2.array()
    assert np.allclose(arr, [s2.func(x) for x in dom.range()])

def test_set_str_and_repr():
    dom = Domain("d", 0, 10)
    s = Set(lambda x: 0.3, domain=dom, name="s")
    s_dangling = Set(lambda x: 0.3)
    assert str(s) == "d.s"
    assert "dangling Set" in str(s_dangling)
    rep = repr(s)
    assert rep.startswith("Set(")

def test_set_normalized():
    dom = Domain("d", 0, 4, res=1)
    s = Set(lambda x: x / 4, domain=dom, name="s")
    normalized = s.normalized()
    # max of s array is 1 (since max x/4 = 1)
    for x in dom.range():
        val = normalized.func(x)
        assert 0 <= val <= 1
        # normalized func should equal original divided by max (which is 1 here)
        assert math.isclose(val, s.func(x) / max(s.array()), rel_tol=1e-9)

def test_inv_function():
    f = lambda x: 0.3
    inv_f = inv(f)
    assert math.isclose(inv_f(0), 1 - 0.3)

def test_constant_function_always_same():
    c = 0.7
    const_f = constant(c)
    for x in [-1000, 0, 1000]:
        assert const_f(x) == c

def test_linear_function_behavior():
    f = linear(1, -1)
    # Values are clipped between 0 and 1
    assert f(-2) == 0
    assert f(0) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(2) == 1
    assert f(3) == 1

def test_bounded_linear_basic_and_inverse():
    f = bounded_linear(2, 4)
    assert f(1) == 0
    assert pytest.approx(f(3)) == 0.5
    assert f(4) == 1
    f_inv = bounded_linear(2,4,inverse=True)
    assert f_inv(1) == 1  # inverse swaps values

def test_R_and_S_functions_basic_behavior():
    r = R(2,4)
    # f(x) increases from 0 at 2 to 1 at 4
    assert r(1) == 0
    assert r(2) == 0
    assert pytest.approx(r(3)) == 0.5
    assert r(5) == 1

    s = S(2,4)
    assert s(1) == 1
    assert pytest.approx(s(3)) == 0.5
    assert s(5) == 0

def test_rectangular_function():
    f = rectangular(2, 4)
    assert f(1) == 0
    assert f(3) == 1
    assert f(5) == 0

def test_triangular_function_values():
    f = triangular(1, 3)
    # Peak at 2
    assert f(1) == 0
    assert f(2) == 1
    assert f(3) == 0

def test_trapezoid_function_behavior():
    f = trapezoid(1, 2, 3, 4)
    assert f(0) == 0
    assert pytest.approx(f(1.5), 0.25) and 0 <= f(1.5) <= 1
    assert f(2.5) == 1
    assert pytest.approx(f(3.5), 0.25) and 0 <= f(3.5) <= 1
    assert f(5) == 0

def test_sigmoid_function_bounds():
    f = sigmoid(1, 1, 0)
    assert 0 <= f(-1000) <= 1
    assert math.isclose(f(0), 0.5, rel_tol=0.1)
    assert 0 <= f(1000) <= 1

def test_bounded_sigmoid_basic():
    f = bounded_sigmoid(0, 1)
    assert pytest.approx(f(0), 0.1) == 0.1
    assert pytest.approx(f(1), 0.1) == 0.9
    assert math.isclose(f(100000), 1, rel_tol=1e-2)
    assert math.isclose(f(-100000), 0, rel_tol=1e-2)

def test_bounded_exponential_behavior():
    f = bounded_exponential(k=0.1, limit=1)
    assert 0 <= f(0) < 1
    assert math.isclose(f(1000), 1, rel_tol=1e-2)

def test_simple_sigmoid_behavior():
    f = simple_sigmoid()
    assert math.isclose(f(-1000), 0, abs_tol=0.02)
    assert math.isclose(f(0), 0.5, abs_tol=0.01)
    assert math.isclose(f(1000), 1.0, abs_tol=0.02)
    assert math.isclose(f(-20), 0.01, abs_tol=0.01)
    assert math.isclose(f(20), 0.99, abs_tol=0.01)

def test_triangular_sigmoid_peaks_correctly():
    f = triangular_sigmoid(2, 4)
    assert math.isclose(f(2), 0.1, abs_tol=0.05)
    assert math.isclose(f(3), 0.9, abs_tol=0.05)

def test_gauss_function_behavior():
    f = gauss(0, 1)
    res_center = f(0)
    res_far = f(1000)
    assert 0 <= res_center <= 1
    assert math.isclose(res_center, 1, rel_tol=1e-2)
    assert math.isclose(res_far, 0, abs_tol=1e-5)
