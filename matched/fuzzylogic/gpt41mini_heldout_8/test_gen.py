import pytest
import math
import numpy as np
from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import inv, normalize, constant, linear, bounded_linear, R, S, rectangular, triangular, trapezoid, sigmoid

# Helper functions for tests

def almost_equal(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_domain_init_and_attributes():
    d = Domain('temperature', 0, 100, res=2)
    assert d._name == 'temperature'
    assert d._low == 0
    assert d._high == 100
    assert d._res == 2
    assert isinstance(d._sets, dict)

    # Setting sets dict
    d2 = Domain('pressure', 10, 50, sets={'hot': Set(constant(0.5))})
    assert 'hot' in d2._sets

def test_domain_call_memberships(monkeypatch):
    d = Domain('temp', 0, 10)
    d.hot = Set(lambda x: 1 if x > 5 else 0)
    d.cold = Set(lambda x: 1 if x < 5 else 0)
    membership = d(7)
    assert set(membership.keys()) == {'hot', 'cold'}
    assert membership['hot'] == 1
    assert membership['cold'] == 0

def test_domain_call_warns_out_of_bounds(recwarn):
    d = Domain('temp', 0, 10)
    d.hot = Set(constant(0))
    _ = d(-1)
    w = recwarn.pop(FuzzyWarning)
    assert "outside of domain" in str(w.message)

def test_domain_str_repr_eq():
    d1 = Domain('temp', 0, 10)
    d1.hot = Set(constant(0))
    d2 = Domain('temp', 0, 10, sets=d1._sets.copy())
    assert str(d1) == 'temp'
    rep = repr(d1)
    assert 'Domain' in rep
    # Equality test
    assert d1 == d2

def test_domain_getattr_and_setattr_and_delattr():
    d = Domain('temp', 0, 10)
    s = Set(constant(0))
    d.freezing = s
    assert d.freezing is s
    # Delete existing set
    del d.freezing
    with pytest.raises(AttributeError):
        _ = d.freezing
    # Del non-set attribute triggers warning
    with pytest.warns(FuzzyWarning):
        del d._name

def test_domain_range_min_max():
    d = Domain('temp', 0, 10, res=2)
    d.low = Set(constant(0))
    d.high = Set(constant(1))
    np.testing.assert_array_equal(d.range(), np.arange(0, 12, 2))
    assert d.min(5) == 0
    assert d.max(5) == 1

def test_set_call_and_invert_and_neg():
    s = Set(lambda x: 0.2 * x)
    assert almost_equal(s(3), 0.6)
    inv_s = ~s
    neg_s = -s
    for x in [0, 0.5, 1, 2]:
        assert almost_equal(inv_s(x), 1 - s(x))
        assert almost_equal(neg_s(x), 1 - s(x))

def test_set_and_or_mul_add_xor_pow():
    s1 = Set(lambda x: min(1, max(0, x/10)))
    s2 = Set(lambda x: max(0, min(1, (10 - x)/10)))
    s_and = s1 & s2
    s_or = s1 | s2
    s_mul = s1 * s2
    s_add = s1 + s2
    s_xor = s1 ^ s2
    s_pow = s1 ** 2

    for x in [0, 2.5, 5, 7.5, 10]:
        assert almost_equal(s_and(x), min(s1(x), s2(x)))
        assert almost_equal(s_or(x), max(s1(x), s2(x)))
        assert almost_equal(s_mul(x), s1(x) * s2(x))
        # bounded_sum ensures output <= 1, so s_add(x) <= 1
        assert 0 <= s_add(x) <= 1
        # simple_disjoint_sum also caps to 1
        assert 0 <= s_xor(x) <= 1
        assert almost_equal(s_pow(x), s1(x)**2)

def test_set_eq_and_comparisons():
    d = Domain('temp', 0, 10)
    s1 = Set(lambda x: x / 10, name='s1', domain=d)
    s2 = Set(lambda x: x / 10, name='s2', domain=d)
    s3 = Set(lambda x: (10 - x) / 10, name='s3', domain=d)
    # Equal sets
    assert s1 == s2
    # Different sets
    assert s1 != s3
    # Less or equal
    assert s1 <= s2
    # Proper subset?
    assert not (s1 < s2)  # equal sets => not proper subset
    # Greater or equal
    assert s2 >= s1
    # Proper superset?
    assert not (s2 > s1)  # equal sets => not proper superset
    # Proper less
    s4 = Set(lambda x: 0.5 * (x/10), name='s4', domain=d)
    assert s4 < s1
    assert s1 > s4

def test_set_len_cardinalities_relative():
    d = Domain('temp', 0, 5, res=1)
    s = Set(lambda x: x / 5, name='s', domain=d)
    length = len(s)
    assert length == 6
    card = s.cardinality()
    rel_card = s.relative_cardinality()
    # Cardinality sum of range 0..1 increasing linearly
    expected_card = sum(x / 5 for x in d.range())
    assert almost_equal(card, expected_card)
    assert almost_equal(rel_card, expected_card / length)
    # No domain raises warning in __len__
    s2 = Set(lambda x: x)
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            _ = len(s2)
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            _ = s2.cardinality()
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            _ = s2.relative_cardinality()

def test_set_concentrated_intensified_dilated_multiplied():
    d = Domain('unit', 0, 1, res=0.1)
    s = Set(lambda x: x, name='s', domain=d)

    concentrated = s.concentrated()
    for x in np.arange(0, 1.1, 0.1):
        val = s(x)
        assert almost_equal(concentrated(x), val * val)

    # Intensified - test function behavior
    intensified = s.intensified()
    for x in np.arange(0, 1.1, 0.1):
        base = s(x)
        if x < 0.5:
            expected = 2 * (base ** 2)
        else:
            expected = 1 - 2 * (1 - base ** 2)
        assert almost_equal(intensified(x), expected)

    dilated = s.dilated()
    for x in np.arange(0, 1.1, 0.1):
        expected = s(x) ** 0.5
        assert almost_equal(dilated(x), expected)

    multiplied = s.multiplied(2)
    for x in np.arange(0, 1.1, 0.1):
        expected = s(x) * 2
        assert almost_equal(multiplied(x), expected)

def test_set_plot_warning(monkeypatch):
    s = Set(lambda x: x)
    # No domain assigned - expect warning in plot()
    with pytest.warns(FuzzyWarning):
        s.plot()

def test_set_array_and_repr_str_normalized():
    d = Domain('temp', 0, 3, res=1)
    s = Set(lambda x: x/3, name='s', domain=d)
    arr = s.array()
    expected = np.array([0/3, 1/3, 2/3, 3/3])
    np.testing.assert_allclose(arr, expected)
    # repr should contain "Set"
    r = repr(s)
    assert r.startswith("Set")
    # str returns domain_name.set_name if possible
    assert str(s) == "temp.s"
    # normalized returns a Set, with max normalized to 1
    norm = s.normalized()
    for x in d.range():
        expected_val = s(x) / max(arr)
        assert almost_equal(norm(x), expected_val)
    # No domain warning in normalized
    s2 = Set(lambda x: x)
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            s2.normalized()


@pytest.mark.parametrize("func, values", [
    (inv(constant(0.2)), [0, 0.5, 1]),
    (constant(0.3), [0, 1, 100]),
    (linear(1, -1), [-2, 0, 1.5, 2, 3]),
    (bounded_linear(2, 3), [1, 2, 2.5, 3, 4]),
    (R(2, 3), [1.5, 2, 2.7, 3, 3.5]),
    (S(2, 3), [1, 2, 2.5, 3, 4]),
    (rectangular(1, 3), [0, 1, 2, 3, 4]),
    (triangular(1, 3), [0, 1, 2, 3, 4]),
    (trapezoid(1, 2, 3, 4), [0, 1.5, 2.5, 3.5, 4.5]),
    (sigmoid(1, 10, 0), [-1, 0, 1]),
])
def test_functions_basic_output_range(func, values):
    for x in values:
        y = func(x)
        # All outputs should be within [0,1]
        assert 0 <= y <= 1


def test_fuzzywarning_raised_on_partial_name_domain():
    d = Domain('temp', 0, 10)
    # Providing only name or only domain in Set should raise
    with pytest.warns(FuzzyWarning):
        Set(lambda x: x, name='test')
    with pytest.warns(FuzzyWarning):
        Set(lambda x: x, domain=d)

def test_domain_add_function_assigns_set():
    d = Domain('temp', 0, 10)
    # Assigning a raw callable creates a Set automatically
    def f(x):
        return 0.5
    d.warm = f
    assert isinstance(d.warm, Set)
    assert d.warm(0) == 0.5

def test_set_eq_raises_if_no_domains():
    s1 = Set(lambda x: x)
    s2 = Set(lambda x: x)
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            s1 == s2

def test_set_comparisons_raises_if_no_domains():
    s1 = Set(lambda x: x)
    s2 = Set(lambda x: x)
    for op in [lambda a,b: a <= b, lambda a,b: a < b, lambda a,b: a >= b, lambda a,b: a > b]:
        with pytest.warns(FuzzyWarning):
            with pytest.raises(FuzzyWarning):
                op(s1, s2)

def test_domain_forbidden_setattr():
    d = Domain('temp', 0, 10)

    # Attempt to set an attribute with non-identifier name causes AssertionError
    with pytest.raises(AssertionError):
        setattr(d, 'not-an-id', constant(1))

def test_domain_deleting_nonexistent_set_warns():
    d = Domain('temp', 0, 10)
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            del d.nonexistent_set
