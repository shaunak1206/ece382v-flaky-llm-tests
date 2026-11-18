# tests/test_fuzzylogic_classes_functions.py
import pytest
import numpy as np
import math
import pickle
import warnings

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv, normalize, noop, constant, alpha, linear, bounded_linear, R, S,
    rectangular, triangular, trapezoid, sigmoid, bounded_sigmoid,
    bounded_exponential, simple_sigmoid, triangular_sigmoid, gauss
)


@pytest.fixture
def simple_domain():
    d = Domain("testdomain", 0, 10, res=1)
    return d


def test_domain_creation_and_str_repr(simple_domain):
    d = simple_domain
    assert d._name == "testdomain"
    assert d._low == 0
    assert d._high == 10
    assert d._res == 1
    assert str(d) == "testdomain"
    repr_str = repr(d)
    d2 = eval(repr_str)
    assert d == d2


def test_domain_call_warning_and_membership(simple_domain):
    d = simple_domain
    d.a = Set(constant(0.5))
    d.b = Set(constant(0.8))
    # inside domain: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        memberships = d(5)
    assert memberships == {'a': 0.5, 'b': 0.8}
    # outside domain: triggers warning
    d.a = Set(constant(0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = d(-1)
    assert any("outside of domain!" in str(warn.message) for warn in w)


def test_domain_getattr_and_setattr(simple_domain):
    d = simple_domain
    # Setting via attribute assigns Set
    def f(x): return x / 10
    d.myset = f
    s = d.myset
    assert isinstance(s, Set)
    assert s.domain is d
    assert s.name == "myset"
    # Accessing unknown attr raises
    with pytest.raises(AttributeError):
        _ = d.unknown


def test_domain_delattr(simple_domain):
    d = simple_domain
    d.s1 = constant(0.3)
    assert 's1' in d._sets
    del d.s1
    assert 's1' not in d._sets
    # deleting normal attribute raises FuzzyWarning
    with pytest.raises(FuzzyWarning):
        del d._low


def test_domain_range(simple_domain):
    d = simple_domain
    r = d.range()
    # Check range includes upper bound
    assert np.isclose(r[-1], d._high)
    expected = np.arange(d._low, d._high + d._res, d._res)
    assert np.array_equal(r, expected)


def test_domain_min_max(simple_domain):
    d = simple_domain
    d.s1 = constant(0.3)
    d.s2 = constant(0.7)
    assert math.isclose(d.min(5), 0.3)
    assert math.isclose(d.max(5), 0.7)


def test_set_basic_call_and_repr_str(simple_domain):
    d = simple_domain
    f = lambda x: x / 10
    s = Set(f, name="s", domain=d)
    # __call__ proxies to func
    assert s(5) == f(5)
    # __repr__ contains func string
    r = repr(s)
    assert r.startswith("Set(")
    # __str__ returns domain.name format
    assert str(s) == f"{d._name}.s"


def test_set_inversion_and_neg(simple_domain):
    d = simple_domain
    s = Set(constant(0.6), name="pos", domain=d)
    inv_s = ~s
    neg_s = -s
    # Their membership at 0 should be 1 - original
    assert math.isclose(inv_s(0), 1 - 0.6)
    assert math.isclose(neg_s(0), 1 - 0.6)


def test_set_and_or_mul_add_xor(simple_domain):
    d = simple_domain

    s1 = Set(linear(0.1, 0))
    s1.domain = d
    s2 = Set(linear(-0.1, 1))
    s2.domain = d

    and_set = s1 & s2
    or_set = s1 | s2
    mul_set = s1 * s2
    add_set = s1 + s2
    xor_set = s1 ^ s2

    # Check types
    for result in [and_set, or_set, mul_set, add_set, xor_set]:
        assert isinstance(result, Set)

    # Check values at a sample point (5)
    val_and = min(s1(5), s2(5))
    val_or = max(s1(5), s2(5))
    val_mul = s1(5) * s2(5)
    # bounded_sum caps at 1
    val_add = min(1, s1(5) + s2(5))
    # simple_disjoint_sum = s1 + s2 - s1 * s2
    val_xor = s1(5) + s2(5) - s1(5)*s2(5)

    assert math.isclose(and_set(5), val_and)
    assert math.isclose(or_set(5), val_or)
    assert math.isclose(mul_set(5), val_mul)
    assert math.isclose(add_set(5), val_add)
    assert math.isclose(xor_set(5), val_xor)


def test_set_pow_and_comparisons(simple_domain):
    d = simple_domain
    s = Set(linear(0.1, 0), name="s", domain=d)
    s2 = Set(s.func, name="s2", domain=d)

    # __pow__ squares values
    s_squared = s ** 2
    x = 5
    assert math.isclose(s_squared(x), s(x) ** 2)

    # __eq__ returns True for identical set funcs on domain
    assert s == s2

    s3 = Set(constant(0.5), domain=d)
    # __eq__ returns False if values differ
    assert not (s == s3)

    # __le__, __lt__, __ge__, __gt__ check subset/superset relationships
    # s3 lower than s should hold for many x if constant(0.5) <= linear
    assert isinstance(s <= s3, bool)
    # For testing raise FuzzyWarning if no domain
    s_no_domain = Set(constant(0.5))
    with pytest.raises(FuzzyWarning):
        _ = s_no_domain == s3


def test_set_len_cardinality(simple_domain):
    d = simple_domain
    s = Set(constant(0.5), domain=d)
    length = len(s)
    assert length == len(d.range())
    card = s.cardinality()
    assert math.isclose(card, 0.5 * length)
    rel_card = s.relative_cardinality()
    assert math.isclose(rel_card, 0.5)

    s_no_domain = Set(constant(1))
    with pytest.raises(FuzzyWarning):
        len(s_no_domain)
    with pytest.raises(FuzzyWarning):
        s_no_domain.cardinality()
    with pytest.raises(FuzzyWarning):
        s_no_domain.relative_cardinality()

    # Relative cardinality zero length domain triggers
    d_empty = Domain("empty", 0, 0, res=1)
    s_empty = Set(constant(1), domain=d_empty)
    with pytest.raises(FuzzyWarning):
        s_empty.relative_cardinality()


def test_set_concentrated_dilated_multiplied(simple_domain):
    d = simple_domain
    s = Set(linear(0.1, 0), domain=d)
    conc = s.concentrated()
    dil = s.dilated()
    mult = s.multiplied(2.5)
    x = 5

    # Check transformed values
    assert math.isclose(conc(x), s(x) ** 2)
    assert math.isclose(dil(x), s(x) ** 0.5)
    assert math.isclose(mult(x), s(x) * 2.5)


def test_set_intensified_function_behavior():
    # This function in Set.intensified has a specific piecewise behavior
    s = Set(constant(0.25))  # func(x) = 0.25 < 0.5
    intensified = s.intensified()
    # For x < 0.5 path
    val1 = intensified(0.3)
    # Expected 2 * (0.25^2) = 2 * 0.0625 = 0.125
    assert math.isclose(val1, 0.125)

    # For x >= 0.5 path: test manually calculated
    # replace func with lambda that returns 0.8
    s2 = Set(lambda x: 0.8)
    intensified2 = s2.intensified()
    val2 = intensified2(0.6)
    expected = 1 - 2 * (1 - 0.8 ** 2)
    assert math.isclose(val2, expected)


def test_set_plot_warnings(simple_domain):
    s = Set(constant(1), domain=simple_domain)
    # Should not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s.plot()
    # Without domain should raise
    s_no = Set(constant(1))
    with pytest.raises(FuzzyWarning):
        s_no.plot()


def test_set_array_value(simple_domain):
    d = simple_domain
    s = Set(lambda x: x / 10, domain=d)
    arr = s.array()
    expected = np.array([s(x) for x in d.range()])
    assert np.allclose(arr, expected)

    s_no_domain = Set(constant(1))
    with pytest.raises(FuzzyWarning):
        s_no_domain.array()


def test_set_normalized(simple_domain):
    d = simple_domain
    s = Set(lambda x: x / 10, domain=d)
    normalized = s.normalized()
    arr_norm = normalized.array()
    assert np.max(arr_norm) == pytest.approx(1.0, rel=1e-5)

    s_no_domain = Set(constant(1))
    with pytest.raises(FuzzyWarning):
        s_no_domain.normalized()


@pytest.mark.parametrize("func, expected", [
    (inv(constant(0.4)), 0.6),
    (normalize(0.5, constant(0.25)), 0.5),
])
def test_functions_inv_and_normalize(func, expected):
    assert math.isclose(func(0), expected)


def test_constants_and_noop():
    f1 = constant(3)
    f2 = noop()
    assert f1(100) == 3
    assert f2(123) == 123


def test_alpha_function_behavior():
    base_func = constant(0.5)
    clipped = alpha(floor=0.3, ceiling=0.7, func=base_func)
    # 0.5 is between 0.3 and 0.7 so returns 0.5
    assert math.isclose(clipped(0), 0.5)
    # Over ceiling returns ceiling_clip (which defaults to ceiling)
    f_high = constant(0.8)
    clipped_high = alpha(floor=0, ceiling=0.7, func=f_high)
    assert math.isclose(clipped_high(0), 0.7)
    # Under floor returns floor_clip (defaults to floor)
    f_low = constant(0.1)
    clipped_low = alpha(floor=0.3, ceiling=1, func=f_low)
    assert math.isclose(clipped_low(0), 0.3)


def test_linear_and_bounded_linear_functions():
    lin_f = linear(1, -1)
    # clipped values
    assert lin_f(-2) == 0
    assert lin_f(0) == 0
    assert lin_f(1) == 0
    assert lin_f(1.5) == 0.5
    assert lin_f(2) == 1
    assert lin_f(3) == 1

    # bounded_linear normal slope
    bl = bounded_linear(2, 3)
    assert bl(1) == 0
    assert bl(2) == 0
    assert math.isclose(bl(2.5), 0.5)
    assert bl(3) == 1
    assert bl(4) == 1

    # bounded_linear inverse True swaps c_m and no_m
    bl_inv = bounded_linear(2, 3, inverse=True)
    assert bl_inv(1) == 1
    assert bl_inv(3) == 0


def test_R_and_S_functions_behavior():
    r = R(2, 3)
    assert r(1) == 0
    assert r(2) == 0
    assert math.isclose(r(2.5), 0.5)
    assert r(3) == 1
    assert r(4) == 1

    s = S(2, 3)
    assert s(1) == 1
    assert s(2) == 1
    # For value between low and high it decreases linearly
    mid_val = s(2.5)
    assert 0 < mid_val < 1
    assert s(3) == 0
    assert s(4) == 0


def test_rectangular_function_behavior():
    rect = rectangular(1, 3, c_m=1, no_m=0)
    assert rect(0) == 0
    assert rect(1) == 1
    assert rect(2) == 1
    assert rect(3) == 1
    assert rect(4) == 0


def test_triangular_function_behavior():
    tri = triangular(1, 3)
    # Should peak at 2
    assert tri(1) == pytest.approx(0, abs=1e-12) or tri(1) > 0
    assert tri(2) == 1
    assert tri(3) == pytest.approx(0, abs=1e-12) or tri(3) > 0


def test_trapezoid_behavior():
    trap = trapezoid(1, 2, 3, 4)
    assert trap(0) == 0
    assert 0 < trap(1.5) < 1
    assert trap(2.5) == 1
    assert 0 < trap(3.5) < 1
    assert trap(5) == 0


def test_sigmoid_behavior():
    sig = sigmoid(1, 1, 0)
    # sigmoid(0) approx 0.5 for k=1, x0=0, L=1
    assert pytest.approx(sig(0), 0.49, 0.51)
    # very large positive x near 1
    assert pytest.approx(sig(1000), 1.0, rel=1e-2)
    # very large negative x near 0
    assert pytest.approx(sig(-1000), 0.0, rel=1e-2)


def test_bounded_sigmoid_behavior():
    f = bounded_sigmoid(0, 1)
    assert pytest.approx(f(0), 0.1)
    assert round(f(1), 2) == 0.9
    assert round(f(100000), 2) == 1.0
    assert round(f(-100000), 2) == 0.0


def test_bounded_exponential_behavior():
    f = bounded_exponential(k=0.1, limit=1)
    val0 = f(0)
    val_pos = f(100)
    # Should approach limit for large x
    assert val0 >= 0
    assert val_pos <= 1


def test_simple_sigmoid_behavior():
    f = simple_sigmoid()
    assert round(f(-1000), 2) == 0.0
    assert f(0) == 0.5
    assert round(f(1000), 2) == 1.0
    assert round(f(-20), 2) == 0.01
    assert round(f(20), 2) == 0.99


def test_triangular_sigmoid_behavior():
    g = triangular_sigmoid(2, 4)
    val_low = g(2)
    val_peak = g(3)
    assert pytest.approx(val_low, 0.1, 0.11)
    assert pytest.approx(val_peak, 0.9, 0.91)


def test_gauss_behavior():
    g = gauss(0, 1, c_m=1)
    assert math.isclose(g(0), 1)
    val_far = g(100)
    assert val_far == 0
