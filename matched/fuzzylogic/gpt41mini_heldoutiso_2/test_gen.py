import pytest
import math
import numpy as np
from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import inv, normalize, constant, linear, bounded_linear, singleton, R, S, trapezoid, triangular, sigmoid, bounded_sigmoid, gauss

def test_domain_basic_properties():
    dom = Domain("temp", 0, 100, res=5)
    assert dom._name == "temp"
    assert dom._low == 0
    assert dom._high == 100
    assert dom._res == 5
    assert isinstance(dom._sets, dict)
    # range includes high + res step
    vals = dom.range()
    assert vals[0] == 0
    assert vals[-1] == 100
    assert all(vals[i] + 5 == vals[i+1] for i in range(len(vals)-1))


def test_domain_call_warning_outside(monkeypatch):
    dom = Domain("test", 0, 10)
    dom.testset = Set(constant(0.5))
    messages = []
    def fake_warn(msg):
        messages.append(str(msg))
    monkeypatch.setattr("fuzzylogic.classes.warn", fake_warn)
    res = dom(11)
    assert res == {"testset": 0.5}
    assert any("outside of domain" in m for m in messages)


def test_domain_str_repr_eq():
    dom1 = Domain("d", 0, 10)
    s1 = Set(constant(0.1))
    dom1.set1 = s1
    dom2 = Domain("d", 0, 10, sets={"set1": s1})
    # str returns name
    assert str(dom1) == "d"
    # repr returns eval-like string
    r = repr(dom1)
    assert r.startswith("Domain(")
    # Equality works
    assert dom1 == dom2


def test_domain_getattr_set_and_attribute_error():
    dom = Domain("test", 0, 5)
    s = Set(constant(0.2))
    dom.foo = s
    # Getting the set from domain works
    assert dom.foo == s
    # attribute error on unknown attribute
    with pytest.raises(AttributeError):
        _ = dom.unknownattr


def test_domain_setattr_set_and_non_set():
    dom = Domain("test", 0, 5)
    # Assigning a plain function auto-wraps in Set
    def f(x): return 0.9
    dom.fset = f
    assert isinstance(dom.fset, Set)
    assert dom.fset.func(0) == 0.9
    # Assign Set directly
    s = Set(constant(0.1))
    dom.sset = s
    assert dom.sset == s


def test_domain_delattr_and_warning(monkeypatch):
    dom = Domain("d", 0, 5)
    s = Set(constant(0.3))
    dom.myset = s
    assert "myset" in dom._sets
    dom.__delattr__("myset")
    assert "myset" not in dom._sets
    with pytest.raises(FuzzyWarning):
        dom.__delattr__("nonsetattr")


def test_set_call_and_invert_negate():
    s = Set(constant(0.6))
    assert s(10) == 0.6
    inv_s = ~s
    assert math.isclose(inv_s(10), 0.4)
    neg_s = -s
    assert math.isclose(neg_s(10), 0.4)


def test_set_binary_operators():
    s1 = Set(constant(0.3))
    s2 = Set(constant(0.7))
    s_and = s1 & s2
    s_or = s1 | s2
    s_mul = s1 * s2
    s_add = s1 + s2
    s_xor = s1 ^ s2
    # All return Set
    for s in [s_and, s_or, s_mul, s_add, s_xor]:
        assert isinstance(s, Set)
    # Check the values according to operator definitions
    assert math.isclose(s_and(0), min(0.3, 0.7))
    assert math.isclose(s_or(0), max(0.3, 0.7))
    # product: 0.3*0.7 = 0.21
    assert math.isclose(s_mul(0), 0.21)
    # bounded_sum: min(1, 0.3 + 0.7) = 1 (approx)
    assert math.isclose(s_add(0), 1, abs_tol=1e-9)
    # simple_disjoint_sum: 0.3 + 0.7 - 0.3*0.7 = 0.3 + 0.7 - 0.21 = 0.79
    assert math.isclose(s_xor(0), 0.79)


def test_set_pow_and_comparison_with_arrays():
    domain = Domain("d", 0, 10, res=2)
    func = constant(0.5)
    s = Set(func)
    s.domain = domain
    # powered set squares value
    s2 = s ** 2
    domain._sets["s"] = s
    domain._sets["s2"] = s2
    s2.domain = domain
    vals_orig = s.array()
    vals_pow = s2.array()
    assert np.allclose(vals_pow, vals_orig ** 2)

    # Equality with same domain and function values
    s3 = Set(func)
    s3.domain = domain
    assert s == s3

    # Inequality with different values
    s4 = Set(constant(0.7))
    s4.domain = domain
    # s < s4
    assert (s < s4) == True
    # s4 > s
    assert (s4 > s) == True
    # s <= s4
    assert (s <= s4) == True
    # s4 >= s
    assert (s4 >= s) == True

    # Comparisons raise warning if no domain
    s_nod = Set(func)
    s2_nod = Set(func)
    with pytest.raises(FuzzyWarning):
        _ = s_nod == s2_nod
    with pytest.raises(FuzzyWarning):
        _ = s_nod <= s2_nod


def test_set_len_cardinalities_rel_cardinality():
    dom = Domain("dom", 0, 4, res=1)
    s = Set(constant(0.25))
    s.domain = dom
    dom._sets["s"] = s
    arr = s.array()
    assert len(s) == len(arr)
    assert s.cardinality() == sum(arr)
    assert math.isclose(s.relative_cardinality(), s.cardinality()/len(arr))
    # Relative cardinality raises warning if domain empty
    dom_empty = Domain("empty", 0, 0, res=1)
    s_empty = Set(constant(0.5))
    s_empty.domain = dom_empty
    with pytest.raises(FuzzyWarning):
        _ = s_empty.relative_cardinality()


def test_set_concentrated_dilated_multiplied():
    s = Set(constant(0.5))
    s.domain = Domain("d", 0, 1, res=0.5)
    s.domain._sets["s"] = s

    s_conc = s.concentrated()
    s_dil = s.dilated()
    s_mul = s.multiplied(2)

    # Concentrated squares the function value (0.5^2=0.25)
    assert math.isclose(s_conc(0), 0.25)
    # Dilated takes square root: sqrt(0.5) approx 0.707
    assert math.isclose(s_dil(0), math.sqrt(0.5))
    # Multiplied multiplies by factor 2
    assert math.isclose(s_mul(0), 1.0)  # capped by no cap, so 1.0


def test_set_intensified_function_behavior():
    # Define a func to test intensified
    def simple_func(x):
        return x  # identity, range arbitrary

    s = Set(simple_func)
    s.domain = Domain("d", 0, 1, res=0.1)
    intensified = s.intensified()
    # For x < 0.5: should be 2 * f(x)^2
    val1 = intensified(0.4)
    expected1 = 2 * (0.4**2)
    assert math.isclose(val1, expected1)
    # For x >= 0.5: 1 - 2(1 - f(x)^2) = 1 - 2 + 2 * f(x)^2 = -1 + 2 f(x)^2
    val2 = intensified(0.6)
    expected2 = 1 - 2 * (1 - 0.6**2)
    assert math.isclose(val2, expected2)


def test_set_plot_without_domain_raises():
    s = Set(constant(0.1))
    with pytest.raises(FuzzyWarning):
        s.plot()


def test_set_array_without_domain_warns():
    s = Set(constant(0.3))
    with pytest.raises(FuzzyWarning):
        s.array()


def test_set_normalized_behavior():
    dom = Domain("d", 0, 2, res=1)
    func = constant(0.5)
    s = Set(func)
    s.domain = dom
    dom._sets["s"] = s
    # normalized returns a set with values multiplied by 1/height
    norm = s.normalized()
    val_orig = s(1)
    val_norm = norm(1)
    assert math.isclose(val_norm, val_orig / max(s.array()))


def test_functions_inv():
    f = constant(0.2)
    f_inv = inv(f)
    assert f_inv(0) == pytest.approx(0.8)
    assert f_inv(1) == pytest.approx(0.8)


def test_functions_constant():
    c = constant(0.7)
    assert c(-100) == 0.7
    assert c(100) == 0.7


def test_functions_linear_and_bounds():
    f = linear(1, -1)
    assert f(-2) == 0
    assert f(0) == 0
    assert f(1.5) == pytest.approx(0.5)
    assert f(3) == 1


def test_functions_bounded_linear_basic():
    f = bounded_linear(2, 3)
    # At 2 should be 0
    assert f(2) == pytest.approx(0)
    # Between 2 and 3
    val = f(2.5)
    assert 0 < val < 1
    # At 3 should be 1
    assert f(3) == pytest.approx(1)
    # Outside bounds
    assert f(1) == pytest.approx(0)
    assert f(4) == pytest.approx(1)


def test_functions_singleton_values():
    f = singleton(2)
    assert f(1) == 0
    assert f(2) == 1


def test_functions_R_and_S_behavior():
    r = R(1, 3)
    s = S(1, 3)
    assert r(0) == 0
    assert math.isclose(r(2), 0.5)
    assert r(4) == 1
    assert s(0) == 1
    assert math.isclose(s(2), 0.5)
    assert s(4) == 0


def test_functions_trapezoid_values():
    f = trapezoid(0, 2, 4, 6)
    # at low bound
    assert f(0) == 0
    # at first climb
    val1 = f(1)
    assert 0 < val1 < 1
    # core
    assert f(3) == 1
    # descending slope
    val2 = f(5)
    assert 0 < val2 < 1
    # beyond high
    assert f(7) == 0


def test_functions_triangular_behavior():
    f = triangular(0, 4, c=2)
    # At low boundary
    assert f(0) == 0
    # At peak
    assert pytest.approx(f(2), 0.0001) == 1
    # At high boundary
    assert f(4) == 0
    # Before low
    assert f(-1) == 0
    # After high
    assert f(5) == 0


def test_functions_sigmoid_behavior():
    f = sigmoid(1, 5, 0)
    # Midpoint approx 0.5
    assert pytest.approx(f(0), 0.5, abs=0.01)
    # Far left small value
    assert f(-1000) < 0.01
    # Far right close to 1
    assert f(1000) > 0.99


def test_functions_bounded_sigmoid_behavior():
    f = bounded_sigmoid(0, 1)
    assert math.isclose(f(0), 0.1, abs_tol=0.01)
    assert math.isclose(f(1), 0.9, abs_tol=0.01)
    assert math.isclose(f(100000), 1.0, abs_tol=0.01)
    assert math.isclose(f(-100000), 0.0, abs_tol=0.01)


def test_functions_gauss_behavior():
    f = gauss(0, 1)
    # Peak at c returns c_m (default 1)
    assert math.isclose(f(0), 1)
    # Away from peak decreases
    assert f(10) < 0.01
    # Negative away from peak similarly low
    assert f(-10) < 0.01
