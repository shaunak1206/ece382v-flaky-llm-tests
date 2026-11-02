import math
import pickle
import warnings
import numpy as np
import pytest

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv, normalize, noop, constant, alpha, linear, bounded_linear, R, S,
    rectangular, triangular, trapezoid, sigmoid, bounded_sigmoid,
    bounded_exponential, simple_sigmoid, triangular_sigmoid, gauss
)

@pytest.mark.parametrize("name, low, high, res", [
    ("temp", 0, 100, 1),
    ("humidity", 0, 50, 0.5),
])
def test_domain_init_and_str_repr_eq(name, low, high, res):
    d1 = Domain(name, low, high, res=res)
    d2 = Domain(name, low, high, res=res)
    assert str(d1) == name
    repr_str = repr(d1)
    # eval of repr produces an equal Domain (except _sets might differ if empty)
    d3 = eval(repr_str)
    assert d1 == d3
    assert d1 == d2

def test_domain_call_warns_outside(monkeypatch):
    domain = Domain("test", 0, 10)
    domain.testset = Set(lambda x: x / 10)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", FuzzyWarning)
        domain(-1)
        domain(11)
        assert any("outside of domain" in str(wrn.message) for wrn in w)

def test_domain_call_returns_correct_memberships():
    d = Domain("d", 0, 5)
    d.low = Set(constant(0.0))
    d.high = Set(constant(1.0))
    res = d(3)
    assert isinstance(res, dict)
    assert res == {"low": 0.0, "high": 1.0}

def test_domain_setattr_getattr_delattr(monkeypatch):
    d = Domain("d", 0, 10)
    s = Set(constant(0.5))
    # Assign a set via attribute
    d.medium = s
    assert d.medium is s
    # The domain attribute of set is assigned correctly
    assert s.domain is d
    # The name is set correctly
    assert s.name == "medium"
    # Deleting a set works
    del d.medium
    with pytest.raises(AttributeError):
        _ = d.medium
    # Deleting a non-set attribute raises FuzzyWarning
    with pytest.raises(FuzzyWarning):
        del d._low

def test_domain_range_includes_high():
    domain = Domain("d", 0, 5, res=1)
    rng = domain.range()
    assert rng[0] == 0
    # The last value should be the high value exactly (not exceeded)
    assert math.isclose(rng[-1], domain._high)

def test_domain_min_max(monkeypatch):
    domain = Domain("d", 0, 5)
    domain.low = Set(constant(0))
    domain.mid = Set(constant(0.5))
    domain.high = Set(constant(1))
    assert domain.min(3) == 0
    assert domain.max(3) == 1

def test_set_call_and_invert(monkeypatch):
    d = Domain("test", 0, 10)
    original = Set(lambda x: x / 10)
    d.s1 = original
    inv_set = ~original
    # Inversion flips membership value
    assert math.isclose(inv_set(5), 1 - original(5))
    # Unary minus also inverts
    neg_set = -original
    assert math.isclose(neg_set(5), 1 - original(5))

def test_set_and_or_mul_add_xor_pow(monkeypatch):
    d = Domain("d", 0, 1, res=0.1)

    f1 = lambda x: 0.5
    f2 = lambda x: 0.7
    s1 = Set(f1)
    s2 = Set(f2)

    d.s1 = s1
    d.s2 = s2

    s_and = s1 & s2
    s_or = s1 | s2
    s_mul = s1 * s2
    s_add = s1 + s2
    s_xor = s1 ^ s2
    s_pow = s1 ** 2

    x = 0.5
    # MIN check
    assert math.isclose(s_and(x), min(f1(x), f2(x)))
    # MAX check
    assert math.isclose(s_or(x), max(f1(x), f2(x)))
    # Product check
    assert math.isclose(s_mul(x), f1(x) * f2(x))
    # Bounded sum check (capped at 1)
    bs = min(1, f1(x) + f2(x))
    assert math.isclose(s_add(x), bs)
    # Simple disjoint sum check per definition: sum - product
    sds = f1(x) + f2(x) - f1(x) * f2(x)
    assert math.isclose(s_xor(x), sds)
    # Power check
    assert math.isclose(s_pow(x), f1(x) ** 2)

def test_set_eq_and_comparisons(monkeypatch):
    d = Domain("d", 0, 1, res=0.1)
    f_const_0_5 = constant(0.5)
    s1 = Set(f_const_0_5)
    s2 = Set(f_const_0_5)
    d.s1 = s1
    d.s2 = s2
    assert s1 == s2
    # Strictly less returns False here - same values
    assert not s1 < s2
    # Less or equal returns True for identical values
    assert s1 <= s2
    # Strictly greater returns False
    assert not s1 > s2
    # Greater or equal returns True
    assert s1 >= s2

def test_set_len_and_cardinalities(monkeypatch):
    d = Domain("d", 0, 10, res=1)
    f = linear(1/10, 0)
    s = Set(f)
    d.s = s
    length = len(s)
    assert length == 11  # from 0 to 10 inclusive
    card = s.cardinality()
    # sum of 0,0.1,0.2,...1
    expected = sum(f(x) for x in range(0, 11))
    assert math.isclose(card, expected)
    rel_card = s.relative_cardinality()
    assert math.isclose(rel_card, expected / length)

def test_set_cardinality_errors(monkeypatch):
    s = Set(constant(0))
    with pytest.raises(FuzzyWarning):
        _ = s.cardinality()
    with pytest.raises(FuzzyWarning):
        _ = len(s)
    with pytest.raises(FuzzyWarning):
        _ = s.relative_cardinality()
    # Relative cardinality no domain but non-empty array will not happen, so skip.

def test_concentrated_and_dilated(monkeypatch):
    d = Domain("d", 0, 1, res=0.5)
    f = constant(0.4)
    s = Set(f)
    d.s = s
    conc = s.concentrated()
    dil = s.dilated()
    # Values should be squared or square rooted
    for x in d.range():
        assert math.isclose(conc(x), s(x) ** 2)
        # sqrt of 0.4 approximated by power 0.5
        assert math.isclose(dil(x), s(x) ** 0.5)

def test_multiplied(monkeypatch):
    d = Domain("d", 0, 1, res=0.1)
    s = Set(constant(0.3))
    d.s = s
    mul = s.multiplied(2)
    for x in d.range():
        assert math.isclose(mul(x), s(x) * 2)

def test_normalized(monkeypatch):
    d = Domain("d", 0, 1, res=0.1)
    f = constant(0.5)
    s = Set(f)
    d.s = s
    norm = s.normalized()
    # max value after normalization should be 1
    arr = norm.array()
    assert np.isclose(arr.max(), 1.0)

def test_array_and_plot(monkeypatch):
    d = Domain("d", 0, 2, res=1)
    s = Set(lambda x: x / 2)
    d.s = s
    arr = s.array()
    expected = np.array([0.0, 0.5, 1.0])
    assert np.allclose(arr, expected)
    # plot() should run without error
    s.plot()

def test_set_init_name_domain_assignment(monkeypatch):
    d = Domain("d", 0, 1)
    def f(x): return x / 10
    s = Set(f, name="s", domain=d)
    # s should be assigned to d.s
    assert d.s is s
    # The name and domain should be set correctly
    assert s.name == "s"
    assert s.domain is d
    # Passing only name or only domain raises warning
    with pytest.warns(FuzzyWarning):
        Set(f, name="name_only")
    with pytest.warns(FuzzyWarning):
        Set(f, domain=d)

def test_set_repr_and_str(monkeypatch):
    f = constant(0.5)
    s = Set(f)
    repr_str = repr(s)
    assert repr_str.startswith("Set(")
    # Without domain and name, str shows dangling Set
    assert "dangling" in str(s)
    d = Domain("d", 0, 1)
    s2 = Set(f, name="s", domain=d)
    assert str(s2) == "d.s"

# --- TESTS FOR FUNCTIONS.PY ---

def test_inv_function_behavior():
    f = constant(0.3)
    invf = inv(f)
    for x in [0, 0.5, 1]:
        assert math.isclose(invf(x), 1 - f(x))

def test_noop_function():
    f = noop()
    for x in [0, 1, -5, 100]:
        assert f(x) == x

def test_constant_function():
    c = 0.7
    f = constant(c)
    for x in [-100, 0, 100]:
        assert f(x) == c

def test_alpha_function_behavior():
    base = constant(0.5)
    f = alpha(floor=0.2, ceiling=0.8, func=base)
    assert f(0) == 0.5  # base value
    # Test >= ceiling returns ceiling_clip (= ceiling default 0.8)
    with pytest.approx(0.8):
        assert f(10) == pytest.approx(0.8)
    # Test <= floor returns floor_clip (= floor default 0.2)
    assert f(-10) == 0.2

def test_linear_function_clipping():
    f = linear(1, -1)  # y = x - 1
    assert f(-2) == 0  # clipped
    assert f(0) == 0   # clipped
    assert f(1) == 0
    assert f(1.5) == 0.5
    assert f(2) == 1
    assert f(3) == 1   # clipped

def test_bounded_linear_behavior():
    f = bounded_linear(2, 3)
    assert f(1) == 0
    assert f(2) == 0
    assert math.isclose(f(2.5), 0.5)
    assert f(3) == 1
    assert f(4) == 1

def test_R_function():
    f = R(2, 3)
    assert f(1) == 0
    assert f(2) == 0
    assert math.isclose(f(2.5), 0.5)
    assert f(3) == 1
    assert f(4) == 1

def test_S_function():
    f = S(2, 3)
    assert f(1) == 1
    assert f(2) == 1
    assert math.isclose(f(2.5), 0.5)
    assert f(3) == 0
    assert f(4) == 0

def test_rectangular_function():
    f = rectangular(2, 4)
    assert f(1) == 0
    assert f(2) == 1
    assert f(3) == 1
    assert f(4) == 1
    assert f(5) == 0

def test_triangular_function_basic():
    f = triangular(1, 3)
    assert f(1) == 0
    assert f(2) == pytest.approx(1)
    assert f(2.5) > 0 and f(2.5) < 1
    assert f(3) == 0

def test_trapezoid_function():
    f = trapezoid(1, 2, 3, 4)
    assert f(0) == 0
    assert 0 < f(1.5) < 1
    assert f(2.5) == 1
    assert 0 < f(3.5) < 1
    assert f(5) == 0

def test_sigmoid_function_extremes():
    f = sigmoid(1, 1, 0)
    # Check values are within (0,1)
    assert 0 <= f(-1000) <= 1
    assert 0 <= f(0) <= 1
    assert 0 <= f(1000) <= 1

def test_bounded_sigmoid_behavior():
    f = bounded_sigmoid(0, 1)
    assert pytest.approx(f(0), 0.01) == 0.1
    assert pytest.approx(f(1), 0.01) == 0.9
    assert f(100000) >= 0.99
    assert f(-100000) <= 0.01

def test_bounded_exponential_behavior():
    f = bounded_exponential(0.1, 1)
    assert abs(f(0)) < 1e-7
    assert 0 <= f(10) <= 1
    assert math.isclose(f(1000), 1, rel_tol=1e-3)

def test_simple_sigmoid_behavior():
    f = simple_sigmoid()
    assert pytest.approx(f(-1000), 0.01) == 0.0
    assert math.isclose(f(0), 0.5)
    assert pytest.approx(f(1000), 0.01) == 1.0

def test_triangular_sigmoid_behavior():
    f = triangular_sigmoid(2, 4)
    assert pytest.approx(f(2), 0.01) == 0.1
    assert pytest.approx(f(3), 0.01) == 0.9

def test_gauss_function_behavior():
    f = gauss(0, 1, c_m=1)
    assert math.isclose(f(0), 1)
    assert 0 <= f(2) <= 1
    val_far = f(1000)
    # Very far from center should be near 0
    assert val_far < 1e-5

