import pytest
import warnings
import math
import numpy as np

from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import inv, constant, normalize, linear, bounded_linear, R, S, rectangular, triangular, trapezoid, sigmoid, bounded_sigmoid, gauss


def test_domain_init_and_str_repr():
    d = Domain("temp", 0, 10, res=1)
    assert str(d) == "temp"
    repr_str = repr(d)
    assert repr_str.startswith("Domain(")
    # round-trip eval would fail because sets empty are empty dict
    d2 = eval(repr_str, {"Domain": Domain, "Set": Set, "dict": dict})
    assert d == d2

def test_domain_call_membership_warning_and_values():
    d = Domain("temp", 0, 5)
    d.cool = Set(constant(0.5))
    d.hot = Set(lambda x: 1 if x > 3 else 0)
    result = d(2)
    # warning expected for x outside domain
    assert "cool" in result and "hot" in result
    # test warning for out of domain
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d(-1)
        assert any(issubclass(warn.category, UserWarning) for warn in w)

def test_domain_getattr_setattr_delattr():
    d = Domain("test", 0, 1)
    fset = Set(constant(0.8))
    d.foo = fset
    assert d.foo is fset
    # value stored in _sets
    assert 'foo' in d._sets and d._sets['foo'] is fset
    # deleting set attribute removes it from _sets
    del d.foo
    assert 'foo' not in d._sets
    # Deleting a non-set attribute should raise FuzzyWarning
    with pytest.raises(FuzzyWarning):
        del d._res

def test_domain_range_inclusive_and_min_max():
    d = Domain("test", 0, 3, res=1)
    d.low = Set(lambda x: x)
    d.high = Set(lambda x: 1 - x/3)
    rng = d.range()
    # The range should include the upper bound
    assert rng[0] == 0
    assert pytest.approx(rng[-1]) == d._high
    # min and max testing at a value
    mn = d.min(1)
    mx = d.max(1)
    # min is min of both funcs at x=1
    expected_min = min(f.func(1) for f in d._sets.values())
    expected_max = max(f.func(1) for f in d._sets.values())
    assert mn == expected_min
    assert mx == expected_max

def test_set_call_and_invert():
    f = constant(0.3)
    s = Set(f)
    assert s(0) == 0.3
    s_inv = ~s
    assert s_inv(0) == pytest.approx(1 - 0.3)

def test_set_eq_and_domain_required_methods():
    d = Domain("d", 0, 10, res=1)
    f1 = linear(0.1, 0)
    f2 = linear(0.1, 0)
    s1 = Set(f1)
    s2 = Set(f2)
    # assign domains
    s1.domain = d
    s2.domain = d
    d.a = s1
    d.b = s2
    # sets with same function and domain should be equal
    assert s1 == s2
    # remove domain of one: checking raises warning
    s2.domain = None
    with pytest.raises(FuzzyWarning):
        s1 == s2

def test_comparison_subsets_and_supersets():
    d = Domain("d", 0, 10, res=1)
    s1 = Set(linear(0.1, 0))
    s2 = Set(constant(1))
    s1.domain = d
    s2.domain = d
    d.s1 = s1
    d.s2 = s2
    # s1 <= s2 always true because linear <= constant 1 for x in [0,10]
    assert s1 <= s2
    # s2 is not subset of s1
    assert not (s2 <= s1)
    # s1 < s2 (proper subset)
    assert s1 < s2
    # s2 > s1 (proper superset)
    assert s2 > s1
    # s2 >= s1
    assert s2 >= s1
    # s1 >= s2 false
    assert not (s1 >= s2)

def test_set_len_cardinality_relative_cardinality():
    d = Domain("d", 0, 5, res=1)
    s = Set(constant(0.5))
    s.domain = d
    d.s = s
    # len matches domain range length
    assert len(s) == len(d.range())
    # cardinality sums all membership values
    expected_sum = sum(s.func(x) for x in d.range())
    assert math.isclose(s.cardinality(), expected_sum)
    # relative_cardinality is avg membership
    rel_card = s.relative_cardinality()
    assert math.isclose(rel_card, expected_sum / len(d.range()))
    # relative_cardinality raises warning if domain empty (simulate)
    s2 = Set(constant(0))
    s2.domain = Domain("empty", 0, 0, res=1)
    with pytest.raises(FuzzyWarning):
        s2.relative_cardinality()

def test_set_concentrated_and_normalized():
    d = Domain("d", 0, 2, res=1)
    s = Set(lambda x: 0.5 * x)
    s.domain = d
    d.s = s
    concentrated = s.concentrated()
    # values squared, test some points explicitly
    for x in d.range():
        orig = s(x)
        conc = concentrated(x)
        assert math.isclose(conc, orig**2)
    # normalized returns set with max normalized to 1
    max_val = max(s.array())
    norm = s.normalized()
    for x in d.range():
        expected = s(x)/max_val
        assert math.isclose(norm(x), expected)

def test_set_operations_and_power():
    f1 = constant(0.2)
    f2 = constant(0.5)
    s1 = Set(f1)
    s2 = Set(f2)
    s_and = s1 & s2
    s_or = s1 | s2
    s_mul = s1 * s2
    s_add = s1 + s2
    s_xor = s1 ^ s2
    s_pow = s1 ** 2
    x = 0
    assert math.isclose(s_and(x), min(f1(x), f2(x)))
    assert math.isclose(s_or(x), max(f1(x), f2(x)))
    assert math.isclose(s_mul(x), f1(x)*f2(x))
    # bounded sum: min(1, a + b)
    expect_bounded_sum = min(1, f1(x) + f2(x))
    assert math.isclose(s_add(x), expect_bounded_sum)
    # simple disjoint sum: a + b - a*b
    expect_disjoint_sum = f1(x) + f2(x) - f1(x)*f2(x)
    assert math.isclose(s_xor(x), expect_disjoint_sum)
    # power
    assert math.isclose(s_pow(x), f1(x)**2)

def test_set_str_and_repr_output():
    d = Domain("foo", 0, 1)
    s = Set(constant(0.5))
    s.domain = d
    d.s = s
    s.name = "s"
    # str returns domain_name.set_name
    assert str(s) == "foo.s"
    r = repr(s)
    assert r.startswith("Set(")
    # dangling Set without domain and name
    s2 = Set(constant(0.8))
    assert str(s2).startswith("dangling Set(")

def test_set_plot_without_domain_warns(monkeypatch):
    s = Set(constant(0.5))
    with pytest.raises(FuzzyWarning):
        s.plot()
    # Patch domain to avoid warning and check plt.plot called
    d = Domain("d", 0, 3, res=1)
    s.domain = d
    d.s = s
    called = {}
    def fake_plot(*args, **kwargs):
        called['yes'] = True
    monkeypatch.setattr("matplotlib.pyplot.plot", fake_plot)
    s.plot()
    assert 'yes' in called

def test_inv_function_correctness():
    f = constant(0.3)
    inv_f = inv(f)
    for x in [0, 0.5, 1]:
        assert math.isclose(inv_f(x), 1 - f(x))

@pytest.mark.parametrize("func,val,expected", [
    (constant(0.5), 0, 0.5),
    (linear(1, -0.5), 1, 0.5),
    (bounded_linear(0, 1), 0.5, 0.5),
    (R(0, 1), 0.5, 0.5),
    (S(0, 1), 0.5, 0.5),
    (rectangular(0, 1), 0.5, 1),
])
def test_membership_functions_return_expected(func, val, expected):
    # Allow tolerance for floating computations
    result = func(val)
    assert math.isclose(result, expected, rel_tol=1e-6)

def test_sigmoid_bounds():
    f = sigmoid(1, 1, 0)
    # f(0) ~ 0.5
    assert 0.49 < f(0) < 0.51
    # sigmoid monotonic increasing
    assert f(-10000) < 0.1
    assert f(10000) > 0.9

def test_bounded_sigmoid_monotonicity_and_limits():
    f = bounded_sigmoid(0, 1)
    assert math.isclose(f(0), 0.1, abs_tol=1e-2)
    assert math.isclose(f(1), 0.9, abs_tol=1e-2)
    assert math.isclose(f(100000), 1.0, abs_tol=1e-2)
    assert math.isclose(f(-100000), 0.0, abs_tol=1e-2)

def test_gauss_monotonic_center():
    center = 0
    b = 1
    f = gauss(center, b)
    # max at center roughly c_m == 1
    mv = f(center)
    assert mv == pytest.approx(1, rel=1e-6)
    # symmetrical and smaller away from center
    left = f(center - 1)
    right = f(center + 1)
    assert left == pytest.approx(right)
    assert left < mv


# The following disables actual plotting for all tests just in case
@pytest.fixture(autouse=True)
def disable_plot(monkeypatch):
    # Disable plt.show and plt.plot
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(plt, "plot", lambda *a, **k: None)

