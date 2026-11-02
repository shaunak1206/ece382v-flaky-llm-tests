import pytest
import math
import numpy as np
import pickle
from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import (
    inv, constant, linear, bounded_linear, R, S, rectangular,
    triangular, trapezoid, sigmoid, bounded_sigmoid, bounded_exponential,
    simple_sigmoid, triangular_sigmoid, gauss,
)


# --- Tests for Domain class ---

def test_domain_creation_and_basic_attributes():
    d = Domain('temp', 0, 10, res=0.5)
    assert d._name == 'temp'
    assert d._low == 0
    assert d._high == 10
    assert d._res == 0.5
    assert isinstance(d._sets, dict)
    assert d._sets == {}


def test_domain_call_warns_out_of_bounds(monkeypatch):
    d = Domain('d', 0, 1)
    s = Set(constant(0.5))
    d.testset = s
    warnings = []

    def fake_warn(msg):
        warnings.append(msg)

    monkeypatch.setattr("fuzzylogic.classes.warn", fake_warn)
    # Within domain
    res = d(0.5)
    assert isinstance(res, dict)
    # Outside domain - triggers warn
    _ = d(2)
    assert warnings and "outside of domain" in warnings[0].lower()


def test_domain_sets_assignment_and_access():
    d = Domain('temp', 0, 10)

    def f(x): return max(0, 1 - abs(x - 5)/5)
    s = Set(f)
    d.cool = s
    assert 'cool' in d._sets
    # Access via attribute
    assert d.cool is s
    # Calling domain returns membership dictionary
    d.hot = Set(constant(0))
    mem = d(5)
    assert mem['cool'] == s(5)
    assert mem['hot'] == 0

    with pytest.raises(AttributeError):
        _ = d.notexist


def test_domain_set_name_domain_link():
    d = Domain('temp', 0, 10)
    s = Set(constant(0.3))
    d.warm = s
    assert s.domain is d
    assert s.name == 'warm'


def test_domain_delattr_removes_fuzzy_set(monkeypatch):
    d = Domain('temp', 0, 10)
    s = Set(constant(0.4))
    d.warm = s
    assert 'warm' in d._sets
    d.__delattr__('warm')
    assert 'warm' not in d._sets

    with pytest.raises(FuzzyWarning):
        d.__delattr__('_name')


def test_domain_range_returns_arange():
    d = Domain('temp', 0, 2, res=0.5)
    rng = d.range()
    expected = np.arange(0, 2 + 0.5, 0.5)
    assert np.allclose(rng, expected)


def test_domain_min_max_functions():
    d = Domain('temp', 0, 10)
    s1 = Set(lambda x: max(0, 1 - abs(x - 3)/3))
    s2 = Set(constant(0.5))
    d.s1 = s1
    d.s2 = s2
    assert math.isclose(d.min(3), 0.5)
    assert math.isclose(d.max(3), 1.0)
    assert d.min(0) < d.max(0) <= 0.5


def test_domain_equality():
    d1 = Domain('temp', 0, 10)
    d2 = Domain('temp', 0, 10)
    s1 = Set(constant(0.6))
    s2 = Set(constant(0.6))
    d1.warm = s1
    d2.warm = s2
    assert d1 == d2

    d3 = Domain('temp', 0, 11)
    assert not (d1 == d3)


# --- Tests for Set class ---

def test_set_callable_and_name_domain():
    f = constant(0.7)
    s = Set(f)
    assert s(0) == 0.7
    s2 = Set(f, name="foo", domain=Domain('d', 0, 1))
    assert s2.name == "foo"
    assert s2.domain._name == 'd'
    assert s2.domain.foo is s2

    with pytest.raises(FuzzyWarning):
        # Providing only name but no domain should warn
        Set(f, name="foo")


def test_set_invert_and_neg_equivalent():
    f = constant(0.3)
    s = Set(f)
    inv_s = ~s
    neg_s = -s
    for v in [0, 0.5, 1]:
        assert math.isclose(inv_s(v), 1 - f(v), rel_tol=1e-6)
        assert math.isclose(neg_s(v), 1 - f(v), rel_tol=1e-6)


def test_set_and_or_operations_return_set():
    d = Domain('temp', 0, 10)
    s1 = Set(lambda x: x/10, name='s1', domain=d)
    s2 = Set(lambda x: 1 - x/10, name='s2', domain=d)
    s_and = s1 & s2
    s_or = s1 | s2
    assert isinstance(s_and, Set)
    assert isinstance(s_or, Set)
    # Check behavior for some values
    for v in [0, 5, 10]:
        assert s_and(v) == min(s1(v), s2(v))
        assert s_or(v) == max(s1(v), s2(v))


def test_set_mul_add_xor_operators_return_set():
    d = Domain('temp', 0, 10)
    s1 = Set(lambda x: 0.8, name='s1', domain=d)
    s2 = Set(lambda x: 0.6, name='s2', domain=d)
    s_mul = s1 * s2
    s_add = s1 + s2
    s_xor = s1 ^ s2
    for v in [0, 5, 10]:
        assert math.isclose(s_mul(v), 0.8 * 0.6)
        bs = min(1, 0.8 + 0.6)
        assert math.isclose(s_add(v), bs)
        sd = 0.8 + 0.6 - 0.8 * 0.6
        assert math.isclose(s_xor(v), sd)


def test_set_pow_returns_function_to_power():
    d = Domain('temp', 0, 10)
    s = Set(lambda x: 0.5, name='s', domain=d)
    s_pow = s ** 2
    for v in [0, 5, 10]:
        assert math.isclose(s_pow(v), 0.5 ** 2)


def test_set_eq_le_lt_ge_gt_comparisons():
    d = Domain('temp', 0, 5, res=1)
    s1 = Set(lambda x: x / 5, name="s1", domain=d)
    s2 = Set(lambda x: (x / 5) ** 2, name="s2", domain=d)
    s3 = Set(lambda x: x / 5, name="s3", domain=d)
    # equality
    assert s1 == s3
    with pytest.raises(FuzzyWarning):
        s_no_domain = Set(lambda x: 1)
        _ = s1 == s_no_domain
    # le, lt, ge, gt tests
    assert (s2 <= s1)  # square is subset (smaller or equal)
    assert (s2 < s1)
    assert (s1 >= s2)
    assert (s1 > s2)


def test_set_len_cardinalities():
    d = Domain('temp', 0, 2, res=1)
    s = Set(lambda x: 1, name="s", domain=d)
    assert len(s) == 3  # 0, 1, 2
    assert s.cardinality() == 3
    assert s.relative_cardinality() == 1

    s0 = Set(lambda x: 0, name="s0", domain=d)
    assert s0.cardinality() == 0
    with pytest.raises(FuzzyWarning):
        s0.relative_cardinality()  # division by zero

    with pytest.raises(FuzzyWarning):
        Set(lambda x: 1)()  # no domain, __len__ triggers warning


def test_set_concentrated_dilated_multiplied():
    d = Domain('temp', 0, 1, res=0.5)
    s = Set(lambda x: x, name="s", domain=d)

    concentrated = s.concentrated()
    dilated = s.dilated()
    multiplied = s.multiplied(2)

    for v in d.range():
        assert math.isclose(concentrated(v), s(v)**2)
        assert math.isclose(dilated(v), s(v)**0.5)
        assert math.isclose(multiplied(v), s(v)*2)


def test_set_intensified_behavior():
    # intensified function has special piecewise definition on membership values (0.5 cutoff)
    d = Domain('temp', 0, 1, res=0.25)
    # Membership returns fixed values for testing
    def membership(x):
        # x interpreted as membership value here for test to trigger if properly passed
        return x
    s = Set(membership, name="s", domain=d)
    intensified = s.intensified()
    # Check for values <0.5 and >=0.5 manually
    for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        orig = val
        if orig < 0.5:
            expected = 2 * (orig**2)
        else:
            expected = 1 - 2*(1 - orig**2)
        assert math.isclose(intensified(orig), expected)


def test_set_array_and_plot(monkeypatch):
    d = Domain('temp', 0, 2, res=1)
    s = Set(lambda x: x / 2, name='s', domain=d)
    arr = s.array()
    expected = np.array([0/2, 1/2, 2/2])
    assert np.allclose(arr, expected)

    # Patch plt.plot to check that plot calls it
    recorded = {}
    def fake_plot(x, y):
        recorded['x'] = x
        recorded['y'] = y
    monkeypatch.setattr('matplotlib.pyplot.plot', fake_plot)
    s.plot()
    assert 'x' in recorded and 'y' in recorded
    assert np.allclose(list(recorded['y']), list(s.array()))


def test_set_str_and_repr():
    d = Domain('temp', 0, 10)
    s = Set(constant(0.6))
    # Dangling set
    s2 = Set(constant(0.5))
    assert s2.__str__().startswith('dangling Set')
    s.domain = d
    s.name = 'warm'
    assert s.__str__() == 'temp.warm'
    repr_str = repr(s)
    assert repr_str.startswith('Set(')


def test_set_normalized():
    d = Domain('temp', 0, 2, res=1)
    s = Set(lambda x: x, name='s', domain=d)
    normalized = s.normalized()
    max_val = max(s.array())
    for v in d.range():
        assert math.isclose(normalized(v), s(v) / max_val)


# --- Tests for functions module ---

def test_inv_function_behavior():
    f = constant(0.3)
    finv = inv(f)
    for v in [-10, 0, 0.5, 1, 10]:
        assert math.isclose(finv(v), 1 - f(v))


def test_constant_function_returns_constant():
    c = 0.7
    f = constant(c)
    for v in [-100, 0, 100]:
        assert f(v) == c


def test_linear_function_clipping():
    f = linear(1, -1)
    assert f(-2) == 0  # clipped low
    assert f(0) == 0
    assert f(1) == 0
    assert math.isclose(f(1.5), 0.5)
    assert f(2) == 1
    assert f(3) == 1  # clipped high


def test_bounded_linear_function_monotonicity():
    f = bounded_linear(2, 3)
    # below low bound
    assert f(1) == 0
    # at low bound
    assert f(2) == 0
    # mid range
    assert math.isclose(f(2.5), 0.5)
    # at high bound
    assert f(3) == 1
    # above high bound
    assert f(4) == 1


def test_R_and_S_functions_behavior():
    r = R(2, 4)
    s = S(2, 4)
    assert r(1) == 0
    assert r(2) == 0
    assert r(3) == 0.5
    assert r(5) == 1
    assert s(1) == 1
    assert 0 < s(3) < 1
    assert s(4) == 0
    assert s(5) == 0


def test_rectangular_function_behavior():
    f = rectangular(1, 3)
    assert f(0) == 0
    assert f(1) == 1
    assert f(2) == 1
    assert f(3) == 1
    assert f(4) == 0


def test_triangular_function_behavior():
    f = triangular(1, 3, c=2)
    assert f(1) == 0
    assert 0 < f(1.5) < 1
    assert f(2) == 1
    assert 0 < f(2.5) < 1
    assert f(3) == 0


def test_trapezoid_function_behavior():
    f = trapezoid(0, 1, 3, 4)
    assert f(-0.1) == 0
    assert 0 < f(0.5) < 1
    assert f(1) == 1
    assert f(2) == 1
    assert f(3) == 1
    assert 0 < f(3.5) < 1
    assert f(4) == 0
    assert f(5) == 0


def test_sigmoid_function_limits():
    f = sigmoid(1, 1, 0)
    assert 0 <= f(-1000) <= 0.01
    assert math.isclose(f(0), 0.5, rel_tol=1e-2)
    assert f(1000) > 0.99


def test_bounded_sigmoid_function_values():
    f = bounded_sigmoid(0, 1)
    # boundary values
    assert math.isclose(f(0), 0.1, abs_tol=1e-5)
    assert round(f(1), 2) == 0.9
    assert round(f(100000), 2) == 1.0
    assert round(f(-100000), 2) == 0.0


def test_bounded_exponential_function_behavior():
    f = bounded_exponential(k=0.1, limit=1)
    assert math.isclose(f(0), 0)
    # for large x, value approaches limit
    assert abs(f(1000) - 1) < 1e-3


def test_simple_sigmoid_behavior():
    f = simple_sigmoid()
    assert round(f(-1000), 2) == 0.0
    assert math.isclose(f(0), 0.5, rel_tol=1e-2)
    assert round(f(1000), 2) == 1.0
    assert round(f(-20), 2) == 0.01
    assert round(f(20), 2) == 0.99


def test_triangular_sigmoid_behavior():
    f = triangular_sigmoid(2, 4)
    assert math.isclose(f(2), 0.1, abs_tol=1e-2)
    assert round(f(3), 2) == 0.9


def test_gauss_function_behavior():
    f = gauss(0, 1)
    # center point x=c=0
    assert math.isclose(f(0), 1)
    # away points produce smaller membership
    val = f(1)
    assert 0 < val < 1
    assert f(100) == 0


def test_inv_function_return_closure():
    def base(x): return 0.4
    invf = inv(base)
    # Should return a callable
    assert callable(invf)
    # The returned function is different from base
    assert invf != base


def test_constant_function_return_closure():
    f = constant(0.9)
    assert callable(f)
    assert f(0) == 0.9
    assert f(12345) == 0.9


def test_alpha_function_clipping_behavior():
    base = constant(0.5)
    clipped = inv(base)
    # alpha function floor=0.4, ceiling=0.6, func=clipped
    from fuzzylogic.functions import alpha
    f = alpha(floor=0.4, ceiling=0.6, func=clipped)
    assert f(0) >= 0
    assert f(0.5) >= 0
    assert 0.4 <= f(0.5) <= 0.6


def test_normalize_function_output():
    base = constant(0.5)
    fnorm = inv(base)
    norm = lambda x: fnorm(x) / 0.5
    from fuzzylogic.functions import normalize
    f = normalize(0.5, fnorm)
    for v in [0, 1]:
        assert math.isclose(f(v), norm(v))


# Done.
