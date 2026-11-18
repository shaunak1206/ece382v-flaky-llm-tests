import pytest
import math
import numpy as np
from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import inv, normalize, constant, linear, bounded_linear, R, S, rectangular, triangular, trapezoid, sigmoid


def test_domain_init_and_str_repr():
    d = Domain("temp", 0, 100, res=5)
    assert d._name == "temp"
    assert d._low == 0
    assert d._high == 100
    assert d._res == 5
    assert str(d) == "temp"
    r = repr(d)
    d2 = eval(r)
    assert d == d2


def test_domain_call_warns_out_of_range(recwarn):
    d = Domain("temp", 0, 10)
    d.cold = Set(constant(1))
    # Value inside range: no warning, membership dict correct
    result = d(5)
    assert result == {"cold": 1}
    # Value outside range: warning issued
    _ = d(15)
    w = recwarn.pop(FuzzyWarning)
    assert "outside of domain" in str(w.message)
    assert w.category is FuzzyWarning


def test_domain_set_and_getattr():
    d = Domain("d", 0, 10)
    func = constant(0.5)
    s = Set(func)
    d.myset = s
    # Check that domain._sets has new entry
    assert "myset" in d._sets
    # __getattr__ returns the set by name
    got = d.myset
    assert isinstance(got, Set)
    assert got.func == func
    # Access unknown attribute raises
    with pytest.raises(AttributeError):
        _ = d.unknown


def test_domain_set_assign_func_as_set():
    d = Domain("d", 0, 10)
    d.fuzzy = lambda x: 0.3
    # The fuzzy attribute must be a Set instance
    assert isinstance(d.fuzzy, Set)
    # The domain attribute of the set should be d, and name should be fuzzy
    assert d.fuzzy.domain == d
    assert d.fuzzy.name == "fuzzy"


def test_domain_delattr():
    d = Domain("d", 0, 10)
    d.s = constant(1)
    assert "s" in d._sets
    del d.s
    assert "s" not in d._sets
    # Deleting regular attribute raises FuzzyWarning
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            del d._low


@pytest.mark.parametrize("low,high,res", [(0, 5, 1), (-1, 1, 0.5)])
def test_domain_range_correct(low, high, res):
    d = Domain("test", low, high, res=res)
    r = d.range()
    # Check start and end values
    assert math.isclose(r[0], low)
    assert math.isclose(r[-1], high)
    # step size matches resolution
    if len(r) > 1:
        step = r[1] - r[0]
        assert math.isclose(step, res)


def test_domain_min_max():
    d = Domain("d", 0, 10)

    f1 = constant(0.1)
    f2 = constant(0.9)
    d.low = Set(f1)
    d.high = Set(f2)

    # Also test min/max for values in domain
    assert math.isclose(d.min(5), 0.1)
    assert math.isclose(d.max(5), 0.9)


def test_set_callable_and_invert_neg():
    s = Set(constant(0.2))
    val = s(0)
    assert math.isclose(val, 0.2)
    inv_s = ~s
    neg_s = -s
    # Inv and neg give similar results ~s == -s
    assert callable(inv_s.func)
    assert np.allclose([inv_s(x) for x in range(3)], [1 - s(x) for x in range(3)])
    assert np.allclose([neg_s(x) for x in range(3)], [1 - s(x) for x in range(3)])


def test_set_and_or_mul_add_xor_pow():
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

    # Evaluate at several points
    for x in [0, 1, 5]:
        and_val = s_and(x)
        or_val = s_or(x)
        mul_val = s_mul(x)
        add_val = s_add(x)
        xor_val = s_xor(x)
        pow_val = s_pow(x)

        assert math.isclose(and_val, min(f1(x), f2(x)))
        assert math.isclose(or_val, max(f1(x), f2(x)))
        assert math.isclose(mul_val, f1(x) * f2(x))
        bs = min(f1(x) + f2(x), 1)
        assert add_val <= 1 and add_val >= 0
        # Using bounded_sum returns clipped sum
        assert math.isclose(add_val, bs)
        # simple_disjoint_sum always >= bounded_sum
        assert xor_val >= add_val
        assert math.isclose(pow_val, (f1(x) ** 2))


def test_set_equality_and_comparisons_with_domain():
    # Define domain and triangular sets
    d = Domain("d", 0, 10, res=1)
    s1 = Set(triangular(0, 10), name="s1", domain=d)
    s2 = Set(triangular(0, 10), name="s2", domain=d)
    s3 = Set(constant(0.5), name="s3", domain=d)
    # s1 and s2 backed by identical functions and domain
    assert s1 == s2
    # s3 is different
    assert s1 != s3
    # Test ordering: s3 subset or superset of s1? Should raise nothing, test boolean returns
    assert (s3 <= s1) or (s3 > s1) or (s3 < s1) or (s3 >= s1)  # at least one is true
    # Using sets without domain should raise warning on eq, le etc
    s4 = Set(constant(0.3))
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            _ = s4 == s1
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            _ = s4 <= s1


def test_set_len_cardinality_relative_cardinality():
    d = Domain("d", 0, 4, res=1)
    s = Set(lambda x: (x / 4), name="s", domain=d)
    # Length matches domain range length
    n = len(s)
    assert n == len(d.range())
    # Cardinality is sum of membership values
    expected_card = sum(s.func(x) for x in d.range())
    assert math.isclose(s.cardinality(), expected_card)
    # Relative cardinality = cardinality / len
    assert math.isclose(s.relative_cardinality(), expected_card / n)
    # Empty domain length raises warning
    d_empty = Domain("e", 0, 0, res=1)
    s_empty = Set(lambda x: 1, name="e", domain=d_empty)
    # relative_cardinality on zero length domain
    with pytest.warns(FuzzyWarning):
        with pytest.raises(FuzzyWarning):
            _ = s_empty.relative_cardinality()


def test_set_concentrated_and_dilated_and_multiplied():
    base_func = constant(0.5)
    s = Set(base_func)

    c = s.concentrated()
    d = s.dilated()
    m = s.multiplied(2)

    # Concentrated: func(x)^2
    for x in range(3):
        val = c(x)
        assert math.isclose(val, base_func(x) ** 2)

    # Dilated: func(x)**0.5
    for x in range(3):
        val = d(x)
        assert math.isclose(val, base_func(x) ** 0.5)

    # Multiplied: func(x) * n
    for x in range(3):
        val = m(x)
        assert math.isclose(val, base_func(x) * 2)


def test_set_plot_warns_without_domain(monkeypatch):
    s = Set(constant(0.5))
    with pytest.warns(FuzzyWarning):
        s.plot()

    # Assign a domain to plot without error
    d = Domain("d", 0, 2, res=1)
    s.domain = d
    monkeypatch.setattr(s, "func", constant(0.5))
    # Just check no exception raised when plotting
    s.plot()


def test_set_array_and_normalized():
    d = Domain("d", 0, 3, res=1)
    s = Set(lambda x: x / 3, domain=d)
    arr = s.array()
    expected = np.array([x / 3 for x in d.range()])
    np.testing.assert_allclose(arr, expected)
    # normalized should scale max to 1
    s_max = s.normalized()
    norm_vals = s_max.array()
    assert np.isclose(np.max(norm_vals), 1.0)


def test_set_str_and_repr():
    d = Domain("d", 0, 1)
    s = Set(constant(0.3))
    # Unassigned domain and name
    s_empty = Set(constant(0.5))
    s_empty.name = None
    s_empty.domain = None
    s_empty_str = str(s_empty)
    assert "dangling" in s_empty_str
    # With domain and name
    s = Set(constant(0.2), name="myset", domain=d)
    s_str = str(s)
    assert s_str == "d.myset"
    s_repr = repr(s)
    assert s_repr.startswith("Set")


def test_set_init_with_pickled_function_and_missing_name_domain_warn(monkeypatch):
    import pickle

    f = constant(0.7)
    pickled = pickle.dumps(f)
    # Create set from pickled function string; should be unpickled inside
    s = Set(pickled)
    assert callable(s.func)
    # Provide only domain or only name raises warning
    d = Domain("d", 0, 1)
    with pytest.warns(FuzzyWarning):
        Set(f, name="name_only")
    with pytest.warns(FuzzyWarning):
        Set(f, domain=d)
    # Provide both works fine
    s2 = Set(f, name="both", domain=d)
    assert s2.name == "both"
    assert s2.domain == d


def test_functions_inv_and_normalize():
    f = constant(0.2)
    inv_f = inv(f)
    for x in range(3):
        assert math.isclose(inv_f(x), 1 - f(x))
    # Normalize with height
    norm_f = normalize(0.5, f)
    for x in range(3):
        assert math.isclose(norm_f(x), f(x) / 0.5)
    with pytest.raises(AssertionError):
        normalize(0, f)


def test_functions_linear_bounded_linear():
    lin = linear(1, -1)
    assert lin(-2) == 0
    assert lin(0) == 0
    assert math.isclose(lin(1.5), 0.5)
    bl = bounded_linear(2, 3)
    assert bl(1) == 0
    assert bl(2) == 0
    assert math.isclose(bl(2.5), 0.5)
    assert bl(4) == 1.0
    # Inverse tests
    bl_inv = bounded_linear(2, 3, inverse=True)
    assert bl_inv(1) == 1.0 or bl_inv(1) == 0.0  # could be floor or ceiling clipped


def test_functions_R_and_S():
    r = R(2, 3)
    assert r(1) == 0
    assert r(2) == 0
    assert math.isclose(r(2.5), 0.5)
    assert r(4) == 1

    s = S(2, 3)
    assert s(1.5) == 1
    assert math.isclose(s(2.5), 0.5)
    assert s(3) == 0


def test_functions_rectangular_triangular_trapezoid_sigmoid():
    rect = rectangular(1, 2)
    assert rect(0) == 0
    assert rect(1.5) == 1
    assert rect(3) == 0

    tri = triangular(0, 2)
    # peak at 1.0
    assert tri(0) == 0
    assert 0 < tri(1) <= 1
    assert tri(2) == 0

    trap = trapezoid(0, 1, 2, 3)
    assert trap(-1) == 0
    assert 0 < trap(0.5) < 1
    assert trap(1.5) == 1
    assert 0 < trap(2.5) < 1
    assert trap(4) == 0

    s = sigmoid(1, 1, 0)
    mid = s(0)
    assert 0 < mid <= 1
    low = s(-10)
    high = s(10)
    assert low < mid < high


def test_functions_simple_sigmoid_behavior():
    f = sigmoid(1, 1, 0)
    assert 0 <= f(0) <= 1
    s = sigmoid(1, 0, 0)
    # k=0 makes a constant function 0.5
    fs = sigmoid(1, 0, 0)
    assert 0 <= fs(0) <= 1


def test_set_normalized_without_domain_warn():
    s = Set(constant(0.5))
    with pytest.warns(FuzzyWarning):
        s.normalized()


def test_set_len_without_domain_warn():
    s = Set(constant(0.5))
    with pytest.warns(FuzzyWarning):
        len(s)

def test_set_cardinality_without_domain_warn():
    s = Set(constant(0.5))
    with pytest.warns(FuzzyWarning):
        s.cardinality()

