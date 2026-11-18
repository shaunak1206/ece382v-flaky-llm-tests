import pytest
from fuzzylogic.classes import Domain, Set, FuzzyWarning
from fuzzylogic.functions import inv, constant, linear, bounded_linear, R, S, rectangular, triangular, trapezoid, sigmoid, normalize

import numpy as np
import warnings

@pytest.fixture
def simple_domain():
    d = Domain('test', 0, 10, res=1)
    return d

@pytest.fixture
def domain_with_sets(simple_domain):
    d = simple_domain
    d.low_set = Set(constant(0.0))
    d.high_set = Set(constant(1.0))
    d.linear_set = Set(linear(0.1, 0))
    return d

def test_domain_initialization():
    d = Domain('temp', 0, 100, res=0.5)
    assert d._name == 'temp'
    assert d._low == 0
    assert d._high == 100
    assert d._res == 0.5
    assert isinstance(d._sets, dict)
    assert len(d._sets) == 0

def test_domain_invalid_initialization():
    with pytest.raises(AssertionError):
        Domain('bad', 10, 0)
    with pytest.raises(AssertionError):
        Domain('bad', 0, 10, res=0)
    with pytest.raises(AssertionError):
        Domain('bad', 0, 10, res=-1)

def test_set_assignment_and_access(simple_domain):
    sd = simple_domain
    s = Set(constant(0.5))
    sd.mid_set = s
    assert 'mid_set' in sd._sets
    assert sd.mid_set is s
    assert callable(sd.mid_set)
    # The Set name and domain are set
    assert s.name == 'mid_set'
    assert s.domain is sd

def test_domain_call_warns_outside(monkeypatch):
    d = Domain('range', 0, 5)
    d.test_set = Set(constant(1))
    warned = []
    def fake_warn(msg):
        warned.append(msg)

    monkeypatch.setattr("fuzzylogic.classes.warn", fake_warn)
    d(6)
    assert any("6 is outside of domain" in w for w in warned)

def test_domain_call_returns_member_dict(domain_with_sets):
    d = domain_with_sets
    x = 3
    mems = d(x)
    assert isinstance(mems, dict)
    # keys are the names of the sets
    assert all(k in d._sets for k in mems.keys())
    # values are floats
    assert all(isinstance(v, float) for v in mems.values())

def test_domain_str_and_repr(simple_domain):
    d = simple_domain
    assert str(d) == d._name
    r = repr(d)
    assert r.startswith("Domain(")
    # eval should recreate equivalent Domain ignoring sets detail (repr of set is func repr)
    d2 = eval(r)
    assert d == d2

def test_domain_eq(simple_domain):
    d1 = simple_domain
    d2 = Domain('test', 0, 10, res=1)
    assert d1 == d2
    d2._name = 'other'
    assert d1 != d2

def test_domain_setattr_sets_set_and_name(domain_with_sets):
    d = domain_with_sets
    # Assigning a raw function assigns a Set automatically
    def func(x): return 0.8
    d.new_set = func
    assert isinstance(d.new_set, Set)
    assert d.new_set.func(0) == pytest.approx(0.8)
    assert d._sets['new_set'] == d.new_set

def test_domain_delattr_removes_set(domain_with_sets):
    d = domain_with_sets
    s = d.high_set
    del d.high_set
    assert 'high_set' not in d._sets
    with pytest.raises(FuzzyWarning):
        del d._high  # deleting a normal domain attr should raise warning

def test_domain_range_includes_upper_bound(simple_domain):
    d = simple_domain
    r = d.range()
    assert r[0] == d._low
    assert r[-1] == d._high

def test_set_call_and_invert(domain_with_sets):
    d = domain_with_sets
    s = d.high_set
    v = s(5)
    s_inv = ~s
    assert isinstance(s_inv, Set)
    v_inv = s_inv(5)
    assert pytest.approx(v_inv) == 1 - v

def test_set_operator_and_or(domain_with_sets):
    d = domain_with_sets
    s1 = d.high_set
    s2 = d.linear_set
    s_and = s1 & s2
    s_or = s1 | s2
    assert isinstance(s_and, Set)
    assert isinstance(s_or, Set)
    x = 5
    assert pytest.approx(s_and(x)) == min(s1(x), s2(x))
    assert pytest.approx(s_or(x)) == max(s1(x), s2(x))

def test_set_equality_and_comparison(domain_with_sets):
    d = domain_with_sets
    s1 = d.high_set
    s2 = Set(constant(1.0))
    s2.domain = d
    # equality on sets with the same outputs over domain range is True
    assert s1 == s2
    # less equal and greater equal
    less_set = Set(constant(0.5))
    less_set.domain = d
    assert less_set <= s1
    assert s1 >= less_set
    assert less_set < s1
    assert s1 > less_set

def test_set_length_cardinality_and_relative_cardinality(domain_with_sets):
    d = domain_with_sets
    s = d.linear_set
    length = len(s)
    assert isinstance(length, int)
    card = s.cardinality()
    assert isinstance(card, float)
    rel_card = s.relative_cardinality()
    assert 0 <= rel_card <= 1

def test_set_methods_concentrated_dilated_multiplied(domain_with_sets):
    d = domain_with_sets
    s = d.linear_set
    c = s.concentrated()
    dlt = s.dilated()
    mult = s.multiplied(2)
    assert isinstance(c, Set)
    assert isinstance(dlt, Set)
    assert isinstance(mult, Set)
    x = 5
    assert pytest.approx(c(x), rel=1) == s(x) ** 2
    assert pytest.approx(dlt(x), rel=1) == s(x) ** 0.5
    assert pytest.approx(mult(x), rel=1) == s(x) * 2

def test_set_plot_raises_without_domain():
    s = Set(constant(1))
    with pytest.raises(FuzzyWarning):
        s.plot()

def test_set_array_returns_numpy_array(domain_with_sets):
    d = domain_with_sets
    s = d.linear_set
    arr = s.array()
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == len(d.range())

def test_set_str_and_repr(domain_with_sets):
    d = domain_with_sets
    s = d.linear_set
    s.name = 'testname'
    s.domain = d
    s_str = str(s)
    s_repr = repr(s)
    assert d._name in s_str
    assert 'Set' in s_repr

def test_set_normalized(domain_with_sets):
    d = domain_with_sets
    s = d.linear_set
    n = s.normalized()
    assert isinstance(n, Set)
    # normalized max must be 1
    arr = n.array()
    assert pytest.approx(max(arr)) == 1

def test_set_init_name_or_domain_but_not_both_raises():
    f = constant(0)
    with pytest.raises(FuzzyWarning):
        Set(f, name="onlyname")
    with pytest.raises(FuzzyWarning):
        Set(f, domain=Domain('d', 0, 1))

def test_invert_neg_operators(domain_with_sets):
    d = domain_with_sets
    s = d.high_set
    s_inv1 = ~s
    s_inv2 = -s
    x = 4
    assert pytest.approx(s_inv1(x)) == pytest.approx(s_inv2(x)) == 1 - s(x)

def test_set_pow_operator(domain_with_sets):
    d = domain_with_sets
    s = d.linear_set
    s_pow = s ** 2
    x = 3
    assert pytest.approx(s_pow(x)) == s(x) ** 2

def test_set_add_mul_xor_operators(domain_with_sets):
    d = domain_with_sets
    s1 = d.high_set
    s2 = d.linear_set
    add_set = s1 + s2
    mul_set = s1 * s2
    xor_set = s1 ^ s2
    x = 2
    # bounded_sum should be s1(x) + s2(x) clipped at 1
    assert add_set(x) <= 2
    # product s1(x) * s2(x)
    assert pytest.approx(mul_set(x)) == s1(x) * s2(x)
    # simple_disjoint_sum: sum minus product
    assert pytest.approx(xor_set(x)) == s1(x) + s2(x) - s1(x)*s2(x)
