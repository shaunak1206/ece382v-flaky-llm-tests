import numpy as np
import pytest
from keplertools import fun

# Helpers for generating consistent orbital conditions
def generate_elliptical_orbit(mu=1.0, a=1.0, e=0.1, I=0.2, O=0.3, w=0.4, E=0.5):
    """Generate A, B arrays and eccentric anomaly array for testing orbElem2vec and vec2orbElem2"""
    A, B = fun.calcAB(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    E_arr = np.array([E])
    mus = mu
    orbElem = (np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    return E_arr, mus, A, B, orbElem

# orbElem2vec tests
def test_orbElem2vec_scalar_inputs_consistency():
    E, mus, A, B, orbElem = generate_elliptical_orbit()
    r1, v1 = fun.orbElem2vec(E[0], mus, orbElem=orbElem)
    r2, v2 = fun.orbElem2vec(E[0], mus, AB=(A, B))
    np.testing.assert_allclose(r1, r2)
    np.testing.assert_allclose(v1, v2)

def test_orbElem2vec_vector_E_scalar_mu():
    E, mus, A, B, orbElem = generate_elliptical_orbit()
    E_vec = np.linspace(0, 2*np.pi, 10)
    r, v = fun.orbElem2vec(E_vec, mus, orbElem=([1.0]*10, [0.1]*10, [0.3]*10, [0.2]*10, [0.4]*10))
    assert r.shape == (3, 10)
    assert v.shape == (3, 10)

def test_orbElem2vec_vector_E_vector_mu():
    E, mus, A, B, orbElem = generate_elliptical_orbit()
    n = 5
    E_vec = np.linspace(0, 2*np.pi, n)
    mus_vec = np.linspace(1, 2, n)
    orbElem = (np.array([1.0]*n), np.array([0.1]*n), np.array([0.3]*n), np.array([0.2]*n), np.array([0.4]*n))
    with pytest.raises(AssertionError):
        # This should raise because orbElem2vec expects AB or orbElem with matching sizes if E and mu vectors
        fun.orbElem2vec(E_vec, mus_vec, AB=(A, B))  # A, B have size 3x1, inconsistent with E and mus sizes
    
    # Proper call
    r, v = fun.orbElem2vec(E_vec, mus_vec, orbElem=orbElem)
    assert r.shape == (3, n)
    assert v.shape == (3, n)

def test_orbElem2vec_returnAB():
    E, mus, A, B, orbElem = generate_elliptical_orbit()
    r, v, (A_out, B_out) = fun.orbElem2vec(E[0], mus, orbElem=orbElem, returnAB=True)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert A_out.shape == A.shape
    assert B_out.shape == B.shape

def test_orbElem2vec_no_inputs():
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.1, 1.0)  # no orbElem or AB supplied

def test_orbElem2vec_both_inputs():
    E, mus, A, B, orbElem = generate_elliptical_orbit()
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, orbElem=orbElem, AB=(A,B))  # both supplied not allowed

def test_orbElem2vec_scalarE_scalarMu_shape_check():
    # Construct A and B with wrong shape for scalar inputs
    A_bad = np.ones((3,2))
    B_bad = np.ones((3,2))
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.1, 1.0, AB=(A_bad, B_bad))

def test_orbElem2vec_vectorE_vectorMu_shape_check():
    n = 4
    E_vec = np.linspace(0,2*np.pi,n)
    mus_vec = np.ones(n)
    # Wrong A,B shapes: should be 3 x n
    A_bad = np.ones((3, n-1))
    B_bad = np.ones((3, n-1))
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E_vec, mus_vec, AB=(A_bad, B_bad))

# vec2orbElem2 tests
def test_vec2orbElem2_basic_consistency():
    # Create a nearly circular orbit (should produce consistent elements)
    mu = 1.32712440018e11  # Solar mu in km^3/s^2 (just example, units arbitrary here)
    a = 1.5e8
    e = 0.0167
    I = 0.0
    O = 0.0
    w = 0.0
    E = 0.0

    A, B = fun.calcAB(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    r, v = fun.orbElem2vec(E, mu, AB=(A, B))
    # Now get elements back from r,v
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = fun.vec2orbElem2(r.flatten(), v.flatten(), mu)
    assert np.allclose(a_out, a, rtol=1e-6)
    assert np.allclose(e_out, e, rtol=1e-6)
    assert np.allclose(I_out, I, atol=1e-8)
    # E, O, w may differ by 2pi shift, check modulo 2pi
    assert np.allclose(np.mod(E_out, 2*np.pi), np.mod(E, 2*np.pi), atol=1e-6)

def test_vec2orbElem2_array_shape_variants():
    # Provide rs and vs as (n,3) and (3,n)
    mu = 1.0
    a = 2.0
    e = 0.5
    I = 0.1
    O = 0.2
    w = 0.3
    E_val = 1.0

    A, B = fun.calcAB(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    r, v = fun.orbElem2vec(E_val, mu, AB=(A, B))

    # Test with flattened vectors (3n x 1)
    rs_flat = r.flatten()
    vs_flat = v.flatten()
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = fun.vec2orbElem2(rs_flat, vs_flat, mu)
    assert a_out.size == 1

    # Test with shape (3,n)
    a_out2, e_out2, E_out2, O_out2, I_out2, w_out2, P_out2, tau_out2 = fun.vec2orbElem2(r, v, mu)
    np.testing.assert_allclose(a_out2, a_out)
    np.testing.assert_allclose(e_out2, e_out)

    # Test with shape (n,3)
    rs_T = r.T
    vs_T = v.T
    a_out3, e_out3, E_out3, O_out3, I_out3, w_out3, P_out3, tau_out3 = fun.vec2orbElem2(rs_T, vs_T, mu)
    np.testing.assert_allclose(a_out3, a_out)
    np.testing.assert_allclose(e_out3, e_out)

def test_vec2orbElem2_invalid_rs_vs_sizes():
    rs = np.arange(8)  # not divisible by 3
    vs = np.arange(8)
    mu = 1.0
    with pytest.raises(AssertionError):
        fun.vec2orbElem2(rs, vs, mu)

def test_vec2orbElem2_mu_size_mismatch():
    rs = np.arange(9)
    vs = np.arange(9)
    mus = np.array([1.0, 2.0])  # size 2 but nplanets=3
    with pytest.raises(AssertionError):
        fun.vec2orbElem2(rs, vs, mus)

def test_vec2orbElem2_mu_scalar_and_array_inputs():
    # mu scalar
    rs = np.array([1,0,0, 0,1,0, 0,0,1])
    vs = np.array([0,1,0, -1,0,0, 0,0,-1])
    mus = 1.0
    a,e,E,O,I,w,P,tau = fun.vec2orbElem2(rs, vs, mus)
    assert a.shape == ()
    # mu array (size 3)
    mus_arr = np.array([1.0,1.0,1.0])
    a2,e2,E2,O2,I2,w2,P2,tau2 = fun.vec2orbElem2(rs, vs, mus_arr)
    assert a2.size == 3

def test_vec2orbElem2_input_dimensions():
    # Create input with 3d array to cause assertion
    rs = np.zeros((3,3,3))
    vs = np.zeros((3,3,3))
    mu = 1.0
    with pytest.raises(AssertionError):
        fun.vec2orbElem2(rs, vs, mu)

# orbElem2vec and vec2orbElem2 integration
def test_vec2orbElem2_and_orbElem2vec_inversion():
    mu = 398600.4418  # Earth mu in km^3/s^2
    a = 7000.0  # km
    e = 0.001
    I = 0.1
    O = 0.2
    w = 0.3
    E_val = 0.4

    orbElem = (np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    r, v = fun.orbElem2vec(E_val, mu, orbElem=orbElem)

    # Recover elements from vectors
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = fun.vec2orbElem2(r.flatten(), v.flatten(), mu)

    # Compare outputs with some tolerance (angles modulo 2pi)
    assert np.allclose(a_out, a, rtol=1e-5)
    assert np.allclose(e_out, e, rtol=1e-4)
    assert np.allclose(I_out, I, atol=1e-2)
    # Angles w and O can differ by 2pi periodicity
    assert np.allclose(np.mod(O_out, 2*np.pi), np.mod(O, 2*np.pi), atol=1e-2)
    assert np.allclose(np.mod(w_out, 2*np.pi), np.mod(w, 2*np.pi), atol=1e-2)

def test_orbElem2vec_vectorized_and_returnAB():
    n = 3
    a = np.array([1.0, 2.0, 1.5])
    e = np.array([0.1, 0.2, 0.3])
    O = np.array([0.0, 0.5, 1.0])
    I = np.array([0.0, 0.1, 0.2])
    w = np.array([0.0, 0.4, 0.6])
    E = np.linspace(0, 2*np.pi, n)
    mus = 1.0

    orbElem = (a, e, O, I, w)
    r, v, (A,B) = fun.orbElem2vec(E, mus, orbElem=orbElem, returnAB=True)
    assert r.shape == (3,n)
    assert v.shape == (3,n)
    assert A.shape == (3,n)
    assert B.shape == (3,n)

def test_orbElem2vec_scalar_with_array_AB():
    # When scalars E and mus but AB mismatch shape raise AssertionError
    E = 1.0
    mus = 1.0
    A = np.ones((3, 2))
    B = np.ones((3, 2))
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, AB=(A,B))

def test_orbElem2vec_invalid_mus_size_for_E_array():
    # Provide array E with mus array of incompatible size
    E = np.linspace(0,1,4)
    mus = np.array([1.0, 2.0])
    orbElem = ([1.0]*4, [0.1]*4, [0.3]*4, [0.2]*4, [0.4]*4)
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, orbElem=orbElem)

def test_orbElem2vec_invalid_orbElem_size_mismatch():
    E = np.linspace(0,2*np.pi,3)
    mus = 1.0
    orbElem = ([1.0, 2.0], [0.1, 0.2], [0.3, 0.4], [0.2, 0.3], [0.4, 0.5])  # size 2, but E size 3
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, orbElem=orbElem)

