import numpy as np
import pytest
from keplertools.fun import orbElem2vec, vec2orbElem

def test_orbElem2vec_basic_scalar_inputs():
    # simple circular orbit in xy-plane
    a = np.array([1.0])
    e = np.array([0.0])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    E = np.array([np.pi/2])
    mu = 1.0

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    # radius should be approximately semi-major axis
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), a, rtol=1e-12)
    # velocity magnitude should be positive
    assert np.all(np.linalg.norm(v, axis=0) > 0)

def test_orbElem2vec_vector_E_scalar_mu():
    # multiple eccentric anomalies for same orbit, scalar mu
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])
    E = np.linspace(0, 2*np.pi, 5)
    mu = 1.0
    
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 5)
    assert v.shape == (3, 5)

def test_orbElem2vec_vector_E_vector_mu():
    # multiple orbits (vector mu), multiple eccentric anomalies (same length)
    a = np.array([1.0, 1.5])
    e = np.array([0.1, 0.2])
    O = np.array([0.2, 0.3])
    I = np.array([0.3, 0.4])
    w = np.array([0.4, 0.5])
    E = np.array([0.1, 1.0])  # length 2
    mu = np.array([1.0, 1.2])
    
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 2)
    assert v.shape == (3, 2)

def test_orbElem2vec_invalid_inputs_raises():
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])
    E = np.array([0.1])
    mu = 1.0
    
    # must supply orbElem or AB
    with pytest.raises(AssertionError):
        orbElem2vec(E, mu)

    # cannot supply both orbElem and AB
    A = np.ones((3,1))
    B = np.ones((3,1))
    with pytest.raises(AssertionError):
        orbElem2vec(E, mu, orbElem=(a,e,O,I,w), AB=(A,B))

    # scalar E with vector mu should raise
    mu_vec = np.array([1.0, 1.1])
    with pytest.raises(AssertionError):
        orbElem2vec(0.1, mu_vec, orbElem=(a,e,O,I,w))

def test_orbElem2vec_return_AB_option():
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])
    E = np.array([0.5])
    mu = 1.0

    r, v, (A, B) = orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)

def test_orbElem2vec_with_AB_input():
    # Calculate A,B then use in orbElem2vec
    a = np.array([1.0, 1.5])
    e = np.array([0.1, 0.2])
    O = np.array([0.3, 0.4])
    I = np.array([0.4, 0.5])
    w = np.array([0.5, 0.6])
    E = np.array([0.1, 1.0])
    mu = np.array([1.0, 1.2])

    from keplertools.fun import calcAB
    A, B = calcAB(a, e, O, I, w)
    r1, v1 = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    r2, v2 = orbElem2vec(E, mu, AB=(A, B))

    np.testing.assert_allclose(r1, r2)
    np.testing.assert_allclose(v1, v2)

def test_vec2orbElem_output_shapes_and_basic_values():
    # Build circular orbit vectors for testing
    a = 1.0
    e = 0.0
    O = 0.0
    I = 0.0
    w = 0.0
    mu = 1.0
    E = np.pi/3

    from keplertools.fun import orbElem2vec
    r, v = orbElem2vec(E, mu, orbElem=(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w])))

    # Test vec2orbElem can recover
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = vec2orbElem(r, v, np.array([mu]))

    assert a_out.shape == (1,)
    assert e_out.shape == (1,)
    assert E_out.shape == (1,)
    assert O_out.shape == (1,)
    assert I_out.shape == (1,)
    assert w_out.shape == (1,)
    assert P_out.shape == (1,)
    assert tau_out.shape == (1,)

    # Semi-major axis close to input
    np.testing.assert_allclose(a_out, a, rtol=1e-6)
    # Eccentricity close to input
    np.testing.assert_allclose(e_out, e, atol=1e-8)
    # Inclination close
    np.testing.assert_allclose(I_out, I, atol=1e-8)

def test_vec2orbElem_with_reshaped_inputs():
    # position and velocity vectors as 3xn and nx3 shapes
    a = 1.0
    e = 0.0
    O = 0.0
    I = 0.0
    w = 0.0
    mu = 1.0
    E = np.pi/4

    r, v = orbElem2vec(np.array([E]), mu, orbElem=(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w])))

    # r.T and v.T shape (1,3)
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = vec2orbElem(r.T, v.T, np.array([mu]))
    a_out2, e_out2, *_ = vec2orbElem(r, v, mu)
    # Both ways should give same shape and values
    np.testing.assert_allclose(a_out, a_out2)
    np.testing.assert_allclose(e_out, e_out2)

def test_vec2orbElem_invalid_sizes_raise():
    r = np.array([1,0,0, 0,1,0])  # 6 elements (2 vectors)
    v = np.array([0,1,0])         # 3 elements, mismatch
    mu = 1.0
    with pytest.raises(AssertionError):
        vec2orbElem(r, v, mu)

def test_vec2orbElem_mu_size_mismatch():
    r, v = orbElem2vec(np.array([0.1]), 1.0, orbElem=(np.array([1.0]), np.array([0.1]), np.array([0.2]), np.array([0.3]), np.array([0.4])))
    mu_vec = np.array([1.0, 2.0])
    with pytest.raises(AssertionError):
        vec2orbElem(r, v, mu_vec)

def test_vec2orbElem_multiple_bodies():
    n = 3
    a = np.array([1.0, 1.2, 1.5])
    e = np.array([0.0, 0.1, 0.2])
    O = np.array([0.0, 0.3, 0.6])
    I = np.array([0.0, 0.2, 0.4])
    w = np.array([0.0, 0.4, 0.8])
    mu = np.array([1.0, 1.1, 1.2])
    E = np.array([0.1, 0.2, 0.3])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = vec2orbElem(r, v, mu)

    np.testing.assert_allclose(a_out, a, rtol=1e-6)
    np.testing.assert_allclose(e_out, e, atol=1e-8)

def test_vec2orbElem_input_shape_transpose_handling():
    # Provide input vectors as nx3 instead of 3xn to check transpose handling
    a = np.array([1.0])
    e = np.array([0.0])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    mu = 1.0
    E = np.array([np.pi/2])

    r3xn, v3xn = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    r_nx3 = r3xn.T
    v_nx3 = v3xn.T

    a_out, e_out, *_ = vec2orbElem(r_nx3, v_nx3, mu)
    np.testing.assert_allclose(a_out, a)

def test_vec2orbElem_checks_multiple_dims():
    # Input with >2 dims should raise
    r = np.ones((3,3,3))
    v = np.ones((3,3,3))
    mu = 1.0
    with pytest.raises(AssertionError):
        vec2orbElem(r, v, mu)

def test_orbElem2vec_scalar_E_and_mu_requirements():
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    mu = 1.0

    # scalar E with scalar mu - OK
    orbElem2vec(0.5, mu, orbElem=(a, e, O, I, w))

    # scalar E with vector mu - should raise
    mu_vec = np.array([1.0, 1.1])
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, mu_vec, orbElem=(a, e, O, I, w))

def test_orbElem2vec_AB_shape_validation():
    # For vector inputs, A and B must have 3xn size
    E = np.array([0.1, 0.2])
    mu = np.array([1.0, 1.1])
    A_good = np.ones((3, 2))
    B_good = np.ones((3, 2))
    # Should work
    orbElem2vec(E, mu, AB=(A_good, B_good))

    # Invalid shapes (3x1 for vector E)
    A_bad = np.ones((3,1))
    B_bad = np.ones((3,1))
    with pytest.raises(AssertionError):
        orbElem2vec(E, mu, AB=(A_bad, B_bad))

def test_orbElem2vec_A_B_scalar_input_size():
    # When E or mu scalar, A and B sizes must be 3
    E = np.array([0.5])
    mu = 1.0
    A = np.ones((3,))  # shape (3,)
    B = np.ones((3,))
    # Convert A,B from 1d to 2d for passing (3,1)
    A2 = A.reshape(3,1)
    B2 = B.reshape(3,1)
    orbElem2vec(E, mu, AB=(A2, B2))
    # Passing (3,) arrays should fail assertion for size check, because it expects 3 or 3*n
    with pytest.raises(AssertionError):
        orbElem2vec(E, mu, AB=(A, B))

def test_orbElem2vec_cosE_sinE_consistency():
    # Using orbElem2vec and ignoring outputs test relation sin^2+cos^2=1 in calculation indirectly
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    mu = 1.0
    E = np.linspace(0, 2*np.pi, 10)

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    # The radius norm should be positive for all E
    r_norm = np.linalg.norm(r, axis=0)
    assert np.all(r_norm > 0)

def test_orbElem2vec_and_vec2orbElem_inverse_relationship():
    # Create orbElem, convert to r,v then back and check parameters close
    a = np.array([1.5, 2.0])
    e = np.array([0.0, 0.3])
    O = np.array([0.1, 0.4])
    I = np.array([0.2, 0.6])
    w = np.array([0.3, 0.7])
    mu = np.array([1.0, 1.2])
    E = np.array([0.2, 1.5])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = vec2orbElem(r, v, mu)

    np.testing.assert_allclose(a_out, a, rtol=1e-6)
    # Eccentricity comparison loose, because vec2orbElem calculates differently
    np.testing.assert_allclose(e_out, e, atol=1e-6)
    np.testing.assert_allclose(I_out, I, atol=1e-6)
    # Argument of pericenter and longitude of ascending node difference modulo 2pi tolerated
    diff_w = np.mod(w_out - w + np.pi, 2*np.pi) - np.pi
    diff_O = np.mod(O_out - O + np.pi, 2*np.pi) - np.pi
    assert np.all(np.abs(diff_w) < 1e-5)
    assert np.all(np.abs(diff_O) < 1e-5)
