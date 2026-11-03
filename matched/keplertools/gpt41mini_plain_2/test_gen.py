import numpy as np
import pytest
from keplertools import fun

# Test data setups for reuse
def simple_2body_vectors():
    # Circular orbit, unit gravitational parameter
    r = np.array([[1.0, 0, 0]]).T  # shape (3,1)
    v = np.array([[0, 1.0, 0]]).T  # shape (3,1)
    mu = 1.0
    return r, v, mu

def elliptical_2body_vectors():
    # Elliptical orbit example vectors with mu=1
    r = np.array([[1.0, 0, 0.1]]).T  # slight out of plane
    v = np.array([[0, 0.9, 0.1]]).T
    mu = 1.0
    return r, v, mu

def multiple_bodies_vectors():
    # Two bodies, stacked position and velocity vectors (n=2)
    rs = np.array([1.0, 0, 0,  # body 1
                   2.0, 0, 1.0])  # body 2
    vs = np.array([0, 1.0, 0,
                   0, 0.5, 0.5])
    mus = np.array([1.0, 0.1])
    return rs, vs, mus


# orbElem2vec tests

def test_orbElem2vec_scalar_scalar_noorbElem():
    # Must raise when orbElem and AB are None
    with pytest.raises(AssertionError):
        fun.orbElem2vec(np.array([0.0]), 1.0, orbElem=None, AB=None)

def test_orbElem2vec_scalar_scalar_orbElem_basic():
    r, v, mu = simple_2body_vectors()
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(r.flatten(), v.flatten(), mu)
    r2, v2 = fun.orbElem2vec(E, mu, orbElem=(a,e,O,I,w))
    assert r2.shape == (3, 1)
    assert v2.shape == (3, 1)
    # Position magnitude close to original radius
    np.testing.assert_allclose(np.linalg.norm(r2, axis=0), np.linalg.norm(r, axis=0), rtol=1e-8)

def test_orbElem2vec_scalar_scalar_AB_basic():
    r, v, mu = simple_2body_vectors()
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(r.flatten(), v.flatten(), mu)
    A, B = fun.calcAB(a,e,O,I,w)
    r2, v2 = fun.orbElem2vec(E, mu, AB=(A,B))
    # Outputs shape
    assert r2.shape == (3, 1)
    assert v2.shape == (3, 1)
    # Position close to original radius vector
    np.testing.assert_allclose(r2.flatten(), r.flatten(), rtol=1e-8)

def test_orbElem2vec_vector_scalar_orbElem():
    # Many eccentric anomalies, scalar mu
    r, v, mu = simple_2body_vectors()
    a, e, E0, O, I, w, P, tau = fun.vec2orbElem(r.flatten(), v.flatten(), mu)
    n = 5
    E = np.linspace(0, 2*np.pi, n)
    a_arr = np.array([a]*n)
    e_arr = np.array([e]*n)
    O_arr = np.array([O]*n)
    I_arr = np.array([I]*n)
    w_arr = np.array([w]*n)
    r2, v2 = fun.orbElem2vec(E, mu, orbElem=(a_arr,e_arr,O_arr,I_arr,w_arr))
    assert r2.shape == (3, n)
    assert v2.shape == (3, n)
    # Test first position close to radius at E0
    r_expected, _ = fun.orbElem2vec(np.array([E0]), mu, orbElem=(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w])))
    np.testing.assert_allclose(r2[:,0], r_expected[:,0], rtol=1e-8)

def test_orbElem2vec_vector_vector_AB():
    # Multi-body case, vector E and mu
    rs, vs, mus = multiple_bodies_vectors()
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(rs, vs, mus)
    A, B = fun.calcAB(a,e,O,I,w)
    r2, v2 = fun.orbElem2vec(E, mus, AB=(A,B))
    assert r2.shape == (3, len(E))
    assert v2.shape == (3, len(E))

def test_orbElem2vec_returnAB_flag():
    r, v, mu = simple_2body_vectors()
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(r.flatten(), v.flatten(), mu)
    out = fun.orbElem2vec(np.array([E]), mu, orbElem=(a,e,O,I,w), returnAB=True)
    assert len(out) == 3
    r2, v2, (A, B) = out
    assert r2.shape == (3,1)
    assert v2.shape == (3,1)
    assert A.shape == (3,1)
    assert B.shape == (3,1)

# vec2orbElem and vec2orbElem2 tests

def test_vec2orbElem_output_shapes_and_types():
    r, v, mu = simple_2body_vectors()
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(r.flatten(), v.flatten(), mu)
    # All outputs must be 1d arrays of length 1
    for arr in (a,e,E,O,I,w,P,tau):
        assert isinstance(arr, np.ndarray)
        assert arr.size == 1
    # a positive, e in [0,1)
    assert a[0] > 0
    assert 0 <= e[0] < 1

def test_vec2orbElem2_output_consistency():
    r, v, mu = simple_2body_vectors()
    a, e, E, O, I, w, P, tau = fun.vec2orbElem2(r.flatten(), v.flatten(), mu)
    # All outputs arrays
    for arr in (a,e,E,O,I,w,P,tau):
        assert isinstance(arr, np.ndarray)
        assert arr.size == 1
    # Check eccentricity in correct range
    assert 0 <= e[0] < 1
    # Semi-major axis positive
    assert a[0] > 0

def test_vec2orbElem_handles_nxn_input_array_shapes():
    # Provide 3 x n shaped array for r and v, check outputs
    r, v, mu = simple_2body_vectors()
    # Convert r,v to 3 x 1 shape explicitly
    r3xn = r
    v3xn = v
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(r3xn, v3xn, mu)
    assert a.size == 1
    assert e.size == 1

def test_vec2orbElem_with_multiple_bodies_vector_mu():
    rs, vs, mus = multiple_bodies_vectors()
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(rs, vs, mus)
    # Length matches number of bodies
    n = int(rs.size / 3)
    for arr in (a,e,E,O,I,w,P,tau):
        assert arr.size == n
    # eccentricities in [0,1)
    assert np.all((e >= 0) & (e < 1))

def test_vec2orbElem2_with_multiple_bodies_scalar_mu():
    rs, vs, mus = multiple_bodies_vectors()
    mu_scalar = 1.0
    # reshape rs and vs to 3 x n
    n = int(rs.size / 3)
    rs_reshaped = rs.reshape(n, 3).T
    vs_reshaped = vs.reshape(n, 3).T
    a, e, E, O, I, w, P, tau = fun.vec2orbElem2(rs_reshaped, vs_reshaped, mu_scalar)
    # length matches number of bodies
    for arr in (a,e,E,O,I,w,P,tau):
        assert arr.size == n
    # eccentricities 0 <= e < 1
    assert np.all((e >= 0) & (e < 1))

def test_vec2orbElem_invalid_shapes_raise():
    r, v, mu = simple_2body_vectors()
    # Provide mismatched sizes for r and v vectors
    r_wrong = np.array([1,0,0,0])
    v_wrong = np.array([0,1,0])
    with pytest.raises(AssertionError):
        fun.vec2orbElem(r_wrong, v_wrong, mu)

def test_vec2orbElem_invalid_mu_size():
    r, v, mu = simple_2body_vectors()
    mus = np.array([1.0, 2.0])  # wrong size for just one body
    with pytest.raises(AssertionError):
        fun.vec2orbElem(r.flatten(), v.flatten(), mus)

def test_orbElem2vec_scalar_mus_scalar_E_requires_AB_sizes():
    r, v, mu = simple_2body_vectors()
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(r.flatten(), v.flatten(), mu)
    A, B = fun.calcAB(a,e,O,I,w)
    # Intentionally provide bad shape A,B for scalar E and scalar mus
    A_bad = np.array([[1,2,3],[4,5,6],[7,8,9]])
    B_bad = A_bad.copy()
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E[0], mu, AB=(A_bad, B_bad))

def test_orbElem2vec_vector_mus_vector_E_requires_correct_AB_shapes():
    rs, vs, mus = multiple_bodies_vectors()
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(rs, vs, mus)
    A, B = fun.calcAB(a,e,O,I,w)
    # Corrupt shape of A (flattened)
    A_bad = A.flatten()
    B_bad = B
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, AB=(A_bad,B_bad))

def test_vec2orbElem_eccentricity_range():
    r, v, mu = elliptical_2body_vectors()
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(r.flatten(), v.flatten(), mu)
    # eccentricity must be >=0 and <1
    assert np.all(e >= 0)
    assert np.all(e < 1)

def test_vec2orbElem2_same_outputs_as_vec2orbElem_for_simple_input():
    r, v, mu = simple_2body_vectors()
    # reshape rs,vs to 3 x n for vec2orbElem2
    n = 1
    rs = r
    vs = v
    out1 = fun.vec2orbElem(rs.flatten(), vs.flatten(), mu)
    out2 = fun.vec2orbElem2(rs, vs, mu)
    # Compare a,e,E,O,I,w to be close
    for x, y in zip(out1, out2):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-14)

def test_vec2orbElem_with_e_zero():
    # Circular orbit e=0 but with small inclination
    r = np.array([1.0, 0, 0])
    v = np.array([0, 1.0, 0])
    mu = 1.0
    # Manually create zero eccentricity in vec2orbElem output by modifying inputs:
    # We'll cheat by forcing e=0 using a circular orbit logic:
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(r, v, mu)
    # e should be close to zero in circular orbit
    assert e[0] < 1e-8

def test_vec2orbElem_invalid_dimensions_rs_vs_more_than_2D():
    r, v, mu = simple_2body_vectors()
    r_3d = np.reshape(r, (3,1,1))  # 3D array
    with pytest.raises(AssertionError):
        fun.vec2orbElem(r_3d, v, mu)

def test_orbElem2vec_with_E_scalar_mus_vector_raises():
    # Or orbElem2vec should raise if E scalar but mus vector
    r, v, mu = simple_2body_vectors()
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(r.flatten(), v.flatten(), mu)
    mus = np.array([1.0, 0.1])  # vector mus
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E[0], mus, orbElem=(a,e,O,I,w))


# eccanom round trip test (basic sanity)
def test_eccanom_roundtrip():
    # For various M and e, check that E solves Kepler's equation within tolerance
    M = np.linspace(0, 2*np.pi, 6)
    e = 0.5
    E, niter = fun.eccanom(M, e, returnIter=True, noc=True)
    assert E.shape == M.shape
    # Check Kepler's equation M = E - e*sin(E) holds approximately
    M_calc = E - e*np.sin(E)
    np.testing.assert_allclose(np.mod(M_calc, 2*np.pi), np.mod(M, 2*np.pi), atol=1e-12)

def test_trueanom_basic():
    # True anomaly calculation test with simple inputs
    E = np.array([0, np.pi/2, np.pi])
    e = 0.5
    nu = fun.trueanom(E, e)
    assert nu.shape == E.shape
    # check nu values are in [0, 2pi)
    assert np.all((nu >= 0) & (nu < 2*np.pi))

def test_trueanom_scalar_expansion():
    E = 0.1
    e = np.array([0.1, 0.2, 0.3])
    nu = fun.trueanom(E, e)
    assert nu.shape == e.shape
    nu2 = fun.trueanom(np.array([E]*3), e)
    np.testing.assert_allclose(nu, nu2)

def test_trueanom_invalid_e_raises():
    E = 0.5
    e = 1.1  # Invalid eccentricity > 1
    with pytest.raises(AssertionError):
        fun.trueanom(E, e)

def test_eccanom_invalid_e_raises():
    M = 0.5
    e = -0.1  # Invalid eccentricity <0
    with pytest.raises(AssertionError):
        fun.eccanom(M, e)

def test_trueanom_negative_output_shifted():
    # Check negative true anomaly are shifted to positive range
    E = np.array([-1.57])  # negative E near -pi/2
    e = 0.1
    nu = fun.trueanom(E, e)
    assert np.all(nu >= 0)

