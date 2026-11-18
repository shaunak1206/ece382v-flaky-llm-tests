import numpy as np
import pytest
from keplertools import fun

# Use some typical mu for tests (e.g. GM for Earth in AU^3/day^2 ~ 0.00029591220828559104)
MU_EARTH = 0.00029591220828559104

def test_orbElem2vec_scalar_single():
    # Simple circular orbit with zero eccentric anomaly
    a = np.array([1.0])
    e = np.array([0.0])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    E = np.array([0.0])  # eccentric anomaly zero
    
    r, v = fun.orbElem2vec(E, MU_EARTH, orbElem=(a,e,O,I,w))
    # Position should be exactly at a*(1 - e) = 1 AU on x-axis with no velocity in position vector
    np.testing.assert_allclose(r[:,0], [a[0]*(1 - e[0]), 0, 0], rtol=1e-10, atol=1e-14)
    # Velocity magnitude should be > 0
    assert np.linalg.norm(v[:,0]) > 0

def test_orbElem2vec_scalar_multipleE():
    # Multiple eccentric anomalies for one body with scalar mu
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.5])
    I = np.array([0.25])
    w = np.array([1.5])
    E = np.linspace(0, 2*np.pi, 10)
    r, v = fun.orbElem2vec(E, MU_EARTH, orbElem=(a,e,O,I,w))
    assert r.shape == (3,len(E))
    assert v.shape == (3,len(E))
    # Positions should have modulated magnitudes ~ a*(1-e*cos(E))
    expected_rmag = a[0]*(1 - e[0]*np.cos(E))
    actual_rmag = np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(actual_rmag, expected_rmag, rtol=1e-10)

def test_orbElem2vec_vector_mu_vector_Es():
    # Multiple bodies orbits, vector mu and vector E (same length)
    n = 5
    a = np.linspace(1,2,n)
    e = np.linspace(0.1,0.4,n)
    O = np.linspace(0, np.pi/2, n)
    I = np.linspace(0, np.pi/3, n)
    w = np.linspace(0, np.pi, n)
    E = np.linspace(0, 2*np.pi, n)
    mus = MU_EARTH*np.linspace(1, 1.2, n)
    
    r, v = fun.orbElem2vec(E, mus, orbElem=(a,e,O,I,w))
    assert r.shape == (3,n)
    assert v.shape == (3,n)

def test_orbElem2vec_invalid_input_no_orbElem_or_AB():
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.0, MU_EARTH)

def test_orbElem2vec_invalid_both_orbElem_and_AB():
    a = np.array([1.0])
    e = np.array([0.0])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    A,B = fun.calcAB(a,e,O,I,w)
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.0, MU_EARTH, orbElem=(a,e,O,I,w), AB=(A,B))

def test_orbElem2vec_AB_computation_match_orbElem():
    a = np.array([1.2])
    e = np.array([0.3])
    O = np.array([0.1])
    I = np.array([0.5])
    w = np.array([1.0])
    E = np.array([np.pi/3])
    # compute A,B from orbElem
    A1,B1 = fun.calcAB(a,e,O,I,w)
    r1, v1 = fun.orbElem2vec(E, MU_EARTH, AB=(A1,B1))
    r2, v2 = fun.orbElem2vec(E, MU_EARTH, orbElem=(a,e,O,I,w))
    np.testing.assert_allclose(r1, r2, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(v1, v2, rtol=1e-14, atol=1e-14)

def test_orbElem2vec_returnAB_option():
    a = np.array([1.0])
    e = np.array([0.2])
    O = np.array([0.3])
    I = np.array([0.4])
    w = np.array([0.5])
    E = np.array([0.7])
    r,v,AB = fun.orbElem2vec(E, MU_EARTH, orbElem=(a,e,O,I,w), returnAB=True)
    assert isinstance(AB, tuple)
    assert len(AB) == 2
    A,B = AB
    assert A.shape == (3,1)
    assert B.shape == (3,1)

def test_vec2orbElem_and_orbElem2vec_consistency():
    # Pick a random orbit, convert to elements and back
    np.random.seed(42)
    n = 7
    rs = np.random.uniform(-1,1,(3,n))
    vs = np.random.uniform(-0.5,0.5,(3,n))
    mus = MU_EARTH

    # flatten to 1d for vec2orbElem input
    rs_vec = rs.T.flatten()
    vs_vec = vs.T.flatten()

    a,e,E,O,I,w,P,tau = fun.vec2orbElem(rs_vec, vs_vec, mus)
    
    # orbElem2vec expects 1D arrays for orbElem members
    orbElem = (a,e,O,I,w)
    r2,v2 = fun.orbElem2vec(E, mus, orbElem=orbElem)
    
    # shapes expected (3,n)
    assert r2.shape == (3,n)
    assert v2.shape == (3,n)
    
    # The round trip position and velocity vectors should be close (within some tolerance)
    np.testing.assert_allclose(r2, rs, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(v2, vs, rtol=1e-5, atol=1e-8)

def test_vec2orbElem2_and_vec2orbElem_similarity():
    # Generate same input for both methods, results should be close
    np.random.seed(0)
    n = 4
    rs = np.random.uniform(-2,2,(3,n))
    vs = np.random.uniform(-1,1,(3,n))
    mus = MU_EARTH

    rs_vec = rs.T.flatten()
    vs_vec = vs.T.flatten()
    
    a2,e2,E2,O2,I2,w2,P2,tau2 = fun.vec2orbElem2(rs_vec, vs_vec, mus)
    a1,e1,E1,O1,I1,w1,P1,tau1 = fun.vec2orbElem(rs_vec, vs_vec, mus)

    for arr1, arr2 in zip((a1,e1,E1,O1,I1,w1,P1,tau1),
                         (a2,e2,E2,O2,I2,w2,P2,tau2)):
        np.testing.assert_allclose(arr1, arr2, rtol=1e-5, atol=1e-8)

@pytest.mark.parametrize("E_val", [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
def test_orbElem2vec_with_various_E(E_val):
    a = np.array([1.5])
    e = np.array([0.3])
    O = np.array([0.9])
    I = np.array([0.4])
    w = np.array([1.2])
    E = np.array([E_val])
    r,v = fun.orbElem2vec(E, MU_EARTH, orbElem=(a,e,O,I,w))
    # position magnitude is expected approximately a*(1 - e*cos(E))
    expected_rmag = a[0]*(1 - e[0]*np.cos(E_val))
    rmag = np.linalg.norm(r[:,0])
    assert abs(rmag - expected_rmag) < 1e-8

def test_orbElem2vec_scalar_mus_vector_E():
    # scalar mus and vector E, orbElem with compatible sizes -> A,B size must be 3x1
    a = np.array([2.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])
    E = np.linspace(0, 2*np.pi, 5)
    r,v = fun.orbElem2vec(E, MU_EARTH, orbElem=(a,e,O,I,w))
    assert r.shape == (3, len(E))
    assert v.shape == (3, len(E))

def test_orbElem2vec_raises_on_bad_AB_size_scalarE():
    # Provide AB with wrong shape for scalar E and scalar mus
    A = np.random.rand(3,2)
    B = np.random.rand(3,2)
    E = 0.0
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, MU_EARTH, AB=(A,B))

def test_orbElem2vec_raises_on_mismatched_mus_E_length():
    # E vector, mus vector of different length
    a = np.array([1.0])
    e = np.array([0.0])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    E = np.array([0.0, 1.0])
    mus = np.array([MU_EARTH])  # length 1, should be length 2 or scalar
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, orbElem=(a,e,O,I,w))

def test_orbElem2vec_scalarE_mus_must_be_scalar():
    a = np.array([1.0])
    e = np.array([0.0])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    E = 1.0
    mus = np.array([MU_EARTH,MU_EARTH])
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, orbElem=(a,e,O,I,w))
