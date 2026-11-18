import numpy as np
import pytest
from keplertools import fun

# Helper constants for tests
mu_earth = 398600.4418  # km^3/s^2 Earth standard gravitational parameter

def test_orbElem2vec_single_scalar_inputs():
    a = 7000.0
    e = 0.01
    O = 0.0
    I = 0.1
    w = 0.2
    E = 1.0
    orbElem = (np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    r, v = fun.orbElem2vec(E, mu_earth, orbElem=orbElem)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    # Positions and velocities should not be all zeros
    assert np.any(np.abs(r) > 0)
    assert np.any(np.abs(v) > 0)

def test_orbElem2vec_vector_eccentric_anomaly_scalar_mu():
    a = np.array([7000.0, 8000.0])
    e = np.array([0.1, 0.2])
    O = np.array([0.0, 0.1])
    I = np.array([0.0, 0.2])
    w = np.array([0.0, 0.3])
    E = np.linspace(0, 2*np.pi, 5)
    orbElem = (a, e, O, I, w)
    r, v = fun.orbElem2vec(E, mu_earth, orbElem=orbElem)
    # When multiple eccentric anomalies but multiple orbitals of different sizes,
    # should assert fail due to dimension mismatch inside function; so test with orbElem sized 1 or compatible
    # Here orbElem size is 2 but E.size=5 so expect assertion error in orbElem2vec
    # So test with single orbElem values (broadcast) for vector E input instead:
    orbElem_broadcast = (np.array([7000.0]), np.array([0.1]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
    r, v = fun.orbElem2vec(E, mu_earth, orbElem=orbElem_broadcast)
    assert r.shape == (3, E.size)
    assert v.shape == (3, E.size)

def test_orbElem2vec_vector_eccentric_anomaly_vector_mu():
    a = np.array([7000.0, 8000.0])
    e = np.array([0.1, 0.2])
    O = np.array([0.0, 0.1])
    I = np.array([0.0, 0.2])
    w = np.array([0.0, 0.3])
    E = np.array([0.1, 0.2])
    orbElem = (a, e, O, I, w)
    mus = np.array([mu_earth, mu_earth*1.5])
    r, v = fun.orbElem2vec(E, mus, orbElem=orbElem)
    assert r.shape == (3, 2)
    assert v.shape == (3, 2)

def test_orbElem2vec_raises_with_neither_orbElem_nor_AB():
    with pytest.raises(AssertionError):
        fun.orbElem2vec(1.0, mu_earth)

def test_orbElem2vec_raises_with_both_orbElem_and_AB():
    a = np.array([7000.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    orbElem = (a, e, O, I, w)
    A, B = fun.calcAB(a, e, O, I, w)
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.0, mu_earth, orbElem=orbElem, AB=(A, B))

def test_orbElem2vec_returnAB_is_true_returns_ABC_tuple():
    a = np.array([7000.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    orbElem = (a, e, O, I, w)
    r, v, (A, B) = fun.orbElem2vec(0.0, mu_earth, orbElem=orbElem, returnAB=True)
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)
    assert np.allclose(np.linalg.norm(A, axis=0), a)
    # Check shapes of r and v
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)

def test_vec2orbElem_simple_circular_equatorial_orbit():
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])
    rs = r.reshape(3,1)
    vs = v.reshape(3,1)
    a,e,E,O,I,w,P,tau = fun.vec2orbElem(rs, vs, mu_earth)
    assert a.shape == (1,)
    assert e.shape == (1,)
    assert E.shape == (1,)
    assert O.shape == (1,)
    assert I.shape == (1,)
    assert w.shape == (1,)
    assert P.shape == (1,)
    assert tau.shape == (1,)
    # For circular, equatorial orbit inclination near zero
    assert np.isclose(I[0], 0, atol=1e-3)
    # eccentricity close to zero
    assert e[0] < 1e-2

def test_vec2orbElem_multiple_vectors_input_shapes():
    # Provide multiple orbits with shape nx3
    rs = np.array([
        [7000.0, 0.0,    0.0],
        [8000.0, 0.0,    0.0]
    ])
    vs = np.array([
        [0.0,    7.5,   0.0],
        [0.0,    7.0,   0.0]
    ])
    mus = np.array([mu_earth, mu_earth])
    a,e,E,O,I,w,P,tau = fun.vec2orbElem(rs, vs, mus)
    assert a.shape == (2,)
    assert e.shape == (2,)

def test_vec2orbElem_accepts_flat_3n_arrays():
    # rs and vs as 3n x 1 vectors
    rs = np.array([7000.0, 0.0, 0.0, 8000.0, 0.0, 0.0])
    vs = np.array([0.0, 7.5, 0.0, 0.0, 7.0, 0.0])
    mus = np.array([mu_earth, mu_earth])
    a,e,E,O,I,w,P,tau = fun.vec2orbElem(rs, vs, mus)
    assert a.shape == (2,)
    assert e.shape == (2,)

def test_vec2orbElem_scalar_mu_vector_positions():
    # Test with scalar mu, multiple 3-vector positions/velocities 
    rs = np.array([
        [7000.0, 0.0, 0.0],
        [7000.0, 7000.0, 0.0]
    ])
    vs = np.array([
        [0.0, 7.5, 0.0],
        [-7.5, 0.0, 0.0]
    ])
    mu = mu_earth
    a,e,E,O,I,w,P,tau = fun.vec2orbElem(rs, vs, mu)
    assert a.shape == (2,)
    assert e.shape == (2,)

def test_vec2orbElem_raises_if_mus_size_mismatch():
    # mus size doesn't match number of orbits
    rs = np.array([
        [7000.0, 0.0, 0.0],
        [7000.0, 7000.0, 0.0]
    ])
    vs = np.array([
        [0.0, 7.5, 0.0],
        [-7.5, 0.0, 0.0]
    ])
    mus = np.array([mu_earth])  # only one mu for two orbits
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, mus)

def test_vec2orbElem2_consistency_with_vec2orbElem():
    # For the same inputs vec2orbElem2 and vec2orbElem return arrays of same shape for a,e,E,O,I,w,P,tau
    rs = np.array([
        [7000.0, 0.0, 0.0],
        [8000.0, 1000.0, 0.0]
    ])
    vs = np.array([
        [0.0, 7.5, 0.0],
        [0.0, 7.2, 0.1]
    ])
    mu = mu_earth
    outputs2 = fun.vec2orbElem2(rs, vs, mu)
    outputs1 = fun.vec2orbElem(rs, vs, mu)
    for o2, o1 in zip(outputs2, outputs1):
        assert np.allclose(o2, o1, atol=1e-3)

def test_vec2orbElem2_accepts_flat_3n_vectors():
    rs = np.array([7000.0, 0.0, 0.0, 8000.0, 1000.0, 0.0])
    vs = np.array([0.0, 7.5, 0.0, 0.0, 7.2, 0.1])
    mu = mu_earth
    a,e,E,O,I,w,P,tau = fun.vec2orbElem2(rs, vs, mu)
    assert a.shape == (2,)
    assert e.shape == (2,)

def test_vec2orbElem2_scalar_mu_vector_positions():
    rs = np.array([
        [7000.0, 0.0, 0.0],
        [8000.0, 1000.0, 0.0]
    ])
    vs = np.array([
        [0.0, 7.5, 0.0],
        [0.0, 7.2, 0.1]
    ])
    mu = mu_earth
    a,e,E,O,I,w,P,tau = fun.vec2orbElem2(rs, vs, mu)
    assert a.shape == (2,)
    assert e.shape == (2,)

def test_vec2orbElem2_raises_if_mus_size_mismatch():
    rs = np.array([
        [7000.0, 0.0, 0.0],
        [8000.0, 1000.0, 0.0]
    ])
    vs = np.array([
        [0.0, 7.5, 0.0],
        [0.0, 7.2, 0.1]
    ])
    mus = np.array([mu_earth])
    with pytest.raises(AssertionError):
        fun.vec2orbElem2(rs, vs, mus)

def test_orbElem2vec_vector_E_scalar_mu_and_AB_input():
    a = np.array([7000.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    orbElem = (a, e, O, I, w)
    A, B = fun.calcAB(a, e, O, I, w)
    E = np.linspace(0, 2*np.pi, 10)
    r, v = fun.orbElem2vec(E, mu_earth, AB=(A, B))
    assert r.shape == (3, E.size)
    assert v.shape == (3, E.size)

def test_orbElem2vec_scalar_E_vector_mu_raises():
    a = np.array([7000.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    orbElem = (a, e, O, I, w)
    E = 0.0  # scalar E
    mus = np.array([mu_earth, mu_earth * 2])
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, orbElem=orbElem)
