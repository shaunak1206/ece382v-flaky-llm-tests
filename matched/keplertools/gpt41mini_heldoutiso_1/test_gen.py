import numpy as np
import pytest
from keplertools.fun import orbElem2vec, vec2orbElem2

def test_vec2orbElem2_single_orbit_basic_consistency():
    # Circular orbit in xy plane with velocity in y dir
    mus = 1.0
    r = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    a,e,E,O,I,w,P,tau = vec2orbElem2(r,v,mus)
    # semi-major axis close to 1
    assert np.isclose(a, 1.0, atol=1e-7)
    # eccentricity close to 0 (circle)
    assert e < 1e-7
    # inclination close to 0 (orbit in xy plane)
    assert np.isclose(I, 0.0, atol=1e-7)
    # eccentric anomaly between 0 and 2pi
    assert 0 <= E < 2*np.pi
    # orbital period positive
    assert P > 0

def test_vec2orbElem2_vectorized_input():
    mus = 1.0
    # Two orbits: one circular at r=1, v=1; one elliptical
    rs = np.array([[1.0, 1.5],
                   [0.0, 0.0],
                   [0.0, 0.0]])
    vs = np.array([[0.0, 0.0],
                   [1.0, 0.5],
                   [0.0, 0.0]])
    # vectors are columns but current shape is 3xn, so transpose rows=3, cols=2
    a,e,E,O,I,w,P,tau = vec2orbElem2(rs,vs,mus)
    assert a.shape == (2,)
    assert e.shape == (2,)
    np.testing.assert_allclose((O >= 0) & (O < 2*np.pi), np.array([True, True]))
    np.testing.assert_allclose((I >= 0) & (I <= np.pi), np.array([True, True]))

def test_vec2orbElem2_wrong_input_shapes():
    mus = 1.0
    r = np.array([1.0, 0.0])  # Only 2 elements
    v = np.array([0.0, 1.0])  # Only 2 elements
    with pytest.raises(AssertionError):
        vec2orbElem2(r,v,mus)

def test_vec2orbElem2_scalar_mus_vector_rv():
    mus = 1.0
    rs = np.array([[1,0,0],[0,1,0]]).T  # 3x2
    vs = np.array([[0,1,0],[1,0,0]]).T
    a,e,E,O,I,w,P,tau = vec2orbElem2(rs,vs,mus)
    assert a.shape == (2,)
    assert e.shape == (2,)

def test_vec2orbElem2_vector_mus_scalar_rv():
    rs = np.array([[1,0,0],[0,1,0]]).T
    vs = np.array([[0,1,0],[1,0,0]]).T
    mus = np.array([1.0, 0.5])
    a,e,E,O,I,w,P,tau = vec2orbElem2(rs,vs,mus)
    assert a.shape == (2,)

def test_orbElem2vec_scalar_single():
    # Simple circular orbit parameters
    a = 1.0
    e = 0.0
    O = 0.0
    I = 0.0
    w = 0.0
    E = 0.5  # rad
    mus = 1.0

    r,v = orbElem2vec(E, mus, orbElem=(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w])))
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    # radius magnitude close to a*(1 - e*cos(E)) = a*1 = 1
    rmag = np.linalg.norm(r[:,0])
    expected_rmag = a*(1 - e*np.cos(E))
    assert np.isclose(rmag, expected_rmag, atol=1e-7)

def test_orbElem2vec_vector_multi():
    n = 5
    a = np.linspace(1, 2, n)
    e = np.linspace(0, 0.5, n)
    O = np.linspace(0, np.pi/2, n)
    I = np.linspace(0, np.pi/4, n)
    w = np.linspace(0, np.pi/3, n)
    E = np.linspace(0, 2*np.pi, n)
    mus = 1.0  # scalar
    
    r,v = orbElem2vec(E, mus, orbElem=(a,e,O,I,w))
    assert r.shape == (3, n)
    assert v.shape == (3, n)
    # test ranges
    assert np.all(r >= -2) and np.all(r <= 2)
    assert np.all(v >= -10) and np.all(v <= 10)

def test_orbElem2vec_returnAB():
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])
    E = np.array([0.5])
    mus = 1.0

    r,v,(A,B) = orbElem2vec(E, mus, orbElem=(a,e,O,I,w), returnAB=True)
    assert r.shape == (3,1)
    assert v.shape == (3,1)
    assert A.shape == (3,1)
    assert B.shape == (3,1)

def test_orbElem2vec_AB_input():
    # Build A and B manually from orbElem then use in orbElem2vec
    a = np.array([1.0])
    e = np.array([0.2])
    O = np.array([0.5])
    I = np.array([0.3])
    w = np.array([1.0])
    from keplertools.fun import calcAB
    A, B = calcAB(a, e, O, I, w)
    E = np.array([0.7])
    mus = 2.0
    
    r,v = orbElem2vec(E, mus, AB=(A,B))
    assert r.shape == (3,1)
    assert v.shape == (3,1)

def test_orbElem2vec_scalarE_vectorA_B_raises():
    a = np.array([1.0, 1.0])
    e = np.array([0.0, 0.1])
    O = np.array([0.0, 0.0])
    I = np.array([0.0, 0.0])
    w = np.array([0.0, 0.0])
    from keplertools.fun import calcAB
    A,B = calcAB(a,e,O,I,w)
    E = 0.1
    mus = 1.0
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, AB=(A,B))

def test_orbElem2vec_invalid_inputs_neither_orbElem_AB():
    E = np.array([0.1])
    mus = 1.0
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus)

def test_orbElem2vec_conflict_orbElem_AB():
    E = 0.1
    mus = 1.0
    orbElem = (np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
    A = np.eye(3)[:, :1]
    B = np.eye(3)[:, :1]
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, orbElem=orbElem, AB=(A,B))

def test_vec2orbElem2_inclination_bounds():
    mus = 1.0
    # Position directly on z axis, velocity perpendicular -> inclination pi/2
    r = np.array([0,0,1])
    v = np.array([1,0,0])
    a,e,E,O,I,w,P,tau = vec2orbElem2(r,v,mus)
    assert np.isclose(I, np.pi/2, atol=1e-7)

def test_vec2orbElem2_node_line_direction():
    mus = 1.0
    r = np.array([1,0,0])
    v = np.array([0,1,0])
    _,_,_,O,_,_,_,_ = vec2orbElem2(r,v,mus)
    assert 0 <= O < 2*np.pi

def test_orbElem2vec_domain_E():
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    mus = 1.0
    # Test with E out of [0, 2pi]
    for E_test in [-1.0, 0.0, 2*np.pi, 3*np.pi]:
        r,v = orbElem2vec(E_test, mus, orbElem=(a,e,O,I,w))
        assert r.shape == (3,1)
        assert v.shape == (3,1)

def test_vec2orbElem2_negative_rs_shape():
    mus = 1.0
    rs = np.array([[-1.0, 0.0, 0.0]])
    vs = np.array([[0.0, 1.0, 0.0]])
    with pytest.raises(AssertionError):
        vec2orbElem2(rs, vs, mus)

def test_vec2orbElem2_scalar_mus_wrong_shape():
    mus = np.array([1.0, 2.0])
    r = np.array([1,0,0])
    v = np.array([0,1,0])
    with pytest.raises(AssertionError):
        vec2orbElem2(r,v,mus)

def test_vec2orbElem2_reshaping_behavior():
    mus = 1.0
    # Flattened input vectors 3n x 1 for 2 bodies
    r = np.array([1.0,0.0,0.0, 0.0,1.0,0.0])
    v = np.array([0.0,1.0,0.0, 1.0,0.0,0.0])
    a,e,E,O,I,w,P,tau = vec2orbElem2(r,v,mus)
    assert a.size == 2
    assert e.size == 2
    assert E.size == 2

def test_vec2orbElem2_invalid_dim_rs():
    mus = 1.0
    rs = np.zeros((2,2,2))  # 3d array invalid
    vs = np.zeros((2,2,2))
    with pytest.raises(AssertionError):
        vec2orbElem2(rs,vs,mus)
