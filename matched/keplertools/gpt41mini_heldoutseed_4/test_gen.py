import numpy as np
import pytest
from keplertools import fun

# For convenience
pi = np.pi

def test_orbElem2vec_scalar_single():
    # Simple circular orbit parameters for one orbit
    a = 1.0
    e = 0.0
    O = 0.0
    I = 0.0
    w = 0.0
    mus = 1.0
    E = pi/2

    rs, vs = fun.orbElem2vec(E, mus, orbElem=(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w])))
    assert rs.shape == (3, 1)
    assert vs.shape == (3, 1)
    # Position at E=pi/2 should be approx (a*cos(E)-ae, a*sin(E)*sqrt(1-e^2)) = (0, a)
    np.testing.assert_allclose(rs.flatten()[0], 0, atol=1e-12)
    np.testing.assert_allclose(rs.flatten()[1], a, atol=1e-12)
    # Velocity magnitude should be nonzero
    assert np.linalg.norm(vs) > 0

def test_orbElem2vec_returnAB_flag():
    a = 1.0
    e = 0.1
    O = 0.2
    I = 0.3
    w = 0.4
    mus = 2.0
    E = np.array([0.1, 1.5, 3.0])

    rs, vs, AB = fun.orbElem2vec(E, mus, orbElem=(np.array([a]*3), np.array([e]*3), np.array([O]*3), np.array([I]*3), np.array([w]*3)), returnAB=True)
    assert isinstance(AB, tuple)
    assert len(AB) == 2
    A, B = AB
    assert A.shape == (3, 3)
    assert B.shape == (3, 3)
    assert rs.shape == (3, 3)
    assert vs.shape == (3, 3)

def test_orbElem2vec_vector_scalar_mu():
    # Multiple positions (varying E) for single body (scalar mus)
    a = 1.5
    e = 0.2
    O = 0.3
    I = 0.4
    w = 0.5
    mus = 3.0
    E = np.linspace(0, 2*pi, 4)

    orbElem = (np.array([a]*len(E)), np.array([e]*len(E)), np.array([O]*len(E)), np.array([I]*len(E)), np.array([w]*len(E)))
    rs, vs = fun.orbElem2vec(E, mus, orbElem=orbElem)
    assert rs.shape == (3, len(E))
    assert vs.shape == (3, len(E))

def test_orbElem2vec_vector_vector_mu():
    # Multiple bodies, with varying mus and varying E
    n = 5
    a = np.linspace(0.8, 1.2, n)
    e = np.linspace(0, 0.5, n)
    O = np.linspace(0, 2*pi, n)
    I = np.linspace(0, pi/2, n)
    w = np.linspace(0, pi, n)
    mus = np.linspace(1, 5, n)
    E = np.linspace(0, 2*pi, n)

    orbElem = (a, e, O, I, w)
    rs, vs = fun.orbElem2vec(E, mus, orbElem=orbElem)
    assert rs.shape == (3, n)
    assert vs.shape == (3, n)

def test_orbElem2vec_AB_input():
    # Use AB input instead of orbElem: calculate AB, then pass and check consistency
    a = 2.0
    e = 0.3
    O = 0.1
    I = 0.2
    w = 0.3
    mus = 4.0
    E = pi/4

    A, B = fun.calcAB(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    rs1, vs1 = fun.orbElem2vec(E, mus, orbElem=(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w])))
    rs2, vs2 = fun.orbElem2vec(E, mus, AB=(A, B))

    np.testing.assert_allclose(rs1, rs2)
    np.testing.assert_allclose(vs1, vs2)

def test_orbElem2vec_invalid_inputs():
    a = 1.0
    e = 0.1
    O = 0.1
    I = 0.1
    w = 0.1
    mus = 1.0
    E = 0.1

    # Both orbElem and AB supplied -> error
    A, B = fun.calcAB(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, orbElem=(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w])), AB=(A, B))

    # Neither orbElem nor AB supplied -> error
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus)

    # Scalar E, vector mus -> error
    mus_vec = np.array([1.0, 2.0])
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.5, mus_vec, orbElem=(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w])))

    # Vector E, mus size mismatch -> error
    E_vec = np.array([0.1, 0.2])
    mus_vec_short = np.array([1.0])
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E_vec, mus_vec_short, orbElem=(np.array([a]*2), np.array([e]*2), np.array([O]*2), np.array([I]*2), np.array([w]*2)))

def test_calcAB_size_mismatch():
    a = np.array([1, 2])
    e = np.array([0.1])
    O = np.array([0.1, 0.2])
    I = np.array([0.1, 0.2])
    w = np.array([0.1, 0.2])

    with pytest.raises(AssertionError):
        fun.calcAB(a, e, O, I, w)

def test_calcAB_properties():
    # Check shape of outputs
    n = 4
    a = np.ones(n)
    e = np.zeros(n)
    O = np.zeros(n)
    I = np.zeros(n)
    w = np.zeros(n)

    A, B = fun.calcAB(a, e, O, I, w)
    assert A.shape == (3, n)
    assert B.shape == (3, n)

    # For circular orbit with zero angles, B should be zero vector since sqrt(1-e^2)=1 and sin(w)=0
    np.testing.assert_allclose(B[0], -a*np.sin(w))
    np.testing.assert_allclose(B[:,0], [0, 0, a[0]*np.sin(I[0])*np.cos(w[0])])

def test_orbElem2vec_consistency_with_vec2orbElem():
    # Round trip test: vec2orbElem2 + orbElem2vec should roughly invert each other
    
    # Setup body parameters
    mus = 1.0
    # Orbit: semi-major axis 1 AU, eccentricity 0.1, inclination 0.3, etc
    a = 1.0
    e = 0.1
    O = 0.5
    I = 0.3
    w = 0.1
    n = 3
    orbElem = (np.array([a]*n), np.array([e]*n), np.array([O]*n), np.array([I]*n), np.array([w]*n))

    E = np.linspace(0, 2*pi, n, endpoint=False)

    # Convert to position/velocity vectors
    rs, vs = fun.orbElem2vec(E, mus, orbElem=orbElem)

    # Convert back
    a2, e2, E2, O2, I2, w2, P2, tau2 = fun.vec2orbElem(rs, vs, mus)

    # Semi-major axis close
    np.testing.assert_allclose(a2, a, atol=1e-7)
    # Eccentricity close
    np.testing.assert_allclose(e2, e, atol=1e-7)
    # Inclinations close
    np.testing.assert_allclose(I2, I, atol=1e-7)
    # Longitude of ascending node close modulo 2pi
    diffO = np.mod(O2 - O + pi, 2*pi) - pi  # difference mapped to (-pi, pi)
    np.testing.assert_allclose(diffO, 0, atol=1e-6)
    # Arguments of pericenter close modulo 2pi
    diffw = np.mod(w2 - w + pi, 2*pi) - pi
    np.testing.assert_allclose(diffw, 0, atol=1e-6)

def test_orbElem2vec_E_and_mu_scalar_different_shapes_error():
    a = 1.0
    e = 0.1
    O = 0.1
    I = 0.1
    w = 0.1
    mus = 1.0
    # E vector length 2, mus scalar ok
    E = np.array([0.1, 0.2])
    orbElem = (np.array([a]*2), np.array([e]*2), np.array([O]*2), np.array([I]*2), np.array([w]*2))

    # A and B created will be of shape 3x2
    # This should be accepted as E vector and mus scalar is allowed
    
    rs, vs = fun.orbElem2vec(E, mus, orbElem=orbElem)
    assert rs.shape == (3, 2)
    assert vs.shape == (3, 2)

def test_orbElem2vec_A_B_vector_E_mu_scalar_size_mismatch():
    # Provide A and B shapes mismatching E shape, should produce assertion error
    E = np.array([0.1, 0.2, 0.3])
    mus = 1.0
    # Provide A, B with size 3, expected size would be 3*len(E) = 9
    A = np.ones((3,1))
    B = np.ones((3,1))
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, AB=(A,B))

def test_orbElem2vec_A_B_scalar_E_vector_mu_vector_size_mismatch():
    # Provide vector mu and E size mismatch with A and B
    E = np.array([0.1, 0.2])
    mus = np.array([1.0, 2.0, 3.0])
    orbElem = (np.array([1.0,1.1]), np.array([0.1,0.1]), np.array([0.1,0.1]), np.array([0.1,0.1]), np.array([0.1,0.1]))
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, orbElem=orbElem)

def test_orbElem2vec_assert_A_B_sizes_scalar_case():
    # Scalar E and mus require A,B to be 3x1 size, test error if different
    E = 0.1
    mus = 1.0
    # A and B vectors with wrong shape (e.g. 3x2)
    A = np.ones((3,2))
    B = np.ones((3,2))
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, AB=(A,B))

def test_orbElem2vec_assert_A_B_sizes_vector_case():
    # Vector E and mus require A,B to be 3xn size
    E = np.array([0.1, 0.2])
    mus = np.array([1.0, 1.0])
    A = np.ones((3,1))  # wrong size: 3x1 instead 3x2
    B = np.ones((3,1))
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, AB=(A,B))

def test_orbElem2vec_A_B_output_consistency():
    # Test that r vs returned aligns with expected from known values
    
    # Define values to create known A,B
    a = 1.0
    e = 0.0
    O = 0.0
    I = 0.0
    w = 0.0
    mus = 1.0
    E = np.array([0.0])

    A, B = fun.calcAB(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    # Manually calculate expected r at E=0: r = A*(cos(E)-e) + B*sin(E) = A*(1-0) + 0 = A
    rs, vs = fun.orbElem2vec(E, mus, AB=(A,B))
    np.testing.assert_allclose(rs, A, atol=1e-14)

def test_orbElem2vec_angles_wrapping():
    # Check that angles > 2pi or <0 do not break calcAB or orbElem2vec
    
    a = 1.0
    e = 0.1
    O = 7 * pi  # > 2pi
    I = -pi/2   # < 0, invalid orbital inclination (but test for calcAB assert handling)
    w = 4 * pi  # > 2pi
    mus = 1.0
    E = 0.5

    # calcAB requires same sizes and doesn't restrict angles explicitly
    with pytest.raises(AssertionError):
        fun.calcAB(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))

    # So fix I to valid range
    I = pi/2 - 0.1
    A, B = fun.calcAB(np.array([a]), np.array([e]), np.array([O % (2*pi)]), np.array([I]), np.array([w % (2*pi)]))

    rs, vs = fun.orbElem2vec(E, mus, AB=(A, B))
    assert rs.shape == (3, 1)
    assert vs.shape == (3, 1)

def test_orbElem2vec_handles_E_scalar_mu_scalar():
    a = 1.0
    e = 0.1
    O = 0.0
    I = 0.0
    w = 0.0
    mus = 1.0
    E = 0.2

    orbElem = (np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    rs, vs = fun.orbElem2vec(E, mus, orbElem=orbElem)
    assert rs.shape == (3, 1)
    assert vs.shape == (3, 1)

def test_orbElem2vec_handles_E_array_mu_scalar():
    a = 1.0
    e = 0.1
    O = 0.0
    I = 0.0
    w = 0.0
    mus = 1.0
    E = np.linspace(0, 2*pi, 10)

    orbElem = (np.array([a]*10), np.array([e]*10), np.array([O]*10), np.array([I]*10), np.array([w]*10))
    rs, vs = fun.orbElem2vec(E, mus, orbElem=orbElem)
    assert rs.shape == (3, 10)
    assert vs.shape == (3, 10)
