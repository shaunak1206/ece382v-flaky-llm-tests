import numpy as np
import pytest

from keplertools import fun

# Convenient constants
MU_SUN = 1.32712440018e11  # km^3/s^2 ~ solar gravitational parameter for testing
EPS = 1e-10

def make_orbit_elements_scalar():
    # Circular orbit, simple case
    a = np.array([7000.0])                           # semi-major axis in km
    e = np.array([0.001])                            # eccentricity
    O = np.array([0.5])                              # longitude of ascending node (rad)
    I = np.array([0.1])                              # inclination (rad)
    w = np.array([1.0])                              # argument of pericenter (rad)
    return a, e, O, I, w


def make_orbit_elements_vector(n=3):
    # Multiple orbits with varying elements
    a = np.linspace(7000, 15000, n)
    e = np.linspace(0.001, 0.1, n)
    O = np.linspace(0, np.pi/2, n)
    I = np.linspace(0, np.pi/6, n)
    w = np.linspace(0, np.pi, n)
    return a, e, O, I, w

@pytest.mark.parametrize("M_vals, e_vals", [
    (0.0, 0.0),               # circular orbit, zero anomaly
    (np.pi/2, 0.1),           # quarter orbit, moderate eccentricity
    (2*np.pi, 0.5),           # full revolution, moderate eccentricity
    (3*np.pi, 0.7),           # > 2pi anomalies
    (np.array([0, np.pi, 3*np.pi/2]), np.array([0.01, 0.1, 0.01])), # vector input
])
def test_eccanom_basic(M_vals, e_vals):
    # Test eccanom returns values close to input, inside domain and shape correct.
    E = fun.eccanom(M_vals, e_vals, noc=True)  # force Python version
    E = np.array(E)
    assert E.shape == np.atleast_1d(M_vals).shape
    # Check eccentric anomaly is between 0 and 2pi
    assert np.all(E >= 0) and np.all(E < 2*np.pi)
    # Check that M and E satisfy Kepler's equation M = E - e sin E (within tolerance)
    M = np.atleast_1d(M_vals) % (2*np.pi)
    M_calc = E - np.array(e_vals) * np.sin(E)
    assert np.allclose(M, M_calc, rtol=0, atol=1e-9)

def test_eccanom_return_iter():
    M, e = 1.0, 0.2
    E, iterations = fun.eccanom(M, e, noc=True, returnIter=True)
    assert isinstance(E, np.ndarray)
    assert isinstance(iterations, int)
    assert iterations > 0
    # Validate same that Kepler's equation holds within tolerance
    assert np.isclose(M % (2*np.pi), E[0] - e*np.sin(E[0]), atol=1e-9)

def test_trueanom_basic_scalar_and_vector():
    # Scalar input
    E = 1.0
    e = 0.1
    nu = fun.trueanom(E, e)
    assert nu.shape == (1,)
    assert (nu >= 0).all() and (nu < 2*np.pi).all()
    # Vector inputs with scalar eccentricity
    E_array = np.linspace(0, 2*np.pi, 5)
    nu_vec = fun.trueanom(E_array, e)
    assert nu_vec.shape == E_array.shape
    # Vector eccentricity with scalar E
    e_array = np.linspace(0, 0.9, 5)
    nu_vec2 = fun.trueanom(E, e_array)
    assert nu_vec2.shape == e_array.shape

@pytest.mark.parametrize("a,e,O,I,w", [
    make_orbit_elements_scalar(),
    make_orbit_elements_vector(5),
])
def test_calcAB_output_shapes_and_types(a,e,O,I,w):
    A,B = fun.calcAB(a,e,O,I,w)
    assert A.shape == (3, len(a))
    assert B.shape == (3, len(a))
    assert A.dtype == np.float64
    assert B.dtype == np.float64
    # Values sanity checks
    a_norm = np.linalg.norm(A, axis=0)
    b_norm = np.linalg.norm(B, axis=0)
    # Semi-major axis norm matches input a
    np.testing.assert_allclose(a_norm, a, rtol=1e-12)
    # b_norm / a ~= sqrt(1 - e^2)
    np.testing.assert_allclose(b_norm / a_norm, np.sqrt(1 - e**2), rtol=1e-12)

@pytest.mark.parametrize("n", [1,3])
def test_vec2orbElem2vec2orbElem_consistency_scalar_and_vector(n):
    # Create dummy position and velocity vectors for test
    # Use simplified circular orbit parameters
    a, e, O, I, w = make_orbit_elements_vector(n)
    # Use kepler's laws to generate position and velocity vectors:
    # Orbital period P
    mus = MU_SUN
    if n > 1:
        mus_arr = np.full(n, mus)
    else:
        mus_arr = mus
    A,B = fun.calcAB(a, e, O, I, w)
    # Eccentric anomaly
    E = np.linspace(0, 2*np.pi, n)
    # Use orbElem2vec to get position and velocity
    r, v = fun.orbElem2vec(E, mus_arr, AB=(A,B))
    # Feed vectors into vec2orbElem2 to get back orbital elements
    out = fun.vec2orbElem2(r, v, mus_arr)
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = out

    # Compare returned and input (allow some error due to numeric precision)
    # Semi-major axes
    np.testing.assert_allclose(a_out, a, rtol=1e-6)
    # Eccentricities
    np.testing.assert_allclose(e_out, e, rtol=1e-6)
    # Longitudes of ascending nodes
    np.testing.assert_allclose(O_out % (2*np.pi), O % (2*np.pi), rtol=1e-6)
    # Inclinations
    np.testing.assert_allclose(I_out, I, rtol=1e-6)
    # Arguments of pericenter
    np.testing.assert_allclose(w_out % (2*np.pi), w % (2*np.pi), rtol=1e-6)
    # Orbital periods ~ (2 pi sqrt(a^3/mu))
    expected_P = 2*np.pi*np.sqrt(a**3/mus)
    np.testing.assert_allclose(P_out, expected_P, rtol=1e-6)

@pytest.mark.parametrize("n", [1,4])
def test_vec2orbElem2_and_vec2orbElem_equivalence(n):
    # Generate random realistic orbit data
    a, e, O, I, w = make_orbit_elements_vector(n)
    mus = MU_SUN
    if n > 1:
        mus_arr = np.full(n, mus)
    else:
        mus_arr = mus
    A, B = fun.calcAB(a, e, O, I, w)
    E = np.linspace(0, 2*np.pi, n)

    # Generate position/velocity vectors with orbElem2vec
    r1, v1 = fun.orbElem2vec(E, mus_arr, AB=(A, B))
    # Use vec2orbElem2 from fun
    out2 = fun.vec2orbElem2(r1, v1, mus_arr)
    # Use vec2orbElem from fun
    out1 = fun.vec2orbElem(r1, v1, mus_arr)

    # Compare outputs semi-major axes
    np.testing.assert_allclose(out1[0], out2[0], rtol=1e-6)
    # Compare eccentricities
    np.testing.assert_allclose(out1[1], out2[1], rtol=1e-6)

@pytest.mark.parametrize("returnAB", [False, True])
def test_orbElem2vec_returnAB_option(returnAB):
    a, e, O, I, w = make_orbit_elements_scalar()
    mus = MU_SUN
    E = np.array([1.0])
    orbElem = (a, e, O, I, w)

    if returnAB:
        r, v, (A, B) = fun.orbElem2vec(E, mus, orbElem=orbElem, returnAB=True)
        assert r.shape == (3, len(E))
        assert v.shape == (3, len(E))
        assert A.shape == (3, len(a))
        assert B.shape == (3, len(a))
    else:
        r, v = fun.orbElem2vec(E, mus, orbElem=orbElem, returnAB=False)
        assert r.shape == (3, len(E))
        assert v.shape == (3, len(E))

def test_orbElem2vec_value_errors():
    a, e, O, I, w = make_orbit_elements_scalar()
    mus = MU_SUN
    E = np.array([0.0, np.pi/2])

    # Fail if neither orbElem nor AB is provided
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus)

    # Fail if both orbElem and AB provided
    A, B = fun.calcAB(a, e, O, I, w)
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, orbElem=(a,e,O,I,w), AB=(A,B))

    # Fail if scalar E with vector mus
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.0, np.array([1.0, 1.0]), orbElem=(a,e,O,I,w))

    # Fail if vector E with mismatched mus size
    with pytest.raises(AssertionError):
        fun.orbElem2vec(np.array([0.0, 1.0]), np.array([1.0]), orbElem=(a,e,O,I,w))

def test_orbElem2vec_scalar_and_vector_E():
    a, e, O, I, w = make_orbit_elements_vector(3)
    mus = MU_SUN
    A,B = fun.calcAB(a,e,O,I,w)
    # Vector E and scalar mu
    E_vec = np.linspace(0, 2*np.pi, 3)
    r,v = fun.orbElem2vec(E_vec, mus, AB=(A,B))
    assert r.shape == (3, len(E_vec))
    assert v.shape == (3, len(E_vec))

    # Scalar E, scalar mu
    r2, v2 = fun.orbElem2vec(E_vec[0], mus, AB=(A[:,0], B[:,0]))
    assert r2.shape == (3,1)
    assert v2.shape == (3,1)

def test_vec2orbElem2_input_shape_transforms():
    a,e,O,I,w = make_orbit_elements_scalar()
    mus = MU_SUN
    A,B = fun.calcAB(a, e, O, I, w)
    E = np.array([1.0])

    r, v = fun.orbElem2vec(E, mus, AB=(A,B))
    # Provide flattened and reshaped inputs to vec2orbElem2
    r_flat = r.flatten()
    v_flat = v.flatten()
    r_3xn = r
    v_3xn = v

    out_flat = fun.vec2orbElem2(r_flat, v_flat, mus)
    out_mat = fun.vec2orbElem2(r_3xn, v_3xn, mus)
    for o1, o2 in zip(out_flat, out_mat):
        np.testing.assert_allclose(o1, o2, rtol=1e-12)

def test_vec2orbElem2_wrong_input_size_raises():
    r = np.array([1,2,3,4,5])  # length not divisible by 3
    v = np.array([1,2,3,4,5,6])
    mus = MU_SUN
    with pytest.raises(AssertionError):
        fun.vec2orbElem2(r, v, mus)

def test_vec2orbElem_wrong_input_size_raises():
    r = np.array([1,2,3,4,5])
    v = np.array([1,2,3,4,5,6])
    mus = MU_SUN
    with pytest.raises(AssertionError):
        fun.vec2orbElem(r, v, mus)

def test_vec2orbElem2_mus_size_mismatch():
    r = np.array([7000., 0., 0., 15000., 0., 0.])
    v = np.array([0., 7.5, 0., 0., 5., 0.])
    mus = np.array([MU_SUN])  # Size 1 but need size 2
    with pytest.raises(AssertionError):
        fun.vec2orbElem2(r, v, mus)

def test_vec2orbElem_mus_size_mismatch():
    r = np.array([7000., 0., 0., 15000., 0., 0.])
    v = np.array([0., 7.5, 0., 0., 5., 0.])
    mus = np.array([MU_SUN])  # Size 1 but need size 2
    with pytest.raises(AssertionError):
        fun.vec2orbElem(r, v, mus)
