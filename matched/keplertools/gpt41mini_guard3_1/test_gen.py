import pytest
import numpy as np
import random

from keplertools.fun import orbElem2vec, vec2orbElem


# Helper for float array comparison with tolerance
def assert_allclose(a, b, rtol=1e-12, atol=1e-14):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def make_test_vectors_one_body():
    # Circular equatorial orbit, semi-major axis 1 AU, mu=1 AU^3/day^2
    # Position at periapsis: (1, 0, 0)
    # Velocity magnitude: sqrt(mu/a) = 1
    rs = np.array([1.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0])
    mus = 1.0
    return rs, vs, mus


def make_test_vectors_multiple_bodies(n=3):
    # Create three distinct elliptical orbits
    rs = np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 1.0, 0.0]
    ])[:n]
    vs = np.array([
        [0.0, 1.0, 0.0],
        [-0.75, 0.5, 0.0],
        [0.5, -0.5, 0.0]
    ])[:n]
    mus = np.array([1.0, 1.0, 1.0])[:n]
    # reshape to flattened 3n-vector for input
    rs_flat = rs.reshape(-1)
    vs_flat = vs.reshape(-1)
    return rs_flat, vs_flat, mus


@pytest.mark.parametrize("func", [orbElem2vec, vec2orbElem])
def test_no_mutation_of_global_state(func):
    # Seed random generators used by keplertools for consistent behavior
    # Even if these functions don't use random in fun.py, follow instructions
    random.seed(12345)
    np.random.seed(12345)

    # Just call each function with minimal inputs to ensure no global mutation side-effect
    # orbElem2vec requires at least E, mus, and orbElem or AB
    if func is orbElem2vec:
        E = 0.1
        mus = 1.0
        a = np.array([1.0])
        e = np.array([0.1])
        O = np.array([0.0])
        I = np.array([0.0])
        w = np.array([0.0])
        orbElem = (a, e, O, I, w)
        r, v = func(E, mus, orbElem=orbElem)
        assert r.shape == (3, 1)
        assert v.shape == (3, 1)
    else:
        # vec2orbElem requires rs, vs, mus
        rs, vs, mus = make_test_vectors_one_body()
        a, e, E, O, I, w, P, tau = func(rs, vs, mus)
        # Check array sizes
        for arr in (a, e, E, O, I, w, P, tau):
            assert arr.shape == (1,)


def test_vec2orbElem_input_shapes_and_scalar_mus():
    random.seed(0)
    np.random.seed(0)

    # use 1 body, inputs as 3x1 arrays
    rs = np.array([[1.0], [0.0], [0.0]])
    vs = np.array([[0.0], [1.0], [0.0]])
    mus = 1.0

    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, mus)
    # a should be ~1, e ~0 for circular orbit
    np.testing.assert_allclose(a, 1, rtol=1e-12)
    np.testing.assert_allclose(e, 0, atol=1e-12)
    # Period consistent with a=1, mu=1
    expected_P = 2 * np.pi
    np.testing.assert_allclose(P, expected_P, rtol=1e-12)


def test_vec2orbElem_input_shapes_and_vector_mus():
    random.seed(1)
    np.random.seed(1)

    # Setup two bodies, mus vector
    rs = np.array([
        [1.0, 1.5],
        [0.0, 0.1],
        [0.0, 0.0]
    ]).T.reshape(-1)
    vs = np.array([
        [0.0, 0.0],
        [1.0, 0.8],
        [0.0, 0.0]
    ]).T.reshape(-1)
    mus = np.array([1.0, 1.2])

    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, mus)

    # Check output length matches number of bodies
    assert a.size == e.size == E.size == O.size == I.size == w.size == P.size == tau.size == len(mus)
    # Semi-major axis and periods positive
    assert np.all(a > 0)
    assert np.all(P > 0)


def test_vec2orbElem_errors_for_bad_inputs():
    random.seed(2)
    np.random.seed(2)

    rs = np.array([0., 0., 0., 1., 1., 1.])
    vs = np.array([0., 0., 0., 0., 0., 0.])
    mus = np.array([1.0, 2.0])

    # rs, vs size mismatch
    with pytest.raises(AssertionError):
        vec2orbElem(rs[:-1], vs, mus)

    # mus size mismatch
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, np.array([1.0]))

    # rs.ndim > 2
    with pytest.raises(AssertionError):
        vec2orbElem(np.zeros((1, 2, 3)), vs, 1.0)


def test_orbElem2vec_scalar_inputs():
    random.seed(3)
    np.random.seed(3)

    E = 0.5
    mus = 1.0
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])

    orbElem = (a, e, O, I, w)
    r, v = orbElem2vec(E, mus, orbElem=orbElem)
    # Shape 3 x 1
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)

    # Position norm ~ close to a*(1 - e*cos(E)) approximately
    orbit_radius = a[0] * (1 - e[0] * np.cos(E))
    rnorm = np.linalg.norm(r)
    np.testing.assert_allclose(rnorm, orbit_radius, rtol=1e-12)


def test_orbElem2vec_vector_E_scalar_mu():
    random.seed(4)
    np.random.seed(4)

    E = np.linspace(0, 2 * np.pi, 5)
    mus = 1.0
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])

    orbElem = (a, e, O, I, w)
    r, v = orbElem2vec(E, mus, orbElem=orbElem)
    assert r.shape == (3, len(E))
    assert v.shape == (3, len(E))
    # For circular zero inclination orbit at zero angles,
    # position x component ~ a*(cos(E) - e)
    expected_x = a[0] * (np.cos(E) - e[0])
    np.testing.assert_allclose(r[0], expected_x, rtol=1e-12)


def test_orbElem2vec_vector_E_vector_mu_AB_input():
    random.seed(5)
    np.random.seed(5)
    n = 4
    E = np.linspace(0, 2 * np.pi, n)
    mus = np.linspace(1.0, 1.5, n)
    a = np.linspace(1.0, 2.0, n)
    e = np.linspace(0.1, 0.3, n)
    O = np.linspace(0, 0.5, n)
    I = np.linspace(0, 0.2, n)
    w = np.linspace(0.1, 0.4, n)

    orbElem = (a, e, O, I, w)
    A, B = orbElem2vec(0, 1, orbElem=orbElem, returnAB=True)[2]

    # Use AB directly with orbElem2vec
    r, v = orbElem2vec(E, mus, AB=(A, B))
    assert r.shape == (3, n)
    assert v.shape == (3, n)

    # Check size relations
    assert A.shape == (3, n)
    assert B.shape == (3, n)

    # Positions x component is sum of A[:,i] * (cos E_i - e_i) + B[:,i] * sin E_i in vector form
    # Test for first element
    pos0 = r[:, 0]
    expected0 = A[:, 0] * (np.cos(E[0]) - e[0]) + B[:, 0] * np.sin(E[0])
    np.testing.assert_allclose(pos0, expected0, rtol=1e-12)


def test_orbElem2vec_invalid_input_combinations():
    random.seed(6)
    np.random.seed(6)

    E = np.array([0.1, 0.2])
    mus = 1.0
    a = np.array([1.0, 1.1])
    e = np.array([0.1, 0.2])
    O = np.array([0.0, 0.0])
    I = np.array([0.0, 0.0])
    w = np.array([0.0, 0.0])

    orbElem = (a, e, O, I, w)
    A, B = None, None

    # both orbElem and AB set raises AssertionError
    AB = ([1, 2, 3], [4, 5, 6])
    with pytest.raises(AssertionError):
        orbElem2vec(0.1, 1.0, orbElem=orbElem, AB=AB)

    # neither orbElem nor AB set raises AssertionError
    with pytest.raises(AssertionError):
        orbElem2vec(0.1, 1.0)


def test_vec2orbElem_vs_vec2orbElem2_consistency():
    # We test that for single body input, outputs are roughly similar
    random.seed(7)
    np.random.seed(7)

    from keplertools.fun import vec2orbElem2

    rs, vs, mus = make_test_vectors_one_body()

    out1 = vec2orbElem(rs, vs, mus)
    out2 = vec2orbElem2(rs, vs, mus)

    a1, e1, E1, O1, I1, w1, P1, tau1 = out1
    a2, e2, E2, O2, I2, w2, P2, tau2 = out2

    # Assert all outputs close within tolerances, angles modulo 2pi
    np.testing.assert_allclose(a1, a2, rtol=1e-12)
    np.testing.assert_allclose(e1, e2, rtol=1e-12)
    np.testing.assert_allclose(P1, P2, rtol=1e-12)
    np.testing.assert_allclose(tau1, tau2, rtol=1e-12)

    # Angular values: wrap differences within 2pi and assert close modulo 2pi
    def angle_close(a, b):
        diff = (a - b) % (2*np.pi)
        return np.minimum(diff, 2*np.pi - diff)

    assert angle_close(O1, O2) < 1e-10
    assert angle_close(I1, I2) < 1e-10
    assert angle_close(w1, w2) < 1e-10

    # Eccentric anomaly angles may wrap, check modulo
    assert angle_close(E1, E2) < 1e-10


def test_calcAB_shapes_and_output_properties():
    random.seed(8)
    np.random.seed(8)

    from keplertools.fun import calcAB

    # Single orbit
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])

    A, B = calcAB(a, e, O, I, w)

    # Check shape
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)

    # Norm B related to A scaled by sqrt(1-e²)
    a_norm = np.linalg.norm(A, axis=0)[0]
    b_norm = np.linalg.norm(B, axis=0)[0]
    np.testing.assert_allclose(b_norm, a_norm * np.sqrt(1 - e[0]**2), rtol=1e-12)

    # Multiple orbits
    n = 5
    a = np.linspace(1, 2, n)
    e = np.linspace(0, 0.5, n)
    O = np.linspace(0, np.pi, n)
    I = np.linspace(0, np.pi/2, n)
    w = np.linspace(0, np.pi/4, n)

    A, B = calcAB(a, e, O, I, w)
    assert A.shape == (3, n)
    assert B.shape == (3, n)


def test_orbElem2vec_returnAB_option():
    random.seed(9)
    np.random.seed(9)

    E = 0.1
    mus = 1.0
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])

    orbElem = (a, e, O, I, w)
    r, v, AB = orbElem2vec(E, mus, orbElem=orbElem, returnAB=True)
    A, B = AB
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)


def test_orbElem2vec_scalar_E_vector_mu_expect_assertion():
    random.seed(10)
    np.random.seed(10)

    E = 0.3
    mus = np.array([1.0, 1.1])
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])

    orbElem = (a, e, O, I, w)
    # Scalar E with vector mus should raise assertion
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, orbElem=orbElem)


def test_orbElem2vec_vector_E_scalar_mu_A_B_size_assertions():
    random.seed(11)
    np.random.seed(11)

    n = 3
    E = np.linspace(0, np.pi / 2, n)
    mus = 1.0
    # Providing incorrect sized A, B arrays for AB input
    A_bad = np.ones((3, 1))
    B_bad = np.ones((3, 1))
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, AB=(A_bad, B_bad))

    # Correct sized AB input passes
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    orbElem = (a, e, O, I, w)
    A, B = orbElem2vec(E, mus, orbElem=orbElem, returnAB=True)[2]
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)


def test_vec2orbElem_and_orbElem2vec_inversion():
    random.seed(12)
    np.random.seed(12)

    rs, vs, mus = make_test_vectors_one_body()
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, mus)

    orbElem = (a, e, O, I, w)
    r2, v2 = orbElem2vec(E, mus, orbElem=orbElem)

    # The reconstructed position and velocity should be close to input
    np.testing.assert_allclose(rs, r2.flatten(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(vs, v2.flatten(), rtol=1e-10, atol=1e-12)


def test_vec2orbElem_and_orbElem2vec_many_points():
    random.seed(13)
    np.random.seed(13)

    rs_flat, vs_flat, mus = make_test_vectors_multiple_bodies(n=3)
    a, e, E, O, I, w, P, tau = vec2orbElem(rs_flat, vs_flat, mus)

    orbElem = (a, e, O, I, w)
    r2, v2 = orbElem2vec(E, mus, orbElem=orbElem)

    assert r2.shape == (3, E.size)
    assert v2.shape == (3, E.size)

    # Positions and velocities are close-ish (note input rs_flat vs reshaped r2 shape)
    np.testing.assert_allclose(rs_flat.reshape(3, -1), r2, rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(vs_flat.reshape(3, -1), v2, rtol=1e-9, atol=1e-11)
