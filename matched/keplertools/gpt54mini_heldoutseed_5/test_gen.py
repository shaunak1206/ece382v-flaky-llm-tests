import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def angle_close(a, b, tol=1e-10):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = np.mod(a - b + np.pi, 2 * np.pi) - np.pi
    return np.all(np.abs(diff) <= tol)


def sample_orbital_elements():
    a = 2.3
    e = 0.31
    O = 1.1
    I = 0.7
    w = 2.2
    return a, e, O, I, w


def test_orbElem2vec_scalar_roundtrip_with_vec2orbElem():
    a, e, O, I, w = sample_orbital_elements()
    E = 1.4
    mu = 1.7

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    out = vec2orbElem(r, v, mu)
    a2, e2, E2, O2, I2, w2, P2, tau2 = out

    assert np.isclose(a2[0], a, rtol=1e-10, atol=1e-10)
    assert np.isclose(e2[0], e, rtol=1e-10, atol=1e-10)
    assert angle_close(E2[0], E, tol=1e-10)
    assert angle_close(O2[0], O, tol=1e-10)
    assert angle_close(I2[0], I, tol=1e-10)
    assert angle_close(w2[0], w, tol=1e-10)
    assert np.isfinite(P2[0])
    assert np.isfinite(tau2[0])


def test_orbElem2vec_returns_expected_shapes_for_scalar_inputs():
    a, e, O, I, w = sample_orbital_elements()
    E = 0.8
    mu = 2.0

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))

    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(v))


def test_orbElem2vec_vector_E_scalar_mu_shape():
    a, e, O, I, w = sample_orbital_elements()
    E = np.array([0.2, 1.0, 2.1, 4.0])
    mu = 1.9

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))

    assert r.shape == (3, E.size)
    assert v.shape == (3, E.size)


def test_orbElem2vec_vector_E_vector_mu_shape():
    n = 4
    E = np.array([0.2, 1.0, 2.1, 4.0])
    mu = np.array([1.9, 2.0, 2.1, 2.2])
    a = np.array([2.3, 2.4, 2.5, 2.6])
    e = np.array([0.31, 0.21, 0.14, 0.28])
    O = np.array([1.1, 0.3, 2.2, 1.7])
    I = np.array([0.7, 1.0, 0.4, 1.2])
    w = np.array([2.2, 1.1, 0.9, 0.2])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))

    assert r.shape == (3, n)
    assert v.shape == (3, n)


def test_orbElem2vec_accepts_AB_input_and_returns_same_as_orbElem_input():
    a, e, O, I, w = sample_orbital_elements()
    E = np.array([0.3, 1.2, 2.4])
    mu = 1.5

    AB = calcAB(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    r1, v1 = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    r2, v2 = orbElem2vec(E, mu, AB=AB)

    assert np.allclose(r1, r2, rtol=1e-12, atol=1e-12)
    assert np.allclose(v1, v2, rtol=1e-12, atol=1e-12)


def test_orbElem2vec_returnAB_matches_calcAB():
    a, e, O, I, w = sample_orbital_elements()
    E = 1.1
    mu = 1.8

    r, v, AB = orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)
    A_ref, B_ref = calcAB(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))

    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.allclose(AB[0], A_ref, rtol=1e-12, atol=1e-12)
    assert np.allclose(AB[1], B_ref, rtol=1e-12, atol=1e-12)


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError, match="either orbElem or AB"):
        orbElem2vec(1.0, 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    a, e, O, I, w = sample_orbital_elements()
    E = 1.0
    mu = 1.0
    AB = calcAB(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))

    with pytest.raises(AssertionError, match="only set orbElem or AB"):
        orbElem2vec(E, mu, orbElem=(a, e, O, I, w), AB=AB)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    a, e, O, I, w = sample_orbital_elements()
    with pytest.raises(AssertionError, match="Scalar E input requires scalar mus input"):
        orbElem2vec(1.0, np.array([1.0, 2.0]), orbElem=(a, e, O, I, w))


def test_orbElem2vec_vector_mus_must_match_E_size():
    a = np.array([2.3, 2.4])
    e = np.array([0.31, 0.21])
    O = np.array([1.1, 0.3])
    I = np.array([0.7, 1.0])
    w = np.array([2.2, 1.1])
    E = np.array([0.3, 1.2])
    mu = np.array([1.5, 1.6, 1.7])

    with pytest.raises(AssertionError, match="mus must be of the same size as E or scalar"):
        orbElem2vec(E, mu, orbElem=(a, e, O, I, w))


def test_vec2orbElem_scalar_inputs_roundtrip_from_orbElem2vec():
    a, e, O, I, w = sample_orbital_elements()
    E = 0.9
    mu = 1.3

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    assert np.isclose(a2[0], a, rtol=1e-10, atol=1e-10)
    assert np.isclose(e2[0], e, rtol=1e-10, atol=1e-10)
    assert angle_close(E2[0], E, tol=1e-10)
    assert angle_close(O2[0], O, tol=1e-10)
    assert angle_close(I2[0], I, tol=1e-10)
    assert angle_close(w2[0], w, tol=1e-10)


def test_vec2orbElem_accepts_3xn_and_nx3_inputs():
    a, e, O, I, w = sample_orbital_elements()
    E = np.array([0.2, 1.4, 2.7])
    mu = 1.6

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    out1 = vec2orbElem(r, v, mu)
    out2 = vec2orbElem(r.T, v.T, mu)

    for i in range(8):
        assert np.allclose(out1[i], out2[i], rtol=1e-12, atol=1e-12)


def test_vec2orbElem_rejects_mismatched_vector_sizes():
    r = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0, 2.0])
    mu = 1.0

    with pytest.raises(AssertionError, match="same size and contain 3n elements"):
        vec2orbElem(r, v, mu)


def test_vec2orbElem_rejects_incorrect_mus_size():
    rs = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 0.0])
    mus = np.array([1.0, 2.0, 3.0])

    with pytest.raises(AssertionError, match="mus must be scalar or of size n"):
        vec2orbElem(rs, vs, mus)


def test_vec2orbElem_multiple_bodies_roundtrip():
    a = np.array([2.3, 2.8])
    e = np.array([0.31, 0.12])
    O = np.array([1.1, 2.2])
    I = np.array([0.7, 1.0])
    w = np.array([2.2, 0.4])
    E = np.array([0.8, 1.9])
    mu = np.array([1.7, 2.4])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    assert np.allclose(a2, a, rtol=1e-10, atol=1e-10)
    assert np.allclose(e2, e, rtol=1e-10, atol=1e-10)
    assert angle_close(E2, E, tol=1e-10)
    assert angle_close(O2, O, tol=1e-10)
    assert angle_close(I2, I, tol=1e-10)
    assert angle_close(w2, w, tol=1e-10)
    assert np.all(np.isfinite(P2))
    assert np.all(np.isfinite(tau2))
