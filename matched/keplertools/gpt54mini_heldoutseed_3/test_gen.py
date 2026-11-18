import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem


def angle_diff(a, b):
    """Smallest signed difference a-b wrapped to [-pi, pi)."""
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def assert_angles_close(actual, expected, atol=1e-10):
    diff = angle_diff(actual, expected)
    np.testing.assert_allclose(diff, 0.0, atol=atol, rtol=0.0)


def sample_orbital_elements():
    a = np.array([1.7, 2.4])
    e = np.array([0.21, 0.37])
    O = np.array([0.7, 1.2])
    I = np.array([0.4, 1.0])
    w = np.array([2.1, 0.8])
    return a, e, O, I, w


def sample_mus():
    return np.array([1.3, 2.1])


def sample_E():
    return np.array([0.9, 2.2])


def test_calcAB_shapes():
    a, e, O, I, w = sample_orbital_elements()
    A, B = calcAB(a, e, O, I, w)
    assert A.shape == (3, 2)
    assert B.shape == (3, 2)


def test_orbElem2vec_scalar_mu_with_orbElem_roundtrip_returns_vectors():
    a, e, O, I, w = sample_orbital_elements()
    E = sample_E()
    mu = 1.7

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 2)
    assert v.shape == (3, 2)

    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    np.testing.assert_allclose(a2, a, rtol=0, atol=1e-10)
    np.testing.assert_allclose(e2, e, rtol=0, atol=1e-10)
    assert_angles_close(E2, E)
    assert_angles_close(O2, O)
    assert_angles_close(I2, I)
    assert_angles_close(w2, w)
    assert np.all(P2 > 0)
    assert np.all(np.isfinite(tau2))


def test_orbElem2vec_scalar_mu_with_AB_input_roundtrip():
    a, e, O, I, w = sample_orbital_elements()
    E = sample_E()
    mu = 1.7
    A, B = calcAB(a, e, O, I, w)

    r1, v1 = orbElem2vec(E, mu, AB=(A, B))
    r2, v2 = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))

    np.testing.assert_allclose(r1, r2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v1, v2, rtol=0, atol=1e-12)


def test_orbElem2vec_returns_ab_when_requested():
    a, e, O, I, w = sample_orbital_elements()
    E = sample_E()
    mu = 1.7

    r, v, (A, B) = orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)
    assert r.shape == (3, 2)
    assert v.shape == (3, 2)
    assert A.shape == (3, 2)
    assert B.shape == (3, 2)

    A2, B2 = calcAB(a, e, O, I, w)
    np.testing.assert_allclose(A, A2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(B, B2, rtol=0, atol=1e-12)


def test_orbElem2vec_vector_mu_vector_E_roundtrip():
    a, e, O, I, w = sample_orbital_elements()
    E = sample_E()
    mus = sample_mus()

    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 2)
    assert v.shape == (3, 2)

    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mus)
    np.testing.assert_allclose(a2, a, rtol=0, atol=1e-10)
    np.testing.assert_allclose(e2, e, rtol=0, atol=1e-10)
    assert_angles_close(E2, E)
    assert_angles_close(O2, O)
    assert_angles_close(I2, I)
    assert_angles_close(w2, w)
    assert np.all(P2 > 0)
    assert np.all(np.isfinite(tau2))


def test_orbElem2vec_scalar_E_scalar_mu_requires_scalar_mu():
    a, e, O, I, w = sample_orbital_elements()
    mu = 1.7
    E = float(sample_E()[0])

    r, v = orbElem2vec(E, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]))
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)


def test_orbElem2vec_scalar_E_vector_mu_is_allowed_with_vector_orbital_elements():
    a, e, O, I, w = sample_orbital_elements()
    mu = np.array([1.3, 2.1])
    E = float(sample_E()[0])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 2)
    assert v.shape == (3, 2)


def test_orbElem2vec_invalid_when_both_orbElem_and_AB_given():
    a, e, O, I, w = sample_orbital_elements()
    E = sample_E()
    mu = 1.7
    A, B = calcAB(a, e, O, I, w)

    with pytest.raises(AssertionError, match="only set orbElem or AB"):
        orbElem2vec(E, mu, orbElem=(a, e, O, I, w), AB=(A, B))


def test_orbElem2vec_invalid_when_missing_both_orbElem_and_AB():
    with pytest.raises(AssertionError, match="supply either orbElem or AB"):
        orbElem2vec(sample_E(), 1.7)


def test_orbElem2vec_invalid_scalar_E_with_vector_mu():
    a, e, O, I, w = sample_orbital_elements()
    E = float(sample_E()[0])
    mus = sample_mus()

    with pytest.raises(AssertionError, match="Scalar E input requires scalar mus input"):
        orbElem2vec(E, mus, orbElem=(a, e, O, I, w))


def test_orbElem2vec_invalid_mus_size_mismatch():
    a, e, O, I, w = sample_orbital_elements()
    E = sample_E()
    mus = np.array([1.3, 2.1, 3.0])

    with pytest.raises(AssertionError, match="mus must be of the same size as E or scalar"):
        orbElem2vec(E, mus, orbElem=(a, e, O, I, w))


def test_vec2orbElem_accepts_nx3_input_and_matches_3xn_input():
    a, e, O, I, w = sample_orbital_elements()
    E = sample_E()
    mu = 1.7

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    r_nx3 = r.T
    v_nx3 = v.T

    out1 = vec2orbElem(r, v, mu)
    out2 = vec2orbElem(r_nx3, v_nx3, mu)

    for arr1, arr2 in zip(out1, out2):
        np.testing.assert_allclose(arr1, arr2, rtol=0, atol=1e-12)


def test_vec2orbElem_invalid_rs_vs_size_mismatch():
    rs = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0])
    mu = 1.0

    with pytest.raises(AssertionError, match="same size and contain 3n elements"):
        vec2orbElem(rs, vs, mu)


def test_vec2orbElem_invalid_mus_size_mismatch():
    a, e, O, I, w = sample_orbital_elements()
    E = sample_E()
    mu = 1.7
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))

    with pytest.raises(AssertionError, match="mus must be scalar or of size n"):
        vec2orbElem(r, v, np.array([1.0, 2.0, 3.0]))


def test_vec2orbElem_roundtrip_with_single_body_vector_input():
    a = np.array([2.2])
    e = np.array([0.15])
    O = np.array([0.9])
    I = np.array([0.5])
    w = np.array([1.4])
    E = np.array([1.1])
    mu = 3.2

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    np.testing.assert_allclose(a2, a, rtol=0, atol=1e-10)
    np.testing.assert_allclose(e2, e, rtol=0, atol=1e-10)
    assert_angles_close(E2, E)
    assert_angles_close(O2, O)
    assert_angles_close(I2, I)
    assert_angles_close(w2, w)
    assert np.all(P2 > 0)
    assert np.all(np.isfinite(tau2))
