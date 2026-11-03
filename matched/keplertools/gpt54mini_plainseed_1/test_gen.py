import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem


def _angle_diff(a, b):
    """Return wrapped absolute difference between angles."""
    return np.abs(np.mod(a - b + np.pi, 2 * np.pi) - np.pi)


def _sample_orbital_elements():
    a = np.array([1.2, 2.5, 0.8])
    e = np.array([0.1, 0.35, 0.05])
    O = np.array([0.2, 1.1, 2.4])
    I = np.array([0.3, 0.9, 1.4])
    w = np.array([0.4, 2.0, 0.7])
    return a, e, O, I, w


def test_calcab_shapes():
    a, e, O, I, w = _sample_orbital_elements()
    A, B = calcAB(a, e, O, I, w)

    assert A.shape == (3, 3)
    assert B.shape == (3, 3)


def test_orbElem2vec_scalar_orbital_elements_round_trip():
    a, e, O, I, w = _sample_orbital_elements()
    idx = 0
    mu = 1.7
    E = 1.3

    r, v = orbElem2vec(E, mu, orbElem=(a[idx], e[idx], O[idx], I[idx], w[idx]))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.allclose(a2, a[idx], rtol=1e-10, atol=1e-10)
    assert np.allclose(e2, e[idx], rtol=1e-10, atol=1e-10)
    assert _angle_diff(E2, np.mod(E, 2 * np.pi)) < 1e-10
    assert _angle_diff(O2, O[idx]) < 1e-10
    assert _angle_diff(I2, I[idx]) < 1e-10
    assert _angle_diff(w2, w[idx]) < 1e-10
    assert np.isfinite(P2).all()
    assert np.isfinite(tau2).all()


def test_orbElem2vec_vector_eccentric_anomaly_scalar_mu_round_trip():
    a, e, O, I, w = _sample_orbital_elements()
    mu = 2.2
    E = np.array([0.15, 1.7, 4.9])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    assert np.allclose(a2, a, rtol=1e-10, atol=1e-10)
    assert np.allclose(e2, e, rtol=1e-10, atol=1e-10)
    assert np.all(_angle_diff(E2, np.mod(E, 2 * np.pi)) < 1e-10)
    assert np.all(_angle_diff(O2, O) < 1e-10)
    assert np.all(_angle_diff(I2, I) < 1e-10)
    assert np.all(_angle_diff(w2, w) < 1e-10)
    assert np.all(np.isfinite(P2))
    assert np.all(np.isfinite(tau2))


def test_orbElem2vec_vector_mu_vector_round_trip():
    a, e, O, I, w = _sample_orbital_elements()
    mu = np.array([1.1, 2.0, 3.4])
    E = np.array([0.5, 1.1, 2.4])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    assert np.allclose(a2, a, rtol=1e-10, atol=1e-10)
    assert np.allclose(e2, e, rtol=1e-10, atol=1e-10)
    assert np.all(_angle_diff(E2, np.mod(E, 2 * np.pi)) < 1e-10)
    assert np.all(_angle_diff(O2, O) < 1e-10)
    assert np.all(_angle_diff(I2, I) < 1e-10)
    assert np.all(_angle_diff(w2, w) < 1e-10)
    assert np.all(P2 > 0)
    assert np.all(np.isfinite(tau2))


def test_orbElem2vec_accepts_AB_input():
    a, e, O, I, w = _sample_orbital_elements()
    mu = 1.3
    E = np.array([0.2, 2.0, 5.1])
    AB = calcAB(a, e, O, I, w)

    r1, v1 = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    r2, v2 = orbElem2vec(E, mu, AB=AB)

    assert np.allclose(r1, r2)
    assert np.allclose(v1, v2)


def test_orbElem2vec_returnAB_matches_calcAB():
    a, e, O, I, w = _sample_orbital_elements()
    mu = 1.9
    E = np.array([0.7, 1.9, 3.3])

    r, v, AB = orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)
    A_expected, B_expected = calcAB(a, e, O, I, w)

    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    assert isinstance(AB, tuple) and len(AB) == 2
    assert np.allclose(AB[0], A_expected)
    assert np.allclose(AB[1], B_expected)


def test_orbElem2vec_scalar_E_and_scalar_mu_shapes():
    a, e, O, I, w = _sample_orbital_elements()
    mu = 1.0
    E = 0.25

    r, v = orbElem2vec(E, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]))

    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.isfinite(r).all()
    assert np.isfinite(v).all()


def test_vec2orbElem_accepts_transposed_input():
    a, e, O, I, w = _sample_orbital_elements()
    mu = 2.6
    E = np.array([0.3, 1.0, 2.5])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r.T, v.T, mu)

    assert np.allclose(a2, a, rtol=1e-10, atol=1e-10)
    assert np.allclose(e2, e, rtol=1e-10, atol=1e-10)
    assert np.all(_angle_diff(E2, np.mod(E, 2 * np.pi)) < 1e-10)
    assert np.all(_angle_diff(O2, O) < 1e-10)
    assert np.all(_angle_diff(I2, I) < 1e-10)
    assert np.all(_angle_diff(w2, w) < 1e-10)
    assert np.all(np.isfinite(P2))
    assert np.all(np.isfinite(tau2))


def test_vec2orbElem_accepts_flattened_input():
    a, e, O, I, w = _sample_orbital_elements()
    mu = 1.8
    E = np.array([0.8, 1.3, 4.2])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r.reshape(-1), v.reshape(-1), mu)

    assert np.allclose(a2, a, rtol=1e-10, atol=1e-10)
    assert np.allclose(e2, e, rtol=1e-10, atol=1e-10)
    assert np.all(_angle_diff(E2, np.mod(E, 2 * np.pi)) < 1e-10)
    assert np.all(_angle_diff(O2, O) < 1e-10)
    assert np.all(_angle_diff(I2, I) < 1e-10)
    assert np.all(_angle_diff(w2, w) < 1e-10)
    assert np.all(np.isfinite(P2))
    assert np.all(np.isfinite(tau2))


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError, match="You must supply either orbElem or AB inputs"):
        orbElem2vec(0.1, 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    a, e, O, I, w = _sample_orbital_elements()
    mu = 1.0
    E = 0.2
    AB = calcAB(a[0], e[0], O[0], I[0], w[0])

    with pytest.raises(AssertionError, match="You can only set orbElem or AB"):
        orbElem2vec(E, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]), AB=AB)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    a, e, O, I, w = _sample_orbital_elements()
    with pytest.raises(AssertionError, match="Scalar E input requires scalar mus input"):
        orbElem2vec(0.1, np.array([1.0, 2.0]), orbElem=(a[0], e[0], O[0], I[0], w[0]))


def test_orbElem2vec_vector_mu_must_match_E_size():
    a, e, O, I, w = _sample_orbital_elements()
    E = np.array([0.1, 0.2, 0.3])
    mu = np.array([1.0, 2.0])

    with pytest.raises(AssertionError, match="mus must be of the same size as E or scalar"):
        orbElem2vec(E, mu, orbElem=(a, e, O, I, w))


def test_vec2orbElem_rejects_non_multiple_of_three():
    rs = np.array([1.0, 2.0, 3.0, 4.0])
    vs = np.array([0.1, 0.2, 0.3, 0.4])

    with pytest.raises(AssertionError, match="rs and vs must be of the same size and contain 3n elements"):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_mus_size_mismatch():
    rs = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    mus = np.array([1.0, 2.0, 3.0])

    with pytest.raises(AssertionError, match="mus must be scalar or of size n"):
        vec2orbElem(rs, vs, mus)


def test_vec2orbElem_rejects_three_dimensional_input():
    rs = np.zeros((3, 1, 1))
    vs = np.zeros((3, 1, 1))

    with pytest.raises(AssertionError, match="rs cannot have more than two dimensions"):
        vec2orbElem(rs, vs, 1.0)


def test_orbElem2vec_and_vec2orbElem_preserve_closed_orbit_relationships():
    a, e, O, I, w = _sample_orbital_elements()
    mu = 4.0
    E = np.array([0.4, 1.8, 5.5])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    assert np.all(a2 > 0)
    assert np.all((e2 >= 0) & (e2 < 1))
    assert np.all((P2 > 0) & np.isfinite(P2))
    assert np.all(np.isfinite(tau2))
    assert np.all(_angle_diff(O2, O) < 1e-10)
    assert np.all(_angle_diff(I2, I) < 1e-10)
    assert np.all(_angle_diff(w2, w) < 1e-10)
    assert np.all(_angle_diff(E2, np.mod(E, 2 * np.pi)) < 1e-10)
