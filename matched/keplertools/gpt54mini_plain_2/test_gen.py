import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def wrap_angle(x):
    return np.mod(x, 2.0 * np.pi)


def angle_close(a, b, atol=1e-10):
    diff = np.mod(a - b + np.pi, 2.0 * np.pi) - np.pi
    return np.allclose(diff, 0.0, atol=atol, rtol=0.0)


def sample_orbital_elements(n=1):
    # Non-degenerate, bound orbits with angles away from singularities.
    a = np.linspace(1.2, 2.0, n)
    e = np.linspace(0.1, 0.4, n)
    O = np.linspace(0.3, 1.1, n)
    I = np.linspace(0.2, 0.9, n)
    w = np.linspace(0.4, 1.4, n)
    E = np.linspace(0.2, 2.4, n)
    mu = 1.7 if n == 1 else np.linspace(1.7, 2.5, n)
    return a, e, O, I, w, E, mu


def test_orbElem2vec_single_body_scalar_inputs_round_trip():
    a, e, O, I, w, E, mu = sample_orbital_elements(1)
    r, v = orbElem2vec(E[0], mu, orbElem=(a[0], e[0], O[0], I[0], w[0]))
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)

    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    assert a2.shape == (1,)
    assert e2.shape == (1,)
    assert E2.shape == (1,)
    assert O2.shape == (1,)
    assert I2.shape == (1,)
    assert w2.shape == (1,)
    assert P2.shape == (1,)
    assert tau2.shape == (1,)

    assert np.allclose(a2[0], a[0], atol=1e-10, rtol=1e-10)
    assert np.allclose(e2[0], e[0], atol=1e-10, rtol=1e-10)
    assert angle_close(E2[0], E[0])
    assert angle_close(O2[0], O[0])
    assert angle_close(I2[0], I[0])
    assert angle_close(w2[0], w[0])


def test_orbElem2vec_vector_eccentric_anomaly_single_body_round_trip():
    a, e, O, I, w, E, mu = sample_orbital_elements(5)
    r, v = orbElem2vec(E, mu[0], orbElem=(a[0], e[0], O[0], I[0], w[0]))
    assert r.shape == (3, 5)
    assert v.shape == (3, 5)

    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu[0])
    assert np.allclose(a2, a[0], atol=1e-10, rtol=1e-10)
    assert np.allclose(e2, e[0], atol=1e-10, rtol=1e-10)
    assert np.allclose(I2, I[0], atol=1e-10, rtol=1e-10)
    assert angle_close(E2, E)
    assert angle_close(O2, O[0])
    assert angle_close(w2, w[0])
    assert np.all(P2 > 0)
    assert np.all(np.isfinite(tau2))


def test_orbElem2vec_vectorized_multi_body_round_trip():
    a, e, O, I, w, E, mu = sample_orbital_elements(4)
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 4)
    assert v.shape == (3, 4)

    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    assert np.allclose(a2, a, atol=1e-10, rtol=1e-10)
    assert np.allclose(e2, e, atol=1e-10, rtol=1e-10)
    assert np.allclose(I2, I, atol=1e-10, rtol=1e-10)
    assert angle_close(E2, E)
    assert angle_close(O2, O)
    assert angle_close(w2, w)
    assert np.all(P2 > 0)
    assert np.all(np.isfinite(tau2))


def test_orbElem2vec_accepts_ab_input_equivalent_to_orbelem():
    a, e, O, I, w, E, mu = sample_orbital_elements(3)
    AB = calcAB(a, e, O, I, w)

    r1, v1 = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    r2, v2 = orbElem2vec(E, mu, AB=AB)

    assert np.allclose(r1, r2, atol=1e-12, rtol=1e-12)
    assert np.allclose(v1, v2, atol=1e-12, rtol=1e-12)


def test_orbElem2vec_returnab_matches_calcab():
    a, e, O, I, w, E, mu = sample_orbital_elements(2)
    r, v, AB = orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)
    A1, B1 = AB
    A2, B2 = calcAB(a, e, O, I, w)

    assert r.shape == (3, 2)
    assert v.shape == (3, 2)
    assert np.allclose(A1, A2, atol=1e-12, rtol=1e-12)
    assert np.allclose(B1, B2, atol=1e-12, rtol=1e-12)


def test_orbElem2vec_scalar_eccentric_anomaly_with_vector_mu_raises():
    a, e, O, I, w, E, mu = sample_orbital_elements(3)
    with pytest.raises(AssertionError):
        orbElem2vec(E[0], mu, orbElem=(a, e, O, I, w))


def test_orbElem2vec_vector_eccentric_anomaly_with_scalar_mu_and_vector_orbelem_raises():
    a, e, O, I, w, E, mu = sample_orbital_elements(3)
    with pytest.raises(AssertionError):
        orbElem2vec(E, mu[0], orbElem=(a, e, O, I, w))


def test_orbElem2vec_rejects_both_orbelem_and_ab():
    a, e, O, I, w, E, mu = sample_orbital_elements(1)
    AB = calcAB(a, e, O, I, w)
    with pytest.raises(AssertionError):
        orbElem2vec(E[0], mu, orbElem=(a[0], e[0], O[0], I[0], w[0]), AB=AB)


def test_orbElem2vec_rejects_missing_orbelem_and_ab():
    _, _, _, _, _, E, mu = sample_orbital_elements(1)
    with pytest.raises(AssertionError):
        orbElem2vec(E[0], mu)


def test_orbElem2vec_rejects_mismatched_mu_vector_length():
    a, e, O, I, w, E, mu = sample_orbital_elements(3)
    bad_mu = np.array([1.0, 2.0])
    with pytest.raises(AssertionError):
        orbElem2vec(E, bad_mu, orbElem=(a, e, O, I, w))


def test_orbElem2vec_rejects_ab_with_incorrect_shape_for_scalar_case():
    a, e, O, I, w, E, mu = sample_orbital_elements(1)
    AB = calcAB(a, e, O, I, w)
    badA = np.hstack([AB[0], AB[0]])
    badB = np.hstack([AB[1], AB[1]])
    with pytest.raises(AssertionError):
        orbElem2vec(E[0], mu, AB=(badA, badB))


def test_vec2orbElem_accepts_row_major_input_shapes():
    a, e, O, I, w, E, mu = sample_orbital_elements(3)
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r.T, v.T, mu)

    assert np.allclose(a2, a, atol=1e-10, rtol=1e-10)
    assert np.allclose(e2, e, atol=1e-10, rtol=1e-10)
    assert angle_close(E2, E)
    assert angle_close(O2, O)
    assert angle_close(I2, I)
    assert angle_close(w2, w)
    assert np.all(P2 > 0)
    assert np.all(np.isfinite(tau2))


def test_vec2orbElem_accepts_flattened_vector_inputs():
    a, e, O, I, w, E, mu = sample_orbital_elements(2)
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r.reshape(-1), v.reshape(-1), mu)

    assert np.allclose(a2, a, atol=1e-10, rtol=1e-10)
    assert np.allclose(e2, e, atol=1e-10, rtol=1e-10)
    assert angle_close(E2, E)
    assert angle_close(O2, O)
    assert angle_close(I2, I)
    assert angle_close(w2, w)
    assert np.all(P2 > 0)
    assert np.all(np.isfinite(tau2))


def test_vec2orbElem_rejects_mismatched_rs_vs_sizes():
    rs = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0])
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_nonmultiple_of_three_elements():
    rs = np.array([1.0, 0.0, 0.0, 2.0])
    vs = np.array([0.0, 1.0, 0.0, 0.0])
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_three_dimensional_input():
    rs = np.zeros((3, 1, 1))
    vs = np.zeros((3, 1, 1))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_wrong_mus_length_for_multiple_bodies():
    a, e, O, I, w, E, mu = sample_orbital_elements(3)
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    with pytest.raises(AssertionError):
        vec2orbElem(r, v, np.array([1.0, 2.0]))
