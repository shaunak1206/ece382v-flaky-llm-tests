import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def angle_close(a, b, atol=1e-10):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return np.allclose(diff, 0.0, atol=atol, rtol=0)


def sample_orbital_elements(n=1):
    a = np.array([2.3, 1.7, 3.1][:n], dtype=float)
    e = np.array([0.2, 0.35, 0.1][:n], dtype=float)
    O = np.array([0.4, 1.1, 2.2][:n], dtype=float)
    I = np.array([0.7, 1.2, 0.3][:n], dtype=float)
    w = np.array([1.0, 0.2, 2.0][:n], dtype=float)
    return a, e, O, I, w


def test_orbElem2vec_scalar_orbElem_roundtrip_single_body():
    mu = 1.0
    E = 1.234
    orbElem = sample_orbital_elements(1)
    r, v = orbElem2vec(E, mu, orbElem=orbElem)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)

    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    assert np.allclose(a2, orbElem[0], atol=1e-10)
    assert np.allclose(e2, orbElem[1], atol=1e-10)
    assert angle_close(E2, np.array([E]) % (2 * np.pi))
    assert angle_close(O2, orbElem[2])
    assert angle_close(I2, orbElem[3])
    assert angle_close(w2, orbElem[4])
    assert np.all(P2 > 0)
    assert np.isfinite(tau2).all()


def test_orbElem2vec_AB_roundtrip_matches_orbElem():
    mu = 1.0
    E = 2.2
    orbElem = sample_orbital_elements(1)
    A, B = calcAB(*orbElem)
    r1, v1 = orbElem2vec(E, mu, orbElem=orbElem)
    r2, v2 = orbElem2vec(E, mu, AB=(A, B))
    assert np.allclose(r1, r2, atol=1e-12)
    assert np.allclose(v1, v2, atol=1e-12)


def test_orbElem2vec_vector_mus_and_vector_elements():
    E = np.array([0.1, 1.1, 2.3], dtype=float)
    mus = np.array([1.0, 1.5, 2.0], dtype=float)
    orbElem = sample_orbital_elements(3)
    r, v = orbElem2vec(E, mus, orbElem=orbElem)
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(v))


def test_orbElem2vec_scalar_E_vector_mu_returns_multiple_bodies():
    E = 1.0
    mus = np.array([1.0, 1.5, 2.0], dtype=float)
    orbElem = sample_orbital_elements(3)
    r, v = orbElem2vec(E, mus, orbElem=orbElem)
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(v))


def test_orbElem2vec_accepts_flat_AB_input_from_calcAB():
    E = np.array([0.8, 1.4], dtype=float)
    mu = 1.0
    orbElem = sample_orbital_elements(2)
    A, B = calcAB(*orbElem)
    r, v = orbElem2vec(E, mu, AB=(A, B))
    assert r.shape == (3, 2)
    assert v.shape == (3, 2)


def test_vec2orbElem_accepts_3xn_input():
    mu = 1.0
    E = np.array([0.2, 1.3, 2.1], dtype=float)
    orbElem = sample_orbital_elements(3)
    r, v = orbElem2vec(E, mu, orbElem=orbElem)
    out = vec2orbElem(r, v, mu)
    assert len(out) == 8
    a, e, E2, O, I, w, P, tau = out
    assert a.shape == (3,)
    assert e.shape == (3,)
    assert angle_close(E2, E % (2 * np.pi))


def test_vec2orbElem_accepts_nx3_input_and_transposes():
    mu = 1.0
    E = np.array([0.25, 1.15], dtype=float)
    orbElem = sample_orbital_elements(2)
    r, v = orbElem2vec(E, mu, orbElem=orbElem)
    rT = r.T
    vT = v.T
    out = vec2orbElem(rT, vT, mu)
    a, e, E2, O, I, w, P, tau = out
    assert a.shape == (2,)
    assert np.allclose(a, orbElem[0], atol=1e-10)
    assert np.allclose(e, orbElem[1], atol=1e-10)
    assert angle_close(E2, E)


def test_vec2orbElem_scalar_mu_roundtrip_vectorized():
    mu = 1.0
    E = np.array([0.4, 0.9, 2.0], dtype=float)
    orbElem = sample_orbital_elements(3)
    r, v = orbElem2vec(E, mu, orbElem=orbElem)
    a, e, E2, O, I, w, P, tau = vec2orbElem(r, v, mu)
    assert np.allclose(a, orbElem[0], atol=1e-10)
    assert np.allclose(e, orbElem[1], atol=1e-10)
    assert angle_close(E2, E)


def test_orbElem2vec_raises_when_missing_orbElem_and_AB():
    with pytest.raises(AssertionError, match="You must supply either orbElem or AB"):
        orbElem2vec(1.0, 1.0)


def test_orbElem2vec_raises_when_both_orbElem_and_AB_given():
    orbElem = sample_orbital_elements(1)
    A, B = calcAB(*orbElem)
    with pytest.raises(AssertionError, match="You can only set orbElem or AB"):
        orbElem2vec(1.0, 1.0, orbElem=orbElem, AB=(A, B))


def test_orbElem2vec_raises_for_scalar_E_with_vector_mu():
    orbElem = sample_orbital_elements(3)
    with pytest.raises(AssertionError, match="Scalar E input requires scalar mus input"):
        orbElem2vec(1.0, np.array([1.0, 2.0, 3.0]), orbElem=orbElem)


def test_orbElem2vec_raises_for_size_mismatch_between_E_and_mu():
    orbElem = sample_orbital_elements(3)
    with pytest.raises(AssertionError, match="mus must be of the same size as E or scalar"):
        orbElem2vec(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0]), orbElem=orbElem)


def test_vec2orbElem_raises_for_nonmultiple_of_three():
    rs = np.array([1.0, 0.0, 0.0, 1.0])
    vs = np.array([0.0, 1.0, 0.0, 0.0])
    with pytest.raises(AssertionError, match="3n elements"):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_raises_for_mus_size_mismatch():
    rs = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
    vs = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0], dtype=float)
    with pytest.raises(AssertionError, match="mus must be scalar or of size n"):
        vec2orbElem(rs, vs, np.array([1.0, 2.0, 3.0]))


def test_vec2orbElem_raises_for_bad_dimensions():
    rs = np.zeros((2, 2, 2), dtype=float)
    vs = np.zeros((2, 2, 2), dtype=float)
    with pytest.raises(AssertionError, match="cannot have more than two dimensions"):
        vec2orbElem(rs, vs, 1.0)
