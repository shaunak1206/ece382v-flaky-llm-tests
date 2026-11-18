import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def _sample_elements(n=3):
    a = np.array([1.2, 2.5, 4.0])[:n]
    e = np.array([0.1, 0.25, 0.4])[:n]
    O = np.array([0.2, 1.1, 2.2])[:n]
    I = np.array([0.3, 0.9, 1.2])[:n]
    w = np.array([0.4, 0.7, 1.3])[:n]
    mu = np.array([1.0, 1.5, 2.0])[:n]
    E = np.array([0.5, 1.0, 2.0])[:n]
    return a, e, O, I, w, mu, E


def _make_state_vectors():
    a, e, O, I, w, mu, E = _sample_elements(3)
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    return a, e, O, I, w, mu, E, r, v


def test_orbElem2vec_scalar_inputs_return_shapes():
    a, e, O, I, w = 2.0, 0.2, 0.3, 0.4, 0.5
    mu = 1.0
    E = 0.7

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))

    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.isfinite(r).all()
    assert np.isfinite(v).all()


def test_orbElem2vec_vector_E_scalar_mu_returnAB():
    a, e, O, I, w = 2.0, 0.15, 0.5, 0.4, 0.3
    mu = 1.0
    E = np.array([0.2, 1.1, 2.4])

    r, v, (A, B) = orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)

    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)
    assert np.isfinite(r).all()
    assert np.isfinite(v).all()


def test_orbElem2vec_vector_E_vector_mu_returnAB_shapes():
    a, e, O, I, w, mu, E = _sample_elements(3)
    r, v, (A, B) = orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)

    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    assert A.shape == (3, 3)
    assert B.shape == (3, 3)


def test_orbElem2vec_raises_when_both_orbElem_and_AB_given():
    a, e, O, I, w, mu, E = _sample_elements(1)
    A, B = calcAB(a, e, O, I, w)

    with pytest.raises(AssertionError, match="You can only set orbElem or AB"):
        orbElem2vec(E[0], mu[0], orbElem=(a[0], e[0], O[0], I[0], w[0]), AB=(A, B))


def test_orbElem2vec_raises_when_neither_orbElem_nor_AB_given():
    with pytest.raises(AssertionError, match="You must supply either orbElem or AB inputs"):
        orbElem2vec(0.5, 1.0)


def test_orbElem2vec_rejects_vector_E_with_vector_mus_size_mismatch():
    a, e, O, I, w = 2.0, 0.15, 0.5, 0.4, 0.3
    E = np.array([0.2, 1.1, 2.4])
    mu = np.array([1.0, 2.0])

    with pytest.raises(AssertionError, match="mus must be of the same size as E or scalar"):
        orbElem2vec(E, mu, orbElem=(a, e, O, I, w))


def test_orbElem2vec_rejects_scalar_E_with_vector_mus():
    a, e, O, I, w = 2.0, 0.15, 0.5, 0.4, 0.3
    E = 0.5
    mu = np.array([1.0, 2.0])

    with pytest.raises(AssertionError, match="Scalar E input requires scalar mus input"):
        orbElem2vec(E, mu, orbElem=(a, e, O, I, w))


def test_orbElem2vec_roundtrip_scalar_case():
    a, e, O, I, w = 3.0, 0.27, 0.8, 0.6, 1.1
    mu = 1.4
    E = 1.2

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    out = vec2orbElem(r.ravel(), v.ravel(), mu)

    a2, e2, E2, O2, I2, w2, P2, tau2 = out

    assert np.isclose(a2[0], a, rtol=1e-9, atol=1e-9)
    assert np.isclose(e2[0], e, rtol=1e-9, atol=1e-9)
    assert np.isclose(E2[0], np.mod(E, 2 * np.pi), rtol=1e-9, atol=1e-9)
    assert np.isclose(I2[0], I, rtol=1e-9, atol=1e-9)
    assert np.isclose(P2[0], 2 * np.pi * np.sqrt(a**3 / mu), rtol=1e-9, atol=1e-9)
    assert np.isfinite(tau2[0])


def test_vec2orbElem_roundtrip_multiple_bodies_3xn_input():
    a, e, O, I, w, mu, E, r, v = _make_state_vectors()

    out = vec2orbElem(r, v, mu)
    a2, e2, E2, O2, I2, w2, P2, tau2 = out

    assert a2.shape == (3,)
    assert e2.shape == (3,)
    assert np.allclose(a2, a, rtol=1e-9, atol=1e-9)
    assert np.allclose(e2, e, rtol=1e-9, atol=1e-9)
    assert np.allclose(np.mod(E2, 2 * np.pi), np.mod(E, 2 * np.pi), rtol=1e-9, atol=1e-9)
    assert np.allclose(np.mod(O2, 2 * np.pi), np.mod(O, 2 * np.pi), rtol=1e-9, atol=1e-9)
    assert np.allclose(np.mod(I2, 2 * np.pi), np.mod(I, 2 * np.pi), rtol=1e-9, atol=1e-9)
    assert np.allclose(np.mod(w2, 2 * np.pi), np.mod(w, 2 * np.pi), rtol=1e-9, atol=1e-9)
    assert np.isfinite(P2).all()
    assert np.isfinite(tau2).all()


def test_vec2orbElem_accepts_nx3_input_and_matches_3xn():
    _, _, _, _, _, mu, _, r, v = _make_state_vectors()

    out1 = vec2orbElem(r, v, mu)
    out2 = vec2orbElem(r.T, v.T, mu)

    for arr1, arr2 in zip(out1, out2):
        assert np.allclose(arr1, arr2, rtol=1e-9, atol=1e-9)


def test_vec2orbElem_accepts_flattened_input():
    _, _, _, _, _, mu, _, r, v = _make_state_vectors()

    out1 = vec2orbElem(r, v, mu)
    out2 = vec2orbElem(r.ravel(), v.ravel(), mu)

    for arr1, arr2 in zip(out1, out2):
        assert np.allclose(arr1, arr2, rtol=1e-9, atol=1e-9)


def test_vec2orbElem_rejects_mismatched_rs_vs_sizes():
    rs = np.array([1.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0, 0.1])

    with pytest.raises(AssertionError, match="rs and vs must be of the same size"):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_mus_size_mismatch():
    rs = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    mus = np.array([1.0])

    with pytest.raises(AssertionError, match="mus must be scalar or of size n"):
        vec2orbElem(rs, vs, mus)


def test_vec2orbElem_rejects_three_dimensional_inputs():
    rs = np.zeros((1, 1, 3))
    vs = np.zeros((1, 1, 3))

    with pytest.raises(AssertionError, match="rs cannot have more than two dimensions"):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_output_values_are_finite_for_valid_input():
    _, _, _, _, _, mu, _, r, v = _make_state_vectors()

    out = vec2orbElem(r, v, mu)

    for arr in out:
        assert np.isfinite(arr).all()


def test_orbElem2vec_uses_AB_input_equivalently_to_orbElem():
    a, e, O, I, w = 2.0, 0.22, 0.9, 0.4, 1.2
    mu = 1.1
    E = np.array([0.2, 1.0])

    r1, v1 = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    A, B = calcAB(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    r2, v2 = orbElem2vec(E, mu, AB=(A, B))

    assert np.allclose(r1, r2, rtol=1e-12, atol=1e-12)
    assert np.allclose(v1, v2, rtol=1e-12, atol=1e-12)
