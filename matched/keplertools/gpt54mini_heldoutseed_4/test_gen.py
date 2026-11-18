import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def _angle_close(a, b, tol=1e-10):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = np.mod(a - b + np.pi, 2 * np.pi) - np.pi
    return np.all(np.abs(diff) <= tol)


def _make_non_degenerate_elements():
    a = np.array([1.7, 2.3])
    e = np.array([0.21, 0.43])
    O = np.array([0.8, 1.4])
    I = np.array([0.5, 1.1])
    w = np.array([1.2, 0.3])
    E = np.array([0.4, 2.1])
    mu = np.array([1.0, 0.8])
    return a, e, O, I, w, E, mu


def test_calcAB_shapes_and_basic_values():
    a, e, O, I, w, _, _ = _make_non_degenerate_elements()
    A, B = calcAB(a, e, O, I, w)
    assert A.shape == (3, 2)
    assert B.shape == (3, 2)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))


def test_orbElem2vec_scalar_single_body_returns_expected_shapes():
    a = 1.9
    e = 0.27
    O = 0.7
    I = 0.9
    w = 1.1
    E = 0.6
    mu = 1.0
    r, v, AB = orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)

    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert isinstance(AB, tuple) and len(AB) == 2
    assert AB[0].shape == (3, 1)
    assert AB[1].shape == (3, 1)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(v))


def test_orbElem2vec_scalar_input_with_AB_matches_orbElem_path():
    a = 2.1
    e = 0.31
    O = 0.4
    I = 1.0
    w = 0.2
    E = 1.3
    mu = 1.5

    r1, v1 = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    A, B = calcAB(np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    r2, v2 = orbElem2vec(E, mu, AB=(A, B))

    np.testing.assert_allclose(r1, r2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(v1, v2, rtol=0, atol=1e-12)


def test_orbElem2vec_vector_scalar_mu_returns_expected_shapes():
    a = np.array([1.6, 2.2, 3.1])
    e = np.array([0.1, 0.25, 0.4])
    O = np.array([0.2, 0.7, 1.3])
    I = np.array([0.4, 0.8, 1.1])
    w = np.array([1.0, 0.5, 0.9])
    E = np.array([0.3, 1.2, 2.0])
    mu = 1.0

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(v))


def test_orbElem2vec_vector_vector_mu_roundtrip_with_vec2orbElem():
    a, e, O, I, w, E, mu = _make_non_degenerate_elements()

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    np.testing.assert_allclose(a2, a, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(e2, e, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(E2, np.mod(E, 2 * np.pi), rtol=1e-10, atol=1e-10)
    assert _angle_close(O2, O)
    assert _angle_close(I2, I)
    assert _angle_close(w2, w)
    assert np.all(P2 > 0)
    assert np.all(np.isfinite(tau2))


def test_orbElem2vec_accepts_vector_E_and_scalar_mu_with_AB():
    a = np.array([1.7, 2.4])
    e = np.array([0.15, 0.35])
    O = np.array([0.1, 1.0])
    I = np.array([0.6, 1.2])
    w = np.array([0.9, 0.2])
    E = np.array([0.25, 1.8])
    mu = 1.2
    A, B = calcAB(a, e, O, I, w)

    r, v = orbElem2vec(E, mu, AB=(A, B))
    assert r.shape == (3, 2)
    assert v.shape == (3, 2)
    np.testing.assert_allclose(np.linalg.norm(r, axis=0) > 0, True)


def test_orbElem2vec_rejects_missing_orbital_inputs():
    with pytest.raises(AssertionError, match="You must supply either orbElem or AB inputs"):
        orbElem2vec(0.1, 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    a = np.array([1.0])
    e = np.array([0.2])
    O = np.array([0.3])
    I = np.array([0.4])
    w = np.array([0.5])
    A, B = calcAB(a, e, O, I, w)
    with pytest.raises(AssertionError, match="You can only set orbElem or AB"):
        orbElem2vec(0.1, 1.0, orbElem=(a, e, O, I, w), AB=(A, B))


def test_orbElem2vec_rejects_scalar_E_with_vector_mu():
    a = np.array([1.0])
    e = np.array([0.2])
    O = np.array([0.3])
    I = np.array([0.4])
    w = np.array([0.5])
    with pytest.raises(AssertionError, match="Scalar E input requires scalar mus input"):
        orbElem2vec(0.1, np.array([1.0]), orbElem=(a, e, O, I, w))


def test_orbElem2vec_rejects_mismatched_vector_sizes():
    a = np.array([1.0, 1.2])
    e = np.array([0.2, 0.3])
    O = np.array([0.3, 0.4])
    I = np.array([0.4, 0.5])
    w = np.array([0.5, 0.6])
    E = np.array([0.1, 0.2])
    with pytest.raises(AssertionError, match="mus must be of the same size as E or scalar"):
        orbElem2vec(E, np.array([1.0, 2.0, 3.0]), orbElem=(a, e, O, I, w))


def test_vec2orbElem_accepts_3xn_input_and_transpose_equivalence():
    a, e, O, I, w, E, mu = _make_non_degenerate_elements()
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))

    a1, e1, E1, O1, I1, w1, P1, tau1 = vec2orbElem(r, v, mu)
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r.T, v.T, mu)

    np.testing.assert_allclose(a1, a2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(e1, e2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(E1, E2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(P1, P2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tau1, tau2, rtol=0, atol=1e-12)
    assert _angle_close(O1, O2)
    assert _angle_close(I1, I2)
    assert _angle_close(w1, w2)


def test_vec2orbElem_accepts_flattened_1d_input():
    a, e, O, I, w, E, mu = _make_non_degenerate_elements()
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r.ravel(), v.ravel(), mu)

    np.testing.assert_allclose(a2, a, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(e2, e, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(E2, np.mod(E, 2 * np.pi), rtol=1e-10, atol=1e-10)
    assert _angle_close(O2, O)
    assert _angle_close(I2, I)
    assert _angle_close(w2, w)


def test_vec2orbElem_rejects_mismatched_r_and_v_sizes():
    r = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0, 0.0])
    with pytest.raises(AssertionError, match="rs and vs must be of the same size"):
        vec2orbElem(r, v, 1.0)


def test_vec2orbElem_rejects_mus_size_mismatch():
    r = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0, 0.0, 0.5, 0.0])
    with pytest.raises(AssertionError, match="mus must be scalar or of size n"):
        vec2orbElem(r, v, np.array([1.0, 2.0, 3.0]))


def test_vec2orbElem_roundtrip_preserves_periods_and_times():
    a, e, O, I, w, E, mu = _make_non_degenerate_elements()
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    expected_P = 2 * np.pi * np.sqrt(a**3 / mu)
    np.testing.assert_allclose(P2, expected_P, rtol=1e-10, atol=1e-10)
    assert np.all(np.isfinite(tau2))
    assert np.all(tau2.shape == a.shape)
    np.testing.assert_allclose(a2, a, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(e2, e, rtol=1e-10, atol=1e-10)


def test_orbElem2vec_raises_for_inconsistent_AB_shape():
    E = np.array([0.1, 0.2])
    mu = 1.0
    A = np.ones((3, 1))
    B = np.ones((3, 2))
    with pytest.raises(AssertionError):
        orbElem2vec(E, mu, AB=(A, B))


def test_vec2orbElem_returns_finite_outputs_for_valid_input():
    a = np.array([2.0])
    e = np.array([0.2])
    O = np.array([0.3])
    I = np.array([0.9])
    w = np.array([1.1])
    E = np.array([0.7])
    mu = 1.0
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    out = vec2orbElem(r, v, mu)

    for arr in out:
        assert np.all(np.isfinite(arr))
