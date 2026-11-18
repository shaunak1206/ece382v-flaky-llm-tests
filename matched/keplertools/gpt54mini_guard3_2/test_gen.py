import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem


def _angle_diff(a, b):
    """Return wrapped angle difference in [-pi, pi]."""
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def _assert_angle_close(actual, expected, tol=1e-10):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    diff = _angle_diff(actual, expected)
    assert np.all(np.abs(diff) <= tol), f"max angle diff={np.max(np.abs(diff))}"


def _sample_orbital_elements():
    a = np.array([1.7, 2.4])
    e = np.array([0.2, 0.55])
    O = np.array([0.7, 2.1])
    I = np.array([0.4, 1.1])
    w = np.array([1.2, 0.3])
    return a, e, O, I, w


def test_calcAB_shapes_and_values():
    a, e, O, I, w = _sample_orbital_elements()
    A, B = calcAB(a, e, O, I, w)
    assert A.shape == (3, 2)
    assert B.shape == (3, 2)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))


def test_calcAB_size_mismatch_raises():
    a, e, O, I, w = _sample_orbital_elements()
    with pytest.raises(AssertionError):
        calcAB(a[:1], e, O, I, w)


def test_orbElem2vec_scalar_roundtrip_through_vec2orbElem():
    mu = 1.0
    a = 1.8
    e = 0.3
    O = 0.9
    I = 0.6
    w = 1.1
    E = 2.0

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)

    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    assert np.allclose(a2, [a], rtol=1e-10, atol=1e-10)
    assert np.allclose(e2, [e], rtol=1e-10, atol=1e-10)
    _assert_angle_close(E2, [E])
    _assert_angle_close(O2, [O])
    _assert_angle_close(I2, [I])
    _assert_angle_close(w2, [w])
    assert np.all(P2 > 0)
    assert np.all(np.isfinite(tau2))


def test_orbElem2vec_vector_mu_scalar_roundtrip():
    mu = 1.0
    a, e, O, I, w = _sample_orbital_elements()
    E = np.array([0.5, 1.9])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 2)
    assert v.shape == (3, 2)

    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    assert np.allclose(a2, a, rtol=1e-10, atol=1e-10)
    assert np.allclose(e2, e, rtol=1e-10, atol=1e-10)
    _assert_angle_close(E2, E)
    _assert_angle_close(O2, O)
    _assert_angle_close(I2, I)
    _assert_angle_close(w2, w)
    assert np.all(P2 > 0)
    assert np.all(np.isfinite(tau2))


def test_orbElem2vec_vector_mu_vector_roundtrip():
    a, e, O, I, w = _sample_orbital_elements()
    E = np.array([0.5, 1.9])
    mus = np.array([1.0, 1.3])

    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mus)
    assert np.allclose(a2, a, rtol=1e-10, atol=1e-10)
    assert np.allclose(e2, e, rtol=1e-10, atol=1e-10)
    _assert_angle_close(E2, E)
    _assert_angle_close(O2, O)
    _assert_angle_close(I2, I)
    _assert_angle_close(w2, w)
    assert np.all(P2 > 0)
    assert np.all(np.isfinite(tau2))


def test_orbElem2vec_accepts_AB_input():
    a, e, O, I, w = _sample_orbital_elements()
    E = np.array([0.3, 1.2])
    mu = 1.0
    A, B = calcAB(a, e, O, I, w)

    r1, v1 = orbElem2vec(E, mu, AB=(A, B))
    r2, v2 = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert np.allclose(r1, r2, rtol=1e-12, atol=1e-12)
    assert np.allclose(v1, v2, rtol=1e-12, atol=1e-12)


def test_orbElem2vec_returnAB_matches_calcAB():
    a, e, O, I, w = _sample_orbital_elements()
    E = np.array([0.2, 1.4])
    mu = 1.0

    r, v, (A, B) = orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)
    A2, B2 = calcAB(a, e, O, I, w)
    assert r.shape == (3, 2)
    assert v.shape == (3, 2)
    assert np.allclose(A, A2, rtol=1e-12, atol=1e-12)
    assert np.allclose(B, B2, rtol=1e-12, atol=1e-12)


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        orbElem2vec(0.1, 1.0)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    a, e, O, I, w = _sample_orbital_elements()
    with pytest.raises(AssertionError):
        orbElem2vec(0.1, np.array([1.0, 1.0]), orbElem=(a[0], e[0], O[0], I[0], w[0]))


def test_orbElem2vec_vector_mu_size_mismatch_raises():
    a, e, O, I, w = _sample_orbital_elements()
    E = np.array([0.1, 0.2])
    mus = np.array([1.0, 1.1, 1.2])
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, orbElem=(a, e, O, I, w))


def test_orbElem2vec_both_orbElem_and_AB_raises():
    a, e, O, I, w = _sample_orbital_elements()
    E = np.array([0.1, 0.2])
    mu = 1.0
    A, B = calcAB(a, e, O, I, w)
    with pytest.raises(AssertionError):
        orbElem2vec(E, mu, orbElem=(a, e, O, I, w), AB=(A, B))


def test_vec2orbElem_rejects_dimension_not_multiple_of_three():
    rs = np.array([1.0, 2.0, 3.0, 4.0])
    vs = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_more_than_two_dimensions():
    rs = np.ones((3, 1, 1))
    vs = np.ones((3, 1, 1))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_bad_mus_length():
    rs = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0, 0.0, 0.5, 0.0])
    mus = np.array([1.0, 1.1, 1.2])
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, mus)


def test_vec2orbElem_transposed_input_gives_same_result():
    a, e, O, I, w = _sample_orbital_elements()
    E = np.array([0.4, 1.6])
    mu = 1.0
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))

    r_t = r.T
    v_t = v.T
    out1 = vec2orbElem(r, v, mu)
    out2 = vec2orbElem(r_t, v_t, mu)

    for arr1, arr2 in zip(out1, out2):
        assert np.allclose(arr1, arr2, rtol=1e-10, atol=1e-10)


def test_vec2orbElem_scalar_mu_and_vector_inputs_preserve_finite_outputs():
    a, e, O, I, w = _sample_orbital_elements()
    E = np.array([0.7, 2.2])
    mu = 1.0
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    out = vec2orbElem(r, v, mu)
    for arr in out:
        assert np.all(np.isfinite(arr))
