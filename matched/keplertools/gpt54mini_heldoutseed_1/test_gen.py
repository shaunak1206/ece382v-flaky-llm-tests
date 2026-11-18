import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem


def _angle_diff(a, b):
    """Smallest absolute difference between angles modulo 2pi."""
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))


def _sample_orbital_elements(n=1):
    a = np.linspace(1.2, 2.3, n)
    e = np.linspace(0.1, 0.4, n)
    O = np.linspace(0.2, 1.4, n)
    I = np.linspace(0.3, 1.0, n)
    w = np.linspace(0.5, 1.7, n)
    return a, e, O, I, w


def test_orbElem2vec_scalar_mu_scalar_E_returnAB_shapes():
    a, e, O, I, w = _sample_orbital_elements(1)
    E = 0.8
    mu = 1.0

    r, v, (A, B) = orbElem2vec(E, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]), returnAB=True)

    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)
    assert np.isfinite(r).all()
    assert np.isfinite(v).all()


def test_orbElem2vec_vector_E_scalar_mu_shapes():
    a, e, O, I, w = _sample_orbital_elements(1)
    E = np.array([0.1, 1.2, 2.0, 4.0])
    mu = 1.0

    r, v = orbElem2vec(E, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]))

    assert r.shape == (3, E.size)
    assert v.shape == (3, E.size)


def test_orbElem2vec_vector_E_vector_mu_shapes():
    n = 4
    a, e, O, I, w = _sample_orbital_elements(n)
    E = np.array([0.1, 1.2, 2.0, 4.0])
    mu = np.linspace(0.8, 1.4, n)

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))

    assert r.shape == (3, n)
    assert v.shape == (3, n)


def test_orbElem2vec_raises_when_missing_orbElem_and_AB():
    with pytest.raises(AssertionError, match="You must supply either orbElem or AB inputs"):
        orbElem2vec(0.5, 1.0)


def test_orbElem2vec_raises_for_scalar_E_with_vector_mu():
    a, e, O, I, w = _sample_orbital_elements(1)
    A, B = calcAB(a, e, O, I, w)
    with pytest.raises(AssertionError, match="Scalar E input requires scalar mus input"):
        orbElem2vec(0.5, np.array([1.0, 2.0]), AB=(A, B))


def test_orbElem2vec_raises_for_both_orbElem_and_AB():
    a, e, O, I, w = _sample_orbital_elements(1)
    A, B = calcAB(a, e, O, I, w)
    with pytest.raises(AssertionError, match="You can only set orbElem or AB"):
        orbElem2vec(0.5, 1.0, orbElem=(a[0], e[0], O[0], I[0], w[0]), AB=(A, B))


def test_orbElem2vec_raises_for_bad_A_B_shape_with_scalar_E():
    E = 0.5
    mu = 1.0
    A = np.zeros((3, 2))
    B = np.zeros((3, 2))
    with pytest.raises(AssertionError, match="A and B must be 3x1"):
        orbElem2vec(E, mu, AB=(A, B))


def test_vec2orbElem_raises_for_mismatched_rs_vs_sizes():
    rs = np.array([1.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    with pytest.raises(AssertionError, match="same size and contain 3n elements"):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_raises_for_bad_mus_size():
    rs = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    mus = np.array([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError, match="mus must be scalar or of size n"):
        vec2orbElem(rs, vs, mus)


def test_vec2orbElem_roundtrip_single_orbit_1d_inputs():
    a, e, O, I, w = _sample_orbital_elements(1)
    mu = 1.0
    E0 = 0.9

    r, v = orbElem2vec(E0, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r.ravel(), v.ravel(), mu)

    assert np.allclose(a2, a[0], rtol=1e-10, atol=1e-10)
    assert np.allclose(e2, e[0], rtol=1e-10, atol=1e-10)
    assert _angle_diff(E2, E0) < 1e-10
    assert _angle_diff(O2, O[0]) < 1e-10
    assert _angle_diff(I2, I[0]) < 1e-10
    assert _angle_diff(w2, w[0]) < 1e-10
    assert np.isfinite(P2).all()
    assert np.isfinite(tau2).all()


def test_vec2orbElem_roundtrip_multiple_orbits_vector_inputs():
    n = 4
    a, e, O, I, w = _sample_orbital_elements(n)
    mu = np.linspace(0.9, 1.5, n)
    E0 = np.array([0.2, 1.0, 2.1, 4.2])

    r, v = orbElem2vec(E0, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    assert a2.shape == (n,)
    assert e2.shape == (n,)
    assert E2.shape == (n,)
    assert O2.shape == (n,)
    assert I2.shape == (n,)
    assert w2.shape == (n,)
    assert np.allclose(a2, a, rtol=1e-10, atol=1e-10)
    assert np.allclose(e2, e, rtol=1e-10, atol=1e-10)
    assert np.all(_angle_diff(E2, E0) < 1e-10)
    assert np.all(_angle_diff(O2, O) < 1e-10)
    assert np.all(_angle_diff(I2, I) < 1e-10)
    assert np.all(_angle_diff(w2, w) < 1e-10)
    assert np.isfinite(P2).all()
    assert np.isfinite(tau2).all()


def test_vec2orbElem_accepts_3xn_and_nx3_inputs_equivalently():
    a, e, O, I, w = _sample_orbital_elements(2)
    mu = np.array([1.0, 1.4])
    E0 = np.array([0.7, 2.4])

    r, v = orbElem2vec(E0, mu, orbElem=(a, e, O, I, w))
    rT = r.T
    vT = v.T

    out1 = vec2orbElem(r, v, mu)
    out2 = vec2orbElem(rT, vT, mu)

    for x1, x2 in zip(out1, out2):
        assert np.allclose(x1, x2, rtol=1e-12, atol=1e-12)


def test_vec2orbElem_returns_same_orbit_for_transposed_1d_stack():
    a, e, O, I, w = _sample_orbital_elements(1)
    mu = 1.2
    E0 = 1.1
    r, v = orbElem2vec(E0, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]))

    out1 = vec2orbElem(r.ravel(), v.ravel(), mu)
    out2 = vec2orbElem(r.reshape(3, 1), v.reshape(3, 1), mu)

    for x1, x2 in zip(out1, out2):
        assert np.allclose(x1, x2, rtol=1e-12, atol=1e-12)


def test_orbElem2vec_and_vec2orbElem_preserve_mu_scalar_and_vector_consistency():
    a, e, O, I, w = _sample_orbital_elements(3)
    mu_scalar = 1.0
    E0 = np.array([0.2, 1.0, 2.7])

    r1, v1 = orbElem2vec(E0, mu_scalar, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, _, _ = vec2orbElem(r1, v1, mu_scalar)

    assert np.allclose(a2, a, rtol=1e-10, atol=1e-10)
    assert np.allclose(e2, e, rtol=1e-10, atol=1e-10)
    assert np.all(_angle_diff(E2, E0) < 1e-10)
    assert np.all(_angle_diff(O2, O) < 1e-10)
    assert np.all(_angle_diff(I2, I) < 1e-10)
    assert np.all(_angle_diff(w2, w) < 1e-10)
