import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem


def _angle_close(a, b, atol=1e-10):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = np.mod(a - b + np.pi, 2 * np.pi) - np.pi
    return np.allclose(diff, 0.0, atol=atol, rtol=0.0)


def _sample_orbital_elements(n=3):
    a = np.array([1.5, 2.1, 3.7][:n], dtype=float)
    e = np.array([0.1, 0.35, 0.6][:n], dtype=float)
    O = np.array([0.2, 1.1, 2.2][:n], dtype=float)
    I = np.array([0.4, 1.0, 0.7][:n], dtype=float)
    w = np.array([0.3, 2.0, 1.5][:n], dtype=float)
    return a, e, O, I, w


def test_orbElem2vec_scalar_roundtrip_single_body():
    a, e, O, I, w = [x[0] for x in _sample_orbital_elements(1)]
    mu = 1.0
    E = 1.234

    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    out = vec2orbElem(rs, vs, mu)

    a2, e2, E2, O2, I2, w2, P2, tau2 = out
    assert rs.shape == (3, 1)
    assert vs.shape == (3, 1)
    assert np.allclose(a2, a)
    assert np.allclose(e2, e)
    assert np.allclose(P2, 2 * np.pi * np.sqrt(a**3 / mu))
    assert np.isfinite(tau2).all()
    assert _angle_close(E2, np.mod(E, 2 * np.pi))
    assert _angle_close(O2, O)
    assert _angle_close(I2, I)
    assert _angle_close(w2, w)


def test_orbElem2vec_vector_E_scalar_mu_shapes():
    a, e, O, I, w = [x[:1] for x in _sample_orbital_elements(1)]
    mu = 1.0
    E = np.array([0.1, 1.2, 2.5, 4.0])

    rs, vs = orbElem2vec(E, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]))
    assert rs.shape == (3, E.size)
    assert vs.shape == (3, E.size)

    out = vec2orbElem(rs, vs, mu)
    a2, e2, E2, O2, I2, w2, P2, tau2 = out
    assert np.allclose(a2, a[0])
    assert np.allclose(e2, e[0])
    assert np.allclose(P2, 2 * np.pi * np.sqrt(a[0] ** 3 / mu))
    assert np.isfinite(tau2).all()
    assert _angle_close(O2, O[0])
    assert _angle_close(I2, I[0])
    assert _angle_close(w2, w[0])
    assert _angle_close(E2, np.mod(E, 2 * np.pi))


def test_orbElem2vec_vectorized_multiple_bodies_roundtrip():
    a, e, O, I, w = _sample_orbital_elements(3)
    mus = np.array([1.0, 1.5, 2.0], dtype=float)
    E = np.array([0.25, 1.0, 2.0], dtype=float)

    rs, vs = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert rs.shape == (3, 3)
    assert vs.shape == (3, 3)

    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(rs, vs, mus)
    assert np.allclose(a2, a)
    assert np.allclose(e2, e)
    assert np.allclose(P2, 2 * np.pi * np.sqrt(a**3 / mus))
    assert np.isfinite(tau2).all()
    assert _angle_close(E2, np.mod(E, 2 * np.pi))
    assert _angle_close(O2, O)
    assert _angle_close(I2, I)
    assert _angle_close(w2, w)


def test_orbElem2vec_accepts_AB_input_and_returnAB():
    a, e, O, I, w = _sample_orbital_elements(1)
    mu = 1.0
    E = 0.7
    A, B = calcAB(a, e, O, I, w)

    rs1, vs1 = orbElem2vec(E, mu, AB=(A, B))
    rs2, vs2, AB_out = orbElem2vec(E, mu, AB=(A, B), returnAB=True)

    assert np.allclose(rs1, rs2)
    assert np.allclose(vs1, vs2)
    assert isinstance(AB_out, tuple) and len(AB_out) == 2
    assert np.allclose(AB_out[0], A)
    assert np.allclose(AB_out[1], B)


def test_orbElem2vec_orbElem_and_AB_mutually_exclusive():
    a, e, O, I, w = _sample_orbital_elements(1)
    mu = 1.0
    E = 0.5
    A, B = calcAB(a, e, O, I, w)

    with pytest.raises(AssertionError, match="You can only set orbElem or AB"):
        orbElem2vec(E, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]), AB=(A, B))


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError, match="You must supply either orbElem or AB inputs"):
        orbElem2vec(0.5, 1.0)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    a, e, O, I, w = _sample_orbital_elements(1)
    A, B = calcAB(a, e, O, I, w)
    with pytest.raises(AssertionError, match="Scalar E input requires scalar mus input"):
        orbElem2vec(0.5, np.array([1.0, 2.0]), AB=(A, B))


def test_orbElem2vec_vector_E_mus_size_mismatch():
    a, e, O, I, w = _sample_orbital_elements(2)
    E = np.array([0.1, 0.2])
    mus = np.array([1.0])  # neither scalar nor same size
    with pytest.raises(AssertionError, match="mus must be of the same size as E or scalar"):
        orbElem2vec(E, mus, orbElem=(a, e, O, I, w))


def test_orbElem2vec_AB_size_mismatch_for_scalar_E():
    E = np.array([0.3])
    mu = 1.0
    A = np.zeros((3, 2))
    B = np.zeros((3, 2))
    with pytest.raises(AssertionError, match="A and B must be 3x1"):
        orbElem2vec(E, mu, AB=(A, B))


def test_vec2orbElem_accepts_one_dimensional_inputs():
    a, e, O, I, w = [x[0] for x in _sample_orbital_elements(1)]
    mu = 1.0
    E = 1.1
    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))

    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(rs.ravel(), vs.ravel(), mu)
    assert np.allclose(a2, a)
    assert np.allclose(e2, e)
    assert _angle_close(E2, np.mod(E, 2 * np.pi))
    assert _angle_close(O2, O)
    assert _angle_close(I2, I)
    assert _angle_close(w2, w)
    assert np.allclose(P2, 2 * np.pi * np.sqrt(a**3 / mu))
    assert np.isfinite(tau2).all()


def test_vec2orbElem_transposes_nx3_inputs():
    a, e, O, I, w = [x[:1] for x in _sample_orbital_elements(1)]
    mu = 1.0
    E = 1.7
    rs, vs = orbElem2vec(E, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]))

    out1 = vec2orbElem(rs, vs, mu)
    out2 = vec2orbElem(rs.T, vs.T, mu)

    for x, y in zip(out1[:6], out2[:6]):
        if np.ndim(x) == 0:
            assert np.allclose(x, y)
        else:
            assert np.allclose(x, y)


def test_vec2orbElem_rejects_incompatible_rs_vs_sizes():
    rs = np.zeros(6)
    vs = np.zeros(3)
    with pytest.raises(AssertionError, match="rs and vs must be of the same size"):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_invalid_mus_size():
    rs = np.zeros(6)
    vs = np.zeros(6)
    mus = np.array([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError, match="mus must be scalar or of size n"):
        vec2orbElem(rs, vs, mus)


def test_vec2orbElem_rejects_more_than_two_dimensions():
    rs = np.zeros((3, 1, 1))
    vs = np.zeros((3, 1, 1))
    with pytest.raises(AssertionError, match="rs cannot have more than two dimensions"):
        vec2orbElem(rs, vs, 1.0)
