import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem


def _arr(*vals):
    return np.array(vals, dtype=float)


def _elements():
    a = _arr(1.0, 2.0, 3.5, 0.8)
    e = _arr(0.1, 0.3, 0.5, 0.7)
    O = _arr(0.5, 1.0, 2.0, 4.0)
    I = _arr(0.3, 0.7, 1.2, 2.5)
    w = _arr(1.0, 2.5, 3.5, 5.0)
    E = _arr(0.3, 1.5, 3.0, 5.0)
    return a, e, O, I, w, E


def test_orbElem2vec_scalar_circular_simple():
    np.random.seed(0)
    orb = (_arr(1.0), _arr(0.0), _arr(0.0), _arr(0.0), _arr(0.0))
    r, v = orbElem2vec(0.0, 1.0, orbElem=orb)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    np.testing.assert_allclose(r[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [0.0, 1.0, 0.0], atol=1e-12)


def test_orbElem2vec_scalar_E_quarter_orbit_circular():
    np.random.seed(0)
    orb = (_arr(2.0), _arr(0.0), _arr(0.0), _arr(0.0), _arr(0.0))
    r, v = orbElem2vec(np.pi / 2, 4.0, orbElem=orb)
    np.testing.assert_allclose(r[:, 0], [0.0, 2.0, 0.0], atol=1e-12)
    speed = np.sqrt(4.0 / 2.0)
    np.testing.assert_allclose(v[:, 0], [-speed, 0.0, 0.0], atol=1e-12)


def test_orbElem2vec_vector_E_scalar_mu_shapes():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    orb = (a[1:2], e[1:2], O[1:2], I[1:2], w[1:2])
    r, v = orbElem2vec(E, 1.0, orbElem=orb)
    assert r.shape == (3, 4)
    assert v.shape == (3, 4)


def test_orbElem2vec_radius_matches_conic_equation():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    mus = _arr(1.0, 2.0, 3.0, 4.0)
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    rmag = np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(rmag, a * (1 - e * np.cos(E)), rtol=1e-10)


def test_orbElem2vec_vis_viva_and_energy():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    mus = _arr(1.0, 2.0, 3.0, 4.0)
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    rmag = np.linalg.norm(r, axis=0)
    v2 = np.sum(v ** 2, axis=0)
    np.testing.assert_allclose(v2, mus * (2.0 / rmag - 1.0 / a), rtol=1e-10)
    np.testing.assert_allclose(v2 / 2 - mus / rmag, -mus / (2 * a), rtol=1e-10)


def test_orbElem2vec_angular_momentum_magnitude():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    mus = _arr(1.0, 2.0, 3.0, 4.0)
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    h = np.cross(r.T, v.T).T
    hmag = np.linalg.norm(h, axis=0)
    np.testing.assert_allclose(hmag, np.sqrt(mus * a * (1 - e ** 2)), rtol=1e-10)
    # h direction gives inclination
    np.testing.assert_allclose(np.arccos(h[2] / hmag), I, atol=1e-10)


def test_orbElem2vec_vector_matches_scalar_calls():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    mus = _arr(1.0, 2.0, 3.0, 4.0)
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    for i in range(4):
        sl = slice(i, i + 1)
        ri, vi = orbElem2vec(float(E[i]), float(mus[i]),
                             orbElem=(a[sl], e[sl], O[sl], I[sl], w[sl]))
        np.testing.assert_allclose(r[:, i], ri[:, 0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(v[:, i], vi[:, 0], rtol=1e-10, atol=1e-12)


def test_orbElem2vec_AB_input_equivalent_to_orbElem():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    mus = _arr(1.0, 2.0, 3.0, 4.0)
    A, B = calcAB(a, e, O, I, w)
    r1, v1 = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    r2, v2 = orbElem2vec(E, mus, AB=(A, B))
    np.testing.assert_allclose(r1, r2, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(v1, v2, rtol=1e-9, atol=1e-12)


def test_orbElem2vec_returnAB():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    mus = _arr(1.0, 2.0, 3.0, 4.0)
    out = orbElem2vec(E, mus, orbElem=(a, e, O, I, w), returnAB=True)
    assert len(out) == 3
    A, B = out[2]
    A0, B0 = calcAB(a, e, O, I, w)
    np.testing.assert_allclose(A, A0)
    np.testing.assert_allclose(B, B0)
    assert len(orbElem2vec(E, mus, orbElem=(a, e, O, I, w))) == 2


def test_orbElem2vec_input_validation():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    orb = (a, e, O, I, w)
    A, B = calcAB(a, e, O, I, w)
    with pytest.raises(AssertionError):
        orbElem2vec(E, 1.0)
    with pytest.raises(AssertionError):
        orbElem2vec(E, _arr(1.0, 1.0, 1.0, 1.0), orbElem=orb, AB=(A, B))
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, _arr(1.0, 2.0), orbElem=(a[:1], e[:1], O[:1], I[:1], w[:1]))
    with pytest.raises(AssertionError):
        orbElem2vec(E, _arr(1.0, 2.0), orbElem=orb)


def test_vec2orbElem_roundtrip_vector_mus():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    mus = _arr(1.0, 2.0, 3.0, 0.5)
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mus)
    np.testing.assert_allclose(a2, a, rtol=1e-8)
    np.testing.assert_allclose(e2, e, rtol=1e-7)
    np.testing.assert_allclose(E2, E, atol=1e-7)
    np.testing.assert_allclose(O2, O, atol=1e-7)
    np.testing.assert_allclose(I2, I, atol=1e-7)
    np.testing.assert_allclose(w2, w, atol=1e-7)


def test_vec2orbElem_roundtrip_scalar_mu():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    r, v = orbElem2vec(E, 1.0, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, 1.0)
    np.testing.assert_allclose(a2, a, rtol=1e-8)
    np.testing.assert_allclose(e2, e, rtol=1e-7)
    np.testing.assert_allclose(O2, O, atol=1e-7)
    np.testing.assert_allclose(w2, w, atol=1e-7)


def test_vec2orbElem_transposed_input_same_result():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    mus = _arr(1.0, 2.0, 3.0, 0.5)
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    res1 = vec2orbElem(r, v, mus)
    res2 = vec2orbElem(r.T.copy(), v.T.copy(), mus)
    assert len(res1) == 8
    for x, y in zip(res1, res2):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


def test_vec2orbElem_period_and_tau():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    mus = _arr(1.0, 2.0, 3.0, 0.5)
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mus)
    np.testing.assert_allclose(P2, 2 * np.pi * np.sqrt(a ** 3 / mus), rtol=1e-8)
    n = np.sqrt(mus / a ** 3)
    np.testing.assert_allclose(tau2, -(E - e * np.sin(E)) / n, rtol=1e-6, atol=1e-8)


def test_vec2orbElem_known_orbit_ranges():
    np.random.seed(0)
    a, e, O, I, w, E = _elements()
    r, v = orbElem2vec(E, 1.0, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, 1.0)
    for ang in (E2, O2, w2):
        assert np.all(ang >= 0) and np.all(ang < 2 * np.pi)
    assert np.all(I2 > 0) and np.all(I2 < np.pi)
    assert np.all(e2 >= 0) and np.all(e2 < 1)
    assert np.all(a2 > 0)


def test_vec2orbElem_input_validation():
    np.random.seed(0)
    rs = np.ones((3, 2))
    vs = np.ones((3, 3))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)
    # wrong mus size
    with pytest.raises(AssertionError):
        vec2orbElem(np.ones((3, 2)), np.ones((3, 2)), np.ones(5))
    # more than two dims
    with pytest.raises(AssertionError):
        vec2orbElem(np.ones((3, 2, 1)), np.ones((3, 2, 1)), 1.0)
    # size not multiple of 3
    with pytest.raises(AssertionError):
        vec2orbElem(np.ones((2, 2)), np.ones((2, 2)), 1.0)
