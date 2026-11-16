import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def _arr(*vals):
    return np.array(vals, dtype=float)


def _single_body(a=2.0, e=0.3, O=1.0, I=0.5, w=2.0):
    return (_arr(a), _arr(e), _arr(O), _arr(I), _arr(w))


def test_periapsis_position_and_velocity():
    mu = 1.0
    a, e = 2.0, 0.5
    r, v = orbElem2vec(_arr(0.0), mu, orbElem=_single_body(a, e, 0.0, 0.0, 0.0))
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    np.testing.assert_allclose(r[:, 0], [a * (1 - e), 0, 0], atol=1e-12)
    vp = np.sqrt(mu / a * (1 + e) / (1 - e))
    np.testing.assert_allclose(v[:, 0], [0, vp, 0], atol=1e-12)


def test_apoapsis_position_and_velocity():
    mu = 1.0
    a, e = 2.0, 0.5
    r, v = orbElem2vec(_arr(np.pi), mu, orbElem=_single_body(a, e, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(r[:, 0], [-a * (1 + e), 0, 0], atol=1e-12)
    va = np.sqrt(mu / a * (1 - e) / (1 + e))
    np.testing.assert_allclose(v[:, 0], [0, -va, 0], atol=1e-12)


def test_circular_orbit_speed_and_radius():
    mu = 3.0
    a = 1.5
    E = np.linspace(0, 2 * np.pi, 9)
    r, v = orbElem2vec(E, mu, orbElem=_single_body(a, 0.0, 0.3, 0.7, 1.1))
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), a, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(v, axis=0), np.sqrt(mu / a), rtol=1e-12)


def test_scalar_E_and_scalar_mu():
    r, v = orbElem2vec(0.7, 1.0, orbElem=_single_body())
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(v))


def test_radius_matches_kepler_equation_of_orbit():
    a, e = 2.0, 0.3
    E = np.linspace(0.1, 6.0, 12)
    r, _ = orbElem2vec(E, 1.0, orbElem=_single_body(a, e))
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), a * (1 - e * np.cos(E)), rtol=1e-12)


def test_energy_and_angular_momentum_conserved():
    mu = 2.5
    a, e = 2.0, 0.4
    E = np.linspace(0.2, 6.0, 10)
    r, v = orbElem2vec(E, mu, orbElem=_single_body(a, e, 0.8, 0.6, 1.9))
    energy = 0.5 * np.sum(v ** 2, axis=0) - mu / np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(energy, -mu / (2 * a), rtol=1e-10)
    h = np.cross(r.T, v.T)
    hmag = np.linalg.norm(h, axis=1)
    np.testing.assert_allclose(hmag, np.sqrt(mu * a * (1 - e ** 2)), rtol=1e-10)
    # direction constant
    np.testing.assert_allclose(h / hmag[:, None], np.tile(h[0] / hmag[0], (len(E), 1)), atol=1e-10)


def test_returnAB_gives_calcAB_output():
    oe = _single_body()
    r, v, AB = orbElem2vec(_arr(0.5, 1.5), 1.0, orbElem=oe, returnAB=True)
    A, B = calcAB(*oe)
    np.testing.assert_allclose(AB[0], A)
    np.testing.assert_allclose(AB[1], B)
    assert r.shape == (3, 2)


def test_AB_input_matches_orbElem_input():
    oe = _single_body(a=1.7, e=0.25, O=0.4, I=1.0, w=3.0)
    E = np.array([0.3, 2.0, 4.5])
    r1, v1, AB = orbElem2vec(E, 1.3, orbElem=oe, returnAB=True)
    r2, v2 = orbElem2vec(E, 1.3, AB=AB)
    np.testing.assert_allclose(r1, r2, atol=1e-10)
    np.testing.assert_allclose(v1, v2, atol=1e-10)


def test_multiple_bodies_vector_mus():
    a = _arr(1.0, 2.0, 3.0)
    e = _arr(0.1, 0.2, 0.3)
    O = _arr(0.5, 1.0, 1.5)
    I = _arr(0.3, 0.6, 0.9)
    w = _arr(1.0, 2.0, 3.0)
    E = _arr(0.5, 1.5, 2.5)
    mus = _arr(1.0, 2.0, 3.0)
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), a * (1 - e * np.cos(E)), rtol=1e-10)
    energy = 0.5 * np.sum(v ** 2, axis=0) - mus / np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(energy, -mus / (2 * a), rtol=1e-10)


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        orbElem2vec(_arr(0.5), 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    oe = _single_body()
    AB = calcAB(*oe)
    with pytest.raises(AssertionError):
        orbElem2vec(_arr(0.5), 1.0, orbElem=oe, AB=AB)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, _arr(1.0, 2.0), orbElem=_single_body())


def test_orbElem2vec_mus_size_mismatch():
    with pytest.raises(AssertionError):
        orbElem2vec(_arr(0.1, 0.2, 0.3), _arr(1.0, 2.0), orbElem=_single_body())


def test_roundtrip_single_body_many_positions():
    mu = 1.7
    a, e, O, I, w = 2.0, 0.3, 1.0, 0.5, 2.0
    E_in = np.array([0.4, 1.2, 2.5, 3.5, 5.0, 6.0])
    r, v = orbElem2vec(E_in, mu, orbElem=_single_body(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    np.testing.assert_allclose(a2, a, rtol=1e-9)
    np.testing.assert_allclose(e2, e, rtol=1e-9)
    np.testing.assert_allclose(E2, E_in, atol=1e-9)
    np.testing.assert_allclose(O2, O, atol=1e-9)
    np.testing.assert_allclose(I2, I, atol=1e-9)
    np.testing.assert_allclose(w2, w, atol=1e-9)
    np.testing.assert_allclose(P2, 2 * np.pi * np.sqrt(a ** 3 / mu), rtol=1e-9)
    np.testing.assert_allclose(tau2, -(E_in - e * np.sin(E_in)) / np.sqrt(mu / a ** 3), atol=1e-9)


def test_roundtrip_multiple_bodies():
    a = _arr(1.0, 2.0, 3.0, 4.0)
    e = _arr(0.1, 0.2, 0.3, 0.4)
    O = _arr(0.5, 1.0, 1.5, 5.5)
    I = _arr(0.3, 0.6, 0.9, 1.2)
    w = _arr(1.0, 2.0, 3.0, 6.0)
    E = _arr(0.5, 1.5, 2.5, 4.0)
    mus = _arr(1.0, 2.0, 3.0, 0.5)
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mus)
    np.testing.assert_allclose(a2, a, rtol=1e-9)
    np.testing.assert_allclose(e2, e, rtol=1e-9)
    np.testing.assert_allclose(E2, E, atol=1e-9)
    np.testing.assert_allclose(O2, O, atol=1e-9)
    np.testing.assert_allclose(I2, I, atol=1e-9)
    np.testing.assert_allclose(w2, w, atol=1e-9)


def test_vec2orbElem_accepts_n_by_3_input():
    mu = 1.0
    E = np.array([0.4, 1.2, 2.5, 3.5])
    r, v = orbElem2vec(E, mu, orbElem=_single_body())
    res1 = vec2orbElem(r, v, mu)
    res2 = vec2orbElem(r.T, v.T, mu)
    for x, y in zip(res1, res2):
        np.testing.assert_allclose(x, y, atol=1e-12)


def test_vec2orbElem_known_circular_like_orbit():
    # near-circular inclined orbit: r=(1,0,0), v=(0,cos(i),sin(i))*sqrt(mu*(1+e)) ...
    mu = 1.0
    rs = np.array([[1.0], [0.0], [0.0]])
    vs = np.array([[0.0], [np.sqrt(1.0 / 1.0) * np.cos(0.4) * 1.1], [np.sqrt(1.0) * np.sin(0.4) * 1.1]])
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, mu)
    # inclination from angular momentum z component
    h = np.cross(rs[:, 0], vs[:, 0])
    np.testing.assert_allclose(I[0], np.arccos(h[2] / np.linalg.norm(h)), atol=1e-12)
    # vis-viva
    np.testing.assert_allclose(a[0], -mu / (2 * (0.5 * np.sum(vs ** 2) - mu)), rtol=1e-12)
    assert 0 <= e[0] < 1
    assert 0 <= E[0] < 2 * np.pi
    assert 0 <= O[0] < 2 * np.pi
    assert 0 <= w[0] < 2 * np.pi


def test_vec2orbElem_size_mismatch_asserts():
    with pytest.raises(AssertionError):
        vec2orbElem(np.ones((3, 2)), np.ones((3, 1)), 1.0)


def test_vec2orbElem_non_multiple_of_three_asserts():
    with pytest.raises(AssertionError):
        vec2orbElem(np.ones((2, 2)), np.ones((2, 2)), 1.0)


def test_vec2orbElem_three_dimensional_input_asserts():
    with pytest.raises(AssertionError):
        vec2orbElem(np.ones((3, 2, 1)), np.ones((3, 2, 1)), 1.0)


def test_vec2orbElem_mus_wrong_size_asserts():
    r, v = orbElem2vec(np.array([0.4, 1.2, 2.5, 3.5]), 1.0, orbElem=_single_body())
    with pytest.raises(AssertionError):
        vec2orbElem(r, v, _arr(1.0, 2.0))
