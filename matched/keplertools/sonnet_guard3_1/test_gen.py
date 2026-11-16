import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def _elems(a, e, O, I, w):
    return (np.float64(a), np.float64(e), np.float64(O), np.float64(I), np.float64(w))


# ---------------------------------------------------------------- orbElem2vec

def test_orbElem2vec_circular_at_periapsis():
    np.random.seed(0)
    r, v = orbElem2vec(0.0, 1.0, orbElem=_elems(1, 0, 0, 0, 0))
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    np.testing.assert_allclose(r[:, 0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [0, 1, 0], atol=1e-12)


def test_orbElem2vec_circular_quarter_orbit():
    np.random.seed(0)
    r, v = orbElem2vec(np.pi / 2, 1.0, orbElem=_elems(1, 0, 0, 0, 0))
    np.testing.assert_allclose(r[:, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [-1, 0, 0], atol=1e-12)


def test_orbElem2vec_polar_orbit_goes_to_z():
    np.random.seed(0)
    r, v = orbElem2vec(np.pi / 2, 1.0, orbElem=_elems(1, 0, 0, np.pi / 2, 0))
    np.testing.assert_allclose(r[:, 0], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [-1, 0, 0], atol=1e-12)


def test_orbElem2vec_eccentric_periapsis_values():
    np.random.seed(0)
    a, e, mu = 2.0, 0.5, 1.0
    r, v = orbElem2vec(0.0, mu, orbElem=_elems(a, e, 0, 0, 0))
    np.testing.assert_allclose(r[:, 0], [a * (1 - e), 0, 0], atol=1e-12)
    expected_speed = np.sqrt(mu / a) * np.sqrt((1 + e) / (1 - e))
    np.testing.assert_allclose(v[:, 0], [0, expected_speed, 0], atol=1e-12)


def test_orbElem2vec_apoapsis_values():
    np.random.seed(0)
    a, e, mu = 2.0, 0.5, 1.0
    r, v = orbElem2vec(np.pi, mu, orbElem=_elems(a, e, 0, 0, 0))
    np.testing.assert_allclose(r[:, 0], [-a * (1 + e), 0, 0], atol=1e-12)
    expected_speed = np.sqrt(mu / a) * np.sqrt((1 - e) / (1 + e))
    np.testing.assert_allclose(v[:, 0], [0, -expected_speed, 0], atol=1e-12)


def test_orbElem2vec_radius_and_vis_viva_for_many_anomalies():
    np.random.seed(0)
    a, e, O, I, w, mu = 1.7, 0.35, 1.1, 0.6, 2.3, 3.0
    E = np.linspace(0.0, 2 * np.pi, 9)
    r, v = orbElem2vec(E, mu, orbElem=_elems(a, e, O, I, w))
    assert r.shape == (3, E.size)
    assert v.shape == (3, E.size)
    rmag = np.linalg.norm(r, axis=0)
    vmag2 = np.sum(v ** 2, axis=0)
    np.testing.assert_allclose(rmag, a * (1 - e * np.cos(E)), rtol=1e-10)
    np.testing.assert_allclose(vmag2, mu * (2.0 / rmag - 1.0 / a), rtol=1e-10)


def test_orbElem2vec_angular_momentum_conserved():
    np.random.seed(0)
    a, e, O, I, w, mu = 2.5, 0.4, 0.7, 1.0, 4.0, 1.0
    E = np.array([0.1, 1.0, 2.0, 3.5, 5.0])
    r, v = orbElem2vec(E, mu, orbElem=_elems(a, e, O, I, w))
    h = np.cross(r.T, v.T)
    expected = np.sqrt(mu * a * (1 - e ** 2))
    np.testing.assert_allclose(np.linalg.norm(h, axis=1), expected, rtol=1e-10)
    np.testing.assert_allclose(h, np.tile(h[0], (E.size, 1)), atol=1e-10)


def test_orbElem2vec_returnAB():
    np.random.seed(0)
    elems = _elems(1.5, 0.2, 0.3, 0.4, 0.5)
    out = orbElem2vec(np.array([0.5, 1.5]), 1.0, orbElem=elems, returnAB=True)
    assert len(out) == 3
    r, v, (A, B) = out
    A2, B2 = calcAB(*elems)
    np.testing.assert_allclose(A, A2)
    np.testing.assert_allclose(B, B2)
    assert r.shape == (3, 2)


def test_orbElem2vec_AB_input_matches_orbElem_input():
    np.random.seed(0)
    elems = _elems(1.3, 0.25, 0.9, 0.8, 1.9)
    E = np.array([0.2, 1.4, 3.0, 4.4])
    r1, v1, AB = orbElem2vec(E, 2.0, orbElem=elems, returnAB=True)
    r2, v2 = orbElem2vec(E, 2.0, AB=AB)
    np.testing.assert_allclose(r1, r2, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(v1, v2, rtol=1e-9, atol=1e-12)


def test_orbElem2vec_multiple_bodies_match_single_body_calls():
    np.random.seed(0)
    a = np.array([1.0, 2.0, 3.5])
    e = np.array([0.1, 0.5, 0.3])
    O = np.array([0.2, 1.2, 2.2])
    I = np.array([0.3, 0.9, 1.5])
    w = np.array([0.4, 2.4, 5.0])
    E = np.array([0.5, 2.0, 4.0])
    mus = np.array([1.0, 2.0, 0.5])
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    for i in range(3):
        ri, vi = orbElem2vec(float(E[i]), float(mus[i]),
                             orbElem=_elems(a[i], e[i], O[i], I[i], w[i]))
        np.testing.assert_allclose(r[:, i], ri[:, 0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(v[:, i], vi[:, 0], rtol=1e-10, atol=1e-12)


def test_orbElem2vec_requires_orbElem_or_AB():
    np.random.seed(0)
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    np.random.seed(0)
    elems = _elems(1, 0.1, 0, 0, 0)
    AB = calcAB(*elems)
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, 1.0, orbElem=elems, AB=AB)


def test_orbElem2vec_scalar_E_requires_scalar_mus():
    np.random.seed(0)
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=_elems(1, 0.1, 0, 0, 0))


def test_orbElem2vec_mus_size_mismatch():
    np.random.seed(0)
    E = np.array([0.1, 0.2, 0.3])
    with pytest.raises(AssertionError):
        orbElem2vec(E, np.array([1.0, 2.0]), orbElem=_elems(1, 0.1, 0, 0, 0))


# ---------------------------------------------------------------- vec2orbElem

def _roundtrip_inputs():
    a = np.array([1.5, 2.0, 0.8, 3.0])
    e = np.array([0.3, 0.6, 0.1, 0.45])
    O = np.array([1.0, 4.0, 2.5, 5.5])
    I = np.array([0.5, 2.0, 1.2, 0.3])
    w = np.array([2.0, 5.0, 0.4, 3.5])
    E = np.array([0.7, 4.0, 2.9, 5.8])
    return a, e, O, I, w, E


def test_vec2orbElem_roundtrip_multiple_bodies():
    np.random.seed(0)
    a, e, O, I, w, E = _roundtrip_inputs()
    mus = np.array([1.0, 1.0, 1.0, 1.0])
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mus)
    np.testing.assert_allclose(a2, a, rtol=1e-8)
    np.testing.assert_allclose(e2, e, rtol=1e-8)
    np.testing.assert_allclose(E2, E, atol=1e-8)
    np.testing.assert_allclose(O2, O, atol=1e-8)
    np.testing.assert_allclose(I2, I, atol=1e-8)
    np.testing.assert_allclose(w2, w, atol=1e-8)


def test_vec2orbElem_accepts_n_by_3_layout():
    np.random.seed(0)
    a, e, O, I, w, E = _roundtrip_inputs()
    mus = np.array([1.0, 1.0, 1.0, 1.0])
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    out_3n = vec2orbElem(r, v, 1.0)
    out_n3 = vec2orbElem(r.T.copy(), v.T.copy(), 1.0)
    for x, y in zip(out_3n, out_n3):
        np.testing.assert_allclose(x, y, rtol=1e-10, atol=1e-12)


def test_vec2orbElem_period_and_tau():
    np.random.seed(0)
    a, e, O, I, w, mu = 2.0, 0.4, 1.0, 0.7, 2.0, 3.0
    E = 1.3
    r, v = orbElem2vec(E, mu, orbElem=_elems(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P, tau = vec2orbElem(r, v, mu)
    np.testing.assert_allclose(P, 2 * np.pi * np.sqrt(a ** 3 / mu), rtol=1e-9)
    n = np.sqrt(mu / a ** 3)
    np.testing.assert_allclose(tau, -(E - e * np.sin(E)) / n, rtol=1e-7, atol=1e-9)


def test_vec2orbElem_output_shapes_and_ranges():
    np.random.seed(0)
    a, e, O, I, w, E = _roundtrip_inputs()
    mus = np.ones(4)
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    out = vec2orbElem(r, v, mus)
    assert len(out) == 8
    for arr in out:
        assert np.asarray(arr).shape == (4,)
    _, _, E2, O2, I2, w2, _, _ = out
    for ang in (E2, O2, w2):
        assert np.all(ang >= 0) and np.all(ang < 2 * np.pi + 1e-12)
    assert np.all(I2 > 0) and np.all(I2 < np.pi)


def test_vec2orbElem_size_mismatch_asserts():
    np.random.seed(0)
    rs = np.ones((3, 2))
    vs = np.ones((3, 1))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_mus_size_mismatch_asserts():
    np.random.seed(0)
    rs = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.5]])
    vs = np.array([[0.0, 0.0], [1.0, 0.7], [0.1, 0.1]])
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, np.array([1.0, 1.0, 1.0]))


def test_vec2orbElem_three_dimensional_input_asserts():
    np.random.seed(0)
    rs = np.ones((3, 1, 1))
    vs = np.ones((3, 1, 1))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)
