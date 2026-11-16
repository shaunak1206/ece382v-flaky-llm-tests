import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def _angdiff(x, y):
    """Smallest absolute difference between angles (handles wrap-around)."""
    d = np.mod(np.asarray(x) - np.asarray(y) + np.pi, 2 * np.pi) - np.pi
    return np.abs(d)


def _single_body_state(a, e, O, I, w, E, mu):
    orb = (np.array([a]), np.array([e]), np.array([O]), np.array([I]), np.array([w]))
    r, v = orbElem2vec(np.array([E]), mu, orbElem=orb)
    return r, v


def test_orbElem2vec_output_shapes_single_body_many_positions():
    E = np.linspace(0.2, 5.5, 5)
    orb = (np.float64(1.5), np.float64(0.3), np.float64(0.5),
           np.float64(0.7), np.float64(1.1))
    r, v = orbElem2vec(E, 1.0, orbElem=orb)
    assert r.shape == (3, 5)
    assert v.shape == (3, 5)


def test_orbElem2vec_scalar_E_scalar_mu_shapes():
    orb = (np.float64(1.5), np.float64(0.3), np.float64(0.5),
           np.float64(0.7), np.float64(1.1))
    r, v = orbElem2vec(0.9, 1.0, orbElem=orb)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(v))


def test_orbElem2vec_multiple_bodies_shapes():
    a = np.array([1.0, 2.0, 0.7, 3.5, 1.2])
    e = np.array([0.1, 0.5, 0.3, 0.7, 0.2])
    O = np.array([0.3, 1.0, 2.0, 4.0, 5.0])
    I = np.array([0.4, 1.0, 1.5, 2.5, 0.9])
    w = np.array([0.1, 2.0, 3.0, 4.0, 5.5])
    E = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    mus = np.array([1.0, 1.5, 0.8, 2.0, 1.0])
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 5)
    assert v.shape == (3, 5)


def test_orbElem2vec_periapsis_distance_and_speed():
    a, e, mu = 2.0, 0.4, 1.3
    r, v = _single_body_state(a, e, 0.6, 0.8, 1.9, 0.0, mu)
    assert np.linalg.norm(r[:, 0]) == pytest.approx(a * (1 - e), rel=1e-12)
    expected_v = np.sqrt(mu / a * (1 + e) / (1 - e))
    assert np.linalg.norm(v[:, 0]) == pytest.approx(expected_v, rel=1e-12)
    # velocity perpendicular to position at periapsis
    assert np.dot(r[:, 0], v[:, 0]) == pytest.approx(0.0, abs=1e-12)


def test_orbElem2vec_apoapsis_distance():
    a, e, mu = 2.0, 0.4, 1.0
    r, v = _single_body_state(a, e, 0.6, 0.8, 1.9, np.pi, mu)
    assert np.linalg.norm(r[:, 0]) == pytest.approx(a * (1 + e), rel=1e-12)
    expected_v = np.sqrt(mu / a * (1 - e) / (1 + e))
    assert np.linalg.norm(v[:, 0]) == pytest.approx(expected_v, rel=1e-12)


def test_orbElem2vec_circular_orbit_constant_radius_and_speed():
    a, mu = 1.7, 2.0
    E = np.linspace(0.1, 6.0, 7)
    orb = (np.float64(a), np.float64(0.0), np.float64(0.4),
           np.float64(1.0), np.float64(0.3))
    r, v = orbElem2vec(E, mu, orbElem=orb)
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), a, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(v, axis=0), np.sqrt(mu / a), rtol=1e-12)


def test_orbElem2vec_energy_and_angular_momentum_conserved():
    a, e, mu = 1.5, 0.35, 0.9
    E = np.linspace(0.1, 6.0, 9)
    orb = (np.float64(a), np.float64(e), np.float64(1.2),
           np.float64(0.6), np.float64(2.2))
    r, v = orbElem2vec(E, mu, orbElem=orb)
    energy = 0.5 * np.sum(v ** 2, axis=0) - mu / np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(energy, -mu / (2 * a), rtol=1e-10)
    h = np.cross(r.T, v.T)
    hmag = np.linalg.norm(h, axis=1)
    np.testing.assert_allclose(hmag, np.sqrt(mu * a * (1 - e ** 2)), rtol=1e-10)
    # Angular momentum direction is constant
    np.testing.assert_allclose(h / hmag[:, None], np.tile(h[0] / hmag[0], (9, 1)),
                               atol=1e-10)


def test_orbElem2vec_returnAB_matches_calcAB():
    orb = (np.array([1.5]), np.array([0.3]), np.array([0.5]),
           np.array([0.7]), np.array([1.1]))
    E = np.array([0.4, 1.0, 2.0])
    r, v, AB = orbElem2vec(E, 1.0, orbElem=orb, returnAB=True)
    A, B = calcAB(*orb)
    np.testing.assert_allclose(AB[0], A)
    np.testing.assert_allclose(AB[1], B)
    assert r.shape == (3, 3)


def test_orbElem2vec_AB_input_equals_orbElem_input():
    orb = (np.array([1.5]), np.array([0.3]), np.array([0.5]),
           np.array([0.7]), np.array([1.1]))
    E = np.array([0.4, 1.0, 2.0, 5.0])
    r1, v1, AB = orbElem2vec(E, 1.0, orbElem=orb, returnAB=True)
    r2, v2 = orbElem2vec(E, 1.0, AB=AB)
    np.testing.assert_allclose(r1, r2, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v1, v2, rtol=1e-10, atol=1e-12)


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.5]), 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    orb = (np.array([1.5]), np.array([0.3]), np.array([0.5]),
           np.array([0.7]), np.array([1.1]))
    A, B = calcAB(*orb)
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.5]), 1.0, orbElem=orb, AB=(A, B))


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    orb = (np.float64(1.5), np.float64(0.3), np.float64(0.5),
           np.float64(0.7), np.float64(1.1))
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=orb)


def test_orbElem2vec_mu_size_mismatch_raises():
    a = np.array([1.0, 2.0, 1.5, 1.2])
    e = np.array([0.1, 0.2, 0.3, 0.4])
    ang = np.array([0.3, 0.5, 0.7, 0.9])
    E = np.array([0.5, 1.0, 1.5, 2.0])
    with pytest.raises(AssertionError):
        orbElem2vec(E, np.array([1.0, 1.0]), orbElem=(a, e, ang, ang, ang))


def test_roundtrip_single_body_recovers_elements():
    a, e, O, I, w, E, mu = 1.5, 0.3, 0.5, 0.7, 1.1, 0.9, 1.0
    r, v = _single_body_state(a, e, O, I, w, E, mu)
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    assert a2[0] == pytest.approx(a, rel=1e-10)
    assert e2[0] == pytest.approx(e, rel=1e-9)
    assert E2[0] == pytest.approx(E, rel=1e-9)
    assert O2[0] == pytest.approx(O, rel=1e-9)
    assert I2[0] == pytest.approx(I, rel=1e-9)
    assert w2[0] == pytest.approx(w, rel=1e-9)


def test_roundtrip_retrograde_orbit_and_large_angles():
    a, e, O, I, w, E, mu = 3.2, 0.6, 5.0, 2.5, 4.0, 5.5, 2.0
    r, v = _single_body_state(a, e, O, I, w, E, mu)
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    assert a2[0] == pytest.approx(a, rel=1e-9)
    assert e2[0] == pytest.approx(e, rel=1e-8)
    assert _angdiff(E2[0], E) < 1e-8
    assert _angdiff(O2[0], O) < 1e-8
    assert I2[0] == pytest.approx(I, rel=1e-9)
    assert _angdiff(w2[0], w) < 1e-8


def test_roundtrip_single_body_many_positions_scalar_mu():
    mu = 1.0
    a, e, O, I, w = 1.5, 0.3, 0.5, 0.7, 1.1
    E = np.array([0.3, 1.0, 2.0, 3.5, 5.0])
    orb = (np.float64(a), np.float64(e), np.float64(O),
           np.float64(I), np.float64(w))
    r, v = orbElem2vec(E, mu, orbElem=orb)
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    np.testing.assert_allclose(a2, a, rtol=1e-9)
    np.testing.assert_allclose(e2, e, rtol=1e-8)
    np.testing.assert_allclose(E2, E, rtol=1e-8)
    np.testing.assert_allclose(O2, O, rtol=1e-8)
    np.testing.assert_allclose(I2, I, rtol=1e-8)
    np.testing.assert_allclose(w2, w, rtol=1e-8)


def test_roundtrip_multiple_bodies_array_mus():
    a = np.array([1.0, 2.0, 0.7, 3.5, 1.2])
    e = np.array([0.1, 0.5, 0.3, 0.7, 0.2])
    O = np.array([0.3, 1.0, 2.0, 4.0, 5.0])
    I = np.array([0.4, 1.0, 1.5, 2.5, 0.9])
    w = np.array([0.1, 2.0, 3.0, 4.0, 5.5])
    E = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    mus = np.array([1.0, 1.5, 0.8, 2.0, 1.0])
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mus)
    np.testing.assert_allclose(a2, a, rtol=1e-9)
    np.testing.assert_allclose(e2, e, rtol=1e-8)
    assert np.all(_angdiff(E2, E) < 1e-8)
    assert np.all(_angdiff(O2, O) < 1e-8)
    np.testing.assert_allclose(I2, I, rtol=1e-8)
    assert np.all(_angdiff(w2, w) < 1e-8)


def test_vec2orbElem_transposed_input_gives_same_result():
    mu = 1.0
    E = np.array([0.3, 1.0, 2.0, 3.5, 5.0])
    orb = (np.float64(1.5), np.float64(0.3), np.float64(0.5),
           np.float64(0.7), np.float64(1.1))
    r, v = orbElem2vec(E, mu, orbElem=orb)
    res1 = vec2orbElem(r, v, mu)
    res2 = vec2orbElem(r.T, v.T, mu)
    for x, y in zip(res1, res2):
        np.testing.assert_allclose(x, y, rtol=1e-12)


def test_vec2orbElem_known_periapsis_state():
    mu = 1.0
    inc = 0.5
    speed = 1.2
    r = np.array([[1.0], [0.0], [0.0]])
    v = np.array([[0.0], [speed * np.cos(inc)], [speed * np.sin(inc)]])
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, mu)
    exp_a = 1.0 / (2.0 - speed ** 2)
    exp_e = speed ** 2 - 1.0
    assert a[0] == pytest.approx(exp_a, rel=1e-12)
    assert e[0] == pytest.approx(exp_e, rel=1e-10)
    assert I[0] == pytest.approx(inc, rel=1e-10)
    assert _angdiff(E[0], 0.0) < 1e-8
    assert _angdiff(O[0], 0.0) < 1e-8
    assert _angdiff(w[0], 0.0) < 1e-8
    assert P[0] == pytest.approx(2 * np.pi * np.sqrt(exp_a ** 3 / mu), rel=1e-12)
    assert abs(tau[0]) < 1e-8


def test_vec2orbElem_period_and_tau_consistent():
    mu = 1.7
    a, e, O, I, w, E = 2.2, 0.45, 1.0, 0.9, 2.0, 1.3
    r, v = _single_body_state(a, e, O, I, w, E, mu)
    a2, e2, E2, O2, I2, w2, P, tau = vec2orbElem(r, v, mu)
    assert P[0] == pytest.approx(2 * np.pi * np.sqrt(a ** 3 / mu), rel=1e-9)
    n = np.sqrt(mu / a ** 3)
    assert tau[0] == pytest.approx(-(E - e * np.sin(E)) / n, rel=1e-8)


def test_vec2orbElem_size_mismatch_raises():
    rs = np.ones((3, 2))
    vs = np.ones((3, 4))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_wrong_mus_size_raises():
    mu = 1.0
    E = np.array([0.3, 1.0])
    mus = np.array([1.0, 1.0])
    orb = (np.array([1.5, 1.2]), np.array([0.3, 0.2]), np.array([0.5, 0.6]),
           np.array([0.7, 0.8]), np.array([1.1, 1.2]))
    r, v = orbElem2vec(E, mus, orbElem=orb)
    with pytest.raises(AssertionError):
        vec2orbElem(r, v, np.array([1.0, 1.0, 1.0, 1.0]))
