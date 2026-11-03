import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem


def _elems(a, e, O, I, w):
    """Build 1-element (or n-element) float arrays for orbital elements."""
    return tuple(np.atleast_1d(np.array(x, dtype=float)) for x in (a, e, O, I, w))


def _wrap_diff(x, y):
    d = np.mod(np.asarray(x) - np.asarray(y) + np.pi, 2 * np.pi) - np.pi
    return d


def test_circular_orbit_at_zero_anomaly():
    orbElem = _elems(1.0, 0.0, 0.0, 0.0, 0.0)
    r, v = orbElem2vec(0.0, 1.0, orbElem=orbElem)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.allclose(r[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(v[:, 0], [0.0, 1.0, 0.0], atol=1e-12)


def test_circular_orbit_quarter_period():
    orbElem = _elems(1.0, 0.0, 0.0, 0.0, 0.0)
    r, v = orbElem2vec(np.pi / 2, 1.0, orbElem=orbElem)
    assert np.allclose(r[:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(v[:, 0], [-1.0, 0.0, 0.0], atol=1e-12)


def test_periapsis_radius_and_speed():
    a, e, mu = 3.0, 0.4, 2.5
    orbElem = _elems(a, e, 0.3, 0.6, 0.9)
    r, v = orbElem2vec(0.0, mu, orbElem=orbElem)
    assert np.linalg.norm(r[:, 0]) == pytest.approx(a * (1 - e), rel=1e-12)
    expected_speed = np.sqrt(mu * (1 + e) / (a * (1 - e)))
    assert np.linalg.norm(v[:, 0]) == pytest.approx(expected_speed, rel=1e-12)


def test_apoapsis_radius():
    a, e, mu = 3.0, 0.4, 2.5
    orbElem = _elems(a, e, 0.3, 0.6, 0.9)
    r, v = orbElem2vec(np.pi, mu, orbElem=orbElem)
    assert np.linalg.norm(r[:, 0]) == pytest.approx(a * (1 + e), rel=1e-12)
    expected_speed = np.sqrt(mu * (1 - e) / (a * (1 + e)))
    assert np.linalg.norm(v[:, 0]) == pytest.approx(expected_speed, rel=1e-12)


def test_vector_E_scalar_mu_shapes_and_radius():
    a, e, mu = 2.0, 0.3, 1.0
    orbElem = _elems(a, e, 0.5, 0.7, 1.1)
    E = np.linspace(0.1, 6.0, 7)
    r, v = orbElem2vec(E, mu, orbElem=orbElem)
    assert r.shape == (3, 7)
    assert v.shape == (3, 7)
    rmag = np.linalg.norm(r, axis=0)
    assert np.allclose(rmag, a * (1 - e * np.cos(E)), rtol=1e-12)


def test_energy_and_angular_momentum_conserved():
    a, e, mu = 2.0, 0.3, 1.7
    orbElem = _elems(a, e, 0.5, 0.7, 1.1)
    E = np.linspace(0.2, 6.0, 9)
    r, v = orbElem2vec(E, mu, orbElem=orbElem)
    rmag = np.linalg.norm(r, axis=0)
    vmag = np.linalg.norm(v, axis=0)
    energy = 0.5 * vmag**2 - mu / rmag
    assert np.allclose(energy, -mu / (2 * a), rtol=1e-10)
    h = np.cross(r.T, v.T)
    hmag = np.linalg.norm(h, axis=1)
    assert np.allclose(hmag, np.sqrt(mu * a * (1 - e**2)), rtol=1e-10)


def test_returnAB_matches_calcAB():
    orbElem = _elems(2.0, 0.3, 0.5, 0.7, 1.1)
    out = orbElem2vec(0.9, 1.0, orbElem=orbElem, returnAB=True)
    assert len(out) == 3
    r, v, (A, B) = out
    A_ref, B_ref = calcAB(*orbElem)
    assert np.allclose(A, A_ref)
    assert np.allclose(B, B_ref)
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)


def test_default_returns_two_outputs():
    orbElem = _elems(2.0, 0.3, 0.5, 0.7, 1.1)
    out = orbElem2vec(0.9, 1.0, orbElem=orbElem)
    assert len(out) == 2


def test_AB_input_matches_orbElem_input():
    orbElem = _elems(2.0, 0.3, 0.5, 0.7, 1.1)
    E = np.array([0.3, 1.2, 4.0])
    r1, v1, AB = orbElem2vec(E, 1.3, orbElem=orbElem, returnAB=True)
    r2, v2 = orbElem2vec(E, 1.3, AB=AB)
    assert np.allclose(r1, r2, atol=1e-10)
    assert np.allclose(v1, v2, atol=1e-10)


def test_multiple_bodies_vector_E_vector_mu():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([0.1, 0.2, 0.3, 0.5])
    O = np.array([0.3, 1.0, 2.0, 4.0])
    I = np.array([0.2, 0.5, 1.0, 2.0])
    w = np.array([0.1, 1.5, 3.0, 5.0])
    mus = np.array([1.0, 2.0, 3.0, 4.0])
    E = np.array([0.5, 1.5, 3.5, 5.5])
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 4)
    assert v.shape == (3, 4)
    rmag = np.linalg.norm(r, axis=0)
    assert np.allclose(rmag, a * (1 - e * np.cos(E)), rtol=1e-10)
    energy = 0.5 * np.linalg.norm(v, axis=0) ** 2 - mus / rmag
    assert np.allclose(energy, -mus / (2 * a), rtol=1e-10)


def test_position_scales_with_semimajor_axis_and_velocity_with_sqrt_mu():
    E = np.array([0.4, 2.0, 5.0])
    r1, v1 = orbElem2vec(E, 1.0, orbElem=_elems(1.0, 0.25, 0.4, 0.8, 2.0))
    r2, v2 = orbElem2vec(E, 4.0, orbElem=_elems(2.0, 0.25, 0.4, 0.8, 2.0))
    assert np.allclose(r2, 2.0 * r1, rtol=1e-10)
    # v ~ sqrt(mu / a): sqrt(4 / 2) = sqrt(2)
    assert np.allclose(v2, np.sqrt(2.0) * v1, rtol=1e-10)


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    orbElem = _elems(2.0, 0.3, 0.5, 0.7, 1.1)
    AB = calcAB(*orbElem)
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, 1.0, orbElem=orbElem, AB=AB)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    orbElem = _elems(2.0, 0.3, 0.5, 0.7, 1.1)
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=orbElem)


def test_orbElem2vec_mu_size_mismatch():
    orbElem = _elems([2.0, 2.0, 2.0], [0.3] * 3, [0.5] * 3, [0.7] * 3, [1.1] * 3)
    E = np.array([0.1, 0.2, 0.3])
    with pytest.raises(AssertionError):
        orbElem2vec(E, np.array([1.0, 1.0]), orbElem=orbElem)


def test_roundtrip_single_orbit():
    a, e, O, I, w, E0, mu = 2.0, 0.3, 0.5, 0.7, 1.1, 0.9, 1.0
    r, v = orbElem2vec(E0, mu, orbElem=_elems(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    assert a2[0] == pytest.approx(a, rel=1e-9)
    assert e2[0] == pytest.approx(e, abs=1e-9)
    assert E2[0] == pytest.approx(E0, abs=1e-9)
    assert O2[0] == pytest.approx(O, abs=1e-9)
    assert I2[0] == pytest.approx(I, abs=1e-9)
    assert w2[0] == pytest.approx(w, abs=1e-9)
    assert P2[0] == pytest.approx(2 * np.pi * np.sqrt(a**3 / mu), rel=1e-9)
    assert tau2[0] == pytest.approx(-(E0 - e * np.sin(E0)) / np.sqrt(mu * a**-3), rel=1e-9)


def test_roundtrip_multiple_bodies_3xn_and_nx3():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    e = np.array([0.1, 0.2, 0.3, 0.5])
    O = np.array([0.3, 1.0, 2.0, 4.0])
    I = np.array([0.2, 0.5, 1.0, 2.0])
    w = np.array([0.1, 1.5, 3.0, 5.0])
    mus = np.array([1.0, 2.0, 3.0, 4.0])
    E = np.array([0.5, 1.5, 3.5, 5.5])
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))

    for rs, vs in ((r, v), (r.T, v.T)):
        a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(rs, vs, mus)
        assert np.allclose(a2, a, rtol=1e-8)
        assert np.allclose(e2, e, atol=1e-8)
        assert np.allclose(_wrap_diff(E2, E), 0, atol=1e-8)
        assert np.allclose(_wrap_diff(O2, O), 0, atol=1e-8)
        assert np.allclose(I2, I, atol=1e-8)
        assert np.allclose(_wrap_diff(w2, w), 0, atol=1e-8)
        assert np.allclose(P2, 2 * np.pi * np.sqrt(a**3 / mus), rtol=1e-8)


def test_roundtrip_same_orbit_many_positions_scalar_mu():
    a, e, O, I, w, mu = 2.5, 0.4, 1.2, 0.9, 2.2, 3.0
    E = np.array([0.3, 1.7, 3.0, 4.4, 5.9])
    r, v = orbElem2vec(E, mu, orbElem=_elems(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    assert np.allclose(a2, a, rtol=1e-8)
    assert np.allclose(e2, e, atol=1e-8)
    assert np.allclose(E2, E, atol=1e-8)
    assert np.allclose(O2, O, atol=1e-8)
    assert np.allclose(I2, I, atol=1e-8)
    assert np.allclose(w2, w, atol=1e-8)


def test_vec2orbElem_known_periapsis_state():
    rs = np.array([[1.0], [0.0], [0.0]])
    vs = np.array([[0.0], [1.0], [0.5]])
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, 1.0)
    assert a[0] == pytest.approx(4.0 / 3.0, rel=1e-12)
    assert e[0] == pytest.approx(0.25, abs=1e-9)
    assert E[0] == pytest.approx(0.0, abs=1e-6)
    assert I[0] == pytest.approx(np.arctan(0.5), abs=1e-9)
    assert P[0] == pytest.approx(2 * np.pi * (4.0 / 3.0) ** 1.5, rel=1e-12)
    assert tau[0] == pytest.approx(0.0, abs=1e-6)


def test_vec2orbElem_output_ranges():
    a, e, O, I, w, mu = 2.0, 0.5, 4.5, 2.5, 5.5, 1.0
    E = np.array([0.1, 2.0, 4.0, 6.0])
    r, v = orbElem2vec(E, mu, orbElem=_elems(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    for ang in (E2, O2, w2):
        assert np.all(ang >= 0) and np.all(ang < 2 * np.pi)
    assert np.all(I2 > 0) and np.all(I2 < np.pi)
    assert np.all(e2 >= 0) and np.all(e2 < 1)


def test_vec2orbElem_rejects_mismatched_sizes():
    rs = np.ones((3, 2))
    vs = np.ones((3, 3))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_wrong_shape_and_mu_size():
    with pytest.raises(AssertionError):
        vec2orbElem(np.ones((2, 6)), np.ones((2, 6)), 1.0)
    rs = np.array([[1.0], [0.0], [0.0]])
    vs = np.array([[0.0], [1.0], [0.5]])
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, np.array([1.0, 1.0]))
