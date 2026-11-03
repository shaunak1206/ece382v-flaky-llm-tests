import math

import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem

MU = 1.3


def _elems(a, e, O, I, w):
    return tuple(np.atleast_1d(np.asarray(x, dtype=float)) for x in (a, e, O, I, w))


def _cols(x):
    return np.asarray(x, dtype=float).reshape(3, -1)


def _first(x):
    return float(np.asarray(x, dtype=float).ravel()[0])


def test_zero_angle_circular_position():
    elems = _elems(2.0, 0.0, 0.0, 0.0, 0.0)
    E = 0.7
    r, _ = orbElem2vec(E, MU, orbElem=elems)
    expected = np.array([2.0 * math.cos(E), 2.0 * math.sin(E), 0.0])
    np.testing.assert_allclose(_cols(r)[:, 0], expected, rtol=1e-12, atol=1e-12)


def test_zero_angle_circular_velocity():
    elems = _elems(2.0, 0.0, 0.0, 0.0, 0.0)
    E = 0.7
    _, v = orbElem2vec(E, MU, orbElem=elems)
    speed = math.sqrt(MU / 2.0)
    expected = speed * np.array([-math.sin(E), math.cos(E), 0.0])
    np.testing.assert_allclose(_cols(v)[:, 0], expected, rtol=1e-12, atol=1e-12)


def test_radius_follows_eccentric_relation():
    a, e = 1.5, 0.3
    elems = _elems(a, e, 0.4, 0.8, 1.9)
    E = np.array([0.0, 0.7, 2.0, 3.5, 5.5])
    r, _ = orbElem2vec(E, MU, orbElem=elems)
    radius = np.linalg.norm(_cols(r), axis=0)
    np.testing.assert_allclose(radius, a * (1.0 - e * np.cos(E)), rtol=1e-12)


def test_vis_viva_energy_identity():
    a, e = 1.5, 0.3
    elems = _elems(a, e, 0.4, 0.8, 1.9)
    E = np.array([0.1, 1.0, 2.0, 3.0, 4.5, 6.0])
    r, v = orbElem2vec(E, MU, orbElem=elems)
    r_mag = np.linalg.norm(_cols(r), axis=0)
    v_sq = np.sum(_cols(v) ** 2, axis=0)
    np.testing.assert_allclose(v_sq, MU * (2.0 / r_mag - 1.0 / a), rtol=1e-10)


def test_periapsis_and_apoapsis_radii():
    a, e = 1.5, 0.3
    elems = _elems(a, e, 0.4, 0.8, 1.9)
    r_peri, _ = orbElem2vec(0.0, MU, orbElem=elems)
    r_apo, _ = orbElem2vec(np.pi, MU, orbElem=elems)
    assert _first(np.linalg.norm(_cols(r_peri), axis=0)) == pytest.approx(a * (1 - e))
    assert _first(np.linalg.norm(_cols(r_apo), axis=0)) == pytest.approx(a * (1 + e))


def test_periapsis_velocity_is_perpendicular_and_scaled():
    a, e = 1.5, 0.3
    elems = _elems(a, e, 0.4, 0.8, 1.9)
    r, v = orbElem2vec(0.0, MU, orbElem=elems)
    r0 = _cols(r)[:, 0]
    v0 = _cols(v)[:, 0]
    assert np.dot(r0, v0) == pytest.approx(0.0, abs=1e-12)
    expected_speed = math.sqrt(MU / a) * math.sqrt((1 + e) / (1 - e))
    assert np.linalg.norm(v0) == pytest.approx(expected_speed, rel=1e-10)


def test_angular_momentum_magnitude():
    a, e = 1.5, 0.3
    elems = _elems(a, e, 0.4, 0.8, 1.9)
    E = np.array([0.4, 1.7, 3.0, 5.2])
    r, v = orbElem2vec(E, MU, orbElem=elems)
    h = np.cross(_cols(r).T, _cols(v).T)
    expected = math.sqrt(MU * a * (1 - e ** 2))
    np.testing.assert_allclose(np.linalg.norm(h, axis=1), expected, rtol=1e-10)


def test_orbit_normal_follows_node_and_inclination():
    O, I = 1.1, 0.6
    elems = _elems(1.7, 0.25, O, I, 2.3)
    E = np.array([0.3, 2.2, 4.0])
    r, v = orbElem2vec(E, MU, orbElem=elems)
    h = np.cross(_cols(r).T, _cols(v).T)
    h_hat = h / np.linalg.norm(h, axis=1, keepdims=True)
    expected = np.array([math.sin(O) * math.sin(I), -math.cos(O) * math.sin(I), math.cos(I)])
    np.testing.assert_allclose(h_hat, np.tile(expected, (3, 1)), atol=1e-10)


def test_periapsis_direction_follows_pericenter_argument():
    O, I, w = 0.8, 0.5, 1.4
    elems = _elems(1.0, 0.4, O, I, w)
    r, _ = orbElem2vec(0.0, MU, orbElem=elems)
    r0 = _cols(r)[:, 0]
    p_hat = r0 / np.linalg.norm(r0)
    expected = np.array([
        math.cos(O) * math.cos(w) - math.sin(O) * math.cos(I) * math.sin(w),
        math.sin(O) * math.cos(w) + math.cos(O) * math.cos(I) * math.sin(w),
        math.sin(I) * math.sin(w),
    ])
    np.testing.assert_allclose(p_hat, expected, atol=1e-12)


def test_eccentric_anomaly_periodic_in_two_pi():
    elems = _elems(1.2, 0.35, 0.3, 0.9, 0.6)
    E = 1.1
    r1, v1 = orbElem2vec(E, MU, orbElem=elems)
    r2, v2 = orbElem2vec(E + 2 * np.pi, MU, orbElem=elems)
    np.testing.assert_allclose(r2, r1, atol=1e-10)
    np.testing.assert_allclose(v2, v1, atol=1e-10)


def test_vector_anomalies_match_scalar_calls():
    elems = _elems(1.2, 0.35, 0.3, 0.9, 0.6)
    E = np.array([0.2, 1.0, 2.5, 4.0])
    r_vec, v_vec = orbElem2vec(E, MU, orbElem=elems)
    assert _cols(r_vec).shape == (3, 4)
    for k, Ek in enumerate(E):
        r_k, v_k = orbElem2vec(Ek, MU, orbElem=elems)
        np.testing.assert_allclose(_cols(r_vec)[:, k], _cols(r_k)[:, 0], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(_cols(v_vec)[:, k], _cols(v_k)[:, 0], rtol=1e-12, atol=1e-14)


def test_per_body_vectors_match_individual_calls():
    elems = _elems([1.0, 2.0, 0.8], [0.1, 0.4, 0.2], [0.3, 1.5, 4.0],
                   [0.2, 0.9, 1.3], [1.0, 2.5, 5.0])
    E = np.array([0.5, 2.0, 3.5])
    mus = np.array([1.0, 2.0, 0.5])
    r_all, v_all = orbElem2vec(E, mus, orbElem=elems)
    assert _cols(r_all).shape == (3, 3)
    for k in range(3):
        sub = tuple(x[k:k + 1] for x in elems)
        r_k, v_k = orbElem2vec(E[k], mus[k], orbElem=sub)
        np.testing.assert_allclose(_cols(r_all)[:, k], _cols(r_k)[:, 0], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(_cols(v_all)[:, k], _cols(v_k)[:, 0], rtol=1e-12, atol=1e-14)


def test_output_shapes():
    elems1 = _elems(1.2, 0.35, 0.3, 0.9, 0.6)
    r1, v1 = orbElem2vec(0.5, MU, orbElem=elems1)
    assert np.shape(r1) == (3, 1)
    assert np.shape(v1) == (3, 1)
    r4, v4 = orbElem2vec(np.linspace(0.1, 3.0, 4), MU, orbElem=elems1)
    assert np.shape(r4) == (3, 4)
    assert np.shape(v4) == (3, 4)
    elems3 = _elems([1.0, 2.0, 0.8], [0.1, 0.4, 0.2], [0.3, 1.5, 4.0],
                    [0.2, 0.9, 1.3], [1.0, 2.5, 5.0])
    r3, v3 = orbElem2vec(np.array([0.5, 2.0, 3.5]), np.array([1.0, 2.0, 0.5]), orbElem=elems3)
    assert np.shape(r3) == (3, 3)
    assert np.shape(v3) == (3, 3)


def test_ab_input_matches_orbelem_input():
    elems = _elems(1.2, 0.35, 0.3, 0.9, 0.6)
    E = 1.4
    r1, v1 = orbElem2vec(E, MU, orbElem=elems)
    r2, v2 = orbElem2vec(E, MU, AB=calcAB(*elems))
    np.testing.assert_allclose(r2, r1, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(v2, v1, rtol=1e-9, atol=1e-12)

    elems3 = _elems([1.0, 2.0, 0.8], [0.1, 0.4, 0.2], [0.3, 1.5, 4.0],
                    [0.2, 0.9, 1.3], [1.0, 2.5, 5.0])
    E3 = np.array([0.5, 2.0, 3.5])
    mus3 = np.array([1.0, 2.0, 0.5])
    r3, v3 = orbElem2vec(E3, mus3, orbElem=elems3)
    r4, v4 = orbElem2vec(E3, mus3, AB=calcAB(*elems3))
    np.testing.assert_allclose(r4, r3, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(v4, v3, rtol=1e-9, atol=1e-12)


def test_return_ab_gives_calcab_matrices():
    elems = _elems(1.2, 0.35, 0.3, 0.9, 0.6)
    out = orbElem2vec(1.4, MU, orbElem=elems, returnAB=True)
    assert len(out) == 3
    r, v, (A, B) = out
    A_ref, B_ref = calcAB(*elems)
    np.testing.assert_allclose(A, A_ref)
    np.testing.assert_allclose(B, B_ref)


def test_default_returns_position_and_velocity_only():
    elems = _elems(1.2, 0.35, 0.3, 0.9, 0.6)
    out = orbElem2vec(1.4, MU, orbElem=elems)
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_round_trip_single_body_through_vec2orbelem():
    a, e, O, I, w, E = 1.7, 0.25, 1.1, 0.6, 2.3, 1.9
    r, v = orbElem2vec(E, MU, orbElem=_elems(a, e, O, I, w))
    a_o, e_o, E_o, O_o, I_o, w_o, P_o, _ = vec2orbElem(r, v, MU)
    assert _first(a_o) == pytest.approx(a, rel=1e-8)
    assert _first(e_o) == pytest.approx(e, rel=1e-8)
    assert _first(E_o) == pytest.approx(E, rel=1e-8)
    assert _first(O_o) == pytest.approx(O, rel=1e-8)
    assert _first(I_o) == pytest.approx(I, rel=1e-8)
    assert _first(w_o) == pytest.approx(w, rel=1e-8)
    assert _first(P_o) == pytest.approx(2 * np.pi * np.sqrt(a ** 3 / MU), rel=1e-8)


def test_round_trip_multi_body_through_vec2orbelem():
    a = np.array([1.0, 2.0, 0.8])
    e = np.array([0.1, 0.4, 0.2])
    O = np.array([0.3, 1.5, 4.0])
    I = np.array([0.2, 0.9, 1.3])
    w = np.array([1.0, 2.5, 5.0])
    E = np.array([0.5, 2.0, 3.5])
    mus = np.array([1.0, 2.0, 0.5])
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    a_o, e_o, E_o, O_o, I_o, w_o, _, _ = vec2orbElem(r, v, mus)
    np.testing.assert_allclose(a_o, a, rtol=1e-8)
    np.testing.assert_allclose(e_o, e, rtol=1e-8)
    np.testing.assert_allclose(E_o, E, rtol=1e-8)
    np.testing.assert_allclose(O_o, O, rtol=1e-8)
    np.testing.assert_allclose(I_o, I, rtol=1e-8)
    np.testing.assert_allclose(w_o, w, rtol=1e-8)


def test_requires_orbelem_or_ab():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, MU)


def test_rejects_orbelem_and_ab_together():
    elems = _elems(1.2, 0.35, 0.3, 0.9, 0.6)
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, MU, orbElem=elems, AB=calcAB(*elems))


def test_scalar_anomaly_rejects_vector_mu():
    elems = _elems(1.2, 0.35, 0.3, 0.9, 0.6)
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=elems)


def test_vector_anomaly_rejects_mismatched_mu_size():
    elems = _elems(1.2, 0.35, 0.3, 0.9, 0.6)
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0]), orbElem=elems)


def test_scalar_mu_rejects_multi_element_elements():
    elems = _elems([1.0, 2.0], [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8])
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2, 0.3]), MU, orbElem=elems)


def test_vector_bodies_reject_single_element_elements():
    elems = _elems(1.2, 0.35, 0.3, 0.9, 0.6)
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0]), orbElem=elems)
