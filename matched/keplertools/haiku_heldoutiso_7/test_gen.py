import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem

RTOL = 1e-9
ATOL = 1e-9


def _multi_elements():
    """Five well-separated bodies with distinct elements, all in valid ranges."""
    a = np.array([1.0, 2.5, 0.7, 4.2, 1.3])
    e = np.array([0.1, 0.4, 0.6, 0.2, 0.85])
    O = np.array([0.3, 1.1, 2.0, 4.0, 5.5])
    I = np.array([0.2, 0.9, 1.7, 2.5, 1.2])
    w = np.array([0.5, 2.2, 3.3, 0.1, 4.4])
    E = np.array([0.4, 1.5, 2.7, 4.1, 5.9])
    mus = np.array([1.0, 2.0, 0.5, 3.0, 1.5])
    return a, e, O, I, w, E, mus


def _single_elements():
    """One body described by size-1 arrays (required by calcAB)."""
    return (np.array([1.3]), np.array([0.35]), np.array([0.8]),
            np.array([0.9]), np.array([2.1]))


def _assert_elements(result, a, e, E, O, I, w, rtol=1e-8, atol=1e-8):
    ra, re, rE, rO, rI, rw = result[:6]
    pairs = [(ra, a), (re, e), (rE, E), (rO, O), (rI, I), (rw, w)]
    for got, expected in pairs:
        np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# orbElem2vec
# ---------------------------------------------------------------------------

def test_orbElem2vec_circular_equatorial_at_E0():
    orb = (np.array([1.0]), np.array([0.0]), np.array([0.0]),
           np.array([0.0]), np.array([0.0]))
    r, v = orbElem2vec(0.0, 1.0, orbElem=orb)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    np.testing.assert_allclose(r[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [0.0, 1.0, 0.0], atol=1e-12)


def test_orbElem2vec_circular_equatorial_at_quarter_orbit():
    orb = (np.array([1.0]), np.array([0.0]), np.array([0.0]),
           np.array([0.0]), np.array([0.0]))
    r, v = orbElem2vec(np.pi / 2, 1.0, orbElem=orb)
    np.testing.assert_allclose(r[:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [-1.0, 0.0, 0.0], atol=1e-12)


def test_orbElem2vec_single_body_returns_3xn_for_array_E():
    orb = _single_elements()
    E = np.linspace(0, 2 * np.pi, 9)[:-1]  # 8 anomalies
    r, v = orbElem2vec(E, 1.7, orbElem=orb)
    assert r.shape == (3, 8)
    assert v.shape == (3, 8)


def test_orbElem2vec_single_body_columns_match_scalar_calls():
    orb = _single_elements()
    E = np.array([0.2, 1.0, 3.0, 5.0])
    mu = 1.7
    r, v = orbElem2vec(E, mu, orbElem=orb)
    for k, Ek in enumerate(E):
        rk, vk = orbElem2vec(float(Ek), mu, orbElem=orb)
        np.testing.assert_allclose(r[:, k], rk[:, 0], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(v[:, k], vk[:, 0], rtol=RTOL, atol=ATOL)


def test_orbElem2vec_multiple_bodies_match_per_body_calls():
    a, e, O, I, w, E, mus = _multi_elements()
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 5)
    assert v.shape == (3, 5)
    for i in range(5):
        orb_i = (a[i:i + 1], e[i:i + 1], O[i:i + 1], I[i:i + 1], w[i:i + 1])
        ri, vi = orbElem2vec(float(E[i]), float(mus[i]), orbElem=orb_i)
        np.testing.assert_allclose(r[:, i], ri[:, 0], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(v[:, i], vi[:, 0], rtol=RTOL, atol=ATOL)


def test_orbElem2vec_speed_satisfies_vis_viva():
    a, e, O, I, w, E, mus = _multi_elements()
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    speed2 = np.sum(v ** 2, axis=0)
    rmag = np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(speed2 / 2 - mus / rmag, -mus / (2 * a),
                               rtol=1e-10, atol=1e-12)


def test_orbElem2vec_angular_momentum_magnitude():
    a, e, O, I, w, E, mus = _multi_elements()
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    h = np.cross(r, v, axis=0)
    hmag = np.linalg.norm(h, axis=0)
    np.testing.assert_allclose(hmag, np.sqrt(mus * a * (1 - e ** 2)),
                               rtol=1e-10, atol=1e-12)


def test_orbElem2vec_radius_matches_focal_distance():
    a, e, O, I, w, E, mus = _multi_elements()
    r, _ = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    rmag = np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(rmag, a * (1 - e * np.cos(E)),
                               rtol=1e-10, atol=1e-12)


def test_orbElem2vec_AB_input_matches_orbElem_input():
    a, e, O, I, w, E, mus = _multi_elements()
    AB = calcAB(a, e, O, I, w)
    r1, v1 = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    r2, v2 = orbElem2vec(E, mus, AB=AB)
    np.testing.assert_allclose(r2, r1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(v2, v1, rtol=RTOL, atol=ATOL)


def test_orbElem2vec_returnAB_gives_three_outputs():
    a, e, O, I, w, E, mus = _multi_elements()
    orb = (a, e, O, I, w)
    out = orbElem2vec(E, mus, orbElem=orb, returnAB=True)
    assert len(out) == 3
    r, v, (A, B) = out
    A_ref, B_ref = calcAB(*orb)
    assert A.shape == (3, 5)
    assert B.shape == (3, 5)
    np.testing.assert_allclose(A, A_ref, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(B, B_ref, rtol=RTOL, atol=ATOL)

    default = orbElem2vec(E, mus, orbElem=orb)
    assert len(default) == 2


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    orb = _single_elements()
    AB = calcAB(*orb)
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, 1.0, orbElem=orb, AB=AB)


def test_orbElem2vec_scalar_E_with_array_mu_raises():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=_single_elements())


def test_orbElem2vec_mu_size_mismatch_raises():
    a, e, O, I, w, _, _ = _multi_elements()
    E = np.array([0.1, 0.2, 0.3])
    mus = np.array([1.0, 2.0])
    orb = (a[:3], e[:3], O[:3], I[:3], w[:3])
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, orbElem=orb)


# ---------------------------------------------------------------------------
# vec2orbElem
# ---------------------------------------------------------------------------

def test_vec2orbElem_recovers_elements_single_body_scalar_mu():
    orb = _single_elements()
    a0, e0, O0, I0, w0 = orb
    E = np.array([0.3, 1.2, 2.5, 4.0, 5.5])
    mu = 1.7
    r, v = orbElem2vec(E, mu, orbElem=orb)
    result = vec2orbElem(r, v, mu)
    m = E.size
    _assert_elements(result,
                     np.full(m, a0[0]), np.full(m, e0[0]), E,
                     np.full(m, O0[0]), np.full(m, I0[0]), np.full(m, w0[0]))
    Erec = result[2]
    assert np.all(Erec >= 0) and np.all(Erec < 2 * np.pi)


def test_vec2orbElem_recovers_elements_multiple_bodies_vector_mu():
    a, e, O, I, w, E, mus = _multi_elements()
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    result = vec2orbElem(r, v, mus)
    _assert_elements(result, a, e, E, O, I, w)


def test_vec2orbElem_accepts_nx3_layout():
    a, e, O, I, w, E, mus = _multi_elements()
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    result = vec2orbElem(r.T, v.T, mus)
    _assert_elements(result, a, e, E, O, I, w)


def test_vec2orbElem_recovers_retrograde_inclination():
    orb = (np.array([2.0]), np.array([0.3]), np.array([1.0]),
           np.array([2.5]), np.array([0.7]))
    E = np.array([0.5, 2.0, 4.5])
    mu = 3.0
    r, v = orbElem2vec(E, mu, orbElem=orb)
    result = vec2orbElem(r, v, mu)
    m = E.size
    _assert_elements(result,
                     np.full(m, 2.0), np.full(m, 0.3), E,
                     np.full(m, 1.0), np.full(m, 2.5), np.full(m, 0.7))


def test_vec2orbElem_period_and_periapsis_time_consistent_with_kepler():
    a, e, O, I, w, E, mus = _multi_elements()
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    result = vec2orbElem(r, v, mus)
    Erec, P, tau = result[2], result[6], result[7]

    P_expected = 2 * np.pi * np.sqrt(a ** 3 / mus)
    np.testing.assert_allclose(P, P_expected, rtol=1e-8)

    n = np.sqrt(mus / a ** 3)
    M = Erec - e * np.sin(Erec)
    np.testing.assert_allclose(tau, -M / n, rtol=1e-8, atol=1e-8)


def test_vec2orbElem_rejects_vs_with_different_size():
    rs = np.ones((3, 3))
    vs = np.ones((3, 4))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_size_not_multiple_of_three():
    rs = np.ones(4)
    vs = np.ones(4)
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_mu_size_mismatch():
    rs = np.ones((3, 3))
    vs = np.ones((3, 3))
    mus = np.array([1.0, 2.0])
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, mus)


def test_vec2orbElem_rejects_three_dimensional_input():
    rs = np.ones((3, 2, 2))
    vs = np.ones((3, 2, 2))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)
