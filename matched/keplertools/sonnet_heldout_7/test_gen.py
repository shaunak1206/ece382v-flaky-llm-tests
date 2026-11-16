import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def _arr(*vals):
    return np.array(vals, dtype=float)


def _single_body_elems(a=1.5, e=0.3, O=0.7, I=0.5, w=1.1):
    return (_arr(a), _arr(e), _arr(O), _arr(I), _arr(w))


def _multi_body_elems():
    a = _arr(1.0, 2.5, 0.8, 5.0)
    e = _arr(0.1, 0.5, 0.25, 0.7)
    O = _arr(0.3, 2.0, 4.0, 5.5)
    I = _arr(0.4, 1.0, 2.0, 0.9)
    w = _arr(0.5, 3.0, 1.2, 6.0)
    return a, e, O, I, w


def test_circular_equatorial_at_E_zero():
    elems = _single_body_elems(a=1.0, e=0.0, O=0.0, I=0.0, w=0.0)
    r, v = orbElem2vec(0.0, 1.0, orbElem=elems)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    np.testing.assert_allclose(r[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [0.0, 1.0, 0.0], atol=1e-12)


def test_circular_equatorial_at_quarter_orbit():
    elems = _single_body_elems(a=1.0, e=0.0, O=0.0, I=0.0, w=0.0)
    r, v = orbElem2vec(np.pi / 2, 1.0, orbElem=elems)
    np.testing.assert_allclose(r[:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [-1.0, 0.0, 0.0], atol=1e-12)


def test_periapsis_distance_and_speed_eccentric_orbit():
    a, e, mu = 2.0, 0.5, 1.0
    elems = _single_body_elems(a=a, e=e, O=0.0, I=0.0, w=0.0)
    r, v = orbElem2vec(np.array([0.0]), mu, orbElem=elems)
    np.testing.assert_allclose(r[:, 0], [a * (1 - e), 0.0, 0.0], atol=1e-12)
    # vis-viva at periapsis
    speed = np.linalg.norm(v[:, 0])
    np.testing.assert_allclose(speed ** 2, mu * (2 / (a * (1 - e)) - 1 / a), rtol=1e-12)
    # velocity at periapsis is perpendicular to r
    assert abs(np.dot(r[:, 0], v[:, 0])) < 1e-12


def test_apoapsis_distance():
    a, e = 2.0, 0.5
    elems = _single_body_elems(a=a, e=e, O=0.0, I=0.0, w=0.0)
    r, v = orbElem2vec(np.array([np.pi]), 1.0, orbElem=elems)
    np.testing.assert_allclose(r[:, 0], [-a * (1 + e), 0.0, 0.0], atol=1e-12)


def test_polar_orbit_geometry():
    a, e = 1.0, 0.4
    elems = _single_body_elems(a=a, e=e, O=0.0, I=np.pi / 2, w=0.0)
    r, v = orbElem2vec(np.array([0.0, np.pi / 2]), 1.0, orbElem=elems)
    assert r.shape == (3, 2)
    np.testing.assert_allclose(r[:, 0], [a * (1 - e), 0.0, 0.0], atol=1e-12)
    b = a * np.sqrt(1 - e ** 2)
    np.testing.assert_allclose(r[:, 1], [-a * e, 0.0, b], atol=1e-12)


def test_single_body_many_positions_shapes_and_invariants():
    a, e, mu = 1.5, 0.3, 1.3
    elems = _single_body_elems(a=a, e=e)
    E = np.linspace(0.1, 6.0, 7)
    r, v = orbElem2vec(E, mu, orbElem=elems)
    assert r.shape == (3, 7)
    assert v.shape == (3, 7)
    rmag = np.linalg.norm(r, axis=0)
    vmag = np.linalg.norm(v, axis=0)
    np.testing.assert_allclose(rmag, a * (1 - e * np.cos(E)), rtol=1e-12)
    # energy conservation
    energy = 0.5 * vmag ** 2 - mu / rmag
    np.testing.assert_allclose(energy, -mu / (2 * a) * np.ones(7), rtol=1e-10)
    # angular momentum magnitude conservation
    h = np.linalg.norm(np.cross(r.T, v.T), axis=1)
    np.testing.assert_allclose(h, np.sqrt(mu * a * (1 - e ** 2)) * np.ones(7), rtol=1e-10)


def test_multiple_bodies_vector_mus_invariants():
    a, e, O, I, w = _multi_body_elems()
    mus = _arr(1.0, 2.0, 0.5, 3.0)
    E = _arr(0.2, 1.5, 3.0, 5.0)
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 4)
    assert v.shape == (3, 4)
    rmag = np.linalg.norm(r, axis=0)
    vmag = np.linalg.norm(v, axis=0)
    np.testing.assert_allclose(rmag, a * (1 - e * np.cos(E)), rtol=1e-10)
    np.testing.assert_allclose(0.5 * vmag ** 2 - mus / rmag, -mus / (2 * a), rtol=1e-9)


def test_AB_input_matches_orbElem_input():
    elems = _single_body_elems()
    E = np.array([0.3, 1.7, 4.2])
    mu = 1.0
    r1, v1 = orbElem2vec(E, mu, orbElem=elems)
    AB = calcAB(*elems)
    r2, v2 = orbElem2vec(E, mu, AB=AB)
    np.testing.assert_allclose(r1, r2, atol=1e-12)
    np.testing.assert_allclose(v1, v2, atol=1e-12)


def test_returnAB_returns_third_output():
    elems = _single_body_elems()
    out = orbElem2vec(np.array([0.5, 1.0]), 1.0, orbElem=elems, returnAB=True)
    assert len(out) == 3
    r, v, (A, B) = out
    A_exp, B_exp = calcAB(*elems)
    np.testing.assert_allclose(A, A_exp)
    np.testing.assert_allclose(B, B_exp)
    out2 = orbElem2vec(np.array([0.5, 1.0]), 1.0, orbElem=elems)
    assert len(out2) == 2


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.5]), 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    elems = _single_body_elems()
    AB = calcAB(*elems)
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.5]), 1.0, orbElem=elems, AB=AB)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    elems = _single_body_elems()
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=elems)


def test_orbElem2vec_mus_size_mismatch():
    elems = _multi_body_elems()
    with pytest.raises(AssertionError):
        orbElem2vec(_arr(0.1, 0.2, 0.3, 0.4), _arr(1.0, 2.0), orbElem=elems)


@pytest.mark.parametrize("I", [0.3, 1.0, 1.5, 2.0, 3.0])
def test_roundtrip_single_body_scalar_mu(I):
    a, e, O, w, mu = 1.7, 0.35, 0.9, 2.2, 1.3
    E_in = np.array([0.4, 2.0, 3.5, 5.5])
    r, v = orbElem2vec(E_in, mu, orbElem=_single_body_elems(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    np.testing.assert_allclose(a2, a * np.ones(4), rtol=1e-9)
    np.testing.assert_allclose(e2, e * np.ones(4), rtol=1e-9)
    np.testing.assert_allclose(E2, E_in, atol=1e-9)
    np.testing.assert_allclose(O2, O * np.ones(4), atol=1e-9)
    np.testing.assert_allclose(I2, I * np.ones(4), atol=1e-9)
    np.testing.assert_allclose(w2, w * np.ones(4), atol=1e-9)


def test_roundtrip_multiple_bodies_vector_mus():
    a, e, O, I, w = _multi_body_elems()
    mus = _arr(1.0, 2.0, 0.5, 3.0)
    E_in = _arr(0.2, 1.5, 3.0, 5.0)
    r, v = orbElem2vec(E_in, mus, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mus)
    np.testing.assert_allclose(a2, a, rtol=1e-8)
    np.testing.assert_allclose(e2, e, rtol=1e-8)
    np.testing.assert_allclose(E2, E_in, atol=1e-8)
    np.testing.assert_allclose(O2, O, atol=1e-8)
    np.testing.assert_allclose(I2, I, atol=1e-8)
    np.testing.assert_allclose(w2, w, atol=1e-8)


def test_vec2orbElem_period_and_tau():
    a, e, mu = 1.7, 0.35, 1.3
    E_in = np.array([0.4, 2.0, 3.5])
    r, v = orbElem2vec(E_in, mu, orbElem=_single_body_elems(a, e))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    np.testing.assert_allclose(P2, 2 * np.pi * np.sqrt(a ** 3 / mu) * np.ones(3), rtol=1e-9)
    expected_tau = -(E_in - e * np.sin(E_in)) / np.sqrt(mu * a ** -3.0)
    np.testing.assert_allclose(tau2, expected_tau, rtol=1e-8, atol=1e-10)


def test_vec2orbElem_angle_ranges_and_output_shapes():
    a, e, O, I, w = _multi_body_elems()
    mus = _arr(1.0, 2.0, 0.5, 3.0)
    E_in = _arr(0.2, 1.5, 3.0, 6.0)
    r, v = orbElem2vec(E_in, mus, orbElem=(a, e, O, I, w))
    out = vec2orbElem(r, v, mus)
    assert len(out) == 8
    for arr in out:
        assert np.asarray(arr).shape == (4,)
    _, _, E2, O2, I2, w2, _, _ = out
    for ang in (E2, O2, w2):
        assert np.all(ang >= 0) and np.all(ang < 2 * np.pi)
    assert np.all(I2 >= 0) and np.all(I2 <= np.pi)


def test_vec2orbElem_accepts_n_by_3_input():
    a, e, O, I, w = _multi_body_elems()
    mus = _arr(1.0, 2.0, 0.5, 3.0)
    E_in = _arr(0.2, 1.5, 3.0, 5.0)
    r, v = orbElem2vec(E_in, mus, orbElem=(a, e, O, I, w))
    out_3xn = vec2orbElem(r, v, mus)
    out_nx3 = vec2orbElem(r.T.copy(), v.T.copy(), mus)
    for x, y in zip(out_3xn, out_nx3):
        np.testing.assert_allclose(x, y, rtol=1e-12)


def test_vec2orbElem_size_validation():
    rs = np.ones((3, 2))
    vs = np.ones((3, 3))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)
    with pytest.raises(AssertionError):
        vec2orbElem(np.ones(4), np.ones(4), 1.0)
