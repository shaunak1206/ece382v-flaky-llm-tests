import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


E_SAMPLES = np.array([0.5, 1.2, 2.5, 4.0, 5.5])


def _elems(a=2.0, e=0.3, O=0.7, I=0.4, w=1.1):
    """Scalar orbital elements as numpy scalars (calcAB needs .size)."""
    return (np.float64(a), np.float64(e), np.float64(O), np.float64(I), np.float64(w))


def _assert_angles_equal(actual, expected, atol=1e-9):
    """Compare angles modulo 2*pi via their cosines and sines."""
    np.testing.assert_allclose(np.cos(actual), np.cos(expected), atol=atol)
    np.testing.assert_allclose(np.sin(actual), np.sin(expected), atol=atol)


# ---------------------------------------------------------------------------
# orbElem2vec: shapes and values
# ---------------------------------------------------------------------------

def test_orbElem2vec_scalar_E_returns_3x1_vectors():
    r, v = orbElem2vec(0.5, 1.0, orbElem=_elems())
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)


def test_orbElem2vec_vector_E_scalar_mu_returns_3xn():
    r, v = orbElem2vec(E_SAMPLES, 1.0, orbElem=_elems())
    assert r.shape == (3, E_SAMPLES.size)
    assert v.shape == (3, E_SAMPLES.size)


def test_orbElem2vec_vector_E_vector_mu_returns_3xn():
    n = 3
    a = np.array([1.5, 2.0, 3.0])
    e = np.array([0.1, 0.3, 0.5])
    O = np.array([0.2, 1.0, 2.5])
    I = np.array([0.3, 0.9, 1.4])
    w = np.array([0.5, 2.0, 4.0])
    E = np.array([0.4, 1.7, 3.1])
    mus = np.array([0.8, 1.0, 2.5])
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, n)
    assert v.shape == (3, n)


def test_orbElem2vec_periapsis_position_and_velocity():
    a, e = 2.0, 0.5
    mu = 1.0
    r, v = orbElem2vec(0.0, mu, orbElem=_elems(a=a, e=e, O=0.0, I=0.0, w=0.0))
    # At E=0 with O=I=w=0 the periapsis lies on +x at distance a(1-e).
    np.testing.assert_allclose(r[:, 0], [a * (1 - e), 0.0, 0.0], atol=1e-12)
    # Speed at periapsis from vis-viva: sqrt(mu/a * (1+e)/(1-e)), along +y.
    expected_speed = np.sqrt(mu / a * (1 + e) / (1 - e))
    np.testing.assert_allclose(v[:, 0], [0.0, expected_speed, 0.0], atol=1e-12)


def test_orbElem2vec_apoapsis_position():
    a, e = 2.0, 0.5
    r, _ = orbElem2vec(np.pi, 1.0, orbElem=_elems(a=a, e=e, O=0.0, I=0.0, w=0.0))
    np.testing.assert_allclose(r[:, 0], [-a * (1 + e), 0.0, 0.0], atol=1e-12)


def test_orbElem2vec_circular_orbit_has_constant_radius():
    a = 2.5
    r, _ = orbElem2vec(E_SAMPLES, 1.0, orbElem=_elems(a=a, e=0.0))
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), a, rtol=1e-12)


def test_orbElem2vec_conserves_specific_energy():
    a, e, mu = 2.0, 0.3, 1.3
    r, v = orbElem2vec(E_SAMPLES, mu, orbElem=_elems(a=a, e=e))
    energy = 0.5 * np.sum(v ** 2, axis=0) - mu / np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(energy, -mu / (2 * a), rtol=1e-10)


def test_orbElem2vec_angular_momentum_magnitude():
    a, e, mu = 2.0, 0.3, 1.3
    r, v = orbElem2vec(E_SAMPLES, mu, orbElem=_elems(a=a, e=e))
    h = np.cross(r.T, v.T)  # shape (n, 3)
    expected = np.sqrt(mu * a * (1 - e ** 2))
    np.testing.assert_allclose(np.linalg.norm(h, axis=1), expected, rtol=1e-10)


def test_orbElem2vec_vector_mu_matches_single_body_calls():
    a = np.array([1.5, 2.0, 3.0])
    e = np.array([0.1, 0.3, 0.5])
    O = np.array([0.2, 1.0, 2.5])
    I = np.array([0.3, 0.9, 1.4])
    w = np.array([0.5, 2.0, 4.0])
    E = np.array([0.4, 1.7, 3.1])
    mus = np.array([0.8, 1.0, 2.5])

    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))

    for i in range(E.size):
        r_i, v_i = orbElem2vec(
            E[i], mus[i], orbElem=(a[i], e[i], O[i], I[i], w[i])
        )
        np.testing.assert_allclose(r[:, i], r_i[:, 0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(v[:, i], v_i[:, 0], rtol=1e-10, atol=1e-12)


def test_orbElem2vec_AB_input_matches_orbElem_input():
    elems = _elems()
    A, B = calcAB(*elems)
    r1, v1 = orbElem2vec(E_SAMPLES, 1.0, orbElem=elems)
    r2, v2 = orbElem2vec(E_SAMPLES, 1.0, AB=(A, B))
    np.testing.assert_allclose(r2, r1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v2, v1, rtol=1e-10, atol=1e-12)


def test_orbElem2vec_returnAB_returns_calcAB_output():
    elems = _elems()
    out = orbElem2vec(E_SAMPLES, 1.0, orbElem=elems, returnAB=True)
    assert len(out) == 3
    r, v, AB = out
    A_expected, B_expected = calcAB(*elems)
    np.testing.assert_allclose(AB[0], A_expected)
    np.testing.assert_allclose(AB[1], B_expected)


# ---------------------------------------------------------------------------
# orbElem2vec: input validation
# ---------------------------------------------------------------------------

def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    A, B = calcAB(*_elems())
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, 1.0, orbElem=_elems(), AB=(A, B))


def test_orbElem2vec_scalar_E_rejects_array_mu():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=_elems())


def test_orbElem2vec_vector_mu_size_mismatch_raises():
    a = np.linspace(1.5, 3.0, 5)
    e = np.linspace(0.1, 0.4, 5)
    O = np.linspace(0.2, 2.0, 5)
    I = np.linspace(0.3, 1.0, 5)
    w = np.linspace(0.5, 2.5, 5)
    with pytest.raises(AssertionError):
        orbElem2vec(E_SAMPLES, np.ones(3), orbElem=(a, e, O, I, w))


# ---------------------------------------------------------------------------
# vec2orbElem: round trips and derived quantities
# ---------------------------------------------------------------------------

def test_vec2orbElem_round_trip_recovers_elements():
    a, e, O, I, w = 2.0, 0.3, 0.7, 0.4, 1.1
    r, v = orbElem2vec(E_SAMPLES, 1.0, orbElem=_elems(a, e, O, I, w))

    a_o, e_o, E_o, O_o, I_o, w_o, P_o, tau_o = vec2orbElem(r, v, 1.0)

    np.testing.assert_allclose(a_o, a, rtol=1e-10)
    np.testing.assert_allclose(e_o, e, rtol=1e-10)
    np.testing.assert_allclose(I_o, I, atol=1e-9)
    _assert_angles_equal(O_o, O)
    _assert_angles_equal(w_o, w)
    _assert_angles_equal(E_o, E_SAMPLES)


def test_vec2orbElem_returns_angles_in_zero_to_two_pi():
    r, v = orbElem2vec(E_SAMPLES, 1.0, orbElem=_elems())
    _, _, E_o, O_o, I_o, w_o, _, _ = vec2orbElem(r, v, 1.0)
    for angle in (E_o, O_o, w_o):
        assert np.all(angle >= 0.0)
        assert np.all(angle < 2 * np.pi)
    assert np.all((I_o >= 0.0) & (I_o <= np.pi))


def test_vec2orbElem_nx3_layout_matches_3xn_layout():
    r, v = orbElem2vec(E_SAMPLES, 1.0, orbElem=_elems())
    out_3xn = vec2orbElem(r, v, 1.0)
    out_nx3 = vec2orbElem(r.T.copy(), v.T.copy(), 1.0)
    for got, expected in zip(out_nx3, out_3xn):
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_vec2orbElem_vector_mus_round_trip():
    a = np.array([1.5, 2.0, 3.0])
    e = np.array([0.1, 0.3, 0.5])
    O = np.array([0.2, 1.0, 2.5])
    I = np.array([0.3, 0.9, 1.4])
    w = np.array([0.5, 2.0, 4.0])
    E = np.array([0.4, 1.7, 3.1])
    mus = np.array([0.8, 1.0, 2.5])

    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    a_o, e_o, E_o, O_o, I_o, w_o, _, _ = vec2orbElem(r, v, mus)

    np.testing.assert_allclose(a_o, a, rtol=1e-10)
    np.testing.assert_allclose(e_o, e, rtol=1e-10)
    np.testing.assert_allclose(I_o, I, atol=1e-9)
    _assert_angles_equal(O_o, O)
    _assert_angles_equal(w_o, w)
    _assert_angles_equal(E_o, E)


def test_vec2orbElem_period_and_time_of_periapsis():
    a, e, mu = 2.0, 0.3, 1.0
    r, v = orbElem2vec(E_SAMPLES, mu, orbElem=_elems(a=a, e=e))
    _, _, E_o, _, _, _, P_o, tau_o = vec2orbElem(r, v, mu)

    expected_period = 2 * np.pi * np.sqrt(a ** 3 / mu)
    np.testing.assert_allclose(P_o, expected_period, rtol=1e-10)

    # tau = -(E - e sin E) / n, with n = sqrt(mu / a^3)
    n = np.sqrt(mu / a ** 3)
    np.testing.assert_allclose(tau_o * n, -(E_o - e * np.sin(E_o)), atol=1e-10)


# ---------------------------------------------------------------------------
# vec2orbElem: input validation
# ---------------------------------------------------------------------------

def test_vec2orbElem_rejects_size_not_multiple_of_three():
    rs = np.ones((2, 4))
    vs = np.ones((2, 4))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_mismatched_rs_vs_sizes():
    rs = np.ones((3, 2))
    vs = np.ones((3, 3))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_mus_of_wrong_size():
    rs = np.ones((3, 4))
    vs = np.ones((3, 4))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, np.ones(3))


def test_vec2orbElem_rejects_three_dimensional_input():
    rs = np.ones((3, 2, 2))
    vs = np.ones((3, 2, 2))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)
