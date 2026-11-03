"""Unit tests for keplertools.fun.orbElem2vec and keplertools.fun.vec2orbElem."""
import numpy as np
import pytest

from keplertools import fun

SMA = 2.5
ECC = 0.3
RAAN = 0.7
INC = 1.1
ARGP = 2.2
MU = 1.0
TWO_PI = 2 * np.pi


def _orb(a=SMA, e=ECC, O=RAAN, I=INC, w=ARGP):
    """Single-orbit element tuple with size-1 arrays (the shape calcAB expects)."""
    return (np.array([a], dtype=float), np.array([e], dtype=float),
            np.array([O], dtype=float), np.array([I], dtype=float),
            np.array([w], dtype=float))


def _multi():
    """Three different orbits as size-3 element arrays."""
    return (np.array([2.5, 1.2, 4.0]), np.array([0.3, 0.1, 0.6]),
            np.array([0.7, 2.0, 5.0]), np.array([1.1, 0.4, 2.5]),
            np.array([2.2, 4.0, 0.5]))


def _states(E, mu=MU, orb=None):
    """Position and velocity (3 x n) of one orbit at several eccentric anomalies."""
    if orb is None:
        orb = _orb()
    return fun.orbElem2vec(np.atleast_1d(np.asarray(E, dtype=float)), mu, orbElem=orb)


def test_circular_orbit_radius_equals_semimajor_axis():
    E = np.array([0.0, 1.0, 2.5, 4.0])
    r, _ = _states(E, orb=_orb(e=0.0))
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), SMA, rtol=1e-12)


def test_circular_orbit_speed_is_sqrt_mu_over_a():
    E = np.array([0.0, 1.0, 2.5, 4.0])
    _, v = _states(E, orb=_orb(e=0.0))
    np.testing.assert_allclose(np.linalg.norm(v, axis=0), np.sqrt(MU / SMA), rtol=1e-12)


def test_periapsis_and_apoapsis_radii():
    r, _ = _states(np.array([0.0, np.pi]))
    radius = np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(radius, [SMA * (1 - ECC), SMA * (1 + ECC)], rtol=1e-12)


def test_reference_position_with_zero_angles():
    E = np.array([0.3, 1.2, 4.0])
    r, _ = _states(E, orb=_orb(O=0.0, I=0.0, w=0.0))
    expected = np.vstack([SMA * (np.cos(E) - ECC),
                          SMA * np.sqrt(1 - ECC ** 2) * np.sin(E),
                          np.zeros_like(E)])
    np.testing.assert_allclose(r, expected, rtol=1e-12, atol=1e-14)


def test_reference_velocity_with_zero_angles():
    E = np.array([0.3, 1.2, 4.0])
    _, v = _states(E, orb=_orb(O=0.0, I=0.0, w=0.0))
    n = np.sqrt(MU / SMA ** 3)
    b = SMA * np.sqrt(1 - ECC ** 2)
    denom = 1 - ECC * np.cos(E)
    expected = np.vstack([-n * SMA * np.sin(E) / denom,
                          n * b * np.cos(E) / denom,
                          np.zeros_like(E)])
    np.testing.assert_allclose(v, expected, rtol=1e-12, atol=1e-14)


def test_zero_inclination_keeps_motion_in_orbital_plane():
    E = np.linspace(0.0, TWO_PI, 9)
    r, v = _states(E, orb=_orb(I=0.0))
    np.testing.assert_allclose(r[2], 0.0, atol=1e-14)
    np.testing.assert_allclose(v[2], 0.0, atol=1e-14)


def test_vis_viva_holds_along_orbit():
    E = np.linspace(0.1, 6.0, 12)
    r, v = _states(E)
    radius = np.linalg.norm(r, axis=0)
    speed_sq = np.sum(v ** 2, axis=0)
    np.testing.assert_allclose(speed_sq, MU * (2.0 / radius - 1.0 / SMA), rtol=1e-10)


def test_angular_momentum_magnitude_matches_orbit_elements():
    E = np.linspace(0.1, 6.0, 12)
    r, v = _states(E)
    h = np.cross(r.T, v.T)
    np.testing.assert_allclose(np.linalg.norm(h, axis=1),
                               np.sqrt(MU * SMA * (1 - ECC ** 2)), rtol=1e-10)


def test_returnAB_gives_three_outputs_matching_calcAB():
    orb = _orb()
    E = np.array([0.2, 1.5, 3.3])
    out = fun.orbElem2vec(E, MU, orbElem=orb, returnAB=True)
    assert len(out) == 3
    r, v, (A, B) = out
    A_ref, B_ref = fun.calcAB(*orb)
    np.testing.assert_allclose(A, A_ref)
    np.testing.assert_allclose(B, B_ref)
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)


def test_AB_input_reproduces_orbElem_input():
    orb = _orb()
    E = np.array([0.2, 1.5, 3.3])
    r_ref, v_ref = fun.orbElem2vec(E, MU, orbElem=orb)
    r, v = fun.orbElem2vec(E, MU, AB=fun.calcAB(*orb))
    np.testing.assert_allclose(r, r_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v, v_ref, rtol=1e-10, atol=1e-12)


def test_multi_body_call_matches_single_body_calls():
    orb = _multi()
    E = np.array([0.5, 2.0, 4.5])
    mus = np.array([1.0, 2.0, 0.5])
    r, v = fun.orbElem2vec(E, mus, orbElem=orb)
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    for j in range(3):
        orb_j = tuple(np.array([x[j]]) for x in orb)
        r_j, v_j = fun.orbElem2vec(np.array([E[j]]), float(mus[j]), orbElem=orb_j)
        np.testing.assert_allclose(r[:, j], r_j[:, 0], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(v[:, j], v_j[:, 0], rtol=1e-12, atol=1e-14)


def test_scalar_E_with_array_mus_raises():
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=_orb())


def test_mismatched_E_and_mus_sizes_raise():
    with pytest.raises(AssertionError):
        fun.orbElem2vec(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0]), orbElem=_orb())


def test_missing_orbElem_and_AB_raises():
    with pytest.raises(AssertionError, match="must supply either"):
        fun.orbElem2vec(0.5, 1.0)


def test_orbElem_and_AB_together_raise():
    orb = _orb()
    with pytest.raises(AssertionError, match="can only set orbElem or AB"):
        fun.orbElem2vec(0.5, 1.0, orbElem=orb, AB=fun.calcAB(*orb))


def test_vector_E_scalar_mu_needs_single_orbit():
    with pytest.raises(AssertionError):
        fun.orbElem2vec(np.array([0.5, 2.0, 4.5]), MU, orbElem=_multi())


def test_vector_E_vector_mu_needs_3n_orbit():
    with pytest.raises(AssertionError):
        fun.orbElem2vec(np.array([0.5, 2.0, 4.5]), np.array([1.0, 2.0, 0.5]), orbElem=_orb())


def test_round_trip_recovers_semimajor_axis_and_eccentricity():
    E = np.array([0.4, 1.7, 3.0])
    r, v = _states(E)
    result = fun.vec2orbElem(r, v, MU)
    np.testing.assert_allclose(result[0], SMA, rtol=1e-9)
    np.testing.assert_allclose(result[1], ECC, rtol=1e-9)


def test_round_trip_recovers_angles_and_eccentric_anomaly():
    E = np.array([0.4, 1.7, 3.0])
    r, v = _states(E)
    result = fun.vec2orbElem(r, v, MU)
    np.testing.assert_allclose(result[2], E, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result[3], RAAN, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result[4], INC, rtol=1e-9)
    np.testing.assert_allclose(result[5], ARGP, rtol=1e-9, atol=1e-12)


def test_round_trip_multiple_bodies_with_vector_mus():
    a, e, O, I, w = _multi()
    E = np.array([0.5, 2.0, 4.5])
    mus = np.array([1.0, 2.0, 0.5])
    r, v = fun.orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    result = fun.vec2orbElem(r, v, mus)
    np.testing.assert_allclose(result[0], a, rtol=1e-9)
    np.testing.assert_allclose(result[1], e, rtol=1e-9)
    np.testing.assert_allclose(result[2], E, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result[3], O, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result[4], I, rtol=1e-9)
    np.testing.assert_allclose(result[5], w, rtol=1e-9, atol=1e-12)


def test_period_obeys_kepler_third_law():
    E = np.array([0.4, 1.7, 3.0])
    r, v = _states(E)
    result = fun.vec2orbElem(r, v, MU)
    a, P = result[0], result[6]
    np.testing.assert_allclose(P, 2 * np.pi * np.sqrt(a ** 3 / MU), rtol=1e-9)


def test_tau_is_negative_mean_anomaly_over_mean_motion():
    E = np.array([0.4, 1.7, 3.0])
    r, v = _states(E)
    tau = fun.vec2orbElem(r, v, MU)[7]
    n = np.sqrt(MU / SMA ** 3)
    expected = -(E - ECC * np.sin(E)) / n
    np.testing.assert_allclose(tau, expected, rtol=1e-8)


def test_vec2orbElem_accepts_nx3_layout():
    E = np.array([0.4, 1.7, 3.0, 5.2])
    r, v = _states(E)
    out_3xn = fun.vec2orbElem(r, v, MU)
    out_nx3 = fun.vec2orbElem(r.T.copy(), v.T.copy(), MU)
    for got, want in zip(out_nx3, out_3xn):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)
