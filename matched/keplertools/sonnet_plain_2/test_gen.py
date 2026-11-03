import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem


# Reference single-body orbit used by several tests.
A_SEMI = np.float64(1.5)
ECC = np.float64(0.3)
OMEGA = np.float64(0.7)
INC = np.float64(0.5)
ARGP = np.float64(1.1)
MU = 1.0


def _elems():
    return (A_SEMI, ECC, OMEGA, INC, ARGP)


def test_orbElem2vec_scalar_output_shapes():
    r, v = orbElem2vec(0.9, MU, orbElem=_elems())
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(v))


def test_orbElem2vec_vector_E_scalar_mu_shape():
    E = np.linspace(0.1, 5.0, 7)
    r, v = orbElem2vec(E, MU, orbElem=_elems())
    assert r.shape == (3, 7)
    assert v.shape == (3, 7)


def test_orbElem2vec_radius_matches_conic_equation():
    E = np.linspace(0.0, 2 * np.pi, 11)
    r, v = orbElem2vec(E, MU, orbElem=_elems())
    rmag = np.linalg.norm(r, axis=0)
    expected = A_SEMI * (1.0 - ECC * np.cos(E))
    np.testing.assert_allclose(rmag, expected, rtol=1e-10)


def test_orbElem2vec_energy_is_conserved():
    E = np.linspace(0.2, 6.0, 9)
    mu = 2.5
    r, v = orbElem2vec(E, mu, orbElem=_elems())
    rmag = np.linalg.norm(r, axis=0)
    v2 = np.sum(v ** 2, axis=0)
    energy = 0.5 * v2 - mu / rmag
    np.testing.assert_allclose(energy, -mu / (2.0 * A_SEMI) * np.ones(E.size), rtol=1e-9)


def test_orbElem2vec_angular_momentum_magnitude():
    E = np.linspace(0.3, 5.5, 8)
    mu = 1.7
    r, v = orbElem2vec(E, mu, orbElem=_elems())
    h = np.cross(r.T, v.T)
    hmag = np.linalg.norm(h, axis=1)
    expected = np.sqrt(mu * A_SEMI * (1.0 - ECC ** 2))
    np.testing.assert_allclose(hmag, expected * np.ones(E.size), rtol=1e-9)


def test_orbElem2vec_periapsis_state():
    r, v = orbElem2vec(0.0, MU, orbElem=_elems())
    rmag = np.linalg.norm(r)
    vmag = np.linalg.norm(v)
    assert rmag == pytest.approx(A_SEMI * (1.0 - ECC), rel=1e-10)
    expected_v = np.sqrt(MU / A_SEMI * (1.0 + ECC) / (1.0 - ECC))
    assert vmag == pytest.approx(expected_v, rel=1e-10)
    # At periapsis position and velocity are perpendicular.
    assert abs(float(np.sum(r * v))) < 1e-10


def test_orbElem2vec_returnAB_gives_three_outputs():
    out = orbElem2vec(0.9, MU, orbElem=_elems(), returnAB=True)
    assert len(out) == 3
    r, v, (A, B) = out
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)
    # A is the eccentricity direction scaled by a.
    assert np.linalg.norm(A) == pytest.approx(A_SEMI, rel=1e-12)
    # B is orthogonal to A.
    assert abs(float(np.sum(A * B))) < 1e-12


def test_orbElem2vec_AB_input_matches_orbElem_input():
    E = np.array([0.4, 1.3, 2.9, 4.4])
    r1, v1, AB = orbElem2vec(E, MU, orbElem=_elems(), returnAB=True)
    r2, v2 = orbElem2vec(E, MU, AB=AB)
    np.testing.assert_allclose(r2, r1, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(v2, v1, rtol=1e-9, atol=1e-12)


def test_orbElem2vec_multiple_bodies_match_single_body_calls():
    a = np.array([1.5, 2.0, 1.1])
    e = np.array([0.1, 0.4, 0.25])
    O = np.array([0.5, 1.5, 2.5])
    I = np.array([0.4, 0.9, 1.2])
    w = np.array([0.6, 1.8, 3.0])
    E = np.array([0.7, 2.2, 4.1])
    mus = np.array([1.0, 2.0, 0.5])

    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)

    for i in range(3):
        ri, vi = orbElem2vec(float(E[i]), float(mus[i]),
                             orbElem=(a[i], e[i], O[i], I[i], w[i]))
        np.testing.assert_allclose(r[:, i], ri[:, 0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(v[:, i], vi[:, 0], rtol=1e-10, atol=1e-12)


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, MU)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    _, _, AB = orbElem2vec(0.5, MU, orbElem=_elems(), returnAB=True)
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, MU, orbElem=_elems(), AB=AB)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=_elems())


def test_orbElem2vec_vector_E_and_mus_need_matching_AB_size():
    E = np.array([0.5, 1.0])
    mus = np.array([1.0, 1.0])
    # Scalar orbital elements give 3x1 A and B, which is wrong for 2 bodies.
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, orbElem=_elems())


def test_vec2orbElem_round_trip_single_body():
    E_in = 0.9
    r, v = orbElem2vec(E_in, MU, orbElem=_elems())
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, MU)

    np.testing.assert_allclose(a, [A_SEMI], rtol=1e-9)
    np.testing.assert_allclose(e, [ECC], rtol=1e-9)
    np.testing.assert_allclose(E, [E_in], rtol=1e-9)
    np.testing.assert_allclose(O, [OMEGA], rtol=1e-9)
    np.testing.assert_allclose(I, [INC], rtol=1e-9)
    np.testing.assert_allclose(w, [ARGP], rtol=1e-9)


def test_vec2orbElem_period_and_periapsis_time():
    E_in = 0.9
    r, v = orbElem2vec(E_in, MU, orbElem=_elems())
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, MU)

    expected_P = 2.0 * np.pi * np.sqrt(A_SEMI ** 3 / MU)
    np.testing.assert_allclose(P, [expected_P], rtol=1e-9)

    expected_tau = -(E_in - ECC * np.sin(E_in)) / np.sqrt(MU * A_SEMI ** -3.)
    np.testing.assert_allclose(tau, [expected_tau], rtol=1e-9)


def test_vec2orbElem_nx3_input_equals_3xn_input():
    E = np.array([0.8, 3.4])
    r, v = orbElem2vec(E, MU, orbElem=_elems())
    res_3xn = vec2orbElem(r, v, MU)
    res_nx3 = vec2orbElem(r.T, v.T, MU)
    for x, y in zip(res_3xn, res_nx3):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


def test_vec2orbElem_output_shapes():
    E = np.array([0.8, 3.4])
    r, v = orbElem2vec(E, MU, orbElem=_elems())
    outputs = vec2orbElem(r, v, MU)
    assert len(outputs) == 8
    for out in outputs:
        assert out.shape == (2,)


def test_vec2orbElem_known_apoapsis_state():
    rs = np.array([[1.0], [0.0], [0.0]])
    vs = np.array([[0.0], [0.7], [0.5]])
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, 1.0)

    # Specific energy = 0.37 - 1 = -0.63 so a = 1 / 1.26.
    np.testing.assert_allclose(a, [1.0 / 1.26], rtol=1e-9)
    # Angular momentum squared = 0.74, so e = sqrt(1 - 0.74 * 1.26) = 0.26.
    np.testing.assert_allclose(e, [0.26], rtol=1e-7)
    # Radial velocity is zero and r > a, so the body sits at apoapsis.
    np.testing.assert_allclose(E, [np.pi], atol=1e-9)
    np.testing.assert_allclose(I, [np.arccos(0.7 / np.sqrt(0.74))], rtol=1e-9)
    np.testing.assert_allclose(O, [0.0], atol=1e-9)


def test_vec2orbElem_multiple_bodies_round_trip_with_vector_mus():
    a_in = np.array([1.5, 2.0, 1.1, 3.0])
    e_in = np.array([0.1, 0.3, 0.5, 0.2])
    O_in = np.array([0.5, 1.0, 2.0, 3.0])
    I_in = np.array([0.4, 0.9, 1.3, 0.7])
    w_in = np.array([0.6, 1.2, 2.2, 4.0])
    E_in = np.array([0.3, 1.2, 2.5, 4.0])
    mus = np.array([1.0, 2.0, 0.5, 3.0])

    r, v = orbElem2vec(E_in, mus, orbElem=(a_in, e_in, O_in, I_in, w_in))
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, mus)

    np.testing.assert_allclose(a, a_in, rtol=1e-8)
    np.testing.assert_allclose(e, e_in, rtol=1e-8)
    np.testing.assert_allclose(E, E_in, rtol=1e-8)
    np.testing.assert_allclose(O, O_in, rtol=1e-8)
    np.testing.assert_allclose(I, I_in, rtol=1e-8)
    np.testing.assert_allclose(w, w_in, rtol=1e-8)
    np.testing.assert_allclose(P, 2.0 * np.pi * np.sqrt(a_in ** 3 / mus), rtol=1e-8)


def test_vec2orbElem_rejects_mismatched_sizes():
    rs = np.zeros(6)
    vs = np.zeros(3)
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_wrong_mus_size():
    E = np.array([0.8, 3.4])
    r, v = orbElem2vec(E, MU, orbElem=_elems())
    with pytest.raises(AssertionError):
        vec2orbElem(r, v, np.array([1.0, 1.0, 1.0]))


def test_vec2orbElem_rejects_three_dimensional_input():
    rs = np.ones((3, 1, 2))
    vs = np.ones((3, 1, 2))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)
