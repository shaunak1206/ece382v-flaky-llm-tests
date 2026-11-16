import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


A_ = 1.5
E_ = 0.3
O_ = 0.7
I_ = 0.5
W_ = 1.1
MU = 1.3


def _single_body(Es, a=A_, e=E_, O=O_, I=I_, w=W_, mu=MU):
    Es = np.array(Es, dtype=float)
    return orbElem2vec(Es, mu, orbElem=(a, e, O, I, w))


def test_circular_orbit_at_periapsis():
    r, v = orbElem2vec(0.0, 1.0, orbElem=(1.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(r.flatten(), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(v.flatten(), [0, 1, 0], atol=1e-12)


def test_circular_orbit_quarter_period():
    r, v = orbElem2vec(np.pi / 2, 1.0, orbElem=(1.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(r.flatten(), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(v.flatten(), [-1, 0, 0], atol=1e-12)


def test_scalar_inputs_shapes():
    r, v = orbElem2vec(0.4, MU, orbElem=(A_, E_, O_, I_, W_))
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)


def test_vector_E_scalar_mu_shapes():
    Es = np.array([0.1, 1.0, 2.0, 4.0])
    r, v = _single_body(Es)
    assert r.shape == (3, 4)
    assert v.shape == (3, 4)


def test_radius_matches_conic_formula():
    Es = np.linspace(0.1, 6.0, 7)
    r, v = _single_body(Es)
    rmag = np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(rmag, A_ * (1 - E_ * np.cos(Es)), rtol=1e-10)


def test_vis_viva_energy_conserved():
    Es = np.linspace(0.1, 6.0, 7)
    r, v = _single_body(Es)
    energy = 0.5 * np.sum(v ** 2, axis=0) - MU / np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(energy, -MU / (2 * A_) * np.ones(Es.size), rtol=1e-10)


def test_angular_momentum_magnitude_and_direction_constant():
    Es = np.linspace(0.2, 5.5, 6)
    r, v = _single_body(Es)
    h = np.cross(r.T, v.T)
    hmag = np.linalg.norm(h, axis=1)
    np.testing.assert_allclose(hmag, np.sqrt(MU * A_ * (1 - E_ ** 2)), rtol=1e-10)
    # orientation fixed by inclination and node
    expected_dir = np.array([np.sin(I_) * np.sin(O_),
                             -np.sin(I_) * np.cos(O_),
                             np.cos(I_)])
    np.testing.assert_allclose(h / hmag[:, None],
                               np.tile(expected_dir, (Es.size, 1)), atol=1e-10)


def test_returnAB_and_AB_input_equivalent():
    Es = np.array([0.3, 1.2, 3.0])
    r1, v1, AB = orbElem2vec(Es, MU, orbElem=(A_, E_, O_, I_, W_), returnAB=True)
    assert len(AB) == 2
    assert AB[0].shape == (3, 1) and AB[1].shape == (3, 1)
    r2, v2 = orbElem2vec(Es, MU, AB=AB)
    np.testing.assert_allclose(r1, r2, atol=1e-10)
    np.testing.assert_allclose(v1, v2, atol=1e-10)


def test_multiple_bodies_match_individual_calls():
    a = np.array([1.0, 2.0, 0.7])
    e = np.array([0.1, 0.5, 0.3])
    O = np.array([0.2, 1.0, 4.0])
    I = np.array([0.3, 1.0, 2.0])
    w = np.array([0.5, 2.5, 5.0])
    Es = np.array([0.4, 2.0, 5.0])
    mus = np.array([1.0, 2.0, 0.5])
    r, v = orbElem2vec(Es, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 3) and v.shape == (3, 3)
    for i in range(3):
        ri, vi = orbElem2vec(Es[i], mus[i],
                             orbElem=(a[i], e[i], O[i], I[i], w[i]))
        np.testing.assert_allclose(r[:, i], ri.flatten(), atol=1e-10)
        np.testing.assert_allclose(v[:, i], vi.flatten(), atol=1e-10)


def test_orbElem2vec_input_validation():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, 1.0)
    orbElem = (A_, E_, O_, I_, W_)
    A, B = calcAB(*[np.array([x]) for x in orbElem])
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, 1.0, orbElem=orbElem, AB=(A, B))
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=orbElem)
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0]), orbElem=orbElem)


def test_roundtrip_single_body_many_positions():
    Es = np.array([0.4, 2.0, 5.0])
    r, v = _single_body(Es)
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, MU)
    np.testing.assert_allclose(a, A_ * np.ones(3), rtol=1e-9)
    np.testing.assert_allclose(e, E_ * np.ones(3), rtol=1e-8)
    np.testing.assert_allclose(E, Es, atol=1e-8)
    np.testing.assert_allclose(O, O_ * np.ones(3), atol=1e-8)
    np.testing.assert_allclose(I, I_ * np.ones(3), atol=1e-8)
    np.testing.assert_allclose(w, W_ * np.ones(3), atol=1e-8)


def test_roundtrip_multiple_bodies_array_mus():
    a0 = np.array([1.0, 2.0, 0.7])
    e0 = np.array([0.1, 0.5, 0.3])
    O0 = np.array([0.2, 1.0, 4.0])
    I0 = np.array([0.3, 1.0, 2.0])  # includes retrograde
    w0 = np.array([0.5, 2.5, 5.0])
    E0 = np.array([0.4, 2.0, 5.0])
    mus = np.array([1.0, 2.0, 0.5])
    r, v = orbElem2vec(E0, mus, orbElem=(a0, e0, O0, I0, w0))
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, mus)
    np.testing.assert_allclose(a, a0, rtol=1e-8)
    np.testing.assert_allclose(e, e0, rtol=1e-7)
    np.testing.assert_allclose(E, E0, atol=1e-7)
    np.testing.assert_allclose(O, O0, atol=1e-7)
    np.testing.assert_allclose(I, I0, atol=1e-7)
    np.testing.assert_allclose(w, w0, atol=1e-7)


def test_vec2orbElem_period_and_tau():
    Es = np.array([0.4, 2.0, 5.0])
    r, v = _single_body(Es)
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, MU)
    np.testing.assert_allclose(P, 2 * np.pi * np.sqrt(A_ ** 3 / MU), rtol=1e-8)
    n = np.sqrt(MU / A_ ** 3)
    np.testing.assert_allclose(tau, -(Es - E_ * np.sin(Es)) / n, atol=1e-7)


def test_vec2orbElem_accepts_nx3_layout():
    Es = np.array([0.4, 2.0])
    r, v = _single_body(Es)  # 3 x 2
    out1 = vec2orbElem(r, v, MU)
    out2 = vec2orbElem(r.T, v.T, MU)
    for x, y in zip(out1, out2):
        np.testing.assert_allclose(x, y, atol=1e-12)


def test_vec2orbElem_known_circular_inclined_orbit_values():
    # slightly eccentric orbit: periapsis along +x, prograde in xy-plane tilted
    mu = 1.0
    r = np.array([[1.0], [0.0], [0.0]])
    vcirc = np.sqrt(mu / 1.0)
    # velocity larger than circular, perpendicular to r, tilted out of plane
    v = np.array([[0.0], [1.1 * vcirc * np.cos(0.4)], [1.1 * vcirc * np.sin(0.4)]])
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, mu)
    np.testing.assert_allclose(I, [0.4], atol=1e-10)
    # r is at periapsis so E = 0
    assert np.isclose(E[0], 0.0, atol=1e-8) or np.isclose(E[0], 2 * np.pi, atol=1e-8)
    np.testing.assert_allclose(e, [0.21], atol=1e-10)
    np.testing.assert_allclose(a, [1.0 / 0.79], rtol=1e-10)
    # position is on the x axis, which is the ascending node line
    assert np.isclose(O[0], 0.0, atol=1e-8) or np.isclose(O[0], 2 * np.pi, atol=1e-8)


def test_vec2orbElem_input_validation():
    with pytest.raises(AssertionError):
        vec2orbElem(np.zeros(4), np.zeros(4), 1.0)
    with pytest.raises(AssertionError):
        vec2orbElem(np.zeros((3, 2)), np.zeros((3, 3)), 1.0)
    with pytest.raises(AssertionError):
        vec2orbElem(np.zeros((3, 2)), np.zeros((3, 2)), np.ones(5))
