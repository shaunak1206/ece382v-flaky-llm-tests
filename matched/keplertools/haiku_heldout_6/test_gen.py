"""Unit tests for keplertools.fun.orbElem2vec and keplertools.fun.vec2orbElem."""
import numpy as np
import pytest

from keplertools import fun

TWO_PI = 2.0 * np.pi


def _orb(a, e, O, I, w):
    """Build an (a, e, O, I, w) tuple of 1-D float arrays, as calcAB requires."""
    return tuple(np.atleast_1d(np.asarray(x, dtype=float)) for x in (a, e, O, I, w))


# ---------------------------------------------------------------------------
# orbElem2vec
# ---------------------------------------------------------------------------

def test_orbElem2vec_perihelion_analytic():
    # a=1, e=0.5, I=90deg, O=0, w=0, E=0 -> perihelion at (0.5, 0, 0)
    # with speed sqrt(3) along +z.
    orb = _orb(1.0, 0.5, 0.0, np.pi / 2, 0.0)
    r, v = fun.orbElem2vec(0.0, 1.0, orbElem=orb)
    np.testing.assert_allclose(r, [[0.5], [0.0], [0.0]], atol=1e-12)
    np.testing.assert_allclose(v, [[0.0], [0.0], [np.sqrt(3.0)]], atol=1e-12)


def test_orbElem2vec_returnAB_gives_calcAB_matrices():
    orb = _orb(1.7, 0.3, 0.9, 0.4, 2.2)
    r, v, AB = fun.orbElem2vec(1.0, 1.0, orbElem=orb, returnAB=True)
    A_expected, B_expected = fun.calcAB(*orb)
    assert len(AB) == 2
    np.testing.assert_allclose(AB[0], A_expected)
    np.testing.assert_allclose(AB[1], B_expected)
    assert r.shape == (3, 1) and v.shape == (3, 1)


def test_AB_input_matches_orbElem_input():
    orb = _orb(1.7, 0.3, 0.9, 0.4, 2.2)
    E = np.linspace(0.0, TWO_PI, 9)
    r1, v1 = fun.orbElem2vec(E, 1.0, orbElem=orb)
    A, B = fun.calcAB(*orb)
    r2, v2 = fun.orbElem2vec(E, 1.0, AB=(A, B))
    np.testing.assert_allclose(r2, r1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v2, v1, rtol=1e-10, atol=1e-12)


def test_circular_orbit_has_constant_radius_and_speed():
    a, mu = 2.0, 1.5
    orb = _orb(a, 0.0, 0.3, 1.0, 0.7)
    E = np.linspace(0.0, TWO_PI, 25)
    r, v = fun.orbElem2vec(E, mu, orbElem=orb)
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), a, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(v, axis=0), np.sqrt(mu / a), rtol=1e-12)


def test_radius_matches_a_one_minus_e_cos_E():
    a, e = 1.3, 0.6
    orb = _orb(a, e, 2.0, 0.8, 0.5)
    E = np.linspace(0.0, TWO_PI, 50)
    r, _ = fun.orbElem2vec(E, 1.0, orbElem=orb)
    expected = a * (1.0 - e * np.cos(E))
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), expected, rtol=1e-12)


def test_energy_is_conserved_with_scalar_mu():
    a, e, mu = 1.7, 0.3, 2.5
    orb = _orb(a, e, 0.9, 0.4, 2.2)
    E = np.linspace(0.0, TWO_PI, 30)
    r, v = fun.orbElem2vec(E, mu, orbElem=orb)
    energy = 0.5 * np.sum(v ** 2, axis=0) - mu / np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(energy, -mu / (2.0 * a), rtol=1e-10)


def test_vector_mu_matches_per_body_single_calls():
    a = np.array([1.0, 2.5, 0.8])
    e = np.array([0.5, 0.2, 0.7])
    O = np.array([0.3, 2.0, 5.0])
    I = np.array([0.7, 1.2, 2.5])
    w = np.array([1.1, 4.0, 0.2])
    E = np.array([0.4, 2.0, 5.5])
    mus = np.array([1.0, 3.0, 0.5])

    r, v = fun.orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 3) and v.shape == (3, 3)

    for i in range(3):
        orb_i = (a[i:i + 1], e[i:i + 1], O[i:i + 1], I[i:i + 1], w[i:i + 1])
        ri, vi = fun.orbElem2vec(E[i], mus[i], orbElem=orb_i)
        np.testing.assert_allclose(r[:, i], ri[:, 0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(v[:, i], vi[:, 0], rtol=1e-10, atol=1e-12)


def test_output_shapes_for_scalar_and_vector_E():
    orb = _orb(1.0, 0.1, 0.2, 0.3, 0.4)
    r, v = fun.orbElem2vec(0.5, 1.0, orbElem=orb)
    assert r.shape == (3, 1) and v.shape == (3, 1)

    r, v = fun.orbElem2vec(np.linspace(0.0, 1.0, 7), 1.0, orbElem=orb)
    assert r.shape == (3, 7) and v.shape == (3, 7)


def test_scalar_E_with_array_mus_raises():
    orb = _orb(1.0, 0.1, 0.2, 0.3, 0.4)
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=orb)


def test_vector_E_and_mus_size_mismatch_raises():
    orb = _orb([1.0, 1.1, 1.2], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2],
               [0.3, 0.3, 0.3], [0.4, 0.4, 0.4])
    E = np.array([0.1, 0.2, 0.3])
    mus = np.array([1.0, 2.0])
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, orbElem=orb)


def test_neither_orbElem_nor_AB_raises():
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.5, 1.0)


def test_both_orbElem_and_AB_raises():
    orb = _orb(1.0, 0.1, 0.2, 0.3, 0.4)
    A, B = fun.calcAB(*orb)
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.5, 1.0, orbElem=orb, AB=(A, B))


# ---------------------------------------------------------------------------
# vec2orbElem
# ---------------------------------------------------------------------------

def test_vec2orbElem_analytic_perihelion_state():
    # Perihelion of a=1, e=0.5, I=90deg, O=0, w=0 orbit with mu=1.
    rs = np.array([[0.5], [0.0], [0.0]])
    vs = np.array([[0.0], [0.0], [np.sqrt(3.0)]])
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(rs, vs, 1.0)

    np.testing.assert_allclose(a, [1.0], atol=1e-10)
    np.testing.assert_allclose(e, [0.5], atol=1e-10)
    np.testing.assert_allclose(E, [0.0], atol=1e-10)
    np.testing.assert_allclose(O, [0.0], atol=1e-10)
    np.testing.assert_allclose(I, [np.pi / 2], atol=1e-10)
    np.testing.assert_allclose(w, [0.0], atol=1e-10)
    np.testing.assert_allclose(P, [TWO_PI], atol=1e-10)
    np.testing.assert_allclose(tau, [0.0], atol=1e-10)


@pytest.mark.parametrize(
    "a, e, O, I, w, E, mu",
    [
        (1.3, 0.2, 0.4, 0.9, 2.1, 1.0, 1.0),
        (2.5, 0.7, 4.0, 2.2, 0.3, 4.5, 3.0),
        (0.8, 0.05, 5.5, 1.5, 5.0, 0.1, 0.5),
    ],
)
def test_round_trip_single_body(a, e, O, I, w, E, mu):
    r, v = fun.orbElem2vec(E, mu, orbElem=_orb(a, e, O, I, w))
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = fun.vec2orbElem(r, v, mu)

    np.testing.assert_allclose(a_out, [a], rtol=1e-9)
    np.testing.assert_allclose(e_out, [e], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(E_out, [np.mod(E, TWO_PI)], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(O_out, [np.mod(O, TWO_PI)], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(I_out, [I], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(w_out, [np.mod(w, TWO_PI)], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(P_out, [TWO_PI * np.sqrt(a ** 3 / mu)], rtol=1e-9)

    n = np.sqrt(mu / a ** 3)
    expected_tau = -(E - e * np.sin(E)) / n
    np.testing.assert_allclose(tau_out, [expected_tau], rtol=1e-7, atol=1e-9)


def test_round_trip_multi_body_3xn():
    a = np.array([1.0, 2.5, 0.8])
    e = np.array([0.5, 0.2, 0.7])
    O = np.array([0.3, 2.0, 5.0])
    I = np.array([0.7, 1.2, 2.5])
    w = np.array([1.1, 4.0, 0.2])
    E = np.array([0.4, 2.0, 5.5])
    mus = np.array([1.0, 3.0, 0.5])

    r, v = fun.orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = fun.vec2orbElem(r, v, mus)

    np.testing.assert_allclose(a_out, a, rtol=1e-9)
    np.testing.assert_allclose(e_out, e, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(E_out, E, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(O_out, O, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(I_out, I, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(w_out, w, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(P_out, TWO_PI * np.sqrt(a ** 3 / mus), rtol=1e-9)
    np.testing.assert_allclose(tau_out, -(E - e * np.sin(E)) / np.sqrt(mus / a ** 3),
                               rtol=1e-7, atol=1e-9)


def test_nx3_layout_matches_3xn_layout():
    a = np.array([1.0, 2.5, 0.8, 1.9])
    e = np.array([0.5, 0.2, 0.7, 0.1])
    O = np.array([0.3, 2.0, 5.0, 1.0])
    I = np.array([0.7, 1.2, 2.5, 0.6])
    w = np.array([1.1, 4.0, 0.2, 3.3])
    E = np.array([0.4, 2.0, 5.5, 3.0])
    mus = np.array([1.0, 3.0, 0.5, 2.0])

    r, v = fun.orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    out_3xn = fun.vec2orbElem(r, v, mus)
    out_nx3 = fun.vec2orbElem(r.T, v.T, mus)

    for got, expected in zip(out_nx3, out_3xn):
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out_nx3[0], a, rtol=1e-9)


def test_scalar_mu_broadcasts_like_array_mu():
    a = np.array([1.0, 2.0, 0.9])
    e = np.array([0.3, 0.6, 0.4])
    O = np.array([0.5, 1.5, 4.0])
    I = np.array([0.8, 1.9, 2.7])
    w = np.array([0.2, 3.0, 5.2])
    E = np.array([1.0, 2.5, 4.0])

    r, v = fun.orbElem2vec(E, np.ones(3), orbElem=(a, e, O, I, w))
    out_scalar = fun.vec2orbElem(r, v, 1.0)
    out_array = fun.vec2orbElem(r, v, np.ones(3))

    for got, expected in zip(out_scalar, out_array):
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out_scalar[0], a, rtol=1e-9)


def test_vec2orbElem_rejects_mismatched_sizes():
    rs = np.ones((3, 2))
    vs = np.ones((3, 1))
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_wrong_mus_size():
    rs = np.ones((3, 3))
    vs = np.ones((3, 3))
    mus = np.array([1.0, 2.0])  # 2 values for 3 bodies
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, mus)


def test_vec2orbElem_rejects_3d_input():
    rs = np.ones((3, 1, 1))
    vs = np.ones((3, 1, 1))
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_input_without_axis_of_length_3():
    rs = np.ones((2, 6))
    vs = np.ones((2, 6))
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, 1.0)
