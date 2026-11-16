import numpy as np
import pytest

from keplertools import fun


def _orb(a, e, O, I, w):
    """Build an orbElem tuple of 1-D arrays, as calcAB requires (.size is used)."""
    return tuple(np.atleast_1d(np.asarray(x, dtype=float)) for x in (a, e, O, I, w))


def assert_angles_close(actual, expected, atol=1e-8):
    """Compare angles modulo 2*pi."""
    diff = np.mod(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float) + np.pi,
                  2 * np.pi) - np.pi
    assert np.allclose(diff, 0.0, atol=atol)


# ---------------------------------------------------------------------------
# vec2orbElem
# ---------------------------------------------------------------------------

def test_vec2orbElem_hand_computed_periapsis():
    # a=1, e=0.5, I=pi/2, O=0, w=0, at periapsis (E=0), mu=1.
    # r = a(1-e) * P = [0.5, 0, 0]; v = sqrt(mu*(1+e)/(a*(1-e))) along z.
    rs = np.array([[0.5], [0.0], [0.0]])
    vs = np.array([[0.0], [0.0], [np.sqrt(3.0)]])
    mu = 1.0

    a, e, E, O, I, w, P, tau = fun.vec2orbElem(rs, vs, mu)

    assert np.allclose(a, 1.0)
    assert np.allclose(e, 0.5)
    assert_angles_close(E, 0.0)
    assert_angles_close(O, 0.0)
    assert np.allclose(I, np.pi / 2)
    assert_angles_close(w, 0.0)
    assert np.allclose(P, 2 * np.pi)
    assert np.allclose(tau, 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "a,e,O,I,w,E,mu",
    [
        (1.0, 0.5, 0.7, 0.5, 1.2, 0.9, 1.0),
        (2.5, 0.1, 4.0, 2.0, 5.5, 3.0, 3.7),
        (0.8, 0.7, 6.0, 1.0, 0.3, 5.5, 0.25),
    ],
)
def test_vec2orbElem_roundtrip_single_body(a, e, O, I, w, E, mu):
    orb = _orb(a, e, O, I, w)
    r, v = fun.orbElem2vec(E, mu, orbElem=orb)

    a2, e2, E2, O2, I2, w2, P2, tau2 = fun.vec2orbElem(r, v, mu)

    assert np.allclose(a2, a, rtol=1e-9, atol=1e-9)
    assert np.allclose(e2, e, rtol=1e-9, atol=1e-9)
    assert_angles_close(E2, E, atol=1e-7)
    assert_angles_close(O2, O, atol=1e-7)
    assert np.allclose(I2, I, atol=1e-9)
    assert_angles_close(w2, w, atol=1e-7)


def test_vec2orbElem_roundtrip_multiple_bodies_vector_mu():
    a = np.array([1.0, 2.0, 0.5])
    e = np.array([0.2, 0.5, 0.8])
    O = np.array([0.1, 2.0, 5.0])
    I = np.array([0.3, 1.2, 2.5])
    w = np.array([1.0, 4.0, 0.2])
    E = np.array([0.4, 2.2, 5.9])
    mus = np.array([1.0, 3.0, 0.5])

    r, v = fun.orbElem2vec(E, mus, orbElem=_orb(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = fun.vec2orbElem(r, v, mus)

    assert a2.shape == (3,)
    assert np.allclose(a2, a, atol=1e-9)
    assert np.allclose(e2, e, atol=1e-9)
    assert_angles_close(E2, E, atol=1e-7)
    assert_angles_close(O2, O, atol=1e-7)
    assert np.allclose(I2, I, atol=1e-9)
    assert_angles_close(w2, w, atol=1e-7)


def test_vec2orbElem_scalar_mu_many_positions_same_orbit():
    a, e = 1.4, 0.3
    E = np.array([0.3, 1.0, 2.5, 4.0, 6.0])
    mu = 2.0

    r, v = fun.orbElem2vec(E, mu, orbElem=_orb(a, e, 0.9, 1.1, 2.2))
    a2, e2, E2, O2, I2, w2, P2, tau2 = fun.vec2orbElem(r, v, mu)

    assert a2.shape == (5,)
    assert np.allclose(a2, a, atol=1e-9)
    assert np.allclose(e2, e, atol=1e-9)
    assert_angles_close(E2, E, atol=1e-7)
    assert np.allclose(O2, O2[0])  # same orbit, same node for every position
    assert np.allclose(I2, I2[0])


def test_vec2orbElem_accepts_nx3_layout():
    mus = np.array([1.0, 2.0])
    E = np.array([0.5, 3.0])
    orb = _orb([1.0, 1.8], [0.3, 0.6], [0.2, 1.5], [0.8, 2.0], [0.4, 3.0])

    r, v = fun.orbElem2vec(E, mus, orbElem=orb)  # 3 x 2

    res_3xn = fun.vec2orbElem(r, v, mus)
    res_nx3 = fun.vec2orbElem(r.T, v.T, mus)

    for x, y in zip(res_3xn, res_nx3):
        assert np.allclose(x, y, atol=1e-12)
    assert np.allclose(res_nx3[0], [1.0, 1.8], atol=1e-9)


def test_vec2orbElem_period_and_time_of_periapsis():
    a, e, O, I, w, E, mu = 2.5, 0.1, 4.0, 2.0, 5.5, 3.0, 3.7
    r, v = fun.orbElem2vec(E, mu, orbElem=_orb(a, e, O, I, w))

    a2, e2, E2, O2, I2, w2, P2, tau2 = fun.vec2orbElem(r, v, mu)

    n = np.sqrt(mu / a**3)
    assert np.allclose(P2, 2 * np.pi / n, rtol=1e-9)
    # tau = -M/n where M = E - e sin E is the mean anomaly
    M = E2 - e * np.sin(E2)
    assert np.allclose(tau2, -M / n, rtol=1e-7, atol=1e-9)


def test_vec2orbElem_output_angles_in_expected_ranges():
    a = np.array([1.0, 2.0, 0.5])
    e = np.array([0.2, 0.5, 0.8])
    O = np.array([0.1, 2.0, 5.0])
    I = np.array([0.3, 1.2, 2.5])
    w = np.array([1.0, 4.0, 0.2])
    E = np.array([0.4, 2.2, 5.9])
    mus = np.array([1.0, 3.0, 0.5])

    r, v = fun.orbElem2vec(E, mus, orbElem=_orb(a, e, O, I, w))
    _, _, E2, O2, I2, w2, _, _ = fun.vec2orbElem(r, v, mus)

    assert np.all((E2 >= 0) & (E2 < 2 * np.pi))
    assert np.all((O2 >= 0) & (O2 < 2 * np.pi))
    assert np.all((w2 >= 0) & (w2 < 2 * np.pi))
    assert np.all((I2 > 0) & (I2 < np.pi))


def test_vec2orbElem_rejects_vs_size_mismatch():
    rs = np.ones((3, 2))
    vs = np.ones((3, 3))
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_size_not_multiple_of_three():
    rs = np.ones((2, 2))  # 4 elements
    vs = np.ones((2, 2))
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_mus_wrong_size():
    rs = np.ones((3, 2))
    vs = np.ones((3, 2))
    mus = np.array([1.0, 1.0, 1.0])  # 3 values for 2 bodies
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, mus)


def test_vec2orbElem_rejects_three_dimensional_rs():
    rs = np.ones((3, 2, 3))
    vs = np.ones((3, 2, 3))
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, 1.0)


# ---------------------------------------------------------------------------
# orbElem2vec
# ---------------------------------------------------------------------------

def test_orbElem2vec_hand_computed_periapsis():
    # a=1, e=0.5, I=pi/2, O=0, w=0, mu=1, E=0 -> r=[0.5,0,0], v=[0,0,sqrt(3)]
    r, v = fun.orbElem2vec(0.0, 1.0, orbElem=_orb(1.0, 0.5, 0.0, np.pi / 2, 0.0))

    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.allclose(r[:, 0], [0.5, 0.0, 0.0], atol=1e-12)
    assert np.allclose(v[:, 0], [0.0, 0.0, np.sqrt(3.0)], atol=1e-12)


def test_orbElem2vec_vis_viva_and_angular_momentum():
    a, e = 1.7, 0.4
    mu = 2.5
    E = np.linspace(0.1, 6.0, 12)

    r, v = fun.orbElem2vec(E, mu, orbElem=_orb(a, e, 0.3, 0.9, 2.1))

    rmag = np.linalg.norm(r, axis=0)
    v2 = np.sum(v**2, axis=0)
    energy = v2 / 2 - mu / rmag
    assert np.allclose(energy, -mu / (2 * a), rtol=1e-10)

    h = np.linalg.norm(np.cross(r.T, v.T), axis=1)
    assert np.allclose(h, np.sqrt(mu * a * (1 - e**2)), rtol=1e-10)


def test_orbElem2vec_AB_input_matches_orbElem_input():
    orb = _orb(1.3, 0.35, 1.0, 0.7, 2.5)
    mu = 1.5
    E = np.linspace(0.0, 2 * np.pi, 7)

    A, B = fun.calcAB(*orb)
    r1, v1 = fun.orbElem2vec(E, mu, AB=(A, B))
    r2, v2 = fun.orbElem2vec(E, mu, orbElem=orb)

    assert np.allclose(r1, r2, atol=1e-10)
    assert np.allclose(v1, v2, atol=1e-10)


def test_orbElem2vec_returnAB_outputs():
    orb = _orb(1.1, 0.2, 0.5, 1.0, 0.8)
    r, v, (A, B) = fun.orbElem2vec(0.7, 1.0, orbElem=orb, returnAB=True)

    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)

    A_ref, B_ref = fun.calcAB(*orb)
    assert np.allclose(A, A_ref)
    assert np.allclose(B, B_ref)


def test_orbElem2vec_scalar_mu_multiple_positions_match_individual_calls():
    orb = _orb(1.2, 0.45, 0.8, 0.6, 1.9)
    mu = 0.9
    E = np.array([0.0, 0.8, 2.0, 3.5, 5.0])

    r, v = fun.orbElem2vec(E, mu, orbElem=orb)
    assert r.shape == (3, 5)
    assert v.shape == (3, 5)

    for i, Ei in enumerate(E):
        ri, vi = fun.orbElem2vec(Ei, mu, orbElem=orb)
        assert np.allclose(r[:, i], ri[:, 0], atol=1e-12)
        assert np.allclose(v[:, i], vi[:, 0], atol=1e-12)


def test_orbElem2vec_vector_bodies_match_individual_calls():
    a = np.array([1.0, 2.0, 0.5])
    e = np.array([0.2, 0.5, 0.8])
    O = np.array([0.1, 2.0, 5.0])
    I = np.array([0.3, 1.2, 2.5])
    w = np.array([1.0, 4.0, 0.2])
    E = np.array([0.4, 2.2, 5.9])
    mus = np.array([1.0, 3.0, 0.5])

    r, v = fun.orbElem2vec(E, mus, orbElem=_orb(a, e, O, I, w))
    assert r.shape == (3, 3)

    for i in range(3):
        ri, vi = fun.orbElem2vec(E[i], mus[i],
                                 orbElem=_orb(a[i], e[i], O[i], I[i], w[i]))
        assert np.allclose(r[:, i], ri[:, 0], atol=1e-12)
        assert np.allclose(v[:, i], vi[:, 0], atol=1e-12)


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.3, 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    orb = _orb(1.0, 0.3, 0.1, 0.2, 0.3)
    A, B = fun.calcAB(*orb)
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.3, 1.0, orbElem=orb, AB=(A, B))


def test_orbElem2vec_rejects_scalar_E_with_array_mu():
    orb = _orb(1.0, 0.3, 0.1, 0.2, 0.3)
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.3, np.array([1.0, 2.0]), orbElem=orb)


def test_orbElem2vec_rejects_mismatched_mu_size():
    E = np.array([0.1, 0.2, 0.3])
    mus = np.array([1.0, 2.0])  # size 2 vs E size 3
    orb = _orb([1.0, 1.1, 1.2], [0.1, 0.2, 0.3], [0.0, 0.1, 0.2],
               [0.5, 0.6, 0.7], [0.1, 0.2, 0.3])
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, orbElem=orb)
