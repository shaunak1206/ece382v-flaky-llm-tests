import numpy as np
import pytest

from keplertools import fun


# Fixed, deterministic test data. No random numbers are used anywhere in this file.
E_SAMPLES = np.array([0.3, 1.7, 3.1, 4.6, 5.8])

A_ARR = np.array([1.0, 1.5, 2.0, 0.9, 1.2])
E_ARR = np.array([0.3, 1.7, 3.1, 4.6, 5.8])
ECC_ARR = np.array([0.1, 0.3, 0.5, 0.2, 0.4])
O_ARR = np.array([1.1, 3.9, 5.2, 0.9, 2.2])
I_ARR = np.array([0.6, 1.9, 2.7, 1.2, 0.8])
W_ARR = np.array([0.4, 2.6, 4.4, 2.2, 5.1])
MU_ARR = np.array([0.5, 1.0, 1.5, 2.0, 2.5])


def _orbit_to_state(E, mu, a, e, O, I, w):
    """Helper: build position/velocity (3 x n) from Keplerian elements."""
    return fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w))


# ---------------------------------------------------------------------------
# orbElem2vec
# ---------------------------------------------------------------------------

def test_orbElem2vec_circular_orbit_at_pericenter_known_vectors():
    # Circular orbit, all angles zero, E=0: r = (a, 0, 0), v = (0, sqrt(mu/a), 0)
    r, v = fun.orbElem2vec(0.0, 1.0, orbElem=(1.0, 0.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(r[:, 0], [1.0, 0.0, 0.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [0.0, 1.0, 0.0], rtol=0, atol=1e-12)


def test_orbElem2vec_node_rotation_moves_pericenter_to_y_axis():
    # With O = pi/2, I = 0, w = 0, E = 0, the pericenter lies along +y.
    r, _ = fun.orbElem2vec(0.0, 1.0, orbElem=(1.0, 0.0, np.pi / 2, 0.0, 0.0))
    np.testing.assert_allclose(r[:, 0], [0.0, 1.0, 0.0], rtol=0, atol=1e-12)


def test_orbElem2vec_radius_matches_conic_equation():
    a, e = 2.0, 0.3
    E = np.array([0.4, 1.3, 2.9, 4.2])
    r, _ = fun.orbElem2vec(E, 1.5, orbElem=(a, e, 0.9, 1.2, 2.2))
    rmag = np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(rmag, a * (1.0 - e * np.cos(E)), rtol=1e-12, atol=1e-12)


def test_orbElem2vec_conserves_energy_and_angular_momentum():
    a, e, mu = 1.7, 0.4, 0.8
    E = np.array([0.4, 1.3, 2.9, 4.2])
    r, v = fun.orbElem2vec(E, mu, orbElem=(a, e, 0.9, 1.2, 2.2))

    rmag = np.linalg.norm(r, axis=0)
    v2 = np.sum(v ** 2, axis=0)
    energy = v2 / 2.0 - mu / rmag
    np.testing.assert_allclose(energy, -mu / (2.0 * a), rtol=1e-10, atol=1e-12)

    h = np.linalg.norm(np.cross(r.T, v.T), axis=1)
    np.testing.assert_allclose(h, np.sqrt(mu * a * (1.0 - e ** 2)), rtol=1e-10, atol=1e-12)


def test_orbElem2vec_scalar_E_gives_column_outputs():
    r, v = fun.orbElem2vec(1.0, 1.0, orbElem=(1.0, 0.2, 0.5, 0.5, 0.5))
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)


def test_orbElem2vec_vector_E_scalar_elements_gives_3xn_outputs():
    r, v = fun.orbElem2vec(E_SAMPLES, 1.3, orbElem=(1.0, 0.2, 0.5, 0.5, 0.5))
    assert r.shape == (3, E_SAMPLES.size)
    assert v.shape == (3, E_SAMPLES.size)


def test_orbElem2vec_vector_mus_matches_per_body_scalar_calls():
    r, v = fun.orbElem2vec(
        E_ARR, MU_ARR, orbElem=(A_ARR, ECC_ARR, O_ARR, I_ARR, W_ARR)
    )
    assert r.shape == (3, E_ARR.size)
    for i in range(E_ARR.size):
        ri, vi = fun.orbElem2vec(
            float(E_ARR[i]),
            float(MU_ARR[i]),
            orbElem=(float(A_ARR[i]), float(ECC_ARR[i]), float(O_ARR[i]),
                     float(I_ARR[i]), float(W_ARR[i])),
        )
        np.testing.assert_allclose(r[:, i], ri[:, 0], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(v[:, i], vi[:, 0], rtol=1e-11, atol=1e-12)


def test_orbElem2vec_AB_input_reproduces_orbElem_result():
    a, e, O, I, w = 1.4, 0.35, 2.0, 0.9, 3.3
    r1, v1, AB = fun.orbElem2vec(
        E_SAMPLES, 1.1, orbElem=(a, e, O, I, w), returnAB=True
    )
    r2, v2 = fun.orbElem2vec(E_SAMPLES, 1.1, AB=AB)
    np.testing.assert_allclose(r2, r1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v2, v1, rtol=1e-10, atol=1e-12)


def test_orbElem2vec_returnAB_gives_three_outputs_with_3x1_AB():
    out = fun.orbElem2vec(0.5, 1.0, orbElem=(1.0, 0.1, 0.2, 0.3, 0.4), returnAB=True)
    assert len(out) == 3
    r, v, AB = out
    A, B = AB
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.5, 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    AB = (np.ones((3, 1)), np.ones((3, 1)))
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.5, 1.0, orbElem=(1.0, 0.1, 0.0, 0.0, 0.0), AB=AB)


def test_orbElem2vec_scalar_E_with_vector_mus_is_rejected():
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=(1.0, 0.1, 0.0, 0.0, 0.0))


def test_orbElem2vec_mismatched_mus_size_is_rejected():
    E = np.array([0.1, 0.2, 0.3])
    mus = np.array([1.0, 2.0])
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mus, orbElem=(1.0, 0.1, 0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# vec2orbElem
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a, e, O, I, w, mu",
    [
        (1.0, 0.2, 1.1, 0.6, 0.4, 1.3),
        (2.5, 0.6, 3.9, 1.9, 2.6, 0.7),
        (0.8, 0.05, 5.2, 2.7, 4.4, 2.1),
    ],
)
def test_vec2orbElem_recovers_elements_from_3xn_input(a, e, O, I, w, mu):
    r, v = _orbit_to_state(E_SAMPLES, mu, a, e, O, I, w)
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = fun.vec2orbElem(r, v, mu)

    n = E_SAMPLES.size
    tol = dict(rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(a_out, np.full(n, a), **tol)
    np.testing.assert_allclose(e_out, np.full(n, e), **tol)
    np.testing.assert_allclose(E_out, E_SAMPLES, **tol)
    np.testing.assert_allclose(O_out, np.full(n, O), **tol)
    np.testing.assert_allclose(I_out, np.full(n, I), **tol)
    np.testing.assert_allclose(w_out, np.full(n, w), **tol)


def test_vec2orbElem_nx3_input_gives_same_result_as_3xn():
    a, e, O, I, w, mu = 1.7, 0.4, 0.9, 1.2, 2.2, 1.0
    r, v = _orbit_to_state(E_SAMPLES, mu, a, e, O, I, w)

    res_3xn = fun.vec2orbElem(r, v, mu)
    res_nx3 = fun.vec2orbElem(r.T.copy(), v.T.copy(), mu)

    for x, y in zip(res_3xn, res_nx3):
        np.testing.assert_allclose(y, x, rtol=1e-12, atol=1e-12)


def test_vec2orbElem_vector_mus_recovers_each_body():
    r, v = fun.orbElem2vec(
        E_ARR, MU_ARR, orbElem=(A_ARR, ECC_ARR, O_ARR, I_ARR, W_ARR)
    )
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = fun.vec2orbElem(r, v, MU_ARR)

    tol = dict(rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(a_out, A_ARR, **tol)
    np.testing.assert_allclose(e_out, ECC_ARR, **tol)
    np.testing.assert_allclose(E_out, E_ARR, **tol)
    np.testing.assert_allclose(O_out, O_ARR, **tol)
    np.testing.assert_allclose(I_out, I_ARR, **tol)
    np.testing.assert_allclose(w_out, W_ARR, **tol)


def test_vec2orbElem_angles_lie_in_expected_ranges():
    a, e, O, I, w, mu = 1.7, 0.4, 5.2, 2.7, 4.4, 1.0
    r, v = _orbit_to_state(E_SAMPLES, mu, a, e, O, I, w)
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = fun.vec2orbElem(r, v, mu)

    assert np.all(a_out > 0)
    assert np.all((e_out >= 0) & (e_out < 1))
    assert np.all((E_out >= 0) & (E_out < 2 * np.pi))
    assert np.all((O_out >= 0) & (O_out < 2 * np.pi))
    assert np.all((w_out >= 0) & (w_out < 2 * np.pi))
    assert np.all((I_out >= 0) & (I_out <= np.pi))
    assert np.all(P_out > 0)


def test_vec2orbElem_period_and_time_of_periapsis_are_consistent():
    a, e, O, I, w, mu = 1.7, 0.4, 0.9, 1.2, 2.2, 1.0
    r, v = _orbit_to_state(E_SAMPLES, mu, a, e, O, I, w)
    _, e_out, E_out, _, _, _, P_out, tau_out = fun.vec2orbElem(r, v, mu)

    n_motion = np.sqrt(mu / a ** 3)
    # Period: P * n = 2*pi
    np.testing.assert_allclose(P_out * n_motion, np.full(E_SAMPLES.size, 2 * np.pi),
                               rtol=1e-9, atol=1e-9)
    # Time of periapsis: tau * n = -(E - e sin E), i.e. -mean anomaly
    np.testing.assert_allclose(tau_out * n_motion,
                               -(E_out - e_out * np.sin(E_out)),
                               rtol=1e-9, atol=1e-9)


def test_vec2orbElem_mismatched_rs_vs_sizes_is_rejected():
    rs = np.ones((3, 2))   # 6 elements
    vs = np.ones((3, 3))   # 9 elements
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_size_not_multiple_of_three_is_rejected():
    rs = np.ones((2, 2))   # 4 elements
    vs = np.ones((2, 2))
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_three_dimensional_arrays():
    rs = np.ones((3, 2, 1))
    vs = np.ones((3, 2, 1))
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_arrays_without_axis_of_length_three():
    rs = np.ones((1, 9))   # size is a multiple of 3, but no axis has length 3
    vs = np.ones((1, 9))
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, 1.0)
