import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem

# Shared test data (all angles in radians, mu in consistent units)
A_ELEM = np.array([2.0, 1.5, 3.0, 0.8])
E_ECC = np.array([0.3, 0.1, 0.6, 0.45])
O_NODE = np.array([1.0, 0.2, 2.5, 5.0])
I_INC = np.array([0.5, 1.1, 0.9, 2.0])
W_PERI = np.array([0.7, 2.0, 4.0, 5.5])
E_ANOM = np.array([0.4, 1.2, 2.5, 4.0])
MU_VEC = np.array([1.0, 2.0, 0.5, 3.0])


def _elements(a, e, O, I, w):
    """Return (a, e, O, I, w) as float ndarrays (1-D)."""
    return tuple(np.atleast_1d(np.asarray(x, dtype=float)) for x in (a, e, O, I, w))


# ---------------------------------------------------------------------------
# orbElem2vec
# ---------------------------------------------------------------------------

def test_orbElem2vec_circular_equatorial_known_values():
    orbElem = _elements([1.0], [0.0], [0.0], [0.0], [0.0])
    E = np.array([0.0, np.pi / 2])
    rs, vs = orbElem2vec(E, 1.0, orbElem=orbElem)

    expected_rs = np.array([[1.0, 0.0],
                            [0.0, 1.0],
                            [0.0, 0.0]])
    expected_vs = np.array([[0.0, -1.0],
                            [1.0, 0.0],
                            [0.0, 0.0]])
    np.testing.assert_allclose(rs, expected_rs, atol=1e-12)
    np.testing.assert_allclose(vs, expected_vs, atol=1e-12)


def test_orbElem2vec_scalar_E_returns_3x1_column():
    orbElem = _elements([2.0], [0.3], [1.0], [0.5], [0.7])
    rs, vs = orbElem2vec(0.0, 1.0, orbElem=orbElem)

    assert rs.shape == (3, 1)
    assert vs.shape == (3, 1)
    # At periapsis (E=0) the radius is a*(1-e)
    np.testing.assert_allclose(np.linalg.norm(rs, axis=0), [2.0 * (1 - 0.3)], rtol=1e-12)


def test_orbElem2vec_returnAB_gives_three_outputs():
    orbElem = _elements([2.0], [0.3], [1.0], [0.5], [0.7])
    result = orbElem2vec(E_ANOM, 1.0, orbElem=orbElem, returnAB=True)

    assert len(result) == 3
    rs, vs, AB = result
    A, B = AB
    assert rs.shape == (3, E_ANOM.size)
    assert vs.shape == (3, E_ANOM.size)
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)


def test_orbElem2vec_AB_input_matches_orbElem_input():
    orbElem = _elements([2.0], [0.3], [1.0], [0.5], [0.7])
    A, B = calcAB(*orbElem)

    rs1, vs1 = orbElem2vec(E_ANOM, 1.0, orbElem=orbElem)
    rs2, vs2 = orbElem2vec(E_ANOM, 1.0, AB=(A, B))

    np.testing.assert_allclose(rs2, rs1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(vs2, vs1, rtol=1e-10, atol=1e-12)


def test_orbElem2vec_vis_viva_scalar_mu():
    mu = 1.3
    a, e = 2.0, 0.3
    orbElem = _elements([a], [e], [1.0], [0.5], [0.7])
    rs, vs = orbElem2vec(E_ANOM, mu, orbElem=orbElem)

    r = np.linalg.norm(rs, axis=0)
    v2 = np.sum(vs ** 2, axis=0)
    np.testing.assert_allclose(v2, mu * (2.0 / r - 1.0 / a), rtol=1e-9)


def test_orbElem2vec_vis_viva_vector_mu_per_body():
    orbElem = _elements(A_ELEM, E_ECC, O_NODE, I_INC, W_PERI)
    rs, vs = orbElem2vec(E_ANOM, MU_VEC, orbElem=orbElem)

    assert rs.shape == (3, 4)
    assert vs.shape == (3, 4)
    r = np.linalg.norm(rs, axis=0)
    v2 = np.sum(vs ** 2, axis=0)
    np.testing.assert_allclose(v2, MU_VEC * (2.0 / r - 1.0 / A_ELEM), rtol=1e-9)


def test_orbElem2vec_angular_momentum_direction_matches_inclination_and_node():
    a, e, O, I, w = 2.0, 0.3, 1.0, 0.5, 0.7
    orbElem = _elements([a], [e], [O], [I], [w])
    rs, vs = orbElem2vec(E_ANOM, 1.0, orbElem=orbElem)

    expected_dir = np.array([np.sin(O) * np.sin(I),
                             -np.cos(O) * np.sin(I),
                             np.cos(I)])
    for i in range(rs.shape[1]):
        h = np.cross(rs[:, i], vs[:, i])
        np.testing.assert_allclose(h / np.linalg.norm(h), expected_dir, atol=1e-10)


def test_orbElem2vec_equatorial_orbit_stays_in_plane():
    orbElem = _elements([2.0], [0.3], [0.4], [0.0], [1.1])
    rs, vs = orbElem2vec(E_ANOM, 1.0, orbElem=orbElem)

    np.testing.assert_allclose(rs[2], 0.0, atol=1e-12)
    np.testing.assert_allclose(vs[2], 0.0, atol=1e-12)


def test_orbElem2vec_missing_orbElem_and_AB_raises():
    with pytest.raises(AssertionError):
        orbElem2vec(E_ANOM, 1.0)


def test_orbElem2vec_both_orbElem_and_AB_raises():
    orbElem = _elements([2.0], [0.3], [1.0], [0.5], [0.7])
    A, B = calcAB(*orbElem)
    with pytest.raises(AssertionError):
        orbElem2vec(E_ANOM, 1.0, orbElem=orbElem, AB=(A, B))


def test_orbElem2vec_scalar_E_with_vector_mu_raises():
    orbElem = _elements([2.0], [0.3], [1.0], [0.5], [0.7])
    with pytest.raises(AssertionError):
        orbElem2vec(0.4, np.array([1.0, 2.0]), orbElem=orbElem)


def test_orbElem2vec_mu_size_mismatch_with_E_raises():
    orbElem = _elements(A_ELEM, E_ECC, O_NODE, I_INC, W_PERI)
    with pytest.raises(AssertionError):
        orbElem2vec(E_ANOM, np.array([1.0, 2.0, 3.0]), orbElem=orbElem)


# ---------------------------------------------------------------------------
# vec2orbElem
# ---------------------------------------------------------------------------

def test_vec2orbElem_round_trip_single_orbit_scalar_mu():
    mu = 1.0
    a0, e0, O0, I0, w0 = 2.0, 0.3, 1.0, 0.5, 0.7
    orbElem = _elements([a0], [e0], [O0], [I0], [w0])
    rs, vs = orbElem2vec(E_ANOM, mu, orbElem=orbElem)

    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, mu)

    np.testing.assert_allclose(a, a0, rtol=1e-9)
    np.testing.assert_allclose(e, e0, rtol=1e-9)
    np.testing.assert_allclose(E, E_ANOM, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(O, O0, rtol=1e-9)
    np.testing.assert_allclose(I, I0, rtol=1e-9)
    np.testing.assert_allclose(w, w0, rtol=1e-9)


def test_vec2orbElem_round_trip_multiple_bodies_vector_mu():
    orbElem = _elements(A_ELEM, E_ECC, O_NODE, I_INC, W_PERI)
    rs, vs = orbElem2vec(E_ANOM, MU_VEC, orbElem=orbElem)

    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, MU_VEC)

    np.testing.assert_allclose(a, A_ELEM, rtol=1e-9)
    np.testing.assert_allclose(e, E_ECC, rtol=1e-9)
    np.testing.assert_allclose(E, E_ANOM, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(O, O_NODE, rtol=1e-9)
    np.testing.assert_allclose(I, I_INC, rtol=1e-9)
    np.testing.assert_allclose(w, W_PERI, rtol=1e-9)


def test_vec2orbElem_nx3_input_matches_3xn_input():
    mu = 1.0
    orbElem = _elements([2.0], [0.3], [1.0], [0.5], [0.7])
    rs, vs = orbElem2vec(E_ANOM, mu, orbElem=orbElem)

    out_3xn = vec2orbElem(rs, vs, mu)
    out_nx3 = vec2orbElem(rs.T, vs.T, mu)

    for a_arr, b_arr in zip(out_3xn, out_nx3):
        np.testing.assert_allclose(b_arr, a_arr, rtol=1e-10, atol=1e-12)


def test_vec2orbElem_period_and_periapsis_time_consistent():
    mu = 1.0
    orbElem = _elements([2.0], [0.3], [1.0], [0.5], [0.7])
    rs, vs = orbElem2vec(E_ANOM, mu, orbElem=orbElem)

    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, mu)

    np.testing.assert_allclose(P, 2 * np.pi * np.sqrt(a ** 3 / mu), rtol=1e-9)
    mean_anomaly = E - e * np.sin(E)
    np.testing.assert_allclose(tau, -mean_anomaly / np.sqrt(mu / a ** 3), rtol=1e-9)


def test_vec2orbElem_periapsis_state_gives_a_and_e():
    mu = 1.0
    a0, e0 = 2.0, 0.3
    orbElem = _elements([a0], [e0], [1.0], [0.5], [0.7])
    rs, vs = orbElem2vec(0.0, mu, orbElem=orbElem)

    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, mu)

    np.testing.assert_allclose(a, [a0], rtol=1e-9)
    np.testing.assert_allclose(e, [e0], rtol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(rs, axis=0), [a0 * (1 - e0)], rtol=1e-9)


def test_vec2orbElem_vs_size_mismatch_raises():
    rs = np.ones((3, 4))
    vs = np.ones((3, 3))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_size_not_multiple_of_three_raises():
    rs = np.ones((2, 4))
    vs = np.ones((2, 4))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_mus_wrong_size_raises():
    orbElem = _elements(A_ELEM, E_ECC, O_NODE, I_INC, W_PERI)
    rs, vs = orbElem2vec(E_ANOM, MU_VEC, orbElem=orbElem)
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, np.array([1.0, 2.0, 3.0]))


def test_vec2orbElem_three_dimensional_rs_raises():
    rs = np.ones((3, 2, 2))
    vs = np.ones((3, 4))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_2d_input_without_dimension_three_raises():
    rs = np.ones((2, 6))
    vs = np.ones((2, 6))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)
