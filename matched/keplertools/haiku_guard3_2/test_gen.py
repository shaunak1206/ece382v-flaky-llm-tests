import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem


def _elements_single():
    """Orbital elements (a, e, O, I, w) for a single body, as size-1 arrays."""
    return (
        np.array([2.0]),
        np.array([0.3]),
        np.array([0.4]),
        np.array([0.7]),
        np.array([1.1]),
    )


def _elements_n():
    """Orbital elements (a, e, O, I, w) for three bodies, as size-3 arrays."""
    return (
        np.array([1.0, 1.5, 2.0]),
        np.array([0.2, 0.4, 0.1]),
        np.array([0.4, 1.0, 2.0]),
        np.array([0.7, 0.2, 2.5]),
        np.array([1.1, 0.3, 4.0]),
    )


def _assert_angles_close(actual, expected, atol=1e-8):
    """Compare angles modulo 2*pi."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    diff = np.arctan2(np.sin(actual - expected), np.cos(actual - expected))
    np.testing.assert_allclose(diff, 0.0, atol=atol)


# ---------------------------------------------------------------------------
# orbElem2vec
# ---------------------------------------------------------------------------

def test_orbElem2vec_circular_orbit_at_E_zero():
    elems = (np.array([1.0]), np.array([0.0]), np.array([0.0]),
             np.array([0.0]), np.array([0.0]))
    r, v = orbElem2vec(0.0, 1.0, orbElem=elems)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    np.testing.assert_allclose(r, [[1.0], [0.0], [0.0]], atol=1e-12)
    np.testing.assert_allclose(v, [[0.0], [1.0], [0.0]], atol=1e-12)


def test_orbElem2vec_circular_orbit_at_quarter_period_point():
    elems = (np.array([1.0]), np.array([0.0]), np.array([0.0]),
             np.array([0.0]), np.array([0.0]))
    r, v = orbElem2vec(np.pi / 2, 1.0, orbElem=elems)
    np.testing.assert_allclose(r, [[0.0], [1.0], [0.0]], atol=1e-12)
    np.testing.assert_allclose(v, [[-1.0], [0.0], [0.0]], atol=1e-12)


def test_orbElem2vec_returnAB_flag_controls_output_length():
    elems = _elements_single()
    out_default = orbElem2vec(0.7, 1.0, orbElem=elems)
    assert len(out_default) == 2

    r, v, (A, B) = orbElem2vec(0.7, 1.0, orbElem=elems, returnAB=True)
    A_ref, B_ref = calcAB(*elems)
    np.testing.assert_allclose(A, A_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(B, B_ref, rtol=1e-12, atol=1e-12)


def test_orbElem2vec_vector_E_scalar_mu_matches_scalar_calls():
    elems = _elements_single()
    E = np.array([0.1, 0.9, 2.0, 3.3])
    mu = 3.0
    r, v = orbElem2vec(E, mu, orbElem=elems)
    assert r.shape == (3, 4)
    assert v.shape == (3, 4)
    for j in range(E.size):
        r_j, v_j = orbElem2vec(E[j], mu, orbElem=elems)
        np.testing.assert_allclose(r[:, j], r_j[:, 0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(v[:, j], v_j[:, 0], rtol=1e-12, atol=1e-12)


def test_orbElem2vec_vector_mu_matches_per_body_calls():
    elems = _elements_n()
    E = np.array([0.3, 1.2, 2.5])
    mus = np.array([1.0, 2.0, 0.5])
    r, v = orbElem2vec(E, mus, orbElem=elems)
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    for j in range(3):
        elems_j = tuple(np.array([x[j]]) for x in elems)
        r_j, v_j = orbElem2vec(E[j], mus[j], orbElem=elems_j)
        np.testing.assert_allclose(r[:, j], r_j[:, 0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(v[:, j], v_j[:, 0], rtol=1e-12, atol=1e-12)


def test_orbElem2vec_AB_input_matches_orbElem_input():
    elems = _elements_single()
    E = np.linspace(0.2, 5.0, 5)
    A, B = calcAB(*elems)
    r1, v1 = orbElem2vec(E, 1.5, orbElem=elems)
    r2, v2 = orbElem2vec(E, 1.5, AB=(A, B))
    assert r2.shape == r1.shape
    np.testing.assert_allclose(r2, r1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v2, v1, rtol=1e-10, atol=1e-12)


def test_orbElem2vec_vis_viva_and_radius_formula():
    elems = _elements_single()  # a = 2.0, e = 0.3
    a, e = 2.0, 0.3
    mu = 3.0
    E = np.linspace(0.1, 6.0, 7)
    r, v = orbElem2vec(E, mu, orbElem=elems)

    r_mag = np.sqrt(np.sum(r ** 2, axis=0))
    v2 = np.sum(v ** 2, axis=0)

    np.testing.assert_allclose(r_mag, a * (1 - e * np.cos(E)), rtol=1e-10)
    np.testing.assert_allclose(v2, mu * (2.0 / r_mag - 1.0 / a), rtol=1e-10)


def test_orbElem2vec_velocity_scales_with_sqrt_of_mu():
    elems = _elements_single()
    E = np.linspace(0.2, 5.5, 6)
    r1, v1 = orbElem2vec(E, 1.0, orbElem=elems)
    r4, v4 = orbElem2vec(E, 4.0, orbElem=elems)
    np.testing.assert_allclose(r4, r1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(v4, 2.0 * v1, rtol=1e-12, atol=1e-12)


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2]), 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    elems = _elements_single()
    AB = calcAB(*elems)
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2]), 1.0, orbElem=elems, AB=AB)


def test_orbElem2vec_rejects_scalar_E_with_array_mu():
    elems = _elements_single()
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=elems)


def test_orbElem2vec_rejects_mu_size_not_matching_E():
    elems = _elements_n()
    E = np.array([0.1, 0.2, 0.3])
    mus = np.array([1.0, 2.0])
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, orbElem=elems)


# ---------------------------------------------------------------------------
# vec2orbElem
# ---------------------------------------------------------------------------

def test_vec2orbElem_analytic_inclined_orbit_at_periapsis():
    # Ellipse with a = 1, e = 0.5, mu = 1, inclination pi/3, node along x-axis,
    # argument of pericenter 0, and the body at periapsis (E = 0).
    I = np.pi / 3
    rs = np.array([[0.5], [0.0], [0.0]])
    vs = np.array([[0.0], [np.sqrt(3) * np.cos(I)], [np.sqrt(3) * np.sin(I)]])

    a, e, E, O, Inc, w, P, tau = vec2orbElem(rs, vs, 1.0)

    np.testing.assert_allclose(a, [1.0], rtol=1e-10)
    np.testing.assert_allclose(e, [0.5], rtol=1e-10)
    np.testing.assert_allclose(Inc, [I], rtol=1e-10)
    _assert_angles_close(O, [0.0])
    _assert_angles_close(w, [0.0])
    _assert_angles_close(E, [0.0])
    np.testing.assert_allclose(P, [2 * np.pi], rtol=1e-10)
    np.testing.assert_allclose(tau, [0.0], atol=1e-10)


def test_vec2orbElem_round_trip_scalar_mu():
    elems = _elements_single()
    a0, e0, O0, I0, w0 = 2.0, 0.3, 0.4, 0.7, 1.1
    mu = 2.5
    E_in = np.array([0.3, 1.2, 2.5, 3.7, 5.1])

    rs, vs = orbElem2vec(E_in, mu, orbElem=elems)
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, mu)

    n = E_in.size
    np.testing.assert_allclose(a, np.full(n, a0), rtol=1e-9)
    np.testing.assert_allclose(e, np.full(n, e0), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(I, np.full(n, I0), rtol=1e-9)
    _assert_angles_close(O, np.full(n, O0))
    _assert_angles_close(w, np.full(n, w0))
    _assert_angles_close(E, E_in)
    np.testing.assert_allclose(P, np.full(n, 2 * np.pi * np.sqrt(a0 ** 3 / mu)),
                               rtol=1e-9)
    np.testing.assert_allclose(
        tau, -(E_in - e0 * np.sin(E_in)) * np.sqrt(a0 ** 3 / mu), rtol=1e-8,
        atol=1e-10)


def test_vec2orbElem_round_trip_vector_mu():
    elems = _elements_n()
    a0, e0, O0, I0, w0 = elems
    E_in = np.array([0.3, 1.2, 2.5])
    mus = np.array([1.0, 2.0, 0.5])

    rs, vs = orbElem2vec(E_in, mus, orbElem=elems)
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, mus)

    np.testing.assert_allclose(a, a0, rtol=1e-9)
    np.testing.assert_allclose(e, e0, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(I, I0, rtol=1e-9)
    _assert_angles_close(O, O0)
    _assert_angles_close(w, w0)
    _assert_angles_close(E, E_in)
    np.testing.assert_allclose(P, 2 * np.pi * np.sqrt(a0 ** 3 / mus), rtol=1e-9)
    np.testing.assert_allclose(
        tau, -(E_in - e0 * np.sin(E_in)) * np.sqrt(a0 ** 3 / mus), rtol=1e-8,
        atol=1e-10)


def test_vec2orbElem_nx3_input_matches_3xn_input():
    elems = _elements_single()
    E = np.linspace(0.4, 5.6, 5)
    mu = 1.7
    rs, vs = orbElem2vec(E, mu, orbElem=elems)  # both 3 x n

    out_3xn = vec2orbElem(rs, vs, mu)
    out_nx3 = vec2orbElem(rs.T, vs.T, mu)  # both n x 3

    assert len(out_3xn) == len(out_nx3) == 8
    for got, ref in zip(out_nx3, out_3xn):
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_vec2orbElem_rejects_mismatched_rs_and_vs_sizes():
    rs = np.zeros((3, 2))
    vs = np.zeros((3, 3))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_more_than_two_dimensions():
    rs = np.ones((3, 1, 2))
    vs = np.ones((3, 1, 2))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_mu_array_of_wrong_size():
    rs = np.ones((3, 3))
    vs = np.ones((3, 3))
    mus = np.array([1.0, 2.0])  # three bodies, but only two mus
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, mus)
