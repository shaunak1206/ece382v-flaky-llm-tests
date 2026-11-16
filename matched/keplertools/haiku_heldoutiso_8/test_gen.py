```python
"""Unit tests for orbElem2vec and vec2orbElem in keplertools/fun.py."""
import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem

TWO_PI = 2.0 * np.pi
ANGLE_TOL = 1e-9


def _scalar_elems(a, e, O, I, w):
    """Orbital elements as numpy scalars (calcAB requires a .size attribute)."""
    return tuple(np.float64(x) for x in (a, e, O, I, w))


def _array_elems(a, e, O, I, w):
    """Orbital elements as float arrays."""
    return tuple(np.asarray(x, dtype=float) for x in (a, e, O, I, w))


def _angle_err(actual, expected):
    """Smallest absolute angular difference, wrapped to [-pi, pi]."""
    diff = np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)
    return np.abs(np.angle(np.exp(1j * diff)))


def assert_angles_close(actual, expected, tol=ANGLE_TOL):
    err = _angle_err(actual, expected)
    assert np.all(err < tol), "angular mismatch: max error %g" % np.max(err)


def _three_body_truth():
    a = np.array([1.0, 2.5, 4.0])
    e = np.array([0.5, 0.3, 0.1])
    O = np.array([0.2, 1.5, 4.0])
    I = np.array([0.9, 1.2, 2.5])
    w = np.array([0.4, 3.0, 5.5])
    E = np.array([1.0, 2.0, 4.5])
    mu = np.array([1.0, 2.0, 3.0])
    return a, e, O, I, w, E, mu


def _two_body_truth():
    a = np.array([1.0, 2.5])
    e = np.array([0.5, 0.3])
    O = np.array([0.2, 1.5])
    I = np.array([0.9, 1.2])
    w = np.array([0.4, 3.0])
    E = np.array([1.0, 2.0])
    mu = np.array([1.0, 2.0])
    return a, e, O, I, w, E, mu


# ---------------------------------------------------------------------------
# orbElem2vec
# ---------------------------------------------------------------------------

def test_orbElem2vec_hand_computed_state_at_quarter_period():
    # a=1, e=0.5, O=0, I=pi/2, w=0, E=pi/2 -> r=(-0.5, 0, sqrt(0.75)), v=(-1, 0, 0)
    elems = _scalar_elems(1.0, 0.5, 0.0, np.pi / 2, 0.0)
    rs, vs = orbElem2vec(np.pi / 2, 1.0, orbElem=elems)
    assert rs.shape == (3, 1)
    assert vs.shape == (3, 1)
    np.testing.assert_allclose(rs.ravel(), [-0.5, 0.0, np.sqrt(0.75)], atol=1e-12)
    np.testing.assert_allclose(vs.ravel(), [-1.0, 0.0, 0.0], atol=1e-12)


def test_orbElem2vec_returnAB_gives_calcAB_output():
    elems = _array_elems(*_three_body_truth()[:5])
    E = np.array([1.0, 2.0, 4.5])
    mus = np.array([1.0, 2.0, 3.0])
    rs, vs, (A, B) = orbElem2vec(E, mus, orbElem=elems, returnAB=True)
    A_ref, B_ref = calcAB(*elems)
    np.testing.assert_allclose(A, A_ref)
    np.testing.assert_allclose(B, B_ref)
    assert rs.shape == (3, 3)
    assert vs.shape == (3, 3)


def test_orbElem2vec_vector_E_scalar_mu_columns_match_scalar_calls():
    elems = _scalar_elems(2.0, 0.4, 0.3, 0.8, 1.1)
    E = np.linspace(0.0, TWO_PI, 7)
    rs, vs = orbElem2vec(E, 1.5, orbElem=elems)
    assert rs.shape == (3, 7)
    assert vs.shape == (3, 7)
    for j, Ej in enumerate(E):
        r1, v1 = orbElem2vec(Ej, 1.5, orbElem=elems)
        np.testing.assert_allclose(rs[:, j], r1.ravel(), atol=1e-12)
        np.testing.assert_allclose(vs[:, j], v1.ravel(), atol=1e-12)


def test_orbElem2vec_vector_E_and_mu_matches_per_body_calls():
    elems_arr = _array_elems(*_three_body_truth()[:5])
    E = np.array([1.0, 2.0, 4.5])
    mus = np.array([1.0, 2.0, 3.0])
    rs, vs = orbElem2vec(E, mus, orbElem=elems_arr)
    assert rs.shape == (3, 3)
    for j in range(3):
        single = tuple(np.float64(x[j]) for x in elems_arr)
        r1, v1 = orbElem2vec(E[j], mus[j], orbElem=single)
        np.testing.assert_allclose(rs[:, j], r1.ravel(), atol=1e-12)
        np.testing.assert_allclose(vs[:, j], v1.ravel(), atol=1e-12)


def test_orbElem2vec_AB_input_matches_orbElem_input_multi_body():
    elems = _array_elems(*_three_body_truth()[:5])
    E = np.array([1.0, 2.0, 4.5])
    mus = np.array([1.0, 2.0, 3.0])
    rs_ref, vs_ref = orbElem2vec(E, mus, orbElem=elems)
    A, B = calcAB(*elems)
    rs, vs = orbElem2vec(E, mus, AB=(A, B))
    np.testing.assert_allclose(rs, rs_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(vs, vs_ref, rtol=1e-10, atol=1e-12)


def test_orbElem2vec_AB_input_scalar_E_matches_orbElem_input():
    elems = _scalar_elems(1.0, 0.5, 0.0, np.pi / 2, 0.0)
    rs_ref, vs_ref = orbElem2vec(0.7, 1.0, orbElem=elems)
    A, B = calcAB(*elems)
    rs, vs = orbElem2vec(0.7, 1.0, AB=(A, B))
    np.testing.assert_allclose(rs, rs_ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(vs, vs_ref, rtol=1e-10, atol=1e-12)


def test_orbElem2vec_radius_matches_conic_equation():
    a, e = 2.0, 0.4
    elems = _scalar_elems(a, e, 0.3, 0.8, 1.1)
    E = np.linspace(0.0, TWO_PI, 13)
    rs, _ = orbElem2vec(E, 1.5, orbElem=elems)
    r = np.linalg.norm(rs, axis=0)
    np.testing.assert_allclose(r, a * (1.0 - e * np.cos(E)), rtol=1e-10, atol=1e-12)


def test_orbElem2vec_satisfies_vis_viva():
    a, mu = 2.0, 1.5
    elems = _scalar_elems(a, 0.4, 0.3, 0.8, 1.1)
    E = np.linspace(0.0, TWO_PI, 13)
    rs, vs = orbElem2vec(E, mu, orbElem=elems)
    r = np.linalg.norm(rs, axis=0)
    v2 = np.sum(vs ** 2, axis=0)
    np.testing.assert_allclose(v2, mu * (2.0 / r - 1.0 / a), rtol=1e-10)


def test_orbElem2vec_equatorial_orbit_stays_in_plane():
    elems = _scalar_elems(1.0, 0.2, 0.7, 0.0, 0.4)
    E = np.linspace(0.0, TWO_PI, 9)
    rs, vs = orbElem2vec(E, 1.0, orbElem=elems)
    np.testing.assert_allclose(rs[2], 0.0, atol=1e-14)
    np.testing.assert_allclose(vs[2], 0.0, atol=1e-14)


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    elems = _scalar_elems(1.0, 0.5, 0.0, np.pi / 2, 0.0)
    A, B = calcAB(*elems)
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, 1.0, orbElem=elems, AB=(A, B))


def test_orbElem2vec_scalar_E_requires_scalar_mus():
    elems = _scalar_elems(1.0, 0.5, 0.0, np.pi / 2, 0.0)
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=elems)


def test_orbElem2vec_rejects_mus_size_mismatch_with_vector_E():
    elems = _array_elems(*_three_body_truth()[:5])
    E = np.array([0.1, 0.2, 0.3])
    with pytest.raises(AssertionError):
        orbElem2vec(E, np.array([1.0, 2.0]), orbElem=elems)


def test_orbElem2vec_rejects_AB_with_wrong_size():
    E = np.array([0.1, 0.2])
    mus = np.array([1.0, 1.0])
    A = np.array([[1.0], [0.0], [0.0]])  # 3x1 but two anomalies requires 3x2
    B = np.array([[0.0], [0.0], [0.5]])
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, AB=(A, B))


# ---------------------------------------------------------------------------
# vec2orbElem
# ---------------------------------------------------------------------------

def test_vec2orbElem_hand_computed_elements_single_body():
    # State from the orbElem2vec hand case: a=1, e=0.5, O=0, I=pi/2, w=0, E=pi/2
    b = np.sqrt(0.75)
    rs = np.array([[-0.5], [0.0], [b]])
    vs = np.array([[-1.0], [0.0], [0.0]])
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, 1.0)
    np.testing.assert_allclose(a, [1.0], rtol=1e-10)
    np.testing.assert_allclose(e, [0.5], rtol=1e-10)
    assert_angles_close(E, [np.pi / 2])
    assert_angles_close(O, [0.0])
    assert_angles_close(I, [np.pi / 2])
    assert_angles_close(w, [0.0])
    np.testing.assert_allclose(P, [TWO_PI], rtol=1e-10)
    np.testing.assert_allclose(tau, [-(np.pi / 2 - 0.5)], rtol=1e-10)


def test_vec2orbElem_recovers_elements_from_3xn_input():
    a_in, e_in, O_in, I_in, w_in, E_in, mu_in = _three_body_truth()
    rs, vs = orbElem2vec(E_in, mu_in, orbElem=(a_in, e_in, O_in, I_in, w_in))
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, mu_in)

    for out in (a, e, E, O, I, w, P, tau):
        assert out.shape == (3,)

    np.testing.assert_allclose(a, a_in, rtol=1e-9)
    np.testing.assert_allclose(e, e_in, rtol=1e-9)
    assert_angles_close(E, E_in)
    assert_angles_close(O, O_in)
    assert_angles_close(I, I_in)
    assert_angles_close(w, w_in)
    np.testing.assert_allclose(P, TWO_PI * np.sqrt(a_in ** 3 / mu_in), rtol=1e-9)
    np.testing.assert_allclose(
        tau, -(E_in - e_in * np.sin(E_in)) / np.sqrt(mu_in / a_in ** 3), rtol=1e-9
    )


def test_vec2orbElem_accepts_nx3_layout():
    a_in, e_in, O_in, I_in, w_in, E_in, mu_in = _two_body_truth()
    rs, vs = orbElem2vec(E_in, mu_in, orbElem=(a_in, e_in, O_in, I_in, w_in))
    out_nx3 = vec2orbElem(rs.T, vs.T, mu_in)
    out_3xn = vec2orbElem(rs, vs, mu_in)
    for got, ref in zip(out_nx3, out_3xn):
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out_nx3[0], a_in, rtol=1e-9)


def test_vec2orbElem_scalar_mus_matches_equal_array_mus():
    a_in, e_in, O_in, I_in, w_in, E_in, _ = _three_body_truth()
    mu_one = np.ones(3)
    rs, vs = orbElem2vec(E_in, mu_one, orbElem=(a_in, e_in, O_in, I_in, w_in))
    out_scalar = vec2orbElem(rs, vs, 1.0)
    out_array = vec2orbElem(rs, vs, mu_one)
    for got, ref in zip(out_scalar, out_array):
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


@pytest.mark.xfail(
    strict=False,
    reason="1-D stacked input computes nplanets as a float and passes it to np.reshape",
)
def test_vec2orbElem_accepts_1d_stacked_input():
    a_in, e_in, O_in, I_in, w_in, E_in, mu_in = _two_body_truth()
    rs, vs = orbElem2vec(E_in, mu_in, orbElem=(a_in, e_in, O_in, I_in, w_in))
    a, *_ = vec2orbElem(rs.T.ravel(), vs.T.ravel(), mu_in)
    np.testing.assert_allclose(a, a_in, rtol=1e-9)


def test_vec2orbElem_rejects_size_mismatch_between_rs_and_vs():
    rs = np.ones((3, 2))
    vs = np.ones((3, 3))
    with pytest.ra