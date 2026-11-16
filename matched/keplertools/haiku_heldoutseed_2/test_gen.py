import numpy as np
import pytest

from keplertools.fun import calcAB, orbElem2vec, vec2orbElem

TWO_PI = 2.0 * np.pi
S_E_HALF = np.sqrt(1.0 - 0.5 ** 2)  # sqrt(1 - e^2) for e = 0.5


def _angle_close(actual, expected, atol=1e-9):
    """Assert two angle arrays agree modulo 2*pi."""
    diff = np.mod(np.asarray(actual) - np.asarray(expected) + np.pi, TWO_PI) - np.pi
    np.testing.assert_allclose(np.abs(diff), 0.0, atol=atol)


def _generic_elements():
    """Multi-body set of orbital elements with no degenerate angles."""
    a = np.array([1.0, 2.5, 0.7])
    e = np.array([0.3, 0.6, 0.1])
    O = np.array([0.4, 2.0, 5.5])
    I = np.array([0.5, 1.2, 2.8])
    w = np.array([1.1, 3.0, 4.5])
    E = np.array([0.3, 2.0, 5.0])
    mu = np.array([1.0, 2.0, 0.5])
    return a, e, O, I, w, E, mu


def _single_orbit_elements():
    """Single orbit with I = O = w = 0, a = 1, e = 0.5, stored as size-1 arrays."""
    return (
        np.array([1.0]),
        np.array([0.5]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
    )


# ---------------------------------------------------------------------------
# orbElem2vec
# ---------------------------------------------------------------------------

def test_orbElem2vec_scalar_E_gives_periapsis_state():
    r, v = orbElem2vec(0.0, 1.0, orbElem=_single_orbit_elements())
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    np.testing.assert_allclose(r[:, 0], [0.5, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [0.0, np.sqrt(3.0), 0.0], atol=1e-12)


def test_orbElem2vec_vector_E_scalar_mu_single_orbit():
    E = np.array([0.0, np.pi / 2, np.pi])
    r, v = orbElem2vec(E, 1.0, orbElem=_single_orbit_elements())
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)

    s = S_E_HALF
    expected_r = np.array([
        [0.5, -0.5, -1.5],
        [0.0, s, 0.0],
        [0.0, 0.0, 0.0],
    ])
    expected_v = np.array([
        [0.0, -1.0, 0.0],
        [2 * s, 0.0, -2 * s / 3],
        [0.0, 0.0, 0.0],
    ])
    np.testing.assert_allclose(r, expected_r, atol=1e-12)
    np.testing.assert_allclose(v, expected_v, atol=1e-12)


def test_orbElem2vec_multi_body_columns_match_single_body_calls():
    a, e, O, I, w, E, mu = _generic_elements()
    r_multi, v_multi = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert r_multi.shape == (3, 3)
    assert v_multi.shape == (3, 3)

    for i in range(3):
        single_elems = (
            np.array([a[i]]),
            np.array([e[i]]),
            np.array([O[i]]),
            np.array([I[i]]),
            np.array([w[i]]),
        )
        r_single, v_single = orbElem2vec(float(E[i]), float(mu[i]), orbElem=single_elems)
        np.testing.assert_allclose(r_multi[:, i], r_single[:, 0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(v_multi[:, i], v_single[:, 0], rtol=1e-10, atol=1e-12)


def test_orbElem2vec_satisfies_vis_viva():
    a, e, O, I, w, E, mu = _generic_elements()
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    rmag = np.sqrt(np.sum(r ** 2, axis=0))
    v2 = np.sum(v ** 2, axis=0)
    np.testing.assert_allclose(v2, mu * (2.0 / rmag - 1.0 / a), rtol=1e-10)


def test_orbElem2vec_angular_momentum_magnitude():
    a, e, O, I, w, E, mu = _generic_elements()
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    h = np.cross(r.T, v.T)  # shape (n, 3)
    hmag = np.linalg.norm(h, axis=1)
    np.testing.assert_allclose(hmag, np.sqrt(mu * a * (1.0 - e ** 2)), rtol=1e-10)


def test_orbElem2vec_default_returns_two_outputs():
    a, e, O, I, w, E, mu = _generic_elements()
    out = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert len(out) == 2


def test_orbElem2vec_returnAB_returns_three_outputs():
    a, e, O, I, w, E, mu = _generic_elements()
    out = orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)
    assert len(out) == 3
    r, v, AB = out
    A, B = AB
    A_ref, B_ref = calcAB(a, e, O, I, w)
    np.testing.assert_allclose(A, A_ref)
    np.testing.assert_allclose(B, B_ref)
    assert A.shape == (3, 3)
    assert B.shape == (3, 3)


def test_orbElem2vec_AB_input_matches_orbElem_input():
    a, e, O, I, w, E, mu = _generic_elements()
    r1, v1 = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    A, B = calcAB(a, e, O, I, w)
    r2, v2 = orbElem2vec(E, mu, AB=(A, B))
    np.testing.assert_allclose(r2, r1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(v2, v1, rtol=1e-10, atol=1e-12)


def test_orbElem2vec_requires_orbElem_or_AB():
    a, e, O, I, w, E, mu = _generic_elements()
    with pytest.raises(AssertionError):
        orbElem2vec(E, mu)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    a, e, O, I, w, E, mu = _generic_elements()
    AB = calcAB(a, e, O, I, w)
    with pytest.raises(AssertionError):
        orbElem2vec(E, mu, orbElem=(a, e, O, I, w), AB=AB)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    with pytest.raises(AssertionError):
        orbElem2vec(0.0, np.array([1.0]), orbElem=_single_orbit_elements())


def test_orbElem2vec_mus_size_mismatch_raises():
    a, e, O, I, w, E, mu = _generic_elements()
    with pytest.raises(AssertionError):
        orbElem2vec(E, np.array([1.0, 2.0]), orbElem=(a, e, O, I, w))


# ---------------------------------------------------------------------------
# vec2orbElem
# ---------------------------------------------------------------------------

def test_vec2orbElem_periapsis_state_analytic():
    rs = np.array([[1.0], [0.0], [0.0]])
    vs = np.array([[0.0], [1.2], [0.0]])
    with np.errstate(divide="ignore", invalid="ignore"):
        a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, 1.0)

    # Vis-viva: a = 1 / (2 - v^2) with v^2 = 1.44
    expected_a = 1.0 / 0.56
    assert a[0] == pytest.approx(expected_a, rel=1e-12)
    assert e[0] == pytest.approx(0.44, rel=1e-10)
    # Starting at periapsis gives E = 0 and tau = 0
    assert E[0] == pytest.approx(0.0, abs=1e-9)
    assert tau[0] == pytest.approx(0.0, abs=1e-9)
    assert P[0] == pytest.approx(2 * np.pi * np.sqrt(expected_a ** 3), rel=1e-12)


def test_vec2orbElem_round_trip_multi_body():
    a, e, O, I, w, E, mu = _generic_elements()
    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(rs, vs, mu)

    np.testing.assert_allclose(a2, a, rtol=1e-9)
    np.testing.assert_allclose(e2, e, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(I2, I, rtol=1e-9, atol=1e-12)
    _angle_close(O2, O)
    _angle_close(w2, w)
    _angle_close(E2, E)


def test_vec2orbElem_round_trip_single_body_column_inputs():
    elems = (
        np.array([1.3]),
        np.array([0.4]),
        np.array([2.2]),
        np.array([0.9]),
        np.array([4.0]),
    )
    E = 0.7
    mu = 1.5
    rs, vs = orbElem2vec(E, mu, orbElem=elems)
    assert rs.shape == (3, 1)
    assert vs.shape == (3, 1)

    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(rs, vs, mu)
    assert a2.shape == (1,)
    np.testing.assert_allclose(a2, elems[0], rtol=1e-9)
    np.testing.assert_allclose(e2, elems[1], rtol=1e-9)
    np.testing.assert_allclose(I2, elems[3], rtol=1e-9)
    _angle_close(O2, elems[2])
    _angle_close(w2, elems[4])
    _angle_close(E2, np.array([E]))


def test_vec2orbElem_period_and_time_of_periapsis_consistent():
    a, e, O, I, w, E, mu = _generic_elements()
    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(rs, vs, mu)

    n = np.sqrt(mu / a ** 3)
    np.testing.assert_allclose(P2, 2 * np.pi / n, rtol=1e-9)
    # tau = -M / n where M = E - e sin E (E taken from the original input)
    expected_tau = -(E - e * np.sin(E)) / n
    np.testing.assert_allclose(tau2, expected_tau, rtol=1e-8, atol=1e-10)


def test_vec2orbElem_transposed_inputs_give_same_result():
    a = np.array([1.0, 1.5, 2.0, 0.8])
    e = np.array([0.2, 0.4, 0.6, 0.1])
    O = np.array([0.1, 1.0, 2.0, 3.0])
    I = np.array([0.3, 0.9, 1.5, 2.5])
    w = np.array([0.5, 1.5, 2.5, 3.5])
    E = np.array([0.1, 1.0, 2.0, 4.0])
    mu = np.array([1.0, 1.0, 2.0, 3.0])

    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))  # 3 x 4
    out_3xn = vec2orbElem(rs, vs, mu)
    out_nx3 = vec2orbElem(np.ascontiguousarray(rs.T), np.ascontiguousarray(vs.T), mu)

    for x, y in zip(out_3xn, out_nx3):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-14)


def test_vec2orbElem_output_ranges():
    a, e, O, I, w, E, mu = _generic_elements()
    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    _, _, E2, O2, I2, w2, _, _ = vec2orbElem(rs, vs, mu)

    for angle in (E2, O2, w2):
        assert np.all(angle >= 0.0)
        assert np.all(angle <= TWO_PI)
    assert np.all(I2 > 0.0)
    assert np.all(I2 < np.pi)


def test_vec2orbElem_mismatched_sizes_raises():
    rs = np.ones((3, 2))
    vs = np.ones((3, 3))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_size_not_multiple_of_three_raises():
    rs = np.ones((2, 2))
    vs = np.ones((2, 2))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_three_dimensional_input():
    rs = np.ones((3, 2, 1))
    vs = np.ones((3, 2, 1))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_mus_wrong_size_raises():
    rs = np.ones((3, 2))
    vs = np.ones((3, 2))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, np.array([1.0, 2.0, 3.0]))
