import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def _single(a, e, O, I, w):
    """Return orbital elements (a, e, O, I, w) as size-1 arrays, as calcAB requires."""
    return (np.array([a], dtype=float),
            np.array([e], dtype=float),
            np.array([O], dtype=float),
            np.array([I], dtype=float),
            np.array([w], dtype=float))


def _assert_same_angle(actual, expected, atol=1e-9):
    np.testing.assert_allclose(np.cos(actual), np.cos(expected), atol=atol)
    np.testing.assert_allclose(np.sin(actual), np.sin(expected), atol=atol)


# ---------------------------------------------------------------------------
# orbElem2vec
# ---------------------------------------------------------------------------

def test_orbElem2vec_scalar_E_scalar_mu_returns_3x1():
    r, v = orbElem2vec(0.7, 1.0, orbElem=_single(1.5, 0.3, 1.1, 0.4, 2.0))
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)


def test_orbElem2vec_equatorial_closed_form():
    # With O = I = w = 0 the perifocal frame coincides with the inertial frame.
    a, e, mu = 2.0, 0.5, 1.0
    E = np.array([0.0, np.pi / 2, np.pi])
    r, v = orbElem2vec(E, mu, orbElem=_single(a, e, 0.0, 0.0, 0.0))

    b = a * np.sqrt(1 - e ** 2)
    rr = a * (1 - e * np.cos(E))
    expected_r = np.vstack((a * (np.cos(E) - e), b * np.sin(E), np.zeros(3)))
    expected_v = np.sqrt(mu * a) / rr * np.vstack(
        (-np.sin(E), (b / a) * np.cos(E), np.zeros(3)))

    assert r.shape == (3, 3)
    assert v.shape == (3, 3)
    np.testing.assert_allclose(r, expected_r, atol=1e-12)
    np.testing.assert_allclose(v, expected_v, atol=1e-12)


def test_orbElem2vec_satisfies_vis_viva_and_angular_momentum():
    a, e, O, I, w, mu = 1.5, 0.3, 1.1, 0.4, 2.0, 1.0
    E = np.linspace(0.1, 6.0, 7)
    r, v = orbElem2vec(E, mu, orbElem=_single(a, e, O, I, w))

    rnorm = np.linalg.norm(r, axis=0)
    v2 = np.sum(v ** 2, axis=0)
    np.testing.assert_allclose(v2, mu * (2.0 / rnorm - 1.0 / a), rtol=1e-10)

    h = np.cross(r.T, v.T)  # (n, 3)
    hmag = np.linalg.norm(h, axis=1)
    np.testing.assert_allclose(hmag, np.sqrt(mu * a * (1 - e ** 2)), rtol=1e-10)
    np.testing.assert_allclose(h[:, 2] / hmag, np.cos(I), rtol=1e-10)


def test_orbElem2vec_vector_E_scalar_mu_shape():
    E = np.linspace(0.0, 2 * np.pi, 9)
    r, v = orbElem2vec(E, 1.0, orbElem=_single(1.2, 0.1, 0.5, 0.9, 0.3))
    assert r.shape == (3, 9)
    assert v.shape == (3, 9)


def test_orbElem2vec_multi_body_matches_single_body_columns():
    a = np.array([1.0, 1.5, 2.0, 3.0])
    e = np.array([0.1, 0.3, 0.5, 0.2])
    O = np.array([0.1, 1.0, 2.0, 3.0])
    I = np.array([0.2, 0.5, 1.0, 2.5])
    w = np.array([0.3, 1.5, 2.5, 4.0])
    mus = np.array([1.0, 2.0, 0.5, 3.0])
    E = np.array([0.4, 1.2, 2.2, 5.0])

    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 4)
    assert v.shape == (3, 4)

    for j in range(4):
        rj, vj = orbElem2vec(E[j], mus[j],
                             orbElem=(a[j:j + 1], e[j:j + 1], O[j:j + 1],
                                      I[j:j + 1], w[j:j + 1]))
        np.testing.assert_allclose(r[:, j], rj[:, 0], atol=1e-12)
        np.testing.assert_allclose(v[:, j], vj[:, 0], atol=1e-12)


def test_orbElem2vec_returnAB_gives_calcAB_output():
    elems = _single(1.5, 0.3, 1.1, 0.4, 2.0)
    E = np.linspace(0.1, 3.0, 4)
    out = orbElem2vec(E, 1.0, orbElem=elems, returnAB=True)
    assert len(out) == 3

    r, v, (A, B) = out
    A_ref, B_ref = calcAB(*elems)
    np.testing.assert_allclose(A, A_ref)
    np.testing.assert_allclose(B, B_ref)
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)


def test_orbElem2vec_AB_input_matches_orbElem_input():
    elems = _single(1.5, 0.3, 1.1, 0.4, 2.0)
    A, B = calcAB(*elems)
    E = np.linspace(0.1, 6.0, 5)

    r1, v1 = orbElem2vec(E, 1.0, orbElem=elems)
    r2, v2 = orbElem2vec(E, 1.0, AB=(A, B))
    np.testing.assert_allclose(r2, r1, atol=1e-10)
    np.testing.assert_allclose(v2, v1, atol=1e-10)


def test_orbElem2vec_raises_when_neither_orbElem_nor_AB_given():
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.5]), 1.0)


def test_orbElem2vec_raises_when_both_orbElem_and_AB_given():
    elems = _single(1.5, 0.3, 1.1, 0.4, 2.0)
    AB = calcAB(*elems)
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.5]), 1.0, orbElem=elems, AB=AB)


def test_orbElem2vec_raises_scalar_E_with_array_mu():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]),
                    orbElem=_single(1.5, 0.3, 1.1, 0.4, 2.0))


def test_orbElem2vec_raises_mu_size_mismatch():
    E = np.array([0.1, 0.2, 0.3])
    orbElem = (np.array([1.0, 1.5, 2.0]), np.array([0.1, 0.2, 0.3]),
               np.zeros(3), np.ones(3) * 0.5, np.zeros(3))
    with pytest.raises(AssertionError):
        orbElem2vec(E, np.array([1.0, 2.0]), orbElem=orbElem)


# ---------------------------------------------------------------------------
# vec2orbElem
# ---------------------------------------------------------------------------

def test_vec2orbElem_recovers_elements_single_body():
    a, e, O, I, w, mu = 1.5, 0.3, 1.1, 0.4, 2.0, 1.0
    E_in = np.linspace(0.1, 6.0, 5)
    r, v = orbElem2vec(E_in, mu, orbElem=_single(a, e, O, I, w))

    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = vec2orbElem(r, v, mu)

    np.testing.assert_allclose(a_out, a, rtol=1e-10)
    np.testing.assert_allclose(e_out, e, rtol=1e-10)
    np.testing.assert_allclose(I_out, I, rtol=1e-10)
    _assert_same_angle(O_out, O)
    _assert_same_angle(w_out, w)
    _assert_same_angle(E_out, E_in)


def test_vec2orbElem_accepts_nx3_layout():
    a, e, O, I, w, mu = 1.5, 0.3, 1.1, 0.4, 2.0, 1.0
    E_in = np.linspace(0.1, 6.0, 5)
    r, v = orbElem2vec(E_in, mu, orbElem=_single(a, e, O, I, w))

    out = vec2orbElem(r.T, v.T, mu)
    a_out, e_out, E_out, O_out, I_out, w_out = out[:6]

    np.testing.assert_allclose(a_out, a, rtol=1e-10)
    np.testing.assert_allclose(e_out, e, rtol=1e-10)
    np.testing.assert_allclose(I_out, I, rtol=1e-10)
    _assert_same_angle(O_out, O)
    _assert_same_angle(w_out, w)
    _assert_same_angle(E_out, E_in)


def test_vec2orbElem_multi_body_with_vector_mus():
    a = np.array([1.0, 1.5, 2.0, 3.0])
    e = np.array([0.1, 0.3, 0.5, 0.2])
    O = np.array([0.1, 1.0, 2.0, 3.0])
    I = np.array([0.2, 0.5, 1.0, 2.5])
    w = np.array([0.3, 1.5, 2.5, 4.0])
    mus = np.array([1.0, 2.0, 0.5, 3.0])
    E_in = np.array([0.4, 1.2, 2.2, 5.0])

    r, v = orbElem2vec(E_in, mus, orbElem=(a, e, O, I, w))
    a_out, e_out, E_out, O_out, I_out, w_out, P_out, tau_out = vec2orbElem(r, v, mus)

    np.testing.assert_allclose(a_out, a, rtol=1e-10)
    np.testing.assert_allclose(e_out, e, rtol=1e-10)
    np.testing.assert_allclose(I_out, I, rtol=1e-10)
    _assert_same_angle(O_out, O)
    _assert_same_angle(w_out, w)
    _assert_same_angle(E_out, E_in)


def test_vec2orbElem_period_and_periapsis_time():
    a, e, O, I, w, mu = 1.5, 0.3, 1.1, 0.4, 2.0, 1.0
    E_in = np.linspace(0.1, 6.0, 5)  # within [0, 2*pi)
    r, v = orbElem2vec(E_in, mu, orbElem=_single(a, e, O, I, w))

    out = vec2orbElem(r, v, mu)
    P_out, tau_out = out[6], out[7]

    n = np.sqrt(mu / a ** 3)
    np.testing.assert_allclose(P_out, 2 * np.pi / n, rtol=1e-10)
    np.testing.assert_allclose(tau_out, -(E_in - e * np.sin(E_in)) / n, rtol=1e-8, atol=1e-10)


def test_vec2orbElem_outputs_have_one_entry_per_body():
    E_in = np.linspace(0.1, 6.0, 5)
    r, v = orbElem2vec(E_in, 1.0, orbElem=_single(1.2, 0.1, 0.5, 0.9, 0.3))
    out = vec2orbElem(r, v, 1.0)
    assert len(out) == 8
    for arr in out:
        assert np.shape(arr) == (5,)


def test_vec2orbElem_raises_when_size_not_multiple_of_3():
    rs = np.zeros((2, 4))
    vs = np.zeros((2, 4))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_raises_when_rs_and_vs_sizes_differ():
    rs = np.ones((3, 4))
    vs = np.ones((3, 3))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_raises_when_mus_size_mismatch():
    rs = np.ones((3, 4))
    vs = np.ones((3, 4))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, np.array([1.0, 2.0]))


def test_vec2orbElem_raises_for_three_dimensional_input():
    rs = np.zeros((3, 2, 2))
    vs = np.zeros((3, 2, 2))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_raises_when_no_dimension_of_length_3():
    rs = np.ones((2, 6))
    vs = np.ones((2, 6))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)
