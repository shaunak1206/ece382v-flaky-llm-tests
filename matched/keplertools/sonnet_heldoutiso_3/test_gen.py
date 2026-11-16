import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def _angdiff(x, y):
    """Smallest absolute difference between angles (rad), elementwise."""
    d = np.mod(np.asarray(x) - np.asarray(y) + np.pi, 2 * np.pi) - np.pi
    return np.abs(d)


def _arr(*vals):
    return np.array(vals, dtype=float)


def test_orbElem2vec_circular_at_zero_anomaly():
    orbElem = (_arr(1.0), _arr(0.0), _arr(0.0), _arr(0.0), _arr(0.0))
    r, v = orbElem2vec(0.0, 1.0, orbElem=orbElem)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    np.testing.assert_allclose(r[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [0.0, 1.0, 0.0], atol=1e-12)


def test_orbElem2vec_circular_quarter_orbit():
    orbElem = (_arr(1.0), _arr(0.0), _arr(0.0), _arr(0.0), _arr(0.0))
    r, v = orbElem2vec(np.pi / 2, 1.0, orbElem=orbElem)
    np.testing.assert_allclose(r[:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [-1.0, 0.0, 0.0], atol=1e-12)


def test_orbElem2vec_periapsis_and_apoapsis_distance():
    a, e = 2.0, 0.25
    orbElem = (_arr(a), _arr(e), _arr(0.3), _arr(0.6), _arr(1.1))
    E = np.array([0.0, np.pi])
    r, v = orbElem2vec(E, 1.0, orbElem=orbElem)
    rmag = np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(rmag, [a * (1 - e), a * (1 + e)], rtol=1e-12)


def test_orbElem2vec_radius_matches_kepler_formula():
    a, e = 1.7, 0.4
    orbElem = (_arr(a), _arr(e), _arr(1.0), _arr(0.8), _arr(2.0))
    E = np.linspace(0.1, 6.0, 7)
    r, v = orbElem2vec(E, 1.3, orbElem=orbElem)
    assert r.shape == (3, E.size)
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), a * (1 - e * np.cos(E)), rtol=1e-10)


def test_orbElem2vec_vis_viva_and_angular_momentum():
    a, e, mu = 1.7, 0.4, 1.3
    orbElem = (_arr(a), _arr(e), _arr(1.0), _arr(0.8), _arr(2.0))
    E = np.linspace(0.2, 5.5, 6)
    r, v = orbElem2vec(E, mu, orbElem=orbElem)
    rmag = np.linalg.norm(r, axis=0)
    v2 = np.sum(v ** 2, axis=0)
    np.testing.assert_allclose(v2, mu * (2.0 / rmag - 1.0 / a), rtol=1e-10)
    h = np.cross(r.T, v.T)
    hmag = np.linalg.norm(h, axis=1)
    np.testing.assert_allclose(hmag, np.sqrt(mu * a * (1 - e ** 2)) * np.ones(E.size), rtol=1e-10)


def test_orbElem2vec_AB_input_matches_orbElem_input():
    orbElem = (_arr(1.5), _arr(0.3), _arr(1.0), _arr(0.5), _arr(0.7))
    A, B = calcAB(*orbElem)
    E = np.array([0.3, 1.2, 4.0])
    r1, v1 = orbElem2vec(E, 2.0, orbElem=orbElem)
    r2, v2 = orbElem2vec(E, 2.0, AB=(A, B))
    np.testing.assert_allclose(r1, r2, atol=1e-10)
    np.testing.assert_allclose(v1, v2, atol=1e-10)


def test_orbElem2vec_returnAB():
    orbElem = (_arr(1.5), _arr(0.3), _arr(1.0), _arr(0.5), _arr(0.7))
    out = orbElem2vec(np.array([0.5, 1.0]), 1.0, orbElem=orbElem, returnAB=True)
    assert len(out) == 3
    r, v, (A, B) = out
    A0, B0 = calcAB(*orbElem)
    np.testing.assert_allclose(A, A0)
    np.testing.assert_allclose(B, B0)
    out2 = orbElem2vec(np.array([0.5, 1.0]), 1.0, orbElem=orbElem)
    assert len(out2) == 2


def test_orbElem2vec_vector_mus_multiple_bodies():
    n = 3
    a = _arr(1.0, 2.0, 3.0)
    e = _arr(0.1, 0.3, 0.5)
    O = _arr(0.2, 1.0, 2.0)
    I = _arr(0.3, 0.9, 1.4)
    w = _arr(0.5, 1.5, 2.5)
    E = _arr(0.4, 2.0, 5.0)
    mus = _arr(1.0, 2.0, 0.5)
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, n)
    assert v.shape == (3, n)
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), a * (1 - e * np.cos(E)), rtol=1e-10)
    v2 = np.sum(v ** 2, axis=0)
    rmag = np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(v2, mus * (2.0 / rmag - 1.0 / a), rtol=1e-10)


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2]), 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    orbElem = (_arr(1.5), _arr(0.3), _arr(1.0), _arr(0.5), _arr(0.7))
    AB = calcAB(*orbElem)
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2]), 1.0, orbElem=orbElem, AB=AB)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    orbElem = (_arr(1.5), _arr(0.3), _arr(1.0), _arr(0.5), _arr(0.7))
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, _arr(1.0, 2.0), orbElem=orbElem)


def test_orbElem2vec_mus_size_mismatch():
    n = 3
    ones = np.ones(n)
    orbElem = (1.5 * ones, 0.3 * ones, 1.0 * ones, 0.5 * ones, 0.7 * ones)
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2, 0.3]), _arr(1.0, 2.0), orbElem=orbElem)


def test_vec2orbElem_known_state():
    mu = 1.0
    i = 0.4
    vmag = 1.2
    rs = np.array([[1.0], [0.0], [0.0]])
    vs = np.array([[0.0], [vmag * np.cos(i)], [vmag * np.sin(i)]])
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, mu)
    np.testing.assert_allclose(a, [1.0 / 0.56], rtol=1e-10)
    np.testing.assert_allclose(e, [0.44], rtol=1e-10)
    np.testing.assert_allclose(I, [i], rtol=1e-10)
    assert _angdiff(E, 0.0)[0] < 1e-6
    assert _angdiff(O, 0.0)[0] < 1e-6
    assert _angdiff(w, 0.0)[0] < 1e-6
    np.testing.assert_allclose(P, 2 * np.pi * np.sqrt(a ** 3 / mu), rtol=1e-10)
    assert abs(tau[0]) < 1e-6


def test_roundtrip_single_body():
    a0, e0, O0, I0, w0, E0, mu = 1.5, 0.3, 1.0, 0.5, 0.7, 1.2, 2.0
    orbElem = (_arr(a0), _arr(e0), _arr(O0), _arr(I0), _arr(w0))
    r, v = orbElem2vec(E0, mu, orbElem=orbElem)
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, mu)
    np.testing.assert_allclose(a, [a0], rtol=1e-10)
    np.testing.assert_allclose(e, [e0], rtol=1e-10)
    np.testing.assert_allclose(I, [I0], rtol=1e-10)
    assert _angdiff(E, E0)[0] < 1e-9
    assert _angdiff(O, O0)[0] < 1e-9
    assert _angdiff(w, w0)[0] < 1e-9
    np.testing.assert_allclose(P, 2 * np.pi * np.sqrt(a0 ** 3 / mu), rtol=1e-10)
    n = np.sqrt(mu / a0 ** 3)
    np.testing.assert_allclose(tau, [-(E0 - e0 * np.sin(E0)) / n], rtol=1e-9)


def test_roundtrip_many_positions_one_body_and_nx3_input():
    a0, e0, O0, I0, w0, mu = 2.2, 0.45, 4.0, 1.0, 5.5, 1.0
    E0 = np.array([0.1, 1.0, 2.5, 3.5, 5.9])
    orbElem = (_arr(a0), _arr(e0), _arr(O0), _arr(I0), _arr(w0))
    r, v = orbElem2vec(E0, mu, orbElem=orbElem)
    assert r.shape == (3, 5)
    # 3 x n input
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, mu)
    np.testing.assert_allclose(a, a0 * np.ones(5), rtol=1e-9)
    np.testing.assert_allclose(e, e0 * np.ones(5), rtol=1e-9)
    np.testing.assert_allclose(I, I0 * np.ones(5), rtol=1e-9)
    assert np.all(_angdiff(E, E0) < 1e-8)
    assert np.all(_angdiff(O, O0) < 1e-8)
    assert np.all(_angdiff(w, w0) < 1e-8)
    # n x 3 input gives same result
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r.T, v.T, mu)
    np.testing.assert_allclose(a2, a)
    np.testing.assert_allclose(e2, e)
    np.testing.assert_allclose(E2, E)
    np.testing.assert_allclose(O2, O)
    np.testing.assert_allclose(I2, I)
    np.testing.assert_allclose(w2, w)


def test_roundtrip_multiple_bodies_vector_mus():
    a0 = _arr(1.0, 2.0, 3.0, 4.0)
    e0 = _arr(0.1, 0.3, 0.5, 0.2)
    O0 = _arr(0.2, 1.0, 2.0, 5.0)
    I0 = _arr(0.3, 0.9, 1.4, 2.0)
    w0 = _arr(0.5, 1.5, 2.5, 4.5)
    E0 = _arr(0.4, 2.0, 5.0, 3.3)
    mus = _arr(1.0, 2.0, 0.5, 3.0)
    r, v = orbElem2vec(E0, mus, orbElem=(a0, e0, O0, I0, w0))
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, mus)
    np.testing.assert_allclose(a, a0, rtol=1e-9)
    np.testing.assert_allclose(e, e0, rtol=1e-9)
    np.testing.assert_allclose(I, I0, rtol=1e-9)
    assert np.all(_angdiff(E, E0) < 1e-8)
    assert np.all(_angdiff(O, O0) < 1e-8)
    assert np.all(_angdiff(w, w0) < 1e-8)
    np.testing.assert_allclose(P, 2 * np.pi * np.sqrt(a0 ** 3 / mus), rtol=1e-9)


def test_vec2orbElem_output_ranges():
    a0 = _arr(1.0, 2.0, 3.0, 4.0)
    e0 = _arr(0.1, 0.3, 0.5, 0.2)
    O0 = _arr(0.2, 1.0, 2.0, 5.0)
    I0 = _arr(0.3, 0.9, 1.4, 2.0)
    w0 = _arr(0.5, 1.5, 2.5, 4.5)
    E0 = _arr(0.4, 2.0, 5.0, 3.3)
    mus = np.ones(4)
    r, v = orbElem2vec(E0, mus, orbElem=(a0, e0, O0, I0, w0))
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, mus)
    for ang in (E, O, w):
        assert np.all(ang >= 0) and np.all(ang <= 2 * np.pi)
    assert np.all(I >= 0) and np.all(I <= np.pi)
    assert np.all(a > 0)
    assert np.all((e >= 0) & (e < 1))
    assert np.all(P > 0)


def test_vec2orbElem_size_mismatch_asserts():
    rs = np.ones(4)
    vs = np.ones(4)
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)
    with pytest.raises(AssertionError):
        vec2orbElem(np.ones((3, 2)), np.ones((3, 3)), 1.0)


def test_vec2orbElem_mus_size_mismatch_asserts():
    rs = np.array([[1.0, 2.0], [0.0, 0.1], [0.1, 0.0]])
    vs = np.array([[0.0, 0.0], [1.0, 0.7], [0.1, 0.2]])
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, np.array([1.0, 1.0, 1.0]))
