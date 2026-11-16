import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def _arr(x):
    return np.array(x, dtype=float, ndmin=1)


def _single_body(a=1.5, e=0.3, O=0.7, I=0.5, w=1.1):
    return (_arr(a), _arr(e), _arr(O), _arr(I), _arr(w))


def _multi_body():
    a = _arr([1.5, 2.0, 0.8])
    e = _arr([0.3, 0.5, 0.1])
    O = _arr([0.7, 2.5, 5.0])
    I = _arr([0.5, 1.2, 2.5])
    w = _arr([1.1, 4.0, 0.3])
    E = _arr([0.9, 3.5, 5.5])
    mus = _arr([1.0, 2.0, 0.5])
    return a, e, O, I, w, E, mus


def test_orbElem2vec_periapsis_radius_and_speed():
    a, e = 1.5, 0.3
    mu = 1.0
    r, v = orbElem2vec(np.array([0.0]), mu, orbElem=_single_body(a=a, e=e))
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.linalg.norm(r) == pytest.approx(a * (1 - e), rel=1e-12)
    expected_v = np.sqrt(mu / a * (1 + e) / (1 - e))
    assert np.linalg.norm(v) == pytest.approx(expected_v, rel=1e-12)


def test_orbElem2vec_apoapsis_radius_and_speed():
    a, e = 2.0, 0.4
    mu = 3.0
    r, v = orbElem2vec(np.array([np.pi]), mu, orbElem=_single_body(a=a, e=e))
    assert np.linalg.norm(r) == pytest.approx(a * (1 + e), rel=1e-12)
    expected_v = np.sqrt(mu / a * (1 - e) / (1 + e))
    assert np.linalg.norm(v) == pytest.approx(expected_v, rel=1e-12)


def test_orbElem2vec_scalar_E_scalar_mu():
    r, v = orbElem2vec(0.5, 1.0, orbElem=_single_body())
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(v))


def test_orbElem2vec_vector_E_scalar_mu_shapes():
    E = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    r, v = orbElem2vec(E, 1.0, orbElem=_single_body())
    assert r.shape == (3, 7)
    assert v.shape == (3, 7)


def test_orbElem2vec_vector_E_vector_mu_shapes():
    a, e, O, I, w, E, mus = _multi_body()
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, 3)
    assert v.shape == (3, 3)


def test_orbElem2vec_vis_viva_and_radius():
    a, e, O, I, w, E, mus = _multi_body()
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    rmag = np.linalg.norm(r, axis=0)
    vmag2 = np.sum(v ** 2, axis=0)
    assert rmag == pytest.approx(a * (1 - e * np.cos(E)), rel=1e-10)
    assert vmag2 == pytest.approx(mus * (2.0 / rmag - 1.0 / a), rel=1e-10)


def test_orbElem2vec_angular_momentum_conserved():
    a, e = 1.7, 0.45
    mu = 2.0
    E = np.linspace(0.1, 6.0, 12)
    r, v = orbElem2vec(E, mu, orbElem=_single_body(a=a, e=e, O=1.0, I=0.8, w=2.0))
    L = np.cross(r.T, v.T)
    Lmag = np.linalg.norm(L, axis=1)
    expected = np.sqrt(mu * a * (1 - e ** 2))
    assert Lmag == pytest.approx(np.full(E.size, expected), rel=1e-10)
    # direction constant
    assert np.allclose(L / Lmag[:, None], (L[0] / Lmag[0])[None, :], atol=1e-10)


def test_orbElem2vec_returnAB():
    orb = _single_body()
    out = orbElem2vec(np.array([0.3]), 1.0, orbElem=orb, returnAB=True)
    assert len(out) == 3
    r, v, (A, B) = out
    A_exp, B_exp = calcAB(*orb)
    assert np.allclose(A, A_exp)
    assert np.allclose(B, B_exp)
    out2 = orbElem2vec(np.array([0.3]), 1.0, orbElem=orb)
    assert len(out2) == 2


def test_orbElem2vec_AB_input_matches_orbElem_input():
    orb = _single_body(a=1.2, e=0.25, O=2.0, I=1.0, w=0.4)
    A, B = calcAB(*orb)
    E = np.array([0.2, 1.5, 4.0])
    r1, v1 = orbElem2vec(E, 1.3, orbElem=orb)
    r2, v2 = orbElem2vec(E, 1.3, AB=(A, B))
    assert np.allclose(r1, r2, atol=1e-10)
    assert np.allclose(v1, v2, atol=1e-10)


def test_orbElem2vec_requires_orbElem_or_AB():
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1]), 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    orb = _single_body()
    AB = calcAB(*orb)
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1]), 1.0, orbElem=orb, AB=AB)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=_single_body())


def test_orbElem2vec_mus_size_mismatch():
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0]),
                    orbElem=_single_body())


def test_vec2orbElem_roundtrip_single_body():
    a, e, O, I, w = _single_body()
    E = np.array([0.9])
    mu = 1.0
    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)
    assert a2 == pytest.approx(a, rel=1e-9)
    assert e2 == pytest.approx(e, rel=1e-9)
    assert E2 == pytest.approx(E, rel=1e-9)
    assert O2 == pytest.approx(O, rel=1e-9)
    assert I2 == pytest.approx(I, rel=1e-9)
    assert w2 == pytest.approx(w, rel=1e-9)


def test_vec2orbElem_roundtrip_multiple_bodies():
    a, e, O, I, w, E, mus = _multi_body()
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mus)
    assert a2 == pytest.approx(a, rel=1e-8)
    assert e2 == pytest.approx(e, rel=1e-8)
    assert E2 == pytest.approx(E, rel=1e-8)
    assert O2 == pytest.approx(O, rel=1e-8)
    assert I2 == pytest.approx(I, rel=1e-8)
    assert w2 == pytest.approx(w, rel=1e-8)


def test_vec2orbElem_period_and_tau():
    a, e, O, I, w, E, mus = _multi_body()
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mus)
    assert P2 == pytest.approx(2 * np.pi * np.sqrt(a ** 3 / mus), rel=1e-8)
    n = np.sqrt(mus / a ** 3)
    assert tau2 == pytest.approx(-(E - e * np.sin(E)) / n, rel=1e-8)


def test_vec2orbElem_output_ranges():
    a, e, O, I, w, E, mus = _multi_body()
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mus)
    for ang in (E2, O2, w2):
        assert np.all(ang >= 0) and np.all(ang < 2 * np.pi)
    assert np.all(I2 > 0) and np.all(I2 < np.pi)
    assert np.all(e2 >= 0) and np.all(e2 < 1)
    assert np.all(a2 > 0)
    assert all(x.shape == (3,) for x in (a2, e2, E2, O2, I2, w2, P2, tau2))


def test_vec2orbElem_transposed_input_equivalent():
    a, e, O, I, w, E, mus = _multi_body()
    E4 = np.array([0.3, 1.0, 2.0, 5.0])
    mus4 = np.array([1.0, 1.5, 2.0, 0.7])
    orb4 = (np.array([1.0, 1.2, 1.4, 2.0]), np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 1.5, 2.5, 3.5]), np.array([0.4, 0.8, 1.2, 1.6]),
            np.array([0.2, 1.2, 2.2, 3.2]))
    r, v = orbElem2vec(E4, mus4, orbElem=orb4)
    out1 = vec2orbElem(r, v, mus4)
    out2 = vec2orbElem(r.T, v.T, mus4)
    for x, y in zip(out1, out2):
        assert np.allclose(x, y)


def test_vec2orbElem_energy_and_semimajor_axis():
    r = np.array([[1.0], [0.0], [0.0]])
    v = np.array([[0.0], [0.6], [0.5]])
    mu = 1.0
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, mu)
    expected_a = -mu / (2 * (0.5 * 0.61 - 1.0))
    assert a[0] == pytest.approx(expected_a, rel=1e-12)
    L = np.cross(r[:, 0], v[:, 0])
    assert e[0] == pytest.approx(np.sqrt(1 - np.dot(L, L) / mu / expected_a), rel=1e-12)
    assert I[0] == pytest.approx(np.arccos(L[2] / np.linalg.norm(L)), rel=1e-12)


def test_vec2orbElem_size_mismatch_asserts():
    rs = np.ones((3, 2))
    vs = np.ones((3, 1))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_not_multiple_of_three_asserts():
    rs = np.ones(4)
    vs = np.ones(4)
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_mus_wrong_size_asserts():
    rs = np.array([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    vs = np.array([[0.0, 0.0], [0.9, 0.7], [0.3, 0.2]])
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, np.array([1.0, 1.0, 1.0]))
