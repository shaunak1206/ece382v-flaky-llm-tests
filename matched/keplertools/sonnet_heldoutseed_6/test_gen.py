import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def _angdiff(x, y):
    """Smallest absolute difference between angles."""
    d = np.mod(np.asarray(x) - np.asarray(y) + np.pi, 2 * np.pi) - np.pi
    return np.abs(d)


def _single_elems(a=1.5, e=0.3, O=0.7, I=0.5, w=1.1):
    return (np.array([a]), np.array([e]), np.array([O]),
            np.array([I]), np.array([w]))


def test_circular_equatorial_E0():
    elems = _single_elems(a=1.0, e=0.0, O=0.0, I=0.0, w=0.0)
    r, v = orbElem2vec(0.0, 1.0, orbElem=elems)
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert np.allclose(r[:, 0], [1, 0, 0])
    assert np.allclose(v[:, 0], [0, 1, 0])


def test_circular_equatorial_E_halfpi():
    elems = _single_elems(a=1.0, e=0.0, O=0.0, I=0.0, w=0.0)
    r, v = orbElem2vec(np.pi / 2, 1.0, orbElem=elems)
    assert np.allclose(r[:, 0], [0, 1, 0], atol=1e-12)
    assert np.allclose(v[:, 0], [-1, 0, 0], atol=1e-12)


def test_periapsis_and_apoapsis_distance():
    a, e = 2.0, 0.4
    elems = _single_elems(a=a, e=e, O=0.3, I=0.6, w=0.9)
    E = np.array([0.0, np.pi])
    r, v = orbElem2vec(E, 1.0, orbElem=elems)
    assert r.shape == (3, 2)
    norms = np.linalg.norm(r, axis=0)
    assert np.allclose(norms, [a * (1 - e), a * (1 + e)])


def test_vis_viva_single_body_many_positions():
    mu = 1.3
    a, e = 1.5, 0.3
    elems = _single_elems(a=a, e=e)
    E = np.linspace(0.1, 6.0, 9)
    r, v = orbElem2vec(E, mu, orbElem=elems)
    rn = np.linalg.norm(r, axis=0)
    vn2 = np.sum(v ** 2, axis=0)
    assert np.allclose(vn2, mu * (2.0 / rn - 1.0 / a))


def test_angular_momentum_magnitude_and_conservation():
    mu = 1.3
    a, e = 1.5, 0.3
    elems = _single_elems(a=a, e=e)
    E = np.linspace(0.2, 5.5, 7)
    r, v = orbElem2vec(E, mu, orbElem=elems)
    L = np.cross(r.T, v.T)
    Lmag = np.linalg.norm(L, axis=1)
    assert np.allclose(Lmag, np.sqrt(mu * a * (1 - e ** 2)))
    # direction constant
    assert np.allclose(L / Lmag[:, None], (L[0] / Lmag[0])[None, :])


def test_multiple_bodies_vector_mus_vis_viva():
    n = 4
    a = np.array([1.0, 1.5, 2.0, 3.0])
    e = np.array([0.1, 0.3, 0.5, 0.2])
    O = np.array([0.2, 1.0, 2.0, 4.0])
    I = np.array([0.3, 0.8, 1.2, 2.0])
    w = np.array([0.5, 1.5, 3.0, 5.0])
    E = np.array([0.4, 1.9, 3.3, 5.5])
    mus = np.array([1.0, 1.2, 0.9, 2.0])
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3, n)
    assert v.shape == (3, n)
    rn = np.linalg.norm(r, axis=0)
    vn2 = np.sum(v ** 2, axis=0)
    assert np.allclose(vn2, mus * (2.0 / rn - 1.0 / a))
    # radius from eccentric anomaly
    assert np.allclose(rn, a * (1 - e * np.cos(E)))


def test_AB_input_matches_orbElem_input():
    elems = _single_elems()
    AB = calcAB(*elems)
    E = np.array([0.5, 1.5, 2.5])
    r1, v1 = orbElem2vec(E, 1.1, orbElem=elems)
    r2, v2 = orbElem2vec(E, 1.1, AB=AB)
    assert np.allclose(r1, r2)
    assert np.allclose(v1, v2)


def test_returnAB_returns_tuple():
    elems = _single_elems()
    out = orbElem2vec(np.array([0.5, 1.0]), 1.0, orbElem=elems, returnAB=True)
    assert len(out) == 3
    r, v, (A, B) = out
    A0, B0 = calcAB(*elems)
    assert np.allclose(A, A0)
    assert np.allclose(B, B0)
    assert r.shape == (3, 2)


def test_returnAB_false_returns_two():
    out = orbElem2vec(0.5, 1.0, orbElem=_single_elems())
    assert len(out) == 2


def test_orbElem2vec_requires_input():
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.5]), 1.0)


def test_orbElem2vec_rejects_both_inputs():
    elems = _single_elems()
    AB = calcAB(*elems)
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.5]), 1.0, orbElem=elems, AB=AB)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=_single_elems())


def test_orbElem2vec_mu_size_mismatch():
    a = np.array([1.0, 1.0, 1.0])
    e = np.array([0.1, 0.2, 0.3])
    O = np.array([0.1, 0.2, 0.3])
    I = np.array([0.5, 0.6, 0.7])
    w = np.array([0.1, 0.2, 0.3])
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2, 0.3]), np.array([1.0, 1.0]),
                    orbElem=(a, e, O, I, w))


def test_vec2orbElem_known_polar_orbit():
    rs = np.array([[0.5], [0.0], [0.0]])
    vs = np.array([[0.0], [0.0], [np.sqrt(3.0)]])
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, 1.0)
    assert np.allclose(a, 1.0)
    assert np.allclose(e, 0.5)
    assert np.allclose(I, np.pi / 2)
    assert _angdiff(E, 0.0)[0] < 1e-8
    assert _angdiff(O, 0.0)[0] < 1e-8
    assert _angdiff(w, 0.0)[0] < 1e-8
    assert np.allclose(P, 2 * np.pi)
    assert abs(tau[0]) < 1e-8


def test_roundtrip_single_body():
    a0, e0, O0, I0, w0 = 1.5, 0.3, 0.7, 0.5, 1.1
    E0 = 0.9
    mu = 1.3
    elems = _single_elems(a0, e0, O0, I0, w0)
    r, v = orbElem2vec(E0, mu, orbElem=elems)
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, mu)
    assert np.allclose(a, a0)
    assert np.allclose(e, e0)
    assert np.allclose(I, I0)
    assert _angdiff(E, E0)[0] < 1e-8
    assert _angdiff(O, O0)[0] < 1e-8
    assert _angdiff(w, w0)[0] < 1e-8
    assert np.allclose(P, 2 * np.pi * np.sqrt(a0 ** 3 / mu))
    n_mean = np.sqrt(mu / a0 ** 3)
    assert np.allclose(-n_mean * tau, E0 - e0 * np.sin(E0))


def test_roundtrip_multiple_bodies_3xn_and_nx3():
    a0 = np.array([1.0, 1.5, 2.0, 3.0])
    e0 = np.array([0.1, 0.3, 0.5, 0.2])
    O0 = np.array([0.2, 1.0, 2.0, 4.0])
    I0 = np.array([0.3, 0.8, 1.2, 2.0])
    w0 = np.array([0.5, 1.5, 3.0, 5.0])
    E0 = np.array([0.4, 1.9, 3.3, 5.5])
    mus = np.array([1.0, 1.2, 0.9, 2.0])
    r, v = orbElem2vec(E0, mus, orbElem=(a0, e0, O0, I0, w0))

    for rr, vv in ((r, v), (r.T, v.T)):
        a, e, E, O, I, w, P, tau = vec2orbElem(rr, vv, mus)
        assert np.allclose(a, a0)
        assert np.allclose(e, e0)
        assert np.allclose(I, I0)
        assert np.all(_angdiff(E, E0) < 1e-7)
        assert np.all(_angdiff(O, O0) < 1e-7)
        assert np.all(_angdiff(w, w0) < 1e-7)
        assert np.allclose(P, 2 * np.pi * np.sqrt(a0 ** 3 / mus))


def test_vec2orbElem_size_validation():
    rs = np.array([[1.0], [0.0], [0.0]])
    vs = np.zeros((3, 2))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)
    with pytest.raises(AssertionError):
        vec2orbElem(np.ones((2, 2)), np.ones((2, 2)), 1.0)


def test_vec2orbElem_mus_size_validation():
    rs = np.array([[0.5], [0.0], [0.0]])
    vs = np.array([[0.0], [0.0], [np.sqrt(3.0)]])
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, np.array([1.0, 1.0]))
