import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


MU = 1.0

# A set of 4 generic (non-degenerate) elliptical orbits.
A_ = np.array([1.0, 2.0, 1.5, 3.0])
E_ = np.array([0.1, 0.3, 0.5, 0.2])
O_ = np.array([0.5, 1.0, 2.0, 4.0])
I_ = np.array([0.3, 0.8, 1.2, 2.0])
W_ = np.array([0.2, 1.5, 3.0, 5.0])
EA_ = np.array([0.4, 2.0, 4.0, 5.5])


def _circular_elems():
    return (np.array([1.0]), np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([0.0]))


def _angdiff(x, y):
    d = np.mod(x - y + np.pi, 2 * np.pi) - np.pi
    return np.abs(d)


def test_orbElem2vec_circular_scalar_E_zero():
    r, v = orbElem2vec(0.0, 1.0, orbElem=_circular_elems())
    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    np.testing.assert_allclose(r[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [0.0, 1.0, 0.0], atol=1e-12)


def test_orbElem2vec_circular_E_quarter_turn():
    r, v = orbElem2vec(np.pi / 2, 1.0, orbElem=_circular_elems())
    np.testing.assert_allclose(r[:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v[:, 0], [-1.0, 0.0, 0.0], atol=1e-12)


def test_orbElem2vec_velocity_scales_with_sqrt_mu():
    r, v = orbElem2vec(0.0, 4.0, orbElem=_circular_elems())
    np.testing.assert_allclose(r[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(v, axis=0), [2.0], rtol=1e-12)


def test_orbElem2vec_vector_E_scalar_mu_shapes():
    E = np.linspace(0, 2 * np.pi, 7)
    r, v = orbElem2vec(E, MU, orbElem=_circular_elems())
    assert r.shape == (3, 7)
    assert v.shape == (3, 7)
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), np.ones(7), rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(v, axis=0), np.ones(7), rtol=1e-12)


def test_orbElem2vec_radius_matches_conic_equation_single_body():
    a = np.array([2.0])
    e = np.array([0.4])
    elems = (a, e, np.array([0.7]), np.array([0.9]), np.array([1.3]))
    E = np.linspace(0.1, 6.0, 9)
    r, v = orbElem2vec(E, MU, orbElem=elems)
    expected = a[0] * (1 - e[0] * np.cos(E))
    np.testing.assert_allclose(np.linalg.norm(r, axis=0), expected, rtol=1e-10)


def test_orbElem2vec_vis_viva_single_body_many_positions():
    a = np.array([2.0])
    e = np.array([0.4])
    mu = 3.5
    elems = (a, e, np.array([0.7]), np.array([0.9]), np.array([1.3]))
    E = np.linspace(0.1, 6.0, 9)
    r, v = orbElem2vec(E, mu, orbElem=elems)
    rm = np.linalg.norm(r, axis=0)
    v2 = np.sum(v ** 2, axis=0)
    np.testing.assert_allclose(v2, mu * (2.0 / rm - 1.0 / a[0]), rtol=1e-10)


def test_orbElem2vec_angular_momentum_conserved():
    a = np.array([1.7])
    e = np.array([0.3])
    elems = (a, e, np.array([2.1]), np.array([0.5]), np.array([4.0]))
    E = np.linspace(0.2, 6.0, 8)
    r, v = orbElem2vec(E, MU, orbElem=elems)
    h = np.cross(r.T, v.T)
    expected_mag = np.sqrt(MU * a[0] * (1 - e[0] ** 2))
    np.testing.assert_allclose(np.linalg.norm(h, axis=1),
                               expected_mag * np.ones(len(E)), rtol=1e-10)
    # direction constant
    np.testing.assert_allclose(h, np.tile(h[0], (len(E), 1)), atol=1e-10)


def test_orbElem2vec_multiple_bodies_vector_mu():
    mus = np.array([1.0, 2.0, 0.5, 4.0])
    r, v = orbElem2vec(EA_, mus, orbElem=(A_, E_, O_, I_, W_))
    assert r.shape == (3, 4)
    assert v.shape == (3, 4)
    rm = np.linalg.norm(r, axis=0)
    np.testing.assert_allclose(rm, A_ * (1 - E_ * np.cos(EA_)), rtol=1e-10)
    v2 = np.sum(v ** 2, axis=0)
    np.testing.assert_allclose(v2, mus * (2.0 / rm - 1.0 / A_), rtol=1e-10)


def test_orbElem2vec_AB_input_matches_orbElem_input():
    elems = (np.array([1.3]), np.array([0.25]), np.array([0.6]),
             np.array([1.1]), np.array([2.2]))
    A, B = calcAB(*elems)
    E = np.linspace(0.3, 5.5, 6)
    r1, v1 = orbElem2vec(E, MU, orbElem=elems)
    r2, v2 = orbElem2vec(E, MU, AB=(A, B))
    np.testing.assert_allclose(r1, r2, atol=1e-10)
    np.testing.assert_allclose(v1, v2, atol=1e-10)


def test_orbElem2vec_returnAB():
    elems = _circular_elems()
    out = orbElem2vec(np.array([0.0, 1.0]), MU, orbElem=elems, returnAB=True)
    assert len(out) == 3
    r, v, (A, B) = out
    A0, B0 = calcAB(*elems)
    np.testing.assert_allclose(A, A0)
    np.testing.assert_allclose(B, B0)
    assert r.shape == (3, 2)


def test_orbElem2vec_requires_some_input():
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2]), MU)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    elems = _circular_elems()
    AB = calcAB(*elems)
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2]), MU, orbElem=elems, AB=AB)


def test_orbElem2vec_scalar_E_requires_scalar_mu():
    with pytest.raises(AssertionError):
        orbElem2vec(0.5, np.array([1.0, 2.0]), orbElem=_circular_elems())


def test_orbElem2vec_mu_size_mismatch():
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0]),
                    orbElem=_circular_elems())


def test_vec2orbElem_known_state():
    # Orbit with periapsis-like state: r=(1,0,0), v tilted out of plane.
    speed = 1.2
    inc = 0.6
    rs = np.array([[1.0, 0.0, 0.0]])
    vs = np.array([[0.0, speed * np.cos(inc), speed * np.sin(inc)]])
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, 1.0)
    a_exp = 1.0 / (2.0 - speed ** 2)
    e_exp = np.sqrt(1 - speed ** 2 * np.cos(inc) ** 2 * 0 - (speed ** 2) * 1.0 / a_exp * 0
                    - 0) if False else np.sqrt(1 - (speed ** 2) / (1.0 / a_exp))
    np.testing.assert_allclose(a, [a_exp], rtol=1e-10)
    np.testing.assert_allclose(e, [e_exp], rtol=1e-10)
    np.testing.assert_allclose(I, [inc], rtol=1e-10)
    np.testing.assert_allclose(P, [2 * np.pi * np.sqrt(a_exp ** 3)], rtol=1e-10)


def test_vec2orbElem_roundtrip_all_elements():
    r, v = orbElem2vec(EA_, MU, orbElem=(A_, E_, O_, I_, W_)) \
        if False else orbElem2vec(EA_, np.full(4, MU), orbElem=(A_, E_, O_, I_, W_))
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, MU)
    np.testing.assert_allclose(a, A_, rtol=1e-8)
    np.testing.assert_allclose(e, E_, rtol=1e-8)
    np.testing.assert_allclose(I, I_, atol=1e-8)
    assert np.all(_angdiff(E, EA_) < 1e-8)
    assert np.all(_angdiff(O, O_) < 1e-8)
    assert np.all(_angdiff(w, W_) < 1e-8)


def test_vec2orbElem_period_and_tau_consistent():
    r, v = orbElem2vec(EA_, np.full(4, MU), orbElem=(A_, E_, O_, I_, W_))
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, MU)
    np.testing.assert_allclose(P, 2 * np.pi * np.sqrt(A_ ** 3 / MU), rtol=1e-8)
    n = np.sqrt(MU / a ** 3)
    np.testing.assert_allclose(tau, -(E - e * np.sin(E)) / n, rtol=1e-10)


def test_vec2orbElem_layouts_equivalent():
    r, v = orbElem2vec(EA_, np.full(4, MU), orbElem=(A_, E_, O_, I_, W_))
    out_3xn = vec2orbElem(r, v, MU)
    out_nx3 = vec2orbElem(r.T.copy(), v.T.copy(), MU)
    for x, y in zip(out_3xn, out_nx3):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12)


def test_vec2orbElem_vector_mus():
    mus = np.array([1.0, 2.0, 0.5, 4.0])
    r, v = orbElem2vec(EA_, mus, orbElem=(A_, E_, O_, I_, W_))
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, mus)
    np.testing.assert_allclose(a, A_, rtol=1e-8)
    np.testing.assert_allclose(e, E_, rtol=1e-8)
    np.testing.assert_allclose(P, 2 * np.pi * np.sqrt(A_ ** 3 / mus), rtol=1e-8)


def test_vec2orbElem_output_ranges():
    r, v = orbElem2vec(EA_, np.full(4, MU), orbElem=(A_, E_, O_, I_, W_))
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, MU)
    for ang in (E, O, w):
        assert np.all(ang >= 0) and np.all(ang < 2 * np.pi)
    assert np.all(I > 0) and np.all(I < np.pi)
    assert np.all(e >= 0) and np.all(e < 1)


def test_vec2orbElem_size_mismatch_raises():
    rs = np.ones((4, 3))
    vs = np.ones((5, 3))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_bad_shape_raises():
    rs = np.ones((2, 6))
    vs = np.ones((2, 6))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, 1.0)
