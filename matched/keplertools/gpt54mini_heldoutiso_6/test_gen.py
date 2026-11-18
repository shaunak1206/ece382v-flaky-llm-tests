import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem


def _ang_diff(a, b):
    """Smallest absolute difference between angles modulo 2π."""
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))


def _normalize_angle(x):
    return np.mod(np.asarray(x), 2 * np.pi)


def _assert_orbital_elements_close(actual, expected, atol=1e-10, rtol=1e-10):
    a, e, E, O, I, w, P, tau = actual
    ea, ee, eE, eO, eI, ew, eP, etau = expected

    np.testing.assert_allclose(a, ea, rtol=rtol, atol=atol)
    np.testing.assert_allclose(e, ee, rtol=rtol, atol=atol)
    np.testing.assert_allclose(E, _normalize_angle(eE), rtol=rtol, atol=atol)
    np.testing.assert_allclose(P, eP, rtol=rtol, atol=atol)
    np.testing.assert_allclose(tau, etau, rtol=rtol, atol=atol)

    np.testing.assert_allclose(_ang_diff(O, eO), 0.0, atol=1e-8)
    np.testing.assert_allclose(_ang_diff(I, eI), 0.0, atol=1e-8)
    np.testing.assert_allclose(_ang_diff(w, ew), 0.0, atol=1e-8)


def test_orbelem2vec_scalar_roundtrip_through_vec2orbelem():
    a = 2.5
    e = 0.3
    O = 0.7
    I = 1.0
    w = 1.4
    E = 0.9
    mu = 1.0

    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    got = vec2orbElem(rs, vs, mu)
    expected = (np.array([a]), np.array([e]), np.array([E]), np.array([O]),
                np.array([I]), np.array([w]), np.array([2 * np.pi * np.sqrt(a**3 / mu)]),
                np.array([-(E - e * np.sin(E)) / np.sqrt(mu * a**-3)]))
    _assert_orbital_elements_close(got, expected)


def test_orbelem2vec_vector_roundtrip_scalar_mu():
    a = np.array([1.5, 2.2, 3.1])
    e = np.array([0.1, 0.25, 0.6])
    O = np.array([0.2, 1.1, 2.4])
    I = np.array([0.3, 0.9, 1.2])
    w = np.array([0.5, 1.7, 0.8])
    E = np.array([0.4, 2.0, 5.5])
    mu = 1.7

    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    got = vec2orbElem(rs, vs, mu)

    expected = (
        a, e, _normalize_angle(E), _normalize_angle(O), _normalize_angle(I), _normalize_angle(w),
        2 * np.pi * np.sqrt(a**3 / mu),
        -(E - e * np.sin(E)) / np.sqrt(mu * a**-3),
    )
    _assert_orbital_elements_close(got, expected)


def test_orbelem2vec_vector_roundtrip_vector_mu():
    a = np.array([1.8, 2.7])
    e = np.array([0.2, 0.4])
    O = np.array([0.1, 1.3])
    I = np.array([0.4, 1.0])
    w = np.array([0.9, 2.0])
    E = np.array([1.0, 4.8])
    mu = np.array([1.0, 1.5])

    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    got = vec2orbElem(rs, vs, mu)

    expected = (
        a, e, _normalize_angle(E), _normalize_angle(O), _normalize_angle(I), _normalize_angle(w),
        2 * np.pi * np.sqrt(a**3 / mu),
        -(E - e * np.sin(E)) / np.sqrt(mu * a**-3),
    )
    _assert_orbital_elements_close(got, expected)


def test_orbelem2vec_returns_ab_when_requested():
    a = 2.0
    e = 0.1
    O = 0.2
    I = 0.3
    w = 0.4
    E = 0.5
    mu = 1.0

    rs, vs, ab = orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)
    assert isinstance(ab, tuple)
    assert len(ab) == 2
    A, B = ab
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)
    assert rs.shape == (3, 1)
    assert vs.shape == (3, 1)


def test_orbelem2vec_accepts_precomputed_ab():
    a = np.array([1.6, 2.1])
    e = np.array([0.2, 0.35])
    O = np.array([0.8, 1.4])
    I = np.array([0.6, 1.1])
    w = np.array([0.3, 2.0])
    E = np.array([1.2, 4.0])
    mu = 1.2

    AB = tuple(np.asarray(x) for x in __import__("keplertools.fun", fromlist=["calcAB"]).calcAB(a, e, O, I, w))
    rs1, vs1 = orbElem2vec(E, mu, AB=AB)
    rs2, vs2 = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))

    np.testing.assert_allclose(rs1, rs2)
    np.testing.assert_allclose(vs1, vs2)


def test_orbelem2vec_accepts_scalar_e_for_vector_E():
    E = np.array([0.2, 1.1, 3.0])
    mu = 1.0
    a = np.array([2.0, 2.0, 2.0])
    e = 0.25
    O = np.array([0.0, 0.5, 1.0])
    I = np.array([0.4, 0.8, 1.2])
    w = np.array([0.3, 0.7, 1.1])

    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert rs.shape == (3, 3)
    assert vs.shape == (3, 3)


def test_orbelem2vec_rejects_missing_orbital_inputs():
    with pytest.raises(AssertionError, match="You must supply either orbElem or AB inputs"):
        orbElem2vec(0.1, 1.0)


def test_orbelem2vec_rejects_both_orbElem_and_AB():
    a = 2.0
    e = 0.2
    O = 0.3
    I = 0.4
    w = 0.5
    A, B = __import__("keplertools.fun", fromlist=["calcAB"]).calcAB(a, e, O, I, w)
    with pytest.raises(AssertionError, match="You can only set orbElem or AB"):
        orbElem2vec(0.1, 1.0, orbElem=(a, e, O, I, w), AB=(A, B))


def test_orbelem2vec_rejects_scalar_E_with_vector_mu():
    a = 2.0
    e = 0.2
    O = 0.3
    I = 0.4
    w = 0.5
    mu = np.array([1.0, 1.2])
    with pytest.raises(AssertionError, match="Scalar E input requires scalar mus input"):
        orbElem2vec(0.1, mu, orbElem=(a, e, O, I, w))


def test_orbelem2vec_rejects_mismatched_mu_size():
    E = np.array([0.2, 0.4, 0.6])
    mu = np.array([1.0, 1.2])  # wrong size
    a = np.array([2.0, 2.1, 2.2])
    e = np.array([0.1, 0.2, 0.3])
    O = np.array([0.3, 0.4, 0.5])
    I = np.array([0.6, 0.7, 0.8])
    w = np.array([0.9, 1.0, 1.1])

    with pytest.raises(AssertionError, match="mus must be of the same size as E or scalar"):
        orbElem2vec(E, mu, orbElem=(a, e, O, I, w))


def test_vec2orbelem_accepts_nx3_input_arrays():
    a = np.array([1.7, 2.3])
    e = np.array([0.15, 0.35])
    O = np.array([0.6, 1.2])
    I = np.array([0.7, 1.0])
    w = np.array([0.2, 2.4])
    E = np.array([0.8, 5.7])
    mu = 1.0

    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    got = vec2orbElem(rs.T, vs.T, mu)
    expected = (
        a, e, _normalize_angle(E), _normalize_angle(O), _normalize_angle(I), _normalize_angle(w),
        2 * np.pi * np.sqrt(a**3 / mu),
        -(E - e * np.sin(E)) / np.sqrt(mu * a**-3),
    )
    _assert_orbital_elements_close(got, expected)


def test_vec2orbelem_accepts_flattened_1d_input_arrays():
    a = 2.4
    e = 0.2
    O = 0.4
    I = 0.8
    w = 1.2
    E = 0.9
    mu = 1.3

    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    got = vec2orbElem(rs.ravel(), vs.ravel(), mu)
    expected = (
        np.array([a]), np.array([e]), np.array([E]), np.array([O]), np.array([I]), np.array([w]),
        np.array([2 * np.pi * np.sqrt(a**3 / mu)]),
        np.array([-(E - e * np.sin(E)) / np.sqrt(mu * a**-3)]),
    )
    _assert_orbital_elements_close(got, expected)


def test_vec2orbelem_rejects_mismatched_vector_sizes():
    rs = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0])
    with pytest.raises(AssertionError, match="rs and vs must be of the same size"):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbelem_rejects_more_than_two_dimensions():
    rs = np.zeros((3, 1, 1))
    vs = np.zeros((3, 1, 1))
    with pytest.raises(AssertionError, match="rs cannot have more than two dimensions"):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbelem_rejects_bad_second_dimension_shape():
    rs = np.zeros((2, 4))
    vs = np.zeros((2, 4))
    with pytest.raises(AssertionError, match="rs must be 3xn or nx3"):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbelem_rejects_mu_size_mismatch():
    rs = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    mus = np.array([1.0, 1.0])  # should be scalar or size 3
    with pytest.raises(AssertionError, match="mus must be scalar or of size n"):
        vec2orbElem(rs, vs, mus)


def test_roundtrip_preserves_vectorized_shapes():
    a = np.array([1.2, 1.8, 2.5, 3.0])
    e = np.array([0.05, 0.15, 0.25, 0.35])
    O = np.array([0.1, 0.6, 1.0, 1.5])
    I = np.array([0.2, 0.5, 0.9, 1.2])
    w = np.array([0.3, 0.7, 1.1, 1.7])
    E = np.array([0.4, 1.2, 2.0, 4.1])
    mu = np.array([1.0, 1.0, 1.0, 1.0])

    rs, vs = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert rs.shape == (3, 4)
    assert vs.shape == (3, 4)

    got = vec2orbElem(rs, vs, mu)
    assert all(arr.shape == (4,) for arr in got)
    np.testing.assert_allclose(got[0], a)
    np.testing.assert_allclose(got[1], e)
    np.testing.assert_allclose(_ang_diff(got[2], O), 0.0, atol=1e-8)
