import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem, calcAB


def _ang_diff(a, b):
    """Return wrapped absolute angle difference in [0, pi]."""
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _sample_orbital_elements():
    # Two non-degenerate elliptical orbits with nontrivial angles.
    a = np.array([1.7, 2.4])
    e = np.array([0.2, 0.6])
    O = np.array([0.4, 1.1])
    I = np.array([0.7, 0.9])
    w = np.array([0.2, 2.0])
    return a, e, O, I, w


def test_calcAB_shapes_and_values():
    a, e, O, I, w = _sample_orbital_elements()
    A, B = calcAB(a, e, O, I, w)

    assert A.shape == (3, 2)
    assert B.shape == (3, 2)

    # Check one column against direct formulas.
    i = 0
    A_expected = np.array([
        a[i] * (np.cos(O[i]) * np.cos(w[i]) - np.sin(O[i]) * np.cos(I[i]) * np.sin(w[i])),
        a[i] * (np.sin(O[i]) * np.cos(w[i]) + np.cos(O[i]) * np.cos(I[i]) * np.sin(w[i])),
        a[i] * np.sin(I[i]) * np.sin(w[i]),
    ])
    B_expected = np.array([
        -a[i] * np.sqrt(1 - e[i] ** 2) * (np.cos(O[i]) * np.sin(w[i]) + np.sin(O[i]) * np.cos(I[i]) * np.cos(w[i])),
        a[i] * np.sqrt(1 - e[i] ** 2) * (-np.sin(O[i]) * np.sin(w[i]) + np.cos(O[i]) * np.cos(I[i]) * np.cos(w[i])),
        a[i] * np.sqrt(1 - e[i] ** 2) * np.sin(I[i]) * np.cos(w[i]),
    ])

    np.testing.assert_allclose(A[:, i], A_expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(B[:, i], B_expected, rtol=1e-13, atol=1e-13)


def test_orbElem2vec_scalar_mu_scalar_E_returnAB():
    a, e, O, I, w = _sample_orbital_elements()
    mu = 1.0
    E = 0.8

    r, v, (A, B) = orbElem2vec(E, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]), returnAB=True)

    assert r.shape == (3, 1)
    assert v.shape == (3, 1)
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)

    # Verify the returned state matches the standard form exactly.
    expected_r = A * (np.cos(E) - e[0]) + B * np.sin(E)
    expected_v = (-A * np.sin(E) + B * np.cos(E)) * (np.sqrt(mu * a[0] ** (-3.)) / (1 - e[0] * np.cos(E)))

    np.testing.assert_allclose(r, expected_r, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(v, expected_v, rtol=1e-13, atol=1e-13)


def test_orbElem2vec_vector_E_scalar_mu_broadcast():
    a, e, O, I, w = _sample_orbital_elements()
    mu = 1.0
    E = np.array([0.1, 1.2, 2.2])

    # Use the first orbit for a scalar orbital element set.
    r, v = orbElem2vec(E, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]))

    assert r.shape == (3, 3)
    assert v.shape == (3, 3)

    A, B = calcAB(a[0], e[0], O[0], I[0], w[0])
    for j, Ej in enumerate(E):
        expected_r = A * (np.cos(Ej) - e[0]) + B * np.sin(Ej)
        expected_v = (-A * np.sin(Ej) + B * np.cos(Ej)) * (np.sqrt(mu * a[0] ** (-3.)) / (1 - e[0] * np.cos(Ej)))
        np.testing.assert_allclose(r[:, [j]], expected_r, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(v[:, [j]], expected_v, rtol=1e-13, atol=1e-13)


def test_orbElem2vec_vector_mu_vector_E_multiple_bodies():
    a, e, O, I, w = _sample_orbital_elements()
    mu = np.array([1.0, 1.5])
    E = np.array([0.8, 1.3])

    r, v, (A, B) = orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)

    assert r.shape == (3, 2)
    assert v.shape == (3, 2)
    assert A.shape == (3, 2)
    assert B.shape == (3, 2)

    for i in range(2):
        expected_r = A[:, [i]] * (np.cos(E[i]) - e[i]) + B[:, [i]] * np.sin(E[i])
        expected_v = (-A[:, [i]] * np.sin(E[i]) + B[:, [i]] * np.cos(E[i])) * (
            np.sqrt(mu[i] * a[i] ** (-3.)) / (1 - e[i] * np.cos(E[i]))
        )
        np.testing.assert_allclose(r[:, [i]], expected_r, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(v[:, [i]], expected_v, rtol=1e-13, atol=1e-13)


def test_orbElem2vec_rejects_missing_orbital_inputs():
    with pytest.raises(AssertionError, match="either orbElem or AB"):
        orbElem2vec(0.1, 1.0)


def test_orbElem2vec_rejects_both_orbElem_and_AB():
    a, e, O, I, w = _sample_orbital_elements()
    A, B = calcAB(a[0], e[0], O[0], I[0], w[0])

    with pytest.raises(AssertionError, match="only set orbElem or AB"):
        orbElem2vec(0.1, 1.0, orbElem=(a[0], e[0], O[0], I[0], w[0]), AB=(A, B))


def test_orbElem2vec_rejects_scalar_E_with_vector_mu():
    a, e, O, I, w = _sample_orbital_elements()
    mu = np.array([1.0, 1.5])

    with pytest.raises(AssertionError, match="Scalar E input requires scalar mus input"):
        orbElem2vec(0.1, mu, orbElem=(a[0], e[0], O[0], I[0], w[0]))


def test_orbElem2vec_rejects_mismatched_vector_sizes():
    a, e, O, I, w = _sample_orbital_elements()
    mu = np.array([1.0, 1.5, 2.0])
    E = np.array([0.1, 0.2])

    with pytest.raises(AssertionError, match="mus must be of the same size as E or scalar"):
        orbElem2vec(E, mu, orbElem=(a[:2], e[:2], O[:2], I[:2], w[:2]))


def test_vec2orbElem_roundtrip_from_orbElem2vec_scalar_mu():
    a, e, O, I, w = _sample_orbital_elements()
    mu = 1.0
    E = np.array([0.8, 1.3])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    np.testing.assert_allclose(a2, a, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(e2, e, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(_ang_diff(E2, np.mod(E, 2 * np.pi)), 0.0, atol=1e-11)
    np.testing.assert_allclose(_ang_diff(O2, O), 0.0, atol=1e-11)
    np.testing.assert_allclose(_ang_diff(I2, I), 0.0, atol=1e-11)
    np.testing.assert_allclose(_ang_diff(w2, w), 0.0, atol=1e-11)

    expected_P = 2 * np.pi * np.sqrt(a ** 3 / mu)
    np.testing.assert_allclose(P2, expected_P, rtol=1e-11, atol=1e-11)
    assert tau2.shape == (2,)


def test_vec2orbElem_accepts_transposed_input_shapes():
    a, e, O, I, w = _sample_orbital_elements()
    mu = 1.0
    E = np.array([0.3, 1.7])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))

    out1 = vec2orbElem(r, v, mu)
    out2 = vec2orbElem(r.T, v.T, mu)

    for arr1, arr2 in zip(out1, out2):
        np.testing.assert_allclose(arr1, arr2, rtol=1e-12, atol=1e-12)


def test_vec2orbElem_vector_mu_roundtrip_two_bodies():
    a, e, O, I, w = _sample_orbital_elements()
    mu = np.array([1.0, 1.5])
    E = np.array([0.8, 1.3])

    r, v = orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    a2, e2, E2, O2, I2, w2, P2, tau2 = vec2orbElem(r, v, mu)

    np.testing.assert_allclose(a2, a, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(e2, e, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(_ang_diff(E2, np.mod(E, 2 * np.pi)), 0.0, atol=1e-11)
    np.testing.assert_allclose(_ang_diff(O2, O), 0.0, atol=1e-11)
    np.testing.assert_allclose(_ang_diff(I2, I), 0.0, atol=1e-11)
    np.testing.assert_allclose(_ang_diff(w2, w), 0.0, atol=1e-11)
    assert P2.shape == (2,)
    assert tau2.shape == (2,)


def test_vec2orbElem_rejects_mismatched_rs_vs_sizes():
    rs = np.array([1.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0, 0.0])

    with pytest.raises(AssertionError, match="same size and contain 3n elements"):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_mismatched_mus_size():
    rs = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    mus = np.array([1.0, 2.0, 3.0])

    with pytest.raises(AssertionError, match="mus must be scalar or of size n"):
        vec2orbElem(rs, vs, mus)


def test_vec2orbElem_rejects_three_dimensional_input():
    rs = np.zeros((1, 1, 3))
    vs = np.zeros((1, 1, 3))

    with pytest.raises(AssertionError, match="cannot have more than two dimensions"):
        vec2orbElem(rs, vs, 1.0)


def test_vec2orbElem_rejects_bad_matrix_shape():
    rs = np.zeros((2, 2))
    vs = np.zeros((2, 2))

    with pytest.raises(AssertionError, match="must be 3xn or nx3"):
        vec2orbElem(rs, vs, 1.0)
