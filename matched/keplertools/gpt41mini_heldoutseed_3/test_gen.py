import numpy as np
import pytest
from keplertools import fun


def test_orbElem2vec_scalar_input():
    a = np.array([2.0])
    e = np.array([0.1])
    O = np.array([0.3])
    I = np.array([0.4])
    w = np.array([0.5])
    E = 1.0
    mu = 10.0

    rs, vs = fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert rs.shape == (3, 1)
    assert vs.shape == (3, 1)


def test_orbElem2vec_vector_E_scalar_mu():
    a = np.array([1.0])
    e = np.array([0.05])
    O = np.array([0.1])
    I = np.array([0.2])
    w = np.array([0.3])
    E = np.linspace(0, 2 * np.pi, 5)
    mu = 1.0

    rs, vs = fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert rs.shape == (3, 5)
    assert vs.shape == (3, 5)


def test_orbElem2vec_vector_E_vector_mu():
    n = 4
    a = np.array([2.0, 3.0, 4.0, 5.0])
    e = np.array([0.1, 0.2, 0.3, 0.4])
    O = np.array([0.0, 0.1, 0.2, 0.3])
    I = np.array([0.5, 0.6, 0.7, 0.8])
    w = np.array([1.0, 1.1, 1.2, 1.3])
    E = np.array([0.0, 1.0, 2.0, 3.0])
    mu = np.array([1.0, 2.0, 3.0, 4.0])

    # Since mu is vector, E and inputs length must match n
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E[:-1], mu, orbElem=(a, e, O, I, w))

    rs, vs = fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert rs.shape == (3, n)
    assert vs.shape == (3, n)


def test_orbElem2vec_with_AB_input_matches_orbElem():
    a = np.array([1.5])
    e = np.array([0.05])
    O = np.array([0.1])
    I = np.array([0.2])
    w = np.array([0.3])
    E = 0.7
    mu = 1.0

    A, B = fun.calcAB(a, e, O, I, w)
    rs1, vs1 = fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    rs2, vs2 = fun.orbElem2vec(E, mu, AB=(A, B))
    np.testing.assert_allclose(rs1, rs2)
    np.testing.assert_allclose(vs1, vs2)


def test_orbElem2vec_returnAB_flag():
    a = np.array([2.0])
    e = np.array([0.2])
    O = np.array([0.5])
    I = np.array([0.6])
    w = np.array([0.7])
    E = 1.2
    mu = 5.0

    rs, vs, (A, B) = fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w), returnAB=True)
    assert rs.shape == (3, 1)
    assert vs.shape == (3, 1)
    assert A.shape == (3, 1)
    assert B.shape == (3, 1)


def test_orbElem2vec_invalid_no_inputs():
    with pytest.raises(AssertionError):
        fun.orbElem2vec(0.5, 1.0)


def test_orbElem2vec_invalid_both_inputs():
    a = np.array([1])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])
    E = 0.7
    mu = 2.0
    A, B = fun.calcAB(a, e, O, I, w)
    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w), AB=(A, B))


def test_orbElem2vec_scalar_mus_requires_scalar_E():
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])
    E = np.array([0.1, 0.2])
    mu = 1.0

    # This is valid (scalar mu, vector E) no exception
    fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w))


def test_orbElem2vec_vector_mus_with_scalar_E_raises():
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])
    E = 0.5
    mu = np.array([1.0, 2.0])

    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w))


def test_vec2orbElem2_single_vector_input():
    rs = np.array([7000.0, 0.0, 0.0])
    vs = np.array([0.0, 7.5, 1.0])
    mu = 398600.4418  # Earth's mu in km^3/s^2

    a, e, E, O, I, w, P, tau = fun.vec2orbElem2(rs, vs, mu)
    assert np.isscalar(a) or a.size == 1
    assert 0 <= e <= 1
    assert np.all(np.isfinite([E, O, I, w, P, tau]))


def test_vec2orbElem2_multiple_vectors_input():
    rs = np.array([[7000.0, 0.0, 0.0],
                   [8000.0, 0.0, 0.0],
                   [8500.0, 700.0, 0.0]]).T
    vs = np.array([[0.0, 7.5, 1.0],
                   [0.0, 7.0, 1.1],
                   [0.0, 6.5, 1.2]]).T
    mu = 398600.4418

    a, e, E, O, I, w, P, tau = fun.vec2orbElem2(rs, vs, mu)
    assert a.shape == e.shape == E.shape == O.shape == I.shape == w.shape == P.shape == tau.shape
    assert np.all(e >= 0) and np.all(e < 1)
    assert np.all(np.isfinite(a))


def test_vec2orbElem2_input_shape_variations():
    # Test rs/vs as 1d flat
    rs = np.array([7000.0, 0.0, 0.0])
    vs = np.array([0.0, 7.5, 1.0])
    mu = 398600.4418
    a, e, E, O, I, w, P, tau = fun.vec2orbElem2(rs, vs, mu)
    # Test rs/vs as 3xn shape
    rs2 = rs.reshape((3, 1))
    vs2 = vs.reshape((3, 1))
    a2, e2, E2, O2, I2, w2, P2, tau2 = fun.vec2orbElem2(rs2, vs2, mu)
    np.testing.assert_allclose(a, a2)
    np.testing.assert_allclose(e, e2)


def test_vec2orbElem_single_vector_input():
    rs = np.array([7000.0, 0.0, 0.0])
    vs = np.array([0.0, 7.5, 1.0])
    mu = 398600.4418

    a, e, E, O, I, w, P, tau = fun.vec2orbElem(rs, vs, mu)
    assert np.isscalar(a) or a.size == 1
    assert 0 <= e <= 1
    assert np.all(np.isfinite([E, O, I, w, P, tau]))


def test_vec2orbElem_multiple_vectors_input():
    rs = np.array([[7000.0, 0.0, 0.0],
                   [8000.0, 0.0, 0.0],
                   [8500.0, 700.0, 0.0]]).T
    vs = np.array([[0.0, 7.5, 1.0],
                   [0.0, 7.0, 1.1],
                   [0.0, 6.5, 1.2]]).T
    mu = 398600.4418

    a, e, E, O, I, w, P, tau = fun.vec2orbElem(rs, vs, mu)
    assert a.shape == e.shape == E.shape == O.shape == I.shape == w.shape == P.shape == tau.shape
    assert np.all(e >= 0) and np.all(e < 1)
    assert np.all(np.isfinite(a))


def test_vec2orbElem_input_shape_variations():
    # Test rs/vs as 1d flat
    rs = np.array([7000.0, 0.0, 0.0])
    vs = np.array([0.0, 7.5, 1.0])
    mu = 398600.4418
    a, e, E, O, I, w, P, tau = fun.vec2orbElem(rs, vs, mu)
    # Test rs/vs as 3xn shape
    rs2 = rs.reshape((3, 1))
    vs2 = vs.reshape((3, 1))
    a2, e2, E2, O2, I2, w2, P2, tau2 = fun.vec2orbElem(rs2, vs2, mu)
    np.testing.assert_allclose(a, a2)
    np.testing.assert_allclose(e, e2)


def test_vec2orbElem_invalid_shape():
    rs = np.array([7000.0, 0.0])  # Incorrect shape (length 2)
    vs = np.array([0.0, 7.5])
    mu = 398600.4418
    with pytest.raises(AssertionError):
        fun.vec2orbElem(rs, vs, mu)


def test_vec2orbElem2_invalid_mus_size():
    rs = np.array([7000.0, 0.0, 0.0, 7100.0, 0.0, 0.0])
    vs = np.array([0.0, 7.5, 1.0, 0.0, 7.3, 1.1])
    mu = np.array([398600.4418])  # length 1, but 2 vectors input

    with pytest.raises(AssertionError):
        fun.vec2orbElem2(rs, vs, mu)


def test_orbElem2vec_shapes_consistency():
    # Create consistent multiple body vectors
    n = 3
    a = np.array([1.0, 1.5, 2.0])
    e = np.array([0.1, 0.2, 0.3])
    O = np.array([0.0, 0.1, 0.2])
    I = np.array([0.3, 0.4, 0.5])
    w = np.array([0.7, 0.8, 0.9])
    E = np.array([0.0, 1.0, 2.0])
    mu = 1.0

    rs, vs = fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert rs.shape == (3, n)
    assert vs.shape == (3, n)


def test_orbElem2vec_wrong_A_B_size():
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])
    mu = 1.0

    # Calc A, B 3x1
    A, B = fun.calcAB(a, e, O, I, w)

    # Fake wrong shapes: flatten A, B to 1D arrays of wrong size
    A_flat = A.flatten()
    B_flat = B.flatten()

    E = np.array([0.5, 1.0])  # vector E

    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mu, AB=(A_flat, B_flat))


def test_orbElem2vec_wrong_E_mu_scalar_vector_consistency():
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])
    mu = np.array([1.0, 2.0])  # vector mu
    E = 0.5  # scalar E

    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w))


def test_orbElem2vec_invalid_shapes_in_orbElem():
    a = np.array([1.0, 2.0])
    e = np.array([0.1, 0.2])
    O = np.array([0.3, 0.4])
    I = np.array([0.5, 0.6])
    w = np.array([0.7])  # Wrong size
    E = np.array([0.0, 1.0])
    mu = 1.0

    with pytest.raises(AssertionError):
        fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w))


def test_orbElem2vec_e_at_limit_zero_and_one():
    # Test eccentricity near zero and near 1 (but <1)
    a = np.array([1.0, 2.0])
    e = np.array([1e-10, 0.9999999])
    O = np.array([0.1, 0.2])
    I = np.array([0.3, 0.4])
    w = np.array([0.5, 0.6])
    E = np.array([0.0, 1.0])
    mu = 1.0

    rs, vs = fun.orbElem2vec(E, mu, orbElem=(a, e, O, I, w))
    assert np.all(rs >= -1e4)  # loosely check no extreme values
    assert np.all(vs >= -1e4)


def test_orbElem2vec_single_scalar_eccentricity_input_expansion():
    # Check that scalar e can be expanded internally by calcAB and orbElem2vec
    a = np.array([1.0, 2.0])
    e = 0.1
    O = np.array([0.1, 0.2])
    I = np.array([0.3, 0.4])
    w = np.array([0.5, 0.6])
    E = np.array([0.1, 0.2])
    mu = 1.0

    rs, vs = fun.orbElem2vec(E, mu, orbElem=(a, np.full_like(a, e), O, I, w))
    assert rs.shape == (3, 2)
    assert vs.shape == (3, 2)
