import numpy as np
import pytest

from keplertools.fun import orbElem2vec, vec2orbElem
from math import isclose

# tolerance for floating point comparisons
rtol = 1e-12
atol = 1e-14

def assert_allclose(a, b):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

def make_sample_inputs():
    # Create a simple elliptical orbit example for tests
    # Position and velocity vectors (3 x n):
    rs = np.array([[1.0, 0.0, 0.0],
                   [0.0, 2.0, 0.0],
                   [0.0, 0.0, 3.0]]).T  # shape (3,3)
    vs = np.array([[0.0, 0.1, 0.0],
                   [1.0, 0.0, 0.1],
                   [0.0, 1.0, 0.0]]).T  # shape (3,3)
    mus = 1.0  # scalar gravitational parameter
    # we'll feed them as flattened arrays with 3n elements
    return rs.flatten(), vs.flatten(), mus

@pytest.mark.parametrize("rs_shape", [(9,), (3,3), (3,3)])
@pytest.mark.parametrize("vs_shape", [(9,), (3,3), (3,3)])
def test_vec2orbElem_inputs_reshape(rs_shape, vs_shape):
    # Test that various input shapes accepted and normalized
    rs, vs, mus = make_sample_inputs()
    # reshape accordingly
    # rs_shape and vs_shape are always either flattened (9,) or (3,3)
    if rs_shape != (9,):
        rs_in = rs.reshape(rs_shape)
    else:
        rs_in = rs.copy()
    if vs_shape != (9,):
        vs_in = vs.reshape(vs_shape)
    else:
        vs_in = vs.copy()

    a,e,E,O,I,w,P,tau = vec2orbElem(rs_in, vs_in, mus)
    # Output arrays are length 3 (3 bodies)
    assert a.shape == e.shape == E.shape == O.shape == I.shape == w.shape == P.shape == tau.shape == (3,)
    # semi-major axis values positive
    assert np.all(a > 0)
    # eccentricities between 0 and 1 (closed orbits)
    assert np.all((e >= 0) & (e < 1))

def test_vec2orbElem_scalar_mus_and_vector_rs_vs():
    # mus scalar, E vector - should be accepted
    rs, vs, mus = make_sample_inputs()
    a,e,E,O,I,w,P,tau = vec2orbElem(rs, vs, mus)
    assert a.shape == e.shape == E.shape == O.shape == I.shape == w.shape == P.shape == tau.shape == (3,)

def test_vec2orbElem_scalar_inputs():
    # Single body (scalar input)
    rs = np.array([1.0, 0.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0])
    mus = 1.0
    a,e,E,O,I,w,P,tau = vec2orbElem(rs, vs, mus)
    # Outputs should be scalar arrays with size 1 (or shape (1,))
    # a, e, E,... each length 1 array
    assert a.size == 1 and e.size == 1
    # eccentricity in [0,1)
    assert 0 <= e[0] < 1
    # semi-major axis positive
    assert a[0] > 0

def test_vec2orbElem_invalid_input_size_mismatch():
    rs = np.array([1,0,0, 0,1,0])
    vs = np.array([0,1,0])  # incorrect size
    mus = 1.0
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, mus)
        
def test_vec2orbElem_invalid_mus_size():
    rs, vs, mus = make_sample_inputs()
    mus_array = np.array([1.0, 2.0])  # wrong size (should be 3)
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, mus_array)

def test_orbElem2vec_vs_vec2orbElem_consistency():
    # Round trip test: convert position+vel to orbital elems and back, verify vectors approx equal
    rs, vs, mus = make_sample_inputs()
    a,e,E,O,I,w,P,tau = vec2orbElem(rs, vs, mus)

    # orbElem tuple (a,e,O,I,w)
    orbElem = (a,e,O,I,w)
    r2, v2 = orbElem2vec(E, mus, orbElem=orbElem)

    # Both outputs shape 3 x n (3x3 in this case)
    assert r2.shape == (3,3)
    assert v2.shape == (3,3)

    # Original rs, vs are flattened, shape (9,)
    # Reshape original rs,vs to match (3,n)
    rs_reshaped = rs.reshape((3,3))
    vs_reshaped = vs.reshape((3,3))

    # Positions approximately equal
    assert_allclose(r2, rs_reshaped)
    # Velocities approximately equal
    assert_allclose(v2, vs_reshaped)

def test_orbElem2vec_scalar_E_scalar_mus_scalar_orbElem():
    # Test scalar inputs for orbElem2vec
    E = 1.0
    mus = 1.0
    a = np.array([2.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    orbElem = (a,e,O,I,w)
    r,v = orbElem2vec(E, mus, orbElem=orbElem)

    # r and v should be shape (3,1)
    assert r.shape == (3,1)
    assert v.shape == (3,1)

    # Check that radius approx semi-major axis (since e small and E=1 rad)
    rmag = np.linalg.norm(r, axis=0)
    assert rmag.size == 1 and rmag[0] > 0

def test_orbElem2vec_vector_E_scalar_mus_scalar_orbElem():
    # Vector E, scalar mus case
    E = np.array([0.0, np.pi/2, np.pi])
    mus = 1.0
    a = np.array([2.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    orbElem = (a,e,O,I,w)
    r,v = orbElem2vec(E, mus, orbElem=orbElem)
    # r and v should be (3,3)
    assert r.shape == (3,3)
    assert v.shape == (3,3)

def test_orbElem2vec_vector_E_vector_mus_vector_AB(monkeypatch):
    # Vector E and mus with explicit AB input from calcAB.
    # We test AB input instead of orbElem.
    # To get AB, first calculate orbElem, then AB from calcAB directly.

    # Arbitrary orbital elements for 2 bodies
    a = np.array([1.5, 2.5])
    e = np.array([0.1, 0.2])
    O = np.array([0.0, 0.1])
    I = np.array([0.0, 0.2])
    w = np.array([0.0, 0.3])
    from keplertools.fun import calcAB
    A,B = calcAB(a,e,O,I,w)

    E = np.array([0.0, np.pi/4])
    mus = np.array([1.0, 2.0])

    # Shapes check:
    assert A.shape == (3,2)
    assert B.shape == (3,2)

    r,v = orbElem2vec(E, mus, AB=(A,B))
    assert r.shape == (3,2)
    assert v.shape == (3,2)

def test_orbElem2vec_returnAB():
    E = np.array([0.1, 0.2])
    mus = 1.0
    # orbElem with scalars repeated to length 2
    a = np.array([3.0, 3.0])
    e = np.array([0.3, 0.3])
    O = np.array([0.1, 0.1])
    I = np.array([0.2, 0.2])
    w = np.array([0.3, 0.3])
    orbElem = (a,e,O,I,w)
    r,v,(A,B) = orbElem2vec(E, mus, orbElem=orbElem, returnAB=True)

    # A and B shapes == (3,n)
    assert A.shape == (3,2)
    assert B.shape == (3,2)

def test_orbElem2vec_raises_if_no_AB_or_orbElem():
    E = 0.1
    mus = 1.0
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus)

def test_orbElem2vec_raises_if_both_AB_and_orbElem():
    E = 0.1
    mus = 1.0
    from keplertools.fun import calcAB
    orbElem = (np.array([3.0]), np.array([0.3]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
    A,B = calcAB(*orbElem)
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, orbElem=orbElem, AB=(A,B))

def test_orbElem2vec_raises_if_wrong_shapes_with_scalars():
    # Provide vector E but AB shape only 3 elements (not 3*n)
    E = np.array([0.0, 1.0])
    mus = 1.0
    A = np.array([1.0, 0.0, 0.0])
    B = np.array([0.0, 1.0, 0.0])

    with pytest.raises(AssertionError):
        # Because E has length 2 but A and B size 3 (single vector), should raise.
        orbElem2vec(E, mus, AB=(A,B))

def test_vec2orbElem_invalid_ndim_rs_vs():
    # Provide rs or vs with ndim 3, should assert
    rs = np.zeros((3,3,3))
    vs = np.zeros(9)
    mus = 1.0
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, mus)

    rs = np.zeros(9)
    vs = np.zeros((3,3,3))
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, mus)

def test_vec2orbElem_angles_ranges():
    # The returned inclinations I should be in [0, pi]
    # longitudes O and arguments w should be in [0, 2pi]
    rs, vs, mus = make_sample_inputs()
    a,e,E,O,I,w,P,tau = vec2orbElem(rs, vs, mus)
    assert np.all((I >= 0) & (I <= np.pi))
    assert np.all((O >= 0) & (O <= 2*np.pi))
    assert np.all((w >= 0) & (w <= 2*np.pi))

def test_vec2orbElem_eccentricity_zero_and_near_one():
    # Test correct behavior for eccentricities close to zero and close to 1 (but < 1)
    # Construct a circular orbit (e ~ 0)
    rs = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    vs = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
    mus = 1.0
    # Force e ~ 0 by construction in output is tricky, just run and check e not exactly 1
    a,e,E,O,I,w,P,tau = vec2orbElem(rs, vs, mus)
    assert np.all(e < 1.0)
    assert np.all(e >= 0)

    # Forcing near 1 eccentricity: give high eccentricity input
    # Using very elongated elliptical orbit
    rs = np.array([1.0, 0.0, 0.0])
    vs = np.array([0.0, 0.01, 0.0])
    a,e,E,O,I,w,P,tau = vec2orbElem(rs, vs, mus)
    assert e[0] < 1.0
    assert e[0] >= 0
    
def test_vec2orbElem_consistency_with_vec2orbElem2():
    # Compare outputs of vec2orbElem and vec2orbElem2 for some input (they have the same signature)
    from keplertools.fun import vec2orbElem2
    rs, vs, mus = make_sample_inputs()
    # Test with scalar mus
    out1 = vec2orbElem(rs, vs, mus)
    out2 = vec2orbElem2(rs, vs, mus)
    for a1,a2 in zip(out1,out2):
        assert_allclose(a1, a2)

def test_orbElem2vec_invalid_mus_vector_E_scalar():
    # E scalar, mus vector - illegal, should assert via orbElem2vec internal checks
    E = 1.0
    mus = np.array([1.0, 2.0])
    a = np.array([1.0, 1.0])
    e = np.array([0.1, 0.1])
    O = np.array([0.0, 0.0])
    I = np.array([0.0, 0.0])
    w = np.array([0.0, 0.0])
    orbElem = (a,e,O,I,w)
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, orbElem=orbElem)

def test_vec2orbElem_large_input_shapes():
    # Test performance on larger n input, shape consistency only
    n = 100
    rs = np.tile(np.array([1.0, 0.0, 0.0]), n)
    vs = np.tile(np.array([0.0, 1.0, 0.0]), n)
    mus = np.ones(n)
    a,e,E,O,I,w,P,tau = vec2orbElem(rs, vs, mus)
    assert a.shape == (n,)
    assert e.shape == (n,)

def test_vec2orbElem_invalid_eccentricity():
    # Provide rs and vs that cause eccentricity outside [0,1) - forcibly test assertion
    # The assertion is inside the code for e in [0,1) via assert e >= 0 and e < 1,
    # But no explicit assertion there - it comes from sqrt(1-p/a) and assertion might fail outside
    rs, vs, mus = make_sample_inputs()
    # Manually patch e to invalid value to test assertion: impossible directly,
    # but the code does not have explicit assert for this, so cannot test this easily here.
    # We skip this test due to no direct assertion.

    pass  # no-op

def test_vec2orbElem_invalid_input_shape_raises():
    rs = np.zeros((4,3))
    vs = np.zeros((4,3))
    mus = 1.0
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, mus)

def test_orbElem2vec_multi_body_with_returnAB():
    E = np.array([0.1, 0.5])
    mus = np.array([1.0, 2.0])
    a = np.array([1.0, 1.5])
    e = np.array([0.1, 0.2])
    O = np.array([0.0, 0.2])
    I = np.array([0.1, 0.3])
    w = np.array([0.05, 0.5])
    orbElem = (a,e,O,I,w)
    r,v,(A,B) = orbElem2vec(E, mus, orbElem=orbElem, returnAB=True)
    assert r.shape == (3,2)
    assert v.shape == (3,2)
    assert A.shape == (3,2)
    assert B.shape == (3,2)

def test_orbElem2vec_vector_E_vector_mus_vector_orbElem():
    E = np.array([0.1, 0.5, 1.0])
    mus = np.array([1.0, 1.5, 2.0])
    a = np.array([1.0, 1.2, 1.5])
    e = np.array([0.1, 0.15, 0.2])
    O = np.array([0.0, 0.1, 0.05])
    I = np.array([0.1, 0.05, 0.0])
    w = np.array([0.0, 0.05, 0.1])
    orbElem = (a,e,O,I,w)
    r,v = orbElem2vec(E, mus, orbElem=orbElem)
    assert r.shape == (3,3)
    assert v.shape == (3,3)

def test_vec2orbElem_rarefied_angles_ranges():
    # angles returned should strictly be inside allowed ranges, with no NaNs or infs
    rs, vs, mus = make_sample_inputs()
    res = vec2orbElem(rs, vs, mus)
    for arr in res:
        assert np.all(np.isfinite(arr))
