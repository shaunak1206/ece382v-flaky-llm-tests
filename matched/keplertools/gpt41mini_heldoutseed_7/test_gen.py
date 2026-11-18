import numpy as np
import pytest
from keplertools.fun import orbElem2vec, vec2orbElem

# Constants for tests
MU_SUN = 1.32712440018e11  # km^3/s^2, approximate solar gravitational parameter
# Use smaller units (AU^3/day^2) for easier testing since code mentions AU and days compatible units:
# 1 AU = 1.495978707e8 km; 1 day = 86400 s
# Convert mu from km^3/s^2 to AU^3/day^2
KM_AU = 1.495978707e8
S_DAY = 86400
MU_AU_DAY = MU_SUN * (S_DAY**2) / (KM_AU**3)  # approx 0.000295912208284 (AU^3/day^2)

def gen_circular_orbit(radius=1.0, mu=MU_AU_DAY):
    """Position and velocity vectors for circular orbit in x-y plane, prograde."""
    r = np.array([radius, 0, 0])
    v_circ = np.sqrt(mu / radius)
    v = np.array([0, v_circ, 0])
    return r, v

def gen_elliptic_orbit(a=1.0, e=0.5, mu=MU_AU_DAY):
    """Generate position and velocity vectors for elliptical orbit, periapsis at x positive axis."""
    # True anomaly: 0 (periapsis)
    # Compute position and velocity in perifocal frame
    p = a * (1 - e**2)
    r_mag = p / (1 + e)
    r_perifocal = np.array([r_mag, 0, 0])
    h = np.sqrt(mu * p)
    v_perifocal = np.array([0, h / r_mag, 0])
    # Rotate to inertial frame assuming zero inclination, zero node, zero argument pericenter (identity)
    r = r_perifocal
    v = v_perifocal
    return r, v

# -------------------------
# Test orbElem2vec and vec2orbElem
# -------------------------

def test_roundtrip_single_body_circular():
    r, v = gen_circular_orbit()
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, MU_AU_DAY)
    # Convert back
    r2, v2 = orbElem2vec(E, MU_AU_DAY, orbElem=(a,e,O,I,w))
    # Positions and velocities should be close
    np.testing.assert_allclose(r, r2.flatten(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(v, v2.flatten(), rtol=1e-12, atol=1e-14)
    # Orbital elements sanity
    assert np.isclose(e, 0, atol=1e-10), "Expected near-zero eccentricity for circular orbit"
    assert I >= 0 and I <= np.pi, "Inclination out of range"

def test_roundtrip_single_body_elliptical():
    r, v = gen_elliptic_orbit()
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, MU_AU_DAY)
    r2, v2 = orbElem2vec(E, MU_AU_DAY, orbElem=(a,e,O,I,w))
    np.testing.assert_allclose(r, r2.flatten(), rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(v, v2.flatten(), rtol=1e-11, atol=1e-13)
    assert e > 0, "Eccentricity should be positive for elliptical orbit"
    assert a > 0, "Semi-major axis should be positive"

def test_vectorized_orbits_single_mu():
    # Create 3 position & velocity vectors for different elliptical orbits
    rs = []
    vs = []
    a_vals = [1.0, 1.5, 2.0]
    e_vals = [0.0, 0.3, 0.6]
    for a,e in zip(a_vals,e_vals):
        r, v = gen_elliptic_orbit(a, e)
        rs.append(r)
        vs.append(v)
    rs = np.array(rs).reshape(-1)
    vs = np.array(vs).reshape(-1)
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, MU_AU_DAY)
    # orbElem2vec with vector E and scalar mu
    r2, v2 = orbElem2vec(E, MU_AU_DAY, orbElem=(a,e,O,I,w))
    # v2 and r2 shape: 3 x n
    assert r2.shape == (3, len(a_vals))
    assert v2.shape == (3, len(a_vals))
    for i in range(len(a_vals)):
        r_orig = rs[3*i:3*i+3]
        r_recov = r2[:,i]
        np.testing.assert_allclose(r_orig, r_recov, rtol=1e-10, atol=1e-12)
        v_orig = vs[3*i:3*i+3]
        v_recov = v2[:,i]
        np.testing.assert_allclose(v_orig, v_recov, rtol=1e-10, atol=1e-12)

def test_vectorized_orbits_vector_mu():
    # Create 2 orbits, 2 different mus for each
    a_vals = np.array([1.0, 1.5])
    e_vals = np.array([0.2, 0.4])
    mus = np.array([MU_AU_DAY, MU_AU_DAY*0.9])
    rs = []
    vs = []
    for a,e in zip(a_vals, e_vals):
        r,v = gen_elliptic_orbit(a,e)
        rs.append(r)
        vs.append(v)
    rs = np.array(rs).reshape(-1)
    vs = np.array(vs).reshape(-1)
    a, e, E, O, I, w, P, tau = vec2orbElem(rs, vs, mus)
    r2, v2 = orbElem2vec(E, mus, orbElem=(a,e,O,I,w))
    assert r2.shape == (3, len(a_vals))
    assert v2.shape == (3, len(a_vals))
    # Test approximate equality because mus differ
    for i in range(len(a_vals)):
        r_orig = rs[3*i:3*i+3]
        r_recov = r2[:,i]
        np.testing.assert_allclose(r_orig, r_recov, rtol=1e-9, atol=1e-11)
        v_orig = vs[3*i:3*i+3]
        v_recov = v2[:,i]
        np.testing.assert_allclose(v_orig, v_recov, rtol=1e-9, atol=1e-11)

def test_orbElem2vec_returnAB():
    r, v = gen_circular_orbit()
    a, e, E, O, I, w, P, tau = vec2orbElem(r, v, MU_AU_DAY)
    r2, v2, AB = orbElem2vec(E, MU_AU_DAY, orbElem=(a,e,O,I,w), returnAB=True)
    A, B = AB
    assert A.shape == (3,1)
    assert B.shape == (3,1)
    assert r2.shape == (3,1)
    assert v2.shape == (3,1)

def test_orbElem2vec_scalar_inputs():
    # Single scalar inputs for everything
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])
    E = np.array([0.1])
    mus = 1.0
    r, v = orbElem2vec(E, mus, orbElem=(a, e, O, I, w))
    assert r.shape == (3,1)
    assert v.shape == (3,1)

def test_orbElem2vec_scalar_E_vector_mu():
    # E scalar, mu vector - should raise AssertionError
    E = 0.2
    mus = np.array([1.0, 2.0])
    a = np.array([1.0, 1.0])
    e = np.array([0.1, 0.1])
    O = np.array([0.1, 0.1])
    I = np.array([0.1, 0.1])
    w = np.array([0.1, 0.1])
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, orbElem=(a,e,O,I,w))

def test_orbElem2vec_vector_E_scalar_mu():
    # E vector, mu scalar, orbElem length 1 (scalars)
    E = np.array([0.1, 0.2, 0.3])
    mus = 1.0
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.0])
    I = np.array([0.0])
    w = np.array([0.0])
    with pytest.raises(AssertionError):
        # Will raise because orbElem components sizes mismatch E sizes
        orbElem2vec(E, mus, orbElem=(a,e,O,I,w))

def test_orbElem2vec_invalid_inputs():
    # Neither orbElem nor AB provided
    with pytest.raises(AssertionError):
        orbElem2vec(np.array([0.1]), 1.0)

def test_orbElem2vec_setting_both_orbElem_AB():
    a = np.array([1.0])
    e = np.array([0.1])
    O = np.array([0.2])
    I = np.array([0.3])
    w = np.array([0.4])
    A = np.array([[1],[0],[0]])
    B = np.array([[0],[1],[0]])
    E = 0.1
    mus = 1.0
    with pytest.raises(AssertionError):
        orbElem2vec(E, mus, orbElem=(a,e,O,I,w), AB=(A,B))

def test_vec2orbElem_input_shapes_errors():
    r = np.array([1,0,0, 0,1,0])  # 6 elements, not multiple of 3 for 2 bodies
    v = np.array([0,1,0, 1,0,0])
    mus = np.array([1.0])
    with pytest.raises(AssertionError):
        vec2orbElem(r, v, mus)

def test_vec2orbElem_mus_size_mismatch():
    r, v = gen_circular_orbit()
    # Stack 2 copies to make 2 orbits
    rs = np.hstack((r, r))
    vs = np.hstack((v, v))
    mus = np.array([1.0])  # size 1, but 2 bodies
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, mus)

def test_vec2orbElem_output_shapes():
    # Single orbit
    r, v = gen_circular_orbit()
    a,e,E,O,I,w,P,tau = vec2orbElem(r, v, MU_AU_DAY)
    assert np.ndim(a) == 0 or a.size == 1
    assert np.ndim(e) == 0 or e.size == 1
    # Multiple orbits
    rs = np.tile(r,3)
    vs = np.tile(v,3)
    a,e,E,O,I,w,P,tau = vec2orbElem(rs, vs, MU_AU_DAY)
    n = 3
    assert a.shape == (n,)
    assert e.shape == (n,)
    assert E.shape == (n,)
    assert O.shape == (n,)
    assert I.shape == (n,)
    assert w.shape == (n,)
    assert P.shape == (n,)
    assert tau.shape == (n,)

def test_vec2orbElem_with_nonstandard_shape():
    # Provide r,v shape nx3 instead of 3xn
    r, v = gen_circular_orbit()
    rs = np.vstack((r, r+0.1, r+0.2))
    vs = np.vstack((v, v+0.1, v+0.2))
    mus = MU_AU_DAY
    # Should accept nx3 shape and process correctly
    a,e,E,O,I,w,P,tau = vec2orbElem(rs, vs, mus)
    assert a.size == 3

def test_vec2orbElem_vs_rs_size_mismatch():
    r, v = gen_circular_orbit()
    rs = np.hstack((r, r))
    vs = np.hstack((v,))  # size mismatch
    mus = 1.0
    with pytest.raises(AssertionError):
        vec2orbElem(rs, vs, mus)

def test_vec2orbElem_soft_limits_inclination():
    # Inclination must be between 0 and pi
    r, v = gen_circular_orbit()
    a,e,E,O,I,w,P,tau = vec2orbElem(r, v, MU_AU_DAY)
    assert 0 <= I <= np.pi

def test_vec2orbElem_in_edge_cases_eccentricity_one():
    # Parabolic case e=1 not fully supported but check no error for e close to 1
    a = np.array([1])
    e = np.array([0.999999])
    O = np.array([0])
    I = np.array([0])
    w = np.array([0])
    E = np.array([np.pi/2])
    mus = 1.0
    # A and B via calcAB
    from keplertools.fun import calcAB
    A, B = calcAB(a,e,O,I,w)
    # This should work, values might be large
    r,v = orbElem2vec(E, mus, AB=(A,B))
    assert r.shape == (3,1)
    assert v.shape == (3,1)

def test_vec2orbElem_correctness_high_eccentricity():
    # For eccentric orbit with e close to 1 but <1
    r, v = gen_elliptic_orbit(a=1.0, e=0.99)
    a,e,E,O,I,w,P,tau = vec2orbElem(r, v, MU_AU_DAY)
    assert np.all(e < 1)
    r2,v2 = orbElem2vec(E, MU_AU_DAY, orbElem=(a,e,O,I,w))
    np.testing.assert_allclose(r, r2.flatten(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(v, v2.flatten(), rtol=1e-8, atol=1e-10)
