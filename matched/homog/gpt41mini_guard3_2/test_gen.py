import numpy as np
import pytest
import homog.homog as homog


@pytest.mark.parametrize("origin,direction", [
    ((1, 2, 3), (0, 0, 1)),
    ((0, 0, 0), (1, 0, 0)),
    ((-1, -1, -1), (1, 1, 1)),
    (np.array([0.1, 0.2, 0.3]), np.array([0.3, 0.2, 0.1])),
])
def test_hray_basic_properties(origin, direction):
    np.random.seed(42)
    r = homog.hray(origin, direction)
    # Shape: should be (..., 4, 2)
    assert r.shape[-2:] == (4, 2)
    # Origin part homogeneous coordinate == 1
    assert np.allclose(r[..., 3, 0], 1)
    # Direction part homogeneous coordinate == 0
    assert np.allclose(r[..., 3, 1], 0)
    # Direction vector should be normalized (3D part)
    dir_norm = np.linalg.norm(r[..., :3, 1], axis=-1)
    np.testing.assert_allclose(dir_norm, 1, atol=1e-7)
    # Origin 3D part matches input (allowing float conversion)
    np.testing.assert_allclose(r[..., :3, 0], homog.hpoint(origin)[..., :3])


def test_hray_broadcast_shape():
    np.random.seed(101)
    # origin shape (2,4), direction shape (4,) broadcast to (2,4)
    origins = np.ones((2, 4, 3))
    directions = np.array([1, 0, 0, 0])
    directions = np.broadcast_to(directions, (4, 3))
    r = homog.hray(origins, directions)
    assert r.shape[:-2] == (2, 4)
    # Each direction normalized
    dir_norms = np.linalg.norm(r[..., :3, 1], axis=-1)
    np.testing.assert_allclose(dir_norms, 1.0, atol=1e-7)


def test_line_line_distance_parallel_lines():
    np.random.seed(202)
    # Two parallel rays offset by (0,1,0), distance should be 1
    origin1 = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    origin2 = origin1 + np.array([0, 1, 0])
    r1 = homog.hray(origin1, direction)
    r2 = homog.hray(origin2, direction)
    dist = homog.line_line_distance(r1, r2)
    # Distance close to 1
    np.testing.assert_allclose(dist, 1, atol=1e-7)


def test_line_line_distance_skew_lines():
    np.random.seed(303)
    # Lines: r1 along x axis at y=0,z=0; r2 along y axis at x=1,z=0
    r1 = homog.hray([0, 0, 0], [1, 0, 0])
    r2 = homog.hray([1, 0, 0], [0, 1, 0])
    dist = homog.line_line_distance(r1, r2)
    # Distance should be 0 (lines actually intersect at [1,0,0])
    np.testing.assert_allclose(dist, 0.0, atol=1e-7)


def test_line_line_distance_nonzero_skew():
    np.random.seed(404)
    # Line 1: origin (0,0,0), direction along x axis
    # Line 2: origin (0,1,1), direction along y axis
    r1 = homog.hray([0, 0, 0], [1, 0, 0])
    r2 = homog.hray([0, 1, 1], [0, 1, 0])
    dist = homog.line_line_distance(r1, r2)
    # Distance is the shortest distance between these skew lines,
    # which is length of vector perpendicular to both direction vectors
    # Vector between origins is (0,1,1)
    # Cross product of directions is (0,0,1)
    # Distance = |dot(vec, cross(direction1,direction2))|/||cross||
    expected_dist = abs(np.dot(np.array([0, 1, 1]), np.cross([1, 0, 0], [0, 1, 0])))
    np.testing.assert_allclose(dist, expected_dist, atol=1e-7)


def test_line_line_distance_identical_lines():
    np.random.seed(505)
    # Same line twice, distance should be zero
    origin = np.array([1, 2, 3])
    direction = np.array([4, 5, 6])
    r1 = homog.hray(origin, direction)
    r2 = homog.hray(origin, direction)
    dist = homog.line_line_distance(r1, r2)
    np.testing.assert_allclose(dist, 0.0, atol=1e-7)


def test_line_line_closest_points_basic():
    np.random.seed(606)
    # Two intersecting lines
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    origin2 = np.array([0, 0, 0])
    direction2 = np.array([0, 1, 0])
    r1 = homog.hray(origin1, direction1)
    r2 = homog.hray(origin2, direction2)
    q1, q2 = homog.line_line_closest_points(r1, r2)
    # The closest points should be at the origin on both lines
    np.testing.assert_allclose(q1, homog.hpoint(origin1), atol=1e-7)
    np.testing.assert_allclose(q2, homog.hpoint(origin2), atol=1e-7)
    # Distance between closest points should be zero
    dist_vec = q1[..., :3] - q2[..., :3]
    dist = np.linalg.norm(dist_vec, axis=-1)
    np.testing.assert_allclose(dist, 0, atol=1e-7)


def test_line_line_closest_points_skew():
    np.random.seed(707)
    # Two skew lines example from line_line_distance_nonzero_skew
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    origin2 = np.array([0, 1, 1])
    direction2 = np.array([0, 1, 0])
    r1 = homog.hray(origin1, direction1)
    r2 = homog.hray(origin2, direction2)
    q1, q2 = homog.line_line_closest_points(r1, r2)
    # Distance between closest points should be same as line_line_distance
    dist = homog.line_line_distance(r1, r2)
    dist_pts = np.linalg.norm(q1[..., :3] - q2[..., :3], axis=-1)
    np.testing.assert_allclose(dist_pts, dist, atol=1e-7)
    # The closest points lie on their respective lines, distance vector perpendicular to directions
    line_vec = q2[..., :3] - q1[..., :3]
    dp1 = np.dot(line_vec, direction1)
    dp2 = np.dot(line_vec, direction2)
    # Should be ~0 within tolerance
    assert abs(dp1) < 1e-7
    assert abs(dp2) < 1e-7


def test_line_line_closest_points_parallel_lines():
    np.random.seed(808)
    origin1 = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    origin2 = origin1 + np.array([0, 1, 0])
    r1 = homog.hray(origin1, direction)
    r2 = homog.hray(origin2, direction)
    with pytest.raises(Exception):
        # The current implementation may error if lines are parallel
        homog.line_line_closest_points(r1, r2)


def test_line_line_distance_pa_scalar_and_array():
    np.random.seed(909)
    # Scalar inputs
    pt1 = np.array([0, 0, 0, 1])
    ax1 = np.array([1, 0, 0, 0])
    pt2 = np.array([0, 1, 0, 1])
    ax2 = np.array([0, 1, 0, 0])
    dist_scalar = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    # Using rays
    r1 = homog.hray(pt1[:3], ax1[:3])
    r2 = homog.hray(pt2[:3], ax2[:3])
    dist_array = homog.line_line_distance(r1, r2)
    np.testing.assert_allclose(dist_scalar, dist_array, atol=1e-7)


def test_line_line_closest_points_pa_consistency():
    np.random.seed(1010)
    pt1 = np.array([0.0, 0.0, 0.0, 1.0])
    ax1 = np.array([1.0, 0.0, 0.0, 0.0])
    pt2 = np.array([0.0, 1.0, 1.0, 1.0])
    ax2 = np.array([0.0, 0.0, 1.0, 0.0])
    q1, q2 = homog.line_line_closest_points_pa(pt1, ax1, pt2, ax2)
    # q1 should be on ray1 line, so vector from pt1 to q1 parallel to ax1
    v1 = q1 - pt1
    cross1 = np.cross(v1[:3], ax1[:3])
    np.testing.assert_allclose(cross1, [0, 0, 0], atol=1e-7)
    # q2 should be on ray2 line, so vector from pt2 to q2 parallel to ax2
    v2 = q2 - pt2
    cross2 = np.cross(v2[:3], ax2[:3])
    np.testing.assert_allclose(cross2, [0, 0, 0], atol=1e-7)


def test_hray_direction_normalized_for_various_inputs():
    np.random.seed(1111)
    # Direction vector not normalized input
    origin = [1.0, 2.0, 3.0]
    direction = [10.0, 0.0, 0.0]
    r = homog.hray(origin, direction)
    dir_norm = np.linalg.norm(r[..., :3, 1], axis=-1)
    np.testing.assert_allclose(dir_norm, 1.0, atol=1e-7)


def test_hray_origin_with_homogeneous_input_returns_same():
    np.random.seed(1212)
    origin = np.array([1, 2, 3, 1])
    direction = np.array([0, 0, 1])
    r = homog.hray(origin, direction)
    # The origin input has len 4, should be same in output at ...,:,0
    np.testing.assert_allclose(r[..., :, 0], homog.hpoint(origin))


def test_line_line_distance_zero_for_intersecting_lines():
    np.random.seed(1313)
    # Lines intersecting at point [1,1,1]
    pt1 = np.array([1, 1, 1, 1])
    ax1 = np.array([1, 0, 0, 0])
    pt2 = np.array([1, 1, 1, 1])
    ax2 = np.array([0, 1, 0, 0])
    r1 = homog.hray(pt1[:3], ax1[:3])
    r2 = homog.hray(pt2[:3], ax2[:3])
    dist = homog.line_line_distance(r1, r2)
    np.testing.assert_allclose(dist, 0, atol=1e-7)


def test_line_line_distance_large_offset():
    np.random.seed(1414)
    # Lines far apart, distance equals offset
    pt1 = np.array([0, 0, 0, 1])
    ax1 = np.array([1, 0, 0, 0])
    pt2 = np.array([0, 100, 0, 1])
    ax2 = np.array([0, 1, 0, 0])
    r1 = homog.hray(pt1[:3], ax1[:3])
    r2 = homog.hray(pt2[:3], ax2[:3])
    dist = homog.line_line_distance(r1, r2)
    np.testing.assert_allclose(dist, 100, atol=1e-7)


def test_line_line_closest_points_multidimensional():
    np.random.seed(1515)
    # Batch input multiple rays
    origins1 = np.array([[0, 0, 0], [1, 1, 1]])
    directions1 = np.array([[1, 0, 0], [0, 1, 0]])
    origins2 = np.array([[0, 1, 0], [2, 2, 2]])
    directions2 = np.array([[0, 1, 0], [1, 0, 0]])
    r1 = homog.hray(origins1, directions1)
    r2 = homog.hray(origins2, directions2)
    q1, q2 = homog.line_line_closest_points(r1, r2)
    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)
    # Distance between closest points should equal line_line_distance for each pair
    dist1 = homog.line_line_distance(r1[0:1], r2[0:1])[0]
    dist2 = homog.line_line_distance(r1[1:2], r2[1:2])[0]
    dist_q1_q2_0 = np.linalg.norm(q1[0, :3] - q2[0, :3])
    dist_q1_q2_1 = np.linalg.norm(q1[1, :3] - q2[1, :3])
    np.testing.assert_allclose(dist1, dist_q1_q2_0, atol=1e-7)
    np.testing.assert_allclose(dist2, dist_q1_q2_1, atol=1e-7)


def test_hray_invalid_direction_length():
    np.random.seed(1616)
    origin = [0, 0, 0]
    direction = [1, 2]  # length 2 invalid
    with pytest.raises(IndexError):
        homog.hray(origin, direction)
    

def test_line_line_distance_pa_same_point_and_axis():
    np.random.seed(1717)
    pt = np.array([1, 2, 3, 1])
    ax = np.array([0, 0, 1, 0])
    dist = homog.line_line_distance_pa(pt, ax, pt, ax)
    assert dist == 0


def test_line_line_closest_points_pa_parallel_axes():
    np.random.seed(1818)
    # Use parallel axes, different points
    pt1 = np.array([0, 0, 0, 1])
    ax1 = np.array([1, 0, 0, 0])
    pt2 = np.array([0, 1, 0, 1])
    ax2 = np.array([1, 0, 0, 0])
    q1, q2 = homog.line_line_closest_points_pa(pt1, ax1, pt2, ax2)
    # Because axes are parallel, closest points are projections onto that axis
    v = pt2[:3] - pt1[:3]
    proj_len = np.dot(v, ax1[:3])
    expected_q1 = pt1[:3] + proj_len * ax1[:3]
    expected_q2 = expected_q1
    np.testing.assert_allclose(q1[:3], expected_q1, atol=1e-7)
    np.testing.assert_allclose(q2[:3], expected_q2, atol=1e-7)
    # Homogeneous coordinate preserved
    assert q1[3] == 1
    assert q2[3] == 1
