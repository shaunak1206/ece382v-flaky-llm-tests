import numpy as np
import pytest
from homog import homog


def test_hray_basic_shape_and_values():
    origin = [1, 2, 3]
    direction = [0, 0, 1]
    r = homog.hray(origin, direction)
    assert r.shape == (4, 2)
    # Check origin (homogeneous coordinate w=1)
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    # Check direction normalized and homogeneous coordinate w=0
    dir_norm = np.linalg.norm(direction)
    np.testing.assert_allclose(r[:3, 1], np.array(direction) / dir_norm)
    assert r[3, 1] == 0


def test_hray_broadcasting():
    origins = np.array([[1, 0, 0], [0, 1, 0]])
    directions = np.array([0, 0, 1])
    r = homog.hray(origins, directions)
    assert r.shape == (2, 4, 2)
    # Each origin proper homogeneous point
    np.testing.assert_allclose(r[0, :, 0], [1, 0, 0, 1])
    np.testing.assert_allclose(r[1, :, 0], [0, 1, 0, 1])
    # Directions should be normalized and same for both
    np.testing.assert_allclose(r[0, :3, 1], [0, 0, 1])
    np.testing.assert_allclose(r[1, :3, 1], [0, 0, 1])
    assert r[0, 3, 1] == 0
    assert r[1, 3, 1] == 0


def test_line_line_distance_parallel_lines():
    # Parallel lines along z-axis, different x offsets
    ray1 = homog.hray([0, 0, 0], [0, 0, 1])
    ray2 = homog.hray([1, 0, 0], [0, 0, 1])
    dist = homog.line_line_distance(ray1, ray2)
    # Distance should be 1 (offset in x-direction)
    assert np.isclose(dist, 1.0)


def test_line_line_distance_intersecting_lines():
    # Intersecting lines: ray1 along x-axis, ray2 along y-axis both crossing at origin
    ray1 = homog.hray([0, 0, 0], [1, 0, 0])
    ray2 = homog.hray([0, 0, 0], [0, 1, 0])
    dist = homog.line_line_distance(ray1, ray2)
    # Intersection at origin, so distance 0
    assert np.isclose(dist, 0.0)


def test_line_line_distance_skew_lines():
    # Two skew lines: first passes through origin along x-axis, second offset along y=1 and along z-axis
    ray1 = homog.hray([0, 0, 0], [1, 0, 0])
    ray2 = homog.hray([0, 1, 0], [0, 0, 1])
    dist = homog.line_line_distance(ray1, ray2)
    # Distance should be 1 (offset along y)
    assert np.isclose(dist, 1.0)


def test_line_line_closest_points_intersecting():
    ray1 = homog.hray([0, 0, 0], [1, 0, 0])
    ray2 = homog.hray([0, 0, 0], [0, 1, 0])
    p1, p2 = homog.line_line_closest_points(ray1, ray2)
    # Closest points at the origin for both lines
    np.testing.assert_allclose(p1, [0, 0, 0, 1])
    np.testing.assert_allclose(p2, [0, 0, 0, 1])


def test_line_line_closest_points_parallel_lines():
    ray1 = homog.hray([0, 0, 0], [0, 0, 1])
    ray2 = homog.hray([1, 0, 0], [0, 0, 1])
    p1, p2 = homog.line_line_closest_points(ray1, ray2)
    # Closest points will have same z-coordinate, x offset of 1
    np.testing.assert_allclose(p1[:3], [0, 0, p1[2]])
    np.testing.assert_allclose(p2[:3], [1, 0, p2[2]])
    assert p1.shape == (4,)
    assert p2.shape == (4,)


def test_line_line_closest_points_skew_lines():
    ray1 = homog.hray([0, 0, 0], [1, 0, 0])
    ray2 = homog.hray([0, 1, 0], [0, 0, 1])
    p1, p2 = homog.line_line_closest_points(ray1, ray2)
    # The closest points lie on lines where distance is minimal
    dist = np.linalg.norm(p1[:3] - p2[:3])
    # This distance should equal line_line_distance
    dist_calc = homog.line_line_distance(ray1, ray2)
    assert np.isclose(dist, dist_calc)


def test_hray_invalid_direction_raises():
    # Direction zero vector should lead to normalization issues
    origin = [0, 0, 0]
    direction = [0, 0, 0]
    with pytest.raises(ZeroDivisionError):
        homog.hray(origin, direction)


def test_line_line_distance_pa_basic():
    # Use points and axes manually with perpendicular directions
    pt1 = np.array([0, 0, 0, 1])
    ax1 = homog.hnormalized(np.array([1, 0, 0, 0]))
    pt2 = np.array([0, 1, 0, 1])
    ax2 = homog.hnormalized(np.array([0, 1, 0, 0]))
    d = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    # These two lines are skew and perpendicular; distance is 1
    assert np.isclose(d, 1.0)


def test_line_line_distance_pa_parallel():
    pt1 = np.array([0, 0, 0, 1])
    ax1 = homog.hnormalized(np.array([0, 0, 1, 0]))
    pt2 = np.array([1, 0, 0, 1])
    ax2 = homog.hnormalized(np.array([0, 0, 1, 0]))
    d = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    # Lines parallel along z, offset 1 along x, distance=1
    assert np.isclose(d, 1.0)


def test_line_line_closest_points_pa_basic():
    pt1 = np.array([0, 0, 0, 1])
    ax1 = homog.hnormalized(np.array([1, 0, 0, 0]))
    pt2 = np.array([0, 1, 0, 1])
    ax2 = homog.hnormalized(np.array([0, 1, 0, 0]))
    q1, q2 = homog.line_line_closest_points_pa(pt1, ax1, pt2, ax2)
    d = np.linalg.norm(q1[:3] - q2[:3])
    # Distance matches line_line_distance_pa output
    dist = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    assert np.isclose(d, dist)


def test_line_line_closest_points_pa_parallel_lines():
    pt1 = np.array([0, 0, 0, 1])
    ax1 = homog.hnormalized(np.array([0, 0, 1, 0]))
    pt2 = np.array([1, 0, 0, 1])
    ax2 = homog.hnormalized(np.array([0, 0, 1, 0]))
    q1, q2 = homog.line_line_closest_points_pa(pt1, ax1, pt2, ax2)
    # Distance between closest points should equal their offset in x
    dist = np.linalg.norm(q1[:3] - q2[:3])
    assert np.isclose(dist, 1.0)


def test_line_line_closest_points_error_on_parallel_rays():
    # The current code comments mention this errors if ax1==ax2
    # We test that it raises or at least does NOT produce NaN
    ray1 = homog.hray([0, 0, 0], [0, 0, 1])
    ray2 = homog.hray([0, 0, 0], [0, 0, 1])
    with pytest.raises(Exception) as excinfo:
        homog.line_line_closest_points(ray1, ray2)
    # It should raise some Exception or ValueError; check type
    assert excinfo.type is not None


def test_hray_direction_normalized():
    # Check that direction vector in hray is normalized to length 1
    origin = np.array([0, 0, 0])
    direction = np.array([1, 2, 2])
    hray_out = homog.hray(origin, direction)
    dir_vec = hray_out[:3, 1]
    norm = np.linalg.norm(dir_vec)
    assert np.isclose(norm, 1.0)


def test_line_line_distance_symmetry():
    # Distance should be symmetric when swapping args
    ray1 = homog.hray([1, 2, 3], [0, 1, 0])
    ray2 = homog.hray([4, 5, 6], [1, 0, 0])
    dist1 = homog.line_line_distance(ray1, ray2)
    dist2 = homog.line_line_distance(ray2, ray1)
    assert np.isclose(dist1, dist2)


def test_line_line_closest_points_shapes_batched():
    origins = np.array([[0, 0, 0], [0, 0, 0]])
    directions = np.array([[1, 0, 0], [0, 1, 0]])
    rays = homog.hray(origins, directions)
    # Self distance and closest points for batch
    for i in range(2):
        p1, p2 = homog.line_line_closest_points(rays[i], rays[(i+1) % 2])
        assert p1.shape == (4,)
        assert p2.shape == (4,)


def test_line_line_distance_zero_length_direction():
    # direction vector zero length should raise or cause error in normalizing hray
    origin = np.array([0, 0, 0])
    direction = np.array([0, 0, 0])
    with pytest.raises(ZeroDivisionError):
        homog.hray(origin, direction)


def test_line_line_distance_different_shapes_error():
    ray1 = homog.hray([0, 0, 0], [1, 0, 0])
    ray2 = homog.hray([[0, 0, 0]], [0, 1, 0])
    # line_line_distance expects compatible shapes, but broadcasting works, so no error expected
    dist = homog.line_line_distance(ray1, ray2)
    assert isinstance(dist, float) or (isinstance(dist, np.ndarray) and dist.shape)


# End of tests
