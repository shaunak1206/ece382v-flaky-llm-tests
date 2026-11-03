import numpy as np
import pytest
from homog import homog

def test_hray_basic_properties():
    origin = np.array([1, 2, 3])
    direction = np.array([4, 5, 6])
    ray = homog.hray(origin, direction)
    # ray shape
    assert ray.shape == (4, 2)
    # origin part is 4-vector with last element 1
    np.testing.assert_allclose(ray[..., :, 0], homog.hpoint(origin))
    assert ray[3, 0] == 1
    # direction part is 4-vector direction normalized with last element 0
    dir_norm = np.linalg.norm(ray[:3, 1])
    np.testing.assert_allclose(dir_norm, 1, atol=1e-7)
    assert ray[3, 1] == 0

def test_hray_broadcasting():
    origins = np.array([[1, 0, 0], [0, 1, 0]])
    directions = np.array([1, 0, 0])
    rays = homog.hray(origins, directions)
    assert rays.shape == (2, 4, 2)
    for i in range(2):
        np.testing.assert_allclose(rays[i, :, 0], homog.hpoint(origins[i]))
        dir_norm = np.linalg.norm(rays[i, :3, 1])
        assert np.isclose(dir_norm, 1, atol=1e-7)

def test_line_line_distance_parallel_lines():
    # Two parallel rays offset along y axis
    ray1 = homog.hray(np.array([0, 0, 0]), np.array([1, 0, 0]))
    ray2 = homog.hray(np.array([0, 1, 0]), np.array([1, 0, 0]))
    dist = homog.line_line_distance(ray1, ray2)
    # Distance should be 1 (along y axis)
    assert np.isclose(dist, 1, atol=1e-7)

def test_line_line_distance_perpendicular():
    ray1 = homog.hray(np.array([0, 0, 0]), np.array([1, 0, 0]))
    ray2 = homog.hray(np.array([0, 1, 0]), np.array([0, 1, 0]))
    dist = homog.line_line_distance(ray1, ray2)
    # Should be 1 (distance from origin to line along y at x=0)
    assert np.isclose(dist, 1, atol=1e-7)

def test_line_line_distance_skew_lines():
    ray1 = homog.hray(np.array([0, 0, 0]), np.array([1, 0, 0]))
    ray2 = homog.hray(np.array([0, 1, 1]), np.array([0, 1, 0]))
    dist = homog.line_line_distance(ray1, ray2)
    # Distance should be sqrt(1^2 + 1^2) = sqrt(2)
    assert np.isclose(dist, np.sqrt(2), atol=1e-7)

def test_line_line_closest_points_simple():
    ray1 = homog.hray([0, 0, 0], [1, 0, 0])
    ray2 = homog.hray([0, 1, 0], [0, 1, 0])
    q1, q2 = homog.line_line_closest_points(ray1, ray2)
    # q1 on x axis, q2 on y=1 axis
    # Closest point on ray1 line is the projection onto x axis
    # They differ by vector along vertical (y) and they should be closest points
    np.testing.assert_allclose(q1[1], 0, atol=1e-7)
    np.testing.assert_allclose(q2[1], 1, atol=1e-7)
    # q1 and q2 differ by vector parallel to cross product of directions
    cross = np.cross(ray1[:3,1], ray2[:3,1])
    diff = q2[:3] - q1[:3]
    dot_with_cross = np.dot(diff, cross)
    assert np.isclose(dot_with_cross, 0, atol=1e-7)

def test_line_line_closest_points_identical_lines():
    # identical rays, closest points are arbitrary but should be equal
    origin = np.array([1, 2, 3])
    direction = np.array([0, 0, 1])
    ray1 = homog.hray(origin, direction)
    ray2 = homog.hray(origin, direction)
    with pytest.raises(Exception):
        # function may error or produce undefined closest points because directions are equal
        homog.line_line_closest_points(ray1, ray2, verbose=1)

def test_hray_invalid_direction_raises():
    origin = [0, 0, 0]
    # zero vector direction invalid for normalization, should raise (during normalization)
    direction = [0, 0, 0]
    with pytest.raises(Exception):
        homog.hray(origin, direction)

def test_line_line_distance_pa_vectorized():
    pt1 = np.array([[0, 0, 0], [0, 0, 0]])
    ax1 = np.array([[1, 0, 0], [1, 0, 0]])
    pt2 = np.array([[0, 1, 0], [0, 2, 0]])
    ax2 = np.array([[0, 1, 0], [0, 1, 0]])
    dist = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    # both distances equal to distance between parallel skew lines
    assert dist.shape == (2,)
    assert np.allclose(dist, [1, 2], atol=1e-7)

def test_line_line_closest_points_pa_basic():
    pt1 = np.array([0, 0, 0])
    ax1 = np.array([1, 0, 0])
    pt2 = np.array([0, 1, 0])
    ax2 = np.array([0, 1, 0])
    Q1, Q2 = homog.line_line_closest_points_pa(pt1, ax1, pt2, ax2)
    # They should be closest points on the two lines
    assert Q1.shape == (3,)
    assert Q2.shape == (3,)
    # vector between Q1 and Q2 should be perpendicular to directions ax1 and ax2
    diff = Q2 - Q1
    assert np.isclose(np.dot(diff, ax1), 0, atol=1e-7)
    assert np.isclose(np.dot(diff, ax2), 0, atol=1e-7)

def test_hray_returns_homogeneous_points():
    origin = np.array([1.5, -2.5, 3.0])
    direction = np.array([0, 0, 5])
    r = homog.hray(origin, direction)
    # Check homogeneous component of origin is 1, direction is 0
    assert r.shape == (4, 2)
    assert r[3, 0] == 1
    assert r[3, 1] == 0
    # Check that direction is normalized
    length_dir = np.linalg.norm(r[:3, 1])
    assert np.isclose(length_dir, 1, atol=1e-7)

def test_line_line_distance_pa_when_cross_zero_direction_parallel():
    pt1 = np.array([0, 0, 0])
    ax1 = np.array([1, 0, 0])
    pt2 = np.array([0, 1e-9, 0])
    ax2 = np.array([1, 0, 0])  # parallel directions
    dist = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    # Should fallback to proj_perp norm which is distance between points projected perpendicular to line
    expected = np.linalg.norm(homog.proj_perp(ax1, pt2 - pt1))
    assert np.isclose(dist, expected, atol=1e-7)
