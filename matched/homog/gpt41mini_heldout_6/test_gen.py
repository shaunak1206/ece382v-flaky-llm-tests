import numpy as np
import pytest
from homog.homog import hray, line_line_distance, line_line_closest_points, hpoint, hnormalized, hdot

def test_hray_basic_structure():
    origin = [1, 2, 3]
    direction = [0, 0, 1]
    r = hray(origin, direction)
    assert r.shape == (4, 2)
    # origin part has w=1
    assert np.allclose(r[..., 3, 0], 1)
    # direction part has w=0 (hnormalized sets 4th coord to 0)
    assert np.allclose(r[..., 3, 1], 0)
    # origin xyz matches input
    np.testing.assert_allclose(r[:3, 0], origin)
    # direction is normalized
    norm_dir = np.linalg.norm(r[:3, 1])
    assert np.isclose(norm_dir, 1)
    # direction is close to input normalized vector
    expected_dir = np.array(direction) / np.linalg.norm(direction)
    np.testing.assert_allclose(r[:3, 1], expected_dir)


def test_hray_broadcasting():
    origins = np.array([[1, 0, 0], [0, 1, 0]])
    direction = [0, 0, 1]
    r = hray(origins, direction)
    assert r.shape == (2, 4, 2)
    for i in range(2):
        np.testing.assert_allclose(r[i, :3, 0], origins[i])
        assert r[i, 3, 0] == 1
    # direction part identical in both rays
    np.testing.assert_allclose(r[0, :3, 1], r[1, :3, 1])
    assert np.isclose(np.linalg.norm(r[0, :3, 1]), 1)


def test_hray_direction_normalization():
    origin = [0, 0, 0]
    direction = [1, 2, 3]
    r = hray(origin, direction)
    norm_dir = np.linalg.norm(r[:3, 1])
    assert np.isclose(norm_dir, 1)
    expected_dir = np.array(direction) / np.linalg.norm(direction)
    np.testing.assert_allclose(r[:3, 1], expected_dir)


def test_line_line_distance_zero_parallel():
    # Two identical rays
    origin = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    r1 = hray(origin, direction)
    r2 = hray(origin, direction)
    dist = line_line_distance(r1, r2)
    # Distance between identical lines is zero
    assert np.isclose(dist, 0)


def test_line_line_distance_non_parallel():
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    origin2 = np.array([0, 1, 0])
    direction2 = np.array([0, 1, 0])
    r1 = hray(origin1, direction1)
    r2 = hray(origin2, direction2)
    dist = line_line_distance(r1, r2)
    # Lines are skew, distance should be 1 (distance between the two parallel lines offset on y-axis)
    assert np.isclose(dist, 1, atol=1e-7)


def test_line_line_distance_perpendicular_cross():
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    origin2 = np.array([0, 0, 1])
    direction2 = np.array([0, 1, 0])
    r1 = hray(origin1, direction1)
    r2 = hray(origin2, direction2)
    dist = line_line_distance(r1, r2)
    # Distance is 1 (z separation)
    assert np.isclose(dist, 1, atol=1e-7)


def test_line_line_closest_points_basic():
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    origin2 = np.array([0, 1, 0])
    direction2 = np.array([0, 1, 0])
    r1 = hray(origin1, direction1)
    r2 = hray(origin2, direction2)
    Q1, Q2 = line_line_closest_points(r1, r2)
    # Closest point on first line should be at y=0, z=0 (somewhere along x-axis)
    assert np.isclose(Q1[1], 0)
    assert np.isclose(Q1[2], 0)
    # Closest point on second line should be at x=0, z=0 (somewhere along y=1 line)
    assert np.isclose(Q2[0], 0)
    assert np.isclose(Q2[2], 0)
    # Distance between closest points should be equal to line_line_distance
    dist = np.linalg.norm(Q1[:3] - Q2[:3])
    direct_dist = line_line_distance(r1, r2)
    assert np.isclose(dist, direct_dist)


def test_line_line_closest_points_parallel():
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    origin2 = np.array([0, 0.5, 0])
    direction2 = np.array([1, 0, 0])
    r1 = hray(origin1, direction1)
    r2 = hray(origin2, direction2)
    Q1, Q2 = line_line_closest_points(r1, r2)
    # Because lines are parallel, closest points on lines are origin projected
    # Direction vectors same normalized
    assert np.allclose(direction1 / np.linalg.norm(direction1), direction2 / np.linalg.norm(direction2))
    dist = np.linalg.norm(Q1[:3] - Q2[:3])
    expected_dist = line_line_distance(r1, r2)
    assert np.isclose(dist, expected_dist)


def test_line_line_closest_points_skew_lines():
    # Use two skew lines that are not parallel and not intersecting
    origin1 = np.array([1, 0, 0])
    direction1 = np.array([0, 1, 0])
    origin2 = np.array([0, 1, 1])
    direction2 = np.array([1, 0, 0])
    r1 = hray(origin1, direction1)
    r2 = hray(origin2, direction2)
    Q1, Q2 = line_line_closest_points(r1, r2)
    dist = np.linalg.norm(Q1[:3] - Q2[:3])
    expected_dist = line_line_distance(r1, r2)
    assert np.isclose(dist, expected_dist)


def test_hray_broadcast_origin_and_direction():
    origins = np.array([[0, 0, 0], [1, 1, 1]])
    directions = np.array([[1, 0, 0], [0, 1, 0]])
    r = hray(origins, directions)
    assert r.shape == (2, 4, 2)
    # Just check basic conditions
    for i in range(2):
        np.testing.assert_allclose(r[i, :3, 0], origins[i])
        assert r[i, 3, 0] == 1
        norm_dir = np.linalg.norm(r[i, :3, 1])
        assert np.isclose(norm_dir, 1)
        expected_dir = directions[i] / np.linalg.norm(directions[i])
        np.testing.assert_allclose(r[i, :3, 1], expected_dir)


def test_line_line_distance_array_inputs():
    # Two arrays of rays, test shape broadcasting and vectorized distance
    origins1 = np.array([[0, 0, 0], [1, 0, 0]])
    directions1 = np.array([[1, 0, 0], [0, 1, 0]])
    origins2 = np.array([[0, 1, 0], [1, 1, 0]])
    directions2 = np.array([[0, 1, 0], [1, 0, 0]])
    r1 = hray(origins1, directions1)
    r2 = hray(origins2, directions2)
    dists = line_line_distance(r1, r2)
    assert dists.shape == (2,)
    assert all(d >= 0 for d in dists)


def test_line_line_closest_points_array_inputs():
    origins1 = np.array([[0, 0, 0], [1, 0, 0]])
    directions1 = np.array([[1, 0, 0], [0, 1, 0]])
    origins2 = np.array([[0, 1, 0], [1, 1, 0]])
    directions2 = np.array([[0, 1, 0], [1, 0, 0]])
    r1 = hray(origins1, directions1)
    r2 = hray(origins2, directions2)
    Q1, Q2 = line_line_closest_points(r1, r2)
    assert Q1.shape == origins1.shape
    assert Q2.shape == origins2.shape
    # Distances match line_line_distance outputs
    dists = np.linalg.norm(Q1 - Q2, axis=-1)
    dists_func = line_line_distance(r1, r2)
    np.testing.assert_allclose(dists, dists_func, rtol=1e-5, atol=1e-8)


def test_hray_direction_zero_vector_raises():
    origin = [0, 0, 0]
    direction = [0, 0, 0]
    with pytest.raises(ZeroDivisionError):
        # Because direction normalization divides by norm zero
        hray(origin, direction)


def test_line_line_distance_with_identical_lines():
    origin = np.array([1, 1, 1])
    direction = np.array([1, 0, 0])
    r1 = hray(origin, direction)
    r2 = hray(origin, direction)
    dist = line_line_distance(r1, r2)
    assert np.isclose(dist, 0)


def test_line_line_closest_points_identical_lines():
    origin = np.array([0, 0, 0])
    direction = np.array([0, 1, 0])
    r1 = hray(origin, direction)
    r2 = hray(origin, direction)
    Q1, Q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(Q1, Q2)

