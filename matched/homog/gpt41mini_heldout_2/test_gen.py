import numpy as np
import pytest

from homog.homog import hray, line_line_distance, line_line_closest_points
from homog.homog import hpoint, hnormalized, hdot, hcross


def test_hray_basic_shapes_and_types():
    origin = np.array([1, 2, 3])
    direction = np.array([0, 0, 1])
    r = hray(origin, direction)
    # Output shape (...,4,2) and correct last coords
    assert r.shape == (4, 2)
    # check origin is hpoint with last w=1
    np.testing.assert_almost_equal(r[:, 0], np.array([1, 2, 3, 1]))
    # direction vector normalized with w=0
    norm_dir = r[:, 1]
    np.testing.assert_almost_equal(norm_dir[3], 0)
    np.testing.assert_almost_equal(np.linalg.norm(norm_dir[:3]), 1)

def test_hray_broadcasting():
    origins = np.array([[0, 0, 0], [1, 1, 1]])
    directions = np.array([[1, 0, 0], [0, 1, 0]])
    r = hray(origins, directions)
    assert r.shape == (2, 4, 2)
    for i in range(2):
        np.testing.assert_almost_equal(r[i, :, 0], np.hstack((origins[i], 1)))
        np.testing.assert_almost_equal(r[i, :, 1][3], 0)
        assert np.isclose(np.linalg.norm(r[i, :3, 1]), 1)

def test_hray_direction_normalization():
    origin = [0, 0, 0]
    direction = [10, 0, 0]
    r = hray(origin, direction)
    np.testing.assert_almost_equal(r[:, 1][3], 0)
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1)

def test_line_line_distance_parallel_lines():
    # Parallel rays along x-axis with offset in y
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    r1 = hray(origin1, direction1)

    origin2 = np.array([0, 1, 0])
    direction2 = np.array([1, 0, 0])
    r2 = hray(origin2, direction2)

    dist = line_line_distance(r1, r2)
    assert np.allclose(dist, 1)

def test_line_line_distance_intersecting_lines():
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    r1 = hray(origin1, direction1)

    origin2 = np.array([0, 0, 0])
    direction2 = np.array([0, 1, 0])
    r2 = hray(origin2, direction2)

    dist = line_line_distance(r1, r2)
    assert np.allclose(dist, 0)

def test_line_line_distance_skew_lines():
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    r1 = hray(origin1, direction1)

    origin2 = np.array([0, 1, 1])
    direction2 = np.array([0, 1, 0])
    r2 = hray(origin2, direction2)

    dist = line_line_distance(r1, r2)
    # Distance should be 1 (difference in z)
    assert np.isclose(dist, 1)

def test_line_line_closest_points_intersecting():
    origin1 = np.array([0, 0, 0])
    d1 = np.array([1, 0, 0])
    ray1 = hray(origin1, d1)

    origin2 = np.array([0, 0, 0])
    d2 = np.array([0, 1, 0])
    ray2 = hray(origin2, d2)

    q1, q2 = line_line_closest_points(ray1, ray2)
    np.testing.assert_almost_equal(q1, q2)
    assert np.allclose(q1, origin1)

def test_line_line_closest_points_skew():
    origin1 = np.array([0, 0, 0])
    d1 = np.array([1, 0, 0])
    ray1 = hray(origin1, d1)

    origin2 = np.array([0, 1, 1])
    d2 = np.array([0, 1, 0])
    ray2 = hray(origin2, d2)

    q1, q2 = line_line_closest_points(ray1, ray2)

    # Distance between closest points equals line_line_distance
    dist = np.linalg.norm(q2 - q1)
    dist_func = line_line_distance(ray1, ray2)
    assert np.isclose(dist, dist_func)

def test_line_line_closest_points_parallel_lines():
    origin1 = np.array([0, 0, 0])
    d1 = np.array([1, 0, 0])
    ray1 = hray(origin1, d1)

    origin2 = np.array([0, 1, 0])
    d2 = np.array([1, 0, 0])
    ray2 = hray(origin2, d2)

    q1, q2 = line_line_closest_points(ray1, ray2)
    # For parallel lines closest points projected from origins with offset
    # Because they are parallel, the distance is the same everywhere
    dist = np.linalg.norm(q2 - q1)
    dist_func = line_line_distance(ray1, ray2)
    assert np.isclose(dist, dist_func)
    # The line between q1 and q2 should be perpendicular to direction
    v = q2 - q1
    # Dot product of v with direction should be ~0 (perp)
    dot1 = np.dot(v[:3], d1/np.linalg.norm(d1))
    dot2 = np.dot(v[:3], d2/np.linalg.norm(d2))
    assert np.isclose(dot1, 0)
    assert np.isclose(dot2, 0)

def test_hray_raises_with_invalid_direction():
    origin = [0, 0, 0]
    direction = [0, 0, 0]  # zero direction vector not valid for normalization
    with pytest.raises(Exception):
        _ = hray(origin, direction)

def test_line_line_distance_multidim():
    origin1 = np.array([[0, 0, 0], [0, 0, 0]])
    d1 = np.array([[1, 0, 0], [1, 0, 0]])
    r1 = hray(origin1, d1)

    origin2 = np.array([[0, 1, 0], [0, 0, 1]])
    d2 = np.array([[1, 0, 0], [0, 1, 0]])
    r2 = hray(origin2, d2)

    dists = line_line_distance(r1, r2)
    expected = np.array([1, 1])
    np.testing.assert_almost_equal(dists, expected)

def test_line_line_closest_points_multidim():
    origin1 = np.array([[0, 0, 0], [0, 0, 0]])
    d1 = np.array([[1, 0, 0], [0, 0, 1]])
    r1 = hray(origin1, d1)

    origin2 = np.array([[0, 1, 0], [0, 1, 1]])
    d2 = np.array([[1, 0, 0], [0, 1, 0]])
    r2 = hray(origin2, d2)

    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)
    # Distances should match
    dists = np.linalg.norm(q2[:, :3] - q1[:, :3], axis=1)
    expected_dists = line_line_distance(r1, r2)
    np.testing.assert_allclose(dists, expected_dists)

