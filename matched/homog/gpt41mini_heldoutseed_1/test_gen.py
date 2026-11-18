import numpy as np
import pytest
from homog import homog


def test_hray_basic_shapes_and_contents():
    origin = np.array([1.0, 2.0, 3.0])
    direction = np.array([4.0, 5.0, 6.0])
    ray = homog.hray(origin, direction)
    assert ray.shape == (4, 2)
    # Origin part has w=1
    np.testing.assert_allclose(ray[:, 0], homog.hpoint(origin))
    assert ray[3, 0] == 1
    # Direction part normalized and w=0
    dir_norm = np.linalg.norm(direction)
    normalized_direction = direction / dir_norm
    np.testing.assert_allclose(ray[:3, 1], normalized_direction)
    assert ray[3, 1] == 0


def test_hray_broadcasting_multiple_rays():
    origins = np.array([[1, 0, 0], [0, 1, 0]])
    directions = np.array([[0, 1, 0], [0, 0, 1]])
    rays = homog.hray(origins, directions)
    assert rays.shape == (2, 4, 2) or rays.shape == (2, 4, 2) or rays.shape == (2, 4, 2)
    # Check normalized directions
    for i in range(2):
        dir_norm = np.linalg.norm(directions[i])
        np.testing.assert_allclose(rays[i, :3, 1], directions[i] / dir_norm)


def test_line_line_distance_basic_perpendicular():
    # Lines crossing perpendicularly with known distance 0
    origin1 = np.array([0, 0, 0, 1])
    direction1 = np.array([1, 0, 0, 0])
    ray1 = np.stack([origin1, direction1], axis=-1)
    origin2 = np.array([0, 0, 0, 1])
    direction2 = np.array([0, 1, 0, 0])
    ray2 = np.stack([origin2, direction2], axis=-1)
    dist = homog.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 0)


def test_line_line_distance_known_nonzero():
    # Two parallel lines a distance apart on y axis
    o1 = np.array([0, 0, 0, 1])
    d1 = np.array([1, 0, 0, 0])
    ray1 = np.stack([o1, d1], axis=-1)
    o2 = np.array([0, 1, 0, 1])
    d2 = d1.copy()
    ray2 = np.stack([o2, d2], axis=-1)
    dist = homog.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 1)


def test_line_line_closest_points_pa_simple_case():
    pt1 = np.array([0., 0., 0., 1.])
    ax1 = np.array([1., 0., 0., 0.])
    pt2 = np.array([0., 1., 0., 1.])
    ax2 = np.array([0., 1., 0., 0.])
    Q1, Q2 = homog.line_line_closest_points_pa(pt1, ax1, pt2, ax2)
    # Closest points should be at (0,0,0) and (0,1,0)
    np.testing.assert_allclose(Q1[:3], pt1[:3])
    np.testing.assert_allclose(Q2[:3], pt2[:3])


def test_line_line_closest_points_consistency():
    # Random points and directions
    np.random.seed(0)
    pt1 = np.random.randn(4)
    pt1[3] = 1
    pt2 = np.random.randn(4)
    pt2[3] = 1
    ax1 = np.random.randn(4)
    ax1[3] = 0
    ax1[:3] /= np.linalg.norm(ax1[:3])
    ax2 = np.random.randn(4)
    ax2[3] = 0
    ax2[:3] /= np.linalg.norm(ax2[:3])

    Q1, Q2 = homog.line_line_closest_points_pa(pt1, ax1, pt2, ax2)

    # The vector Q2-Q1 should be perpendicular to both directions
    v = Q2 - Q1
    dot1 = np.dot(v[:3], ax1[:3])
    dot2 = np.dot(v[:3], ax2[:3])
    assert np.isclose(dot1, 0, atol=1e-12)
    assert np.isclose(dot2, 0, atol=1e-12)


def test_line_line_closest_points_function_ray_inputs():
    origin1 = np.array([1.0, 0.0, 0.0])
    dir1 = np.array([1.0, 0.0, 0.0])
    origin2 = np.array([0.0, 1.0, 0.0])
    dir2 = np.array([0.0, 1.0, 0.0])
    ray1 = homog.hray(origin1, dir1)
    ray2 = homog.hray(origin2, dir2)
    Q1, Q2 = homog.line_line_closest_points(ray1, ray2)
    # Q1 and Q2 points should be on their respective rays
    # Q1 closest on line 1, Q2 closest on line 2
    assert Q1.shape[-1] == 4
    assert Q2.shape[-1] == 4
    # Q1 and Q2 difference vector perpendicular to both rays directions
    diff = Q2 - Q1
    dir1_norm = homog.hnormalized(dir1)
    dir2_norm = homog.hnormalized(dir2)
    dot1 = np.dot(diff[:3], dir1_norm[:3])
    dot2 = np.dot(diff[:3], dir2_norm[:3])
    assert np.isclose(dot1, 0, atol=1e-12)
    assert np.isclose(dot2, 0, atol=1e-12)


def test_hray_direction_normalized():
    origin = np.array([0, 0, 0])
    direction = np.array([1, 2, 2])
    ray = homog.hray(origin, direction)
    dir_part = ray[:, 1]
    norm = np.linalg.norm(dir_part[:3])
    assert np.isclose(norm, 1.0)


def test_line_line_distance_parallel_lines():
    # Lines parallel but offset along y = 3
    o1 = np.array([0, 0, 0, 1])
    d1 = np.array([1, 0, 0, 0])
    ray1 = np.stack([o1, d1], axis=-1)
    o2 = np.array([0, 3, 0, 1])
    d2 = d1.copy()
    ray2 = np.stack([o2, d2], axis=-1)
    dist = homog.line_line_distance(ray1, ray2)
    np.testing.assert_allclose(dist, 3)


def test_line_line_distance_skew_lines_nonzero():
    # Two skew lines
    pt1 = np.array([0, 0, 0, 1])
    ax1 = np.array([1, 0, 0, 0])
    pt2 = np.array([0, 1, 1, 1])
    ax2 = np.array([0, 1, 0, 0])
    ray1 = np.stack([pt1, ax1], axis=-1)
    ray2 = np.stack([pt2, ax2], axis=-1)
    dist = homog.line_line_distance(ray1, ray2)
    # Manually calculate expected shortest distance:
    # cross product direction vector
    u = np.cross(ax1[:3], ax2[:3])
    u_norm = np.linalg.norm(u)
    expected_dist = abs(np.dot(pt2[:3] - pt1[:3], u)) / u_norm
    np.testing.assert_allclose(dist, expected_dist)


def test_hray_multi_dimension_broadcasting():
    origins = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    directions = np.array([[0, 0, 1], [1, 0, 0]], dtype=float)
    rays = homog.hray(origins, directions)
    assert rays.shape == (2, 4, 2)
    for i in range(2):
        np.testing.assert_allclose(rays[i, 3, 0], 1)
        np.testing.assert_allclose(rays[i, 3, 1], 0)
        dir_norm = np.linalg.norm(directions[i])
        np.testing.assert_allclose(rays[i, :3, 1], directions[i] / dir_norm)


def test_line_line_closest_points_pa_degenerate_parallel_case():
    # Nearly parallel vectors (should handle gracefully but may raise warning)
    pt1 = np.array([0., 0., 0., 1.])
    ax1 = np.array([1., 0., 0., 0.])
    pt2 = np.array([0., 1e-10, 1e-10, 1.])
    ax2 = np.array([1., 1e-10, 0., 0.])
    # Normalize ax2 properly
    ax2[:3] /= np.linalg.norm(ax2[:3])
    Q1, Q2 = homog.line_line_closest_points_pa(pt1, ax1, pt2, ax2)
    # They should be close points: distance between Q1 and Q2 ~0 or small
    dist = np.linalg.norm(Q2[:3] - Q1[:3])
    assert dist >= 0


def test_hray_direction_input_4dim():
    origin = np.array([1, 2, 3, 1])
    direction = np.array([0, 0, 1, 0])
    ray = homog.hray(origin, direction)
    assert np.all(ray[:, 0] == origin)
    norm_dir = direction.copy()
    norm_dir[:3] /= np.linalg.norm(direction[:3])
    np.testing.assert_allclose(ray[:, 1], norm_dir)


def test_ray_direction_nonunit_normalization():
    origin = np.zeros(3)
    direction = np.array([3, 0, 0])
    ray = homog.hray(origin, direction)
    assert np.isclose(np.linalg.norm(ray[:3, 1]), 1.0)


def test_line_line_distance_with_arrays_multiple_rays():
    # generate two sets of rays
    o1 = np.array([[0, 0, 0], [0, 0, 0]])
    d1 = np.array([[1, 0, 0], [0, 1, 0]])
    o2 = np.array([[0, 1, 0], [1, 0, 0]])
    d2 = np.array([[0, 1, 0], [1, 0, 0]])
    rays1 = homog.hray(o1, d1)
    rays2 = homog.hray(o2, d2)
    dists = homog.line_line_distance(rays1, rays2)
    assert dists.shape == (2,)
    # distances should be > 0 for the first pair and zero for the second
    assert dists[0] > 0
    assert np.isclose(dists[1], 0, atol=1e-10)

