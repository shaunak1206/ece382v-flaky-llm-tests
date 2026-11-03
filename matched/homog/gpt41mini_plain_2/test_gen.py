import numpy as np
import pytest
from homog import homog


def test_hray_basic_shapes():
    origin = [1, 2, 3]
    direction = [0, 0, 1]
    ray = homog.hray(origin, direction)
    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:3, 0], origin)
    assert ray[3, 0] == 1
    # direction normalized and set in second column
    expected_dir = homog.hnormalized(direction)
    np.testing.assert_allclose(ray[:, 1], expected_dir)


def test_hray_broadcast_shape():
    origins = np.array([[0, 0, 0], [1, 1, 1]])
    directions = np.array([[1, 0, 0], [0, 1, 0]])
    rays = homog.hray(origins, directions)
    assert rays.shape[:-2] == (2,)
    np.testing.assert_allclose(rays[0, :3, 0], origins[0])
    np.testing.assert_allclose(rays[1, :3, 0], origins[1])
    np.testing.assert_allclose(rays[0, :, 1], homog.hnormalized(directions[0]))
    np.testing.assert_allclose(rays[1, :, 1], homog.hnormalized(directions[1]))


def test_line_line_distance_parallel():
    # Two parallel lines along x-axis at different y values
    pt1 = np.array([0, 0, 0, 1])
    pt2 = np.array([0, 1, 0, 1])
    ax1 = np.array([1, 0, 0, 0])
    ax2 = np.array([1, 0, 0, 0])
    ray1 = homog.hray(pt1[:3], ax1[:3])
    ray2 = homog.hray(pt2[:3], ax2[:3])
    dist = homog.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 1.0)


def test_line_line_distance_nonparallel():
    # lines intersecting at origin but offset along others
    pt1 = np.array([0, 0, 0, 1])
    pt2 = np.array([0, 1, 0, 1])
    ax1 = np.array([1, 0, 0, 0])
    ax2 = np.array([0, 1, 0, 0])
    ray1 = homog.hray(pt1[:3], ax1[:3])
    ray2 = homog.hray(pt2[:3], ax2[:3])
    dist = homog.line_line_distance(ray1, ray2)
    # The shortest distance is 0 because the point on the first line (0,0,0)
    # and the point on the second line (0,1,0) form a right angle with the direction vectors.
    assert np.isclose(dist, 0)


def test_line_line_closest_points_pa_simple():
    pt1 = np.array([[0, 0, 0]])
    pt2 = np.array([[0, 1, 0]])
    ax1 = homog.hnormalized(np.array([[1, 0, 0]]))
    ax2 = homog.hnormalized(np.array([[0, 1, 0]]))
    Q1, Q2 = homog.line_line_closest_points_pa(pt1, ax1, pt2, ax2)
    # Q1 and Q2 should be closest points on lines, Q1 on x-axis line, Q2 on y-axis line
    # For these orthogonal lines, closest points should be projections onto origin
    np.testing.assert_allclose(Q1, np.array([[0, 0, 0]]), atol=1e-7)
    np.testing.assert_allclose(Q2, np.array([[0, 1, 0]]), atol=1e-7)


def test_line_line_closest_points_identical_rays():
    pt = np.array([1, 2, 3])
    ax = homog.hnormalized(np.array([0, 1, 0]))
    r1 = homog.hray(pt, ax)
    r2 = homog.hray(pt, ax)
    Q1, Q2 = homog.line_line_closest_points(r1, r2)
    # Closest points should be exactly the same points
    np.testing.assert_allclose(Q1, pt)
    np.testing.assert_allclose(Q2, pt)


def test_line_line_closest_points_nonparallel():
    pt1 = np.array([0, 0, 0])
    pt2 = np.array([0, 1, 0])
    ax1 = homog.hnormalized(np.array([1, 0, 0]))
    ax2 = homog.hnormalized(np.array([0, 1, 0]))
    ray1 = homog.hray(pt1, ax1)
    ray2 = homog.hray(pt2, ax2)
    Q1, Q2 = homog.line_line_closest_points(ray1, ray2)
    # Q1 should be origin, Q2 should be pt2
    np.testing.assert_allclose(Q1[:3], pt1)
    np.testing.assert_allclose(Q2[:3], pt2)


def test_hray_requires_hpoint_and_hnormalized():
    # Origin must be converted with hpoint and direction normalized
    origin = [1, 0, 0]
    direction = [2, 0, 0]
    ray = homog.hray(origin, direction)
    assert ray.shape == (4, 2)
    assert ray[3, 0] == 1
    assert np.isclose(np.linalg.norm(ray[:3, 1]), 1)


def test_line_line_distance_pa_parallel_lines_zero_distance():
    pt1 = np.array([[0., 0., 0.]])
    pt2 = np.array([[0., 0., 0.]])
    ax1 = homog.hnormalized(np.array([[1., 0., 0.]]))
    ax2 = homog.hnormalized(np.array([[1., 0., 0.]]))
    # Lines are the same so distance=0
    dist = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    assert np.isclose(dist, 0)


def test_line_line_distance_pa_general_case():
    pt1 = np.array([[0., 0., 0.]])
    pt2 = np.array([[0., 1., 0.]])
    ax1 = homog.hnormalized(np.array([[1., 0., 0.]]))
    ax2 = homog.hnormalized(np.array([[0., 1., 0.]]))
    dist = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    # Should be 0 since the lines intersect at (0, 0, 0)
    assert np.isclose(dist, 0)


def test_line_line_distance_with_array_inputs():
    pt1 = np.array([[0., 0., 0.], [1., 0., 0.]])
    pt2 = np.array([[0., 1., 0.], [1., 1., 0.]])
    ax1 = homog.hnormalized(np.array([[1., 0., 0.], [0., 1., 0.]]))
    ax2 = homog.hnormalized(np.array([[0., 1., 0.], [1., 0., 0.]]))
    dist = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    assert dist.shape == (2,)
    assert np.all(dist >= 0)


def test_line_line_closest_points_pa_parallel_lines():
    pt = np.array([[0., 0., 0.]])
    ax = homog.hnormalized(np.array([[1., 0., 0.]]))
    Q1, Q2 = homog.line_line_closest_points_pa(pt, ax, pt + np.array([[0, 1, 0]]), ax)
    # When lines are parallel, closest points will lie along pt1 and pt2 projected along axis
    # We check shape and that points are distinct
    assert Q1.shape == (1, 3)
    assert Q2.shape == (1, 3)
    assert not np.allclose(Q1, Q2)


def test_line_line_closest_points_pa_verbose_output(capfd):
    pt1 = np.array([[0., 0., 0.]])
    pt2 = np.array([[0., 1., 0.]])
    ax1 = homog.hnormalized(np.array([[1., 0., 0.]]))
    ax2 = homog.hnormalized(np.array([[0., 1., 0.]]))
    _ = homog.line_line_closest_points_pa(pt1, ax1, pt2, ax2, verbose=1)
    out, err = capfd.readouterr()
    assert 'C21' in out
    assert 't1' in out
    assert 'Q1' in out


def test_hray_direction_normalization_consistency():
    direction = [2, 2, 0]
    ray = homog.hray([0, 0, 0], direction)
    expected_dir = direction / np.linalg.norm(direction)
    np.testing.assert_allclose(ray[:3, 1], expected_dir)


def test_line_line_distance_identical_rays_zero():
    origin = [0, 0, 0]
    direction = [1, 0, 0]
    ray1 = homog.hray(origin, direction)
    ray2 = homog.hray(origin, direction)
    d = homog.line_line_distance(ray1, ray2)
    assert np.isclose(d, 0)


def test_line_line_distance_perpendicular_offset():
    # Two lines perpendicular and offset by 1 in z direction
    pt1 = [0, 0, 0]
    ax1 = [1, 0, 0]
    pt2 = [0, 0, 1]
    ax2 = [0, 1, 0]
    ray1 = homog.hray(pt1, ax1)
    ray2 = homog.hray(pt2, ax2)
    dist = homog.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 1)


def test_line_line_closest_points_pa_identical_lines():
    pt1 = np.array([[1., 0., 0.]])
    ax1 = homog.hnormalized(np.array([[1., 0., 0.]]))
    Q1, Q2 = homog.line_line_closest_points_pa(pt1, ax1, pt1, ax1)
    # Closest points coincide
    np.testing.assert_allclose(Q1, Q2)


def test_line_line_closest_points_same_points_different_directions():
    pt = np.array([1., 2., 3.])
    ax1 = homog.hnormalized(np.array([1., 0., 0.]))
    ax2 = homog.hnormalized(np.array([0., 1., 0.]))
    ray1 = homog.hray(pt, ax1)
    ray2 = homog.hray(pt, ax2)
    Q1, Q2 = homog.line_line_closest_points(ray1, ray2)
    # Points are on lines closest to each other; expect Q1 and Q2 distinct in general
    assert Q1.shape == (4,)
    assert Q2.shape == (4,)


def test_line_line_distance_broadcasting_error():
    pt1 = np.array([0., 0., 0.])
    ax1 = np.array([1., 0., 0.])
    pt2 = np.array([[0., 1., 0.], [1., 0., 0.]])
    ax2 = np.array([[0., 1., 0.], [0., 1., 0.]])
    ray1 = homog.hray(pt1, ax1)
    ray2 = homog.hray(pt2, ax2)
    # Should work with broadcasting shape (2,) for ray2 and (,) for ray1
    dist = homog.line_line_distance(ray1, ray2)
    assert dist.shape == (2,)
    assert np.all(dist >= 0)
