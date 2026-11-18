import numpy as np
import pytest

from homog import homog as h


def test_hray_creates_valid_ray_from_3d_inputs():
    ray = h.hray([1, 2, 3], [0, 0, 5])
    assert ray.shape == (4, 2)
    assert np.allclose(ray[:3, 0], [1, 2, 3])
    assert ray[3, 0] == 1
    assert np.allclose(ray[:3, 1], [0, 0, 1])
    assert ray[3, 1] == 0
    assert h.is_valid_rays(ray)


def test_hray_preserves_4d_point_and_normalizes_direction():
    origin = np.array([1.0, 2.0, 3.0, 1.0])
    direction = np.array([3.0, 0.0, 4.0, 0.0])
    ray = h.hray(origin, direction)
    assert np.allclose(ray[:, 0], origin)
    assert np.allclose(ray[:3, 1], [0.6, 0.0, 0.8])
    assert ray[3, 1] == 0
    assert h.is_valid_rays(ray)


def test_hray_broadcasts_origin_and_direction_arrays():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    directions = np.array([[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    rays = h.hray(origins, directions)
    assert rays.shape == (2, 4, 2)
    assert np.allclose(rays[0, :3, 0], [0.0, 0.0, 0.0])
    assert np.allclose(rays[1, :3, 0], [1.0, 2.0, 3.0])
    assert np.allclose(rays[0, :3, 1], [0.0, 0.0, 1.0])
    assert np.allclose(rays[1, :3, 1], [0.0, 1.0, 0.0])
    assert np.allclose(rays[..., 3, 0], 1.0)
    assert np.allclose(rays[..., 3, 1], 0.0)


def test_hray_accepts_already_homogeneous_inputs():
    origin = np.array([1.0, 2.0, 3.0, 1.0])
    direction = np.array([0.0, 1.0, 0.0, 0.0])
    ray = h.hray(origin, direction)
    assert np.allclose(ray[:, 0], origin)
    assert np.allclose(ray[:, 1], direction)
    assert h.is_valid_rays(ray)


def test_hray_rejects_invalid_direction_length():
    with pytest.raises(ValueError):
        h.hray([1, 2, 3], [1, 2])


def test_line_line_distance_intersecting_lines_is_zero():
    ray1 = h.hray([0, 0, 0], [1, 0, 0])
    ray2 = h.hray([0, 0, 0], [0, 1, 0])
    dist = h.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 0.0)


def test_line_line_distance_skew_lines_matches_expected_value():
    # x-axis through origin and y-axis through z=1 have distance 1.
    ray1 = h.hray([0, 0, 0], [1, 0, 0])
    ray2 = h.hray([0, 0, 1], [0, 1, 0])
    dist = h.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 1.0)


def test_line_line_distance_parallel_lines_uses_perpendicular_offset():
    ray1 = h.hray([0, 0, 0], [1, 0, 0])
    ray2 = h.hray([0, 2, 0], [5, 0, 0])  # parallel, non-unit direction
    dist = h.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 2.0)


def test_line_line_distance_antiparallel_lines_uses_perpendicular_offset():
    ray1 = h.hray([0, 0, 0], [0, 0, 1])
    ray2 = h.hray([3, 4, 0], [0, 0, -2])
    dist = h.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 5.0)


def test_line_line_distance_is_symmetric_for_general_lines():
    ray1 = h.hray([1, 0, 0], [1, 2, 0])
    ray2 = h.hray([0, 1, 1], [0, 1, 3])
    d12 = h.line_line_distance(ray1, ray2)
    d21 = h.line_line_distance(ray2, ray1)
    assert np.isclose(d12, d21)


def test_line_line_distance_batched_inputs_return_broadcasted_shape():
    ray1 = h.hray(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    ray2 = h.hray(
        np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    dist = h.line_line_distance(ray1, ray2)
    assert dist.shape == (2,)
    assert np.allclose(dist, [1.0, 2.0])


def test_line_line_closest_points_intersecting_lines_return_same_point():
    ray1 = h.hray([0, 0, 0], [1, 0, 0])
    ray2 = h.hray([0, 0, 0], [0, 1, 0])
    q1, q2 = h.line_line_closest_points(ray1, ray2)
    assert q1.shape == (4,)
    assert q2.shape == (4,)
    assert np.allclose(q1, q2)
    assert np.allclose(q1[:3], [0, 0, 0])
    assert np.isclose(q1[3], 1.0)
    assert np.isclose(q2[3], 1.0)


def test_line_line_closest_points_skew_lines_have_expected_midpoint():
    # x-axis through origin and y-axis through z=1.
    ray1 = h.hray([0, 0, 0], [1, 0, 0])
    ray2 = h.hray([0, 0, 1], [0, 1, 0])
    q1, q2 = h.line_line_closest_points(ray1, ray2)
    assert np.allclose(q1[:3], [0, 0, 0])
    assert np.allclose(q2[:3], [0, 0, 1])
    assert np.isclose(h.line_line_distance(ray1, ray2), np.linalg.norm(q2[:3] - q1[:3]))


def test_line_line_closest_points_on_general_skew_lines_are_orthogonal_to_axes():
    ray1 = h.hray([1, 2, 3], [1, 1, 0])
    ray2 = h.hray([4, 0, 1], [0, 1, 1])
    q1, q2 = h.line_line_closest_points(ray1, ray2)
    delta = q2[:3] - q1[:3]
    ax1 = ray1[:3, 1]
    ax2 = ray2[:3, 1]
    assert np.isclose(np.dot(delta, ax1), 0.0, atol=1e-7)
    assert np.isclose(np.dot(delta, ax2), 0.0, atol=1e-7)


def test_line_line_closest_points_batched_inputs_return_broadcasted_shape():
    ray1 = h.hray(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    ray2 = h.hray(
        np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 1.0]]),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    q1, q2 = h.line_line_closest_points(ray1, ray2)
    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)
    d = h.line_line_distance(ray1, ray2)
    assert d.shape == (2,)
    assert np.allclose(np.linalg.norm(q2[:, :3] - q1[:, :3], axis=-1), d)


def test_line_line_closest_points_parallel_lines_returns_finite_points():
    ray1 = h.hray([0, 0, 0], [1, 0, 0])
    ray2 = h.hray([0, 2, 0], [2, 0, 0])
    q1, q2 = h.line_line_closest_points(ray1, ray2)
    assert np.all(np.isfinite(q1))
    assert np.all(np.isfinite(q2))
    assert np.isclose(h.line_line_distance(ray1, ray2), 2.0)
