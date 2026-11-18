import numpy as np
import pytest

from homog.homog import (
    hray,
    hnormalized,
    hpoint,
    line_line_closest_points,
    line_line_distance,
)


def test_hray_creates_expected_shape_and_homogeneous_structure():
    origin = [1.0, 2.0, 3.0]
    direction = [0.0, 0.0, 2.0]

    ray = hray(origin, direction)

    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:3, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(ray[3, 0], 1.0)
    np.testing.assert_allclose(ray[:3, 1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(ray[3, 1], 0.0)


def test_hray_uses_hpoint_for_origin_passes_through_4d_origin_unchanged():
    origin = [4.0, 5.0, 6.0, 1.0]
    direction = [1.0, 0.0, 0.0]

    ray = hray(origin, direction)

    np.testing.assert_allclose(ray[:, 0], np.array(origin))
    np.testing.assert_allclose(ray[:3, 1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(ray[3, 1], 0.0)


def test_hray_broadcasts_over_batch_dimensions():
    origin = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    direction = np.array([0.0, 1.0, 0.0])

    rays = hray(origin, direction)

    assert rays.shape == (2, 4, 2)
    np.testing.assert_allclose(rays[0, :3, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(rays[1, :3, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(rays[..., :3, 1], np.array([[0.0, 1.0, 0.0]] * 2))


def test_hray_accepts_numpy_arrays_and_does_not_modify_inputs():
    origin = np.array([7.0, 8.0, 9.0], dtype=float)
    direction = np.array([0.0, 3.0, 0.0], dtype=float)
    origin_copy = origin.copy()
    direction_copy = direction.copy()

    _ = hray(origin, direction)

    np.testing.assert_array_equal(origin, origin_copy)
    np.testing.assert_array_equal(direction, direction_copy)


def test_hray_normalizes_direction_vector():
    ray = hray([0.0, 0.0, 0.0], [0.0, 4.0, 0.0])
    np.testing.assert_allclose(np.linalg.norm(ray[:3, 1]), 1.0)
    np.testing.assert_allclose(ray[:3, 1], [0.0, 1.0, 0.0])


def test_line_line_distance_parallel_lines_uses_perpendicular_offset():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 3.0, 0.0], [1.0, 0.0, 0.0])

    dist = line_line_distance(ray1, ray2)

    np.testing.assert_allclose(dist, 3.0)


def test_line_line_distance_intersecting_lines_is_zero():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    dist = line_line_distance(ray1, ray2)

    np.testing.assert_allclose(dist, 0.0)


def test_line_line_distance_skew_lines_matches_known_example():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 1.0, 1.0], [0.0, 1.0, 0.0])

    dist = line_line_distance(ray1, ray2)

    np.testing.assert_allclose(dist, 1.0)


def test_line_line_distance_is_symmetric_for_general_lines():
    ray1 = hray([1.0, 2.0, 3.0], [1.0, 1.0, 0.0])
    ray2 = hray([-2.0, 4.0, 1.0], [0.0, 1.0, 1.0])

    d12 = line_line_distance(ray1, ray2)
    d21 = line_line_distance(ray2, ray1)

    np.testing.assert_allclose(d12, d21)


def test_line_line_distance_broadcasts_over_batches():
    ray1 = np.stack(
        [
            hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            hray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ],
        axis=0,
    )
    ray2 = hray([0.0, 2.0, 0.0], [1.0, 0.0, 0.0])

    dist = line_line_distance(ray1, ray2)

    assert dist.shape == (2,)
    np.testing.assert_allclose(dist, [2.0, 0.0])


def test_line_line_closest_points_intersecting_lines_return_same_point():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1, q2)
    np.testing.assert_allclose(q1[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(q1[3], 1.0)


def test_line_line_closest_points_for_parallel_lines_returns_endpoints_on_each_line():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 3.0, 0.0], [1.0, 0.0, 0.0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(q2[:3], [0.0, 3.0, 0.0])
    np.testing.assert_allclose(q1[3], 1.0)
    np.testing.assert_allclose(q2[3], 1.0)
    np.testing.assert_allclose(line_line_distance(ray1, ray2), 3.0)


def test_line_line_closest_points_between_skew_lines_are_orthogonal_to_both_directions():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 1.0, 1.0], [0.0, 1.0, 0.0])

    q1, q2 = line_line_closest_points(ray1, ray2)
    d = q2[:3] - q1[:3]

    np.testing.assert_allclose(np.dot(d, ray1[:3, 1]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(d, ray2[:3, 1]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(d), line_line_distance(ray1, ray2))


def test_line_line_closest_points_are_symmetric_up_to_swapping_outputs():
    ray1 = hray([1.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    ray2 = hray([-1.0, 2.0, 1.0], [0.0, 1.0, 1.0])

    q1a, q2a = line_line_closest_points(ray1, ray2)
    q1b, q2b = line_line_closest_points(ray2, ray1)

    np.testing.assert_allclose(q1a, q2b)
    np.testing.assert_allclose(q2a, q1b)


def test_line_line_closest_points_broadcasts_over_batches():
    ray1 = np.stack(
        [
            hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            hray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ],
        axis=0,
    )
    ray2 = hray([0.0, 1.0, 0.0], [0.0, 1.0, 0.0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)
    np.testing.assert_allclose(q1[0, :3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(q2[0, :3], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(q1[1, :3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(q2[1, :3], [0.0, 0.0, 0.0])


def test_line_line_closest_points_collinear_case_returns_finite_values():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([2.0, 0.0, 0.0], [2.0, 0.0, 0.0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    assert np.all(np.isfinite(q1))
    assert np.all(np.isfinite(q2))
    np.testing.assert_allclose(q1[3], 1.0)
    np.testing.assert_allclose(q2[3], 1.0)
    np.testing.assert_allclose(line_line_distance(ray1, ray2), 0.0, atol=1e-12)


def test_line_line_distance_does_not_depend_on_ray_point_parameterization():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray1_shifted = hray([5.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 4.0, 0.0], [1.0, 1.0, 0.0])

    d1 = line_line_distance(ray1, ray2)
    d2 = line_line_distance(ray1_shifted, ray2)

    np.testing.assert_allclose(d1, d2)
