import numpy as np
import pytest

from homog.homog import hray, hnormalized, is_valid_rays
from homog.homog import line_line_distance, line_line_closest_points


def test_hray_normalizes_direction_and_sets_homogeneous_coordinates():
    ray = hray([1, 2, 3], [0, 0, 5])

    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:3, 0], [1, 2, 3])
    np.testing.assert_allclose(ray[3, 0], 1)
    np.testing.assert_allclose(ray[:3, 1], [0, 0, 1])
    np.testing.assert_allclose(ray[3, 1], 0)
    assert is_valid_rays(ray)


def test_hray_preserves_broadcast_shape_for_batched_origins():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    ray = hray(origins, [0, 1, 0])

    assert ray.shape == (2, 4, 2)
    np.testing.assert_allclose(ray[:, :3, 0], np.array([[0, 0, 0], [1, 2, 3]]))
    np.testing.assert_allclose(ray[:, :3, 1], np.array([[0, 1, 0], [0, 1, 0]]))
    np.testing.assert_allclose(ray[:, 3, 0], [1, 1])
    np.testing.assert_allclose(ray[:, 3, 1], [0, 0])
    assert is_valid_rays(ray)


def test_hray_accepts_four_component_inputs_without_modifying_them():
    origin = np.array([1.0, 2.0, 3.0, 1.0])
    direction = np.array([0.0, 3.0, 4.0, 0.0])

    ray = hray(origin, direction)

    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:, 0], origin)
    np.testing.assert_allclose(ray[:3, 1], np.array([0.0, 0.6, 0.8]))
    np.testing.assert_allclose(ray[3, 1], 0.0)


def test_hray_handles_array_of_directions_broadcast_with_single_origin():
    origin = [1.0, 0.0, 0.0]
    directions = np.array([[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]])

    rays = hray(origin, directions)

    assert rays.shape == (2, 4, 2)
    np.testing.assert_allclose(rays[:, :3, 0], np.array([[1, 0, 0], [1, 0, 0]]))
    np.testing.assert_allclose(rays[:, :3, 1], np.array([[0, 0, 1], [0, 1, 0]]))


def test_line_line_distance_intersecting_lines_is_zero():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 0, 0], [0, 1, 0])

    dist = line_line_distance(ray1, ray2)

    assert dist == pytest.approx(0.0)


def test_line_line_distance_skew_lines_known_distance():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 1, 1], [0, 1, 0])

    dist = line_line_distance(ray1, ray2)

    assert dist == pytest.approx(1.0)


def test_line_line_distance_parallel_lines_returns_perpendicular_offset():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 2, 0], [5, 0, 0])

    dist = line_line_distance(ray1, ray2)

    assert dist == pytest.approx(2.0)


def test_line_line_distance_antiparallel_lines_behaves_like_parallel():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, -3, 0], [-1, 0, 0])

    dist = line_line_distance(ray1, ray2)

    assert dist == pytest.approx(3.0)


def test_line_line_distance_vectorized_over_multiple_pairs():
    ray1 = np.stack(
        [
            hray([0, 0, 0], [1, 0, 0]),
            hray([0, 0, 0], [1, 0, 0]),
        ],
        axis=0,
    )
    ray2 = np.stack(
        [
            hray([0, 0, 0], [0, 1, 0]),  # intersecting
            hray([0, 1, 1], [0, 1, 0]),  # skew, distance 1
        ],
        axis=0,
    )

    dist = line_line_distance(ray1, ray2)

    np.testing.assert_allclose(dist, np.array([0.0, 1.0]))


def test_line_line_closest_points_intersecting_lines_are_the_intersection_point():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 0, 0], [0, 1, 0])

    p1, p2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(p1, p2)
    np.testing.assert_allclose(p1[:3], [0, 0, 0])
    np.testing.assert_allclose(p1[3], 1)


def test_line_line_closest_points_skew_lines_return_points_on_each_line():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 1, 1], [0, 1, 0])

    p1, p2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(p1[:3], [0, 0, 0])
    np.testing.assert_allclose(p2[:3], [0, 0, 1])
    np.testing.assert_allclose(p1[3], 1)
    np.testing.assert_allclose(p2[3], 1)


def test_line_line_closest_points_skew_segment_is_orthogonal_to_both_lines():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 1, 1], [0, 1, 0])

    p1, p2 = line_line_closest_points(ray1, ray2)
    segment = p2[:3] - p1[:3]

    assert np.dot(segment, ray1[:3, 1]) == pytest.approx(0.0)
    assert np.dot(segment, ray2[:3, 1]) == pytest.approx(0.0)
    assert np.linalg.norm(segment) == pytest.approx(1.0)


def test_line_line_closest_points_are_symmetric_when_swapping_inputs():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 1, 1], [0, 1, 0])

    p1a, p2a = line_line_closest_points(ray1, ray2)
    p2b, p1b = line_line_closest_points(ray2, ray1)

    np.testing.assert_allclose(p1a, p1b)
    np.testing.assert_allclose(p2a, p2b)


def test_line_line_closest_points_parallel_lines_return_finite_values():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 2, 0], [3, 0, 0])

    p1, p2 = line_line_closest_points(ray1, ray2)

    assert np.all(np.isfinite(p1))
    assert np.all(np.isfinite(p2))
