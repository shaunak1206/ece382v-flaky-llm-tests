import numpy as np
import pytest

from homog.homog import hray, line_line_distance, line_line_closest_points


def test_hray_from_3d_origin_and_direction_basic():
    origin = np.array([1.0, 2.0, 3.0])
    direction = np.array([0.0, 0.0, 2.0])

    ray = hray(origin, direction)

    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:3, 0], origin)
    np.testing.assert_allclose(ray[3, 0], 1.0)
    np.testing.assert_allclose(ray[:3, 1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(ray[3, 1], 0.0)


def test_hray_accepts_4d_origin_and_direction():
    origin = np.array([1.0, 2.0, 3.0, 1.0])
    direction = np.array([4.0, 0.0, 0.0, 0.0])

    ray = hray(origin, direction)

    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:, 0], origin)
    np.testing.assert_allclose(ray[:, 1], np.array([1.0, 0.0, 0.0, 0.0]))


def test_hray_normalizes_direction_vector():
    ray = hray([0, 0, 0], [3, 4, 0])
    np.testing.assert_allclose(np.linalg.norm(ray[:3, 1]), 1.0)
    np.testing.assert_allclose(ray[:3, 1], np.array([0.6, 0.8, 0.0]))


def test_hray_broadcasts_origin_and_direction_arrays():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    direction = np.array([0.0, 1.0, 0.0])

    rays = hray(origins, direction)

    assert rays.shape == (2, 4, 2)
    np.testing.assert_allclose(rays[0, :3, 0], origins[0])
    np.testing.assert_allclose(rays[1, :3, 0], origins[1])
    np.testing.assert_allclose(rays[:, :3, 1], np.array([[0.0, 1.0, 0.0]] * 2))


def test_hray_preserves_broadcasted_direction_origin_shapes():
    origins = np.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])  # (2, 1, 3)
    directions = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])  # (2, 1, 3)

    rays = hray(origins, directions)

    assert rays.shape == (2, 1, 4, 2)
    np.testing.assert_allclose(rays[0, 0, :3, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(rays[1, 0, :3, 0], [1.0, 0.0, 0.0])


def test_line_line_distance_intersecting_lines_is_zero():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 0, 0], [0, 1, 0])

    dist = line_line_distance(ray1, ray2)

    assert dist == pytest.approx(0.0)


def test_line_line_distance_parallel_offset_lines():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 1, 0], [1, 0, 0])

    dist = line_line_distance(ray1, ray2)

    assert dist == pytest.approx(1.0)


def test_line_line_distance_skew_lines():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 1, 1], [0, 1, 0])

    dist = line_line_distance(ray1, ray2)

    assert dist == pytest.approx(1.0)


def test_line_line_distance_antiparallel_lines_use_perpendicular_distance():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 2, 0], [-5, 0, 0])

    dist = line_line_distance(ray1, ray2)

    assert dist == pytest.approx(2.0)


def test_line_line_distance_with_broadcasted_inputs():
    ray1 = np.stack(
        [
            hray([0, 0, 0], [1, 0, 0]),
            hray([0, 0, 0], [1, 0, 0]),
        ],
        axis=0,
    )
    ray2 = hray([0, 1, 0], [1, 0, 0])

    dist = line_line_distance(ray1, ray2)

    np.testing.assert_allclose(dist, np.array([1.0, 1.0]))


def test_line_line_closest_points_intersecting_lines_same_point():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 0, 0], [0, 1, 0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1, q2)
    np.testing.assert_allclose(q1[:3], [0.0, 0.0, 0.0])


def test_line_line_closest_points_skew_lines_expected_points():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 1, 1], [0, 1, 0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(q2[:3], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(q1[:3] - q2[:3], np.array([0.0, 0.0, -1.0]))


def test_line_line_closest_points_parallel_lines_return_points_on_each_line():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 2, 0], [1, 0, 0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    # The implementation is not defined for parallel lines, but it should still
    # return finite points.
    assert np.isfinite(q1).all()
    assert np.isfinite(q2).all()
    np.testing.assert_allclose(q1[3], 1.0)
    np.testing.assert_allclose(q2[3], 1.0)


def test_line_line_closest_points_points_lie_on_the_original_lines():
    ray1 = hray([1, 2, 3], [1, 1, 0])
    ray2 = hray([4, -1, 2], [0, 1, 1])

    q1, q2 = line_line_closest_points(ray1, ray2)

    # q1 should be on ray1 and q2 on ray2:
    # (q - p) should be orthogonal to the line direction for closest points.
    np.testing.assert_allclose(np.dot(q1[:3] - ray1[:3, 0], ray1[:3, 1]), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.dot(q2[:3] - ray2[:3, 0], ray2[:3, 1]), 0.0, atol=1e-10)


def test_line_line_closest_points_are_symmetric_in_the_inputs():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 1, 1], [0, 1, 0])

    q1a, q2a = line_line_closest_points(ray1, ray2)
    q2b, q1b = line_line_closest_points(ray2, ray1)

    np.testing.assert_allclose(q1a, q1b)
    np.testing.assert_allclose(q2a, q2b)
