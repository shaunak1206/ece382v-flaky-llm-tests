import numpy as np
import pytest

from homog import homog as hg


def test_hray_from_3d_inputs_returns_valid_ray_shape_and_w_row():
    origin = [1.0, 2.0, 3.0]
    direction = [0.0, 0.0, 5.0]

    ray = hg.hray(origin, direction)

    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:3, 0], np.array(origin, dtype=float))
    np.testing.assert_allclose(ray[3, 0], 1.0)
    np.testing.assert_allclose(ray[3, 1], 0.0)
    np.testing.assert_allclose(np.linalg.norm(ray[:3, 1]), 1.0)
    assert hg.is_valid_rays(ray)


def test_hray_passes_through_4d_origin_unchanged_except_normalization():
    origin = np.array([1.0, 2.0, 3.0, 1.0])
    direction = np.array([0.0, 3.0, 0.0, 0.0])

    ray = hg.hray(origin, direction)

    np.testing.assert_allclose(ray[:, 0], origin)
    np.testing.assert_allclose(ray[:3, 1], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(ray[3, 1], 0.0)


def test_hray_broadcasts_over_batch_dimension():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    directions = np.array([[0.0, 0.0, 2.0], [0.0, 4.0, 0.0]])

    rays = hg.hray(origins, directions)

    assert rays.shape == (2, 4, 2)
    np.testing.assert_allclose(rays[0, :3, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(rays[1, :3, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.linalg.norm(rays[0, :3, 1]), 1.0)
    np.testing.assert_allclose(np.linalg.norm(rays[1, :3, 1]), 1.0)
    assert hg.is_valid_rays(rays)


def test_hray_accepts_already_homogeneous_direction():
    origin = [0.0, 0.0, 0.0]
    direction = [1.0, 2.0, 2.0, 0.0]

    ray = hg.hray(origin, direction)

    np.testing.assert_allclose(ray[:3, 1], np.array(direction[:3]) / 3.0)
    np.testing.assert_allclose(ray[3, 1], 0.0)


def test_hray_raises_on_invalid_direction_shape():
    with pytest.raises((IndexError, ValueError)):
        hg.hray([0.0, 0.0, 0.0], [1.0, 2.0])


def test_line_line_distance_intersecting_lines_is_zero():
    ray1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hg.hray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    dist = hg.line_line_distance(ray1, ray2)

    np.testing.assert_allclose(dist, 0.0)


def test_line_line_distance_skew_lines_matches_expected_value():
    ray1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hg.hray([0.0, 1.0, 1.0], [0.0, 1.0, 0.0])

    dist = hg.line_line_distance(ray1, ray2)

    np.testing.assert_allclose(dist, 1.0)


def test_line_line_distance_parallel_lines_uses_perpendicular_separation():
    ray1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hg.hray([0.0, 2.5, 0.0], [3.0, 0.0, 0.0])

    dist = hg.line_line_distance(ray1, ray2)

    np.testing.assert_allclose(dist, 2.5)


def test_line_line_distance_antiparallel_lines_is_distance_between_lines():
    ray1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hg.hray([0.0, -4.0, 0.0], [-2.0, 0.0, 0.0])

    dist = hg.line_line_distance(ray1, ray2)

    np.testing.assert_allclose(dist, 4.0)


def test_line_line_distance_broadcasts_over_batches():
    ray1 = hg.hray(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    ray2 = hg.hray(
        np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
    )

    dist = hg.line_line_distance(ray1, ray2)

    np.testing.assert_allclose(dist, np.array([0.0, 0.0]))


def test_line_line_closest_points_intersecting_lines_return_same_point():
    ray1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hg.hray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    q1, q2 = hg.line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1, q2)
    np.testing.assert_allclose(q1[:3], [0.0, 0.0, 0.0])


def test_line_line_closest_points_skew_lines_have_expected_endpoints():
    ray1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hg.hray([0.0, 1.0, 1.0], [0.0, 1.0, 0.0])

    q1, q2 = hg.line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1[:3], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(q2[:3], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(hg.line_line_distance(ray1, ray2), np.linalg.norm(q2[:3] - q1[:3]))


def test_line_line_closest_points_batch_shape():
    ray1 = hg.hray(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    ray2 = hg.hray(
        np.array([[0.0, 1.0, 1.0], [0.0, 2.0, 3.0]]),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )

    q1, q2 = hg.line_line_closest_points(ray1, ray2)

    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)
    np.testing.assert_allclose(q1[:, 3], 1.0)
    np.testing.assert_allclose(q2[:, 3], 1.0)


def test_line_line_closest_points_parallel_lines_returns_finite_or_nan_without_crashing():
    ray1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hg.hray([0.0, 2.0, 0.0], [2.0, 0.0, 0.0])

    q1, q2 = hg.line_line_closest_points(ray1, ray2)

    assert q1.shape == (4,)
    assert q2.shape == (4,)
    assert np.all(np.isfinite(q1) | np.isnan(q1))
    assert np.all(np.isfinite(q2) | np.isnan(q2))
