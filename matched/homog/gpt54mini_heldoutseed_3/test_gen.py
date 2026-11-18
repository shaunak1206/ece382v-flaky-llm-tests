import numpy as np
import pytest

from homog.homog import hray, line_line_distance, line_line_closest_points


def make_ray(point, direction):
    return hray(np.asarray(point, dtype=float), np.asarray(direction, dtype=float))


def test_hray_basic_shape_and_homogeneous_structure():
    ray = hray([1, 2, 3], [4, 0, 0])

    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:3, 0], [1, 2, 3])
    assert ray[3, 0] == 1
    np.testing.assert_allclose(ray[:3, 1], [1, 0, 0])
    assert ray[3, 1] == 0


def test_hray_normalizes_direction_vector():
    ray = hray([0, 0, 0], [0, 3, 4])

    np.testing.assert_allclose(ray[:3, 1], [0, 0.6, 0.8])
    np.testing.assert_allclose(np.linalg.norm(ray[:3, 1]), 1.0)


def test_hray_accepts_4d_origin_and_direction_without_changing_them_except_normalization():
    origin = np.array([5.0, -2.0, 7.5, 1.0])
    direction = np.array([0.0, 10.0, 0.0, 0.0])

    ray = hray(origin, direction)

    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:, 0], origin)
    np.testing.assert_allclose(ray[:3, 1], [0.0, 1.0, 0.0])
    assert ray[3, 1] == 0


def test_hray_broadcasts_batched_inputs():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])

    rays = hray(origins, directions)

    assert rays.shape == (2, 4, 2)
    np.testing.assert_allclose(rays[0, :3, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(rays[1, :3, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(rays[0, :3, 1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rays[1, :3, 1], [0.0, 0.0, 1.0])


def test_hray_preserves_broadcasted_origin_against_single_direction():
    origins = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ray = hray(origins, [0.0, 1.0, 0.0])

    assert ray.shape == (2, 4, 2)
    np.testing.assert_allclose(ray[:, :3, 0], origins)
    np.testing.assert_allclose(ray[:, :3, 1], np.array([[0.0, 1.0, 0.0],
                                                         [0.0, 1.0, 0.0]]))


def test_line_line_distance_intersecting_lines_is_zero():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 0, 0], [0, 1, 0])

    dist = line_line_distance(ray1, ray2)

    assert np.isclose(dist, 0.0)


def test_line_line_distance_perpendicular_skew_lines():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 1, 1], [0, 1, 0])

    dist = line_line_distance(ray1, ray2)

    assert np.isclose(dist, 1.0)


def test_line_line_distance_parallel_lines_uses_point_to_line_distance():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 2, 0], [3, 0, 0])

    dist = line_line_distance(ray1, ray2)

    assert np.isclose(dist, 2.0)


def test_line_line_distance_anti_parallel_lines_match_parallel_distance():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, -3, 0], [-5, 0, 0])

    dist = line_line_distance(ray1, ray2)

    assert np.isclose(dist, 3.0)


def test_line_line_distance_nearly_parallel_prefers_parallel_branch():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 4, 0], [1.0, 1e-6, 0.0])

    dist = line_line_distance(ray1, ray2)

    assert np.isclose(dist, 4.0, atol=1e-6)


def test_line_line_distance_batched_inputs():
    ray1 = hray(np.array([[0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0]]),
                np.array([[1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0]]))
    ray2 = hray(np.array([[0.0, 1.0, 1.0],
                          [0.0, 2.0, 0.0]]),
                np.array([[0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0]]))

    dist = line_line_distance(ray1, ray2)

    np.testing.assert_allclose(dist, [1.0, 2.0])


def test_line_line_closest_points_for_intersecting_lines_return_same_point():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 0, 0], [0, 1, 0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(q2[:3], [0.0, 0.0, 0.0])


def test_line_line_closest_points_for_skew_lines_are_correct_and_perpendicular():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 1, 1], [0, 1, 0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(q2[:3], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(q2[:3] - q1[:3], [0.0, 0.0, 1.0])
    assert np.isclose(np.dot(q2[:3] - q1[:3], ray1[:3, 1]), 0.0)
    assert np.isclose(np.dot(q2[:3] - q1[:3], ray2[:3, 1]), 0.0)


def test_line_line_closest_points_return_points_on_each_line():
    ray1 = make_ray([1, 2, 3], [1, 0, 0])
    ray2 = make_ray([4, 5, 6], [0, 1, 0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    # q1 should lie on ray1 line and q2 on ray2 line.
    assert np.isclose(q1[1], 2.0)
    assert np.isclose(q1[2], 3.0)
    assert np.isclose(q2[0], 4.0)
    assert np.isclose(q2[2], 6.0)


def test_line_line_closest_points_batched_inputs():
    ray1 = hray(np.array([[0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0]]),
                np.array([[1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0]]))
    ray2 = hray(np.array([[0.0, 1.0, 1.0],
                          [0.0, 2.0, 1.0]]),
                np.array([[0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0]]))

    q1, q2 = line_line_closest_points(ray1, ray2)

    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)
    np.testing.assert_allclose(q1[:, :3], np.array([[0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(q2[:, :3], np.array([[0.0, 0.0, 1.0],
                                                    [0.0, 0.0, 1.0]]))


def test_line_line_closest_points_parallel_lines_raise_or_produce_nonfinite():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 2, 0], [1, 0, 0])

    # The implementation is documented as failing for ax1 == ax2; accept either
    # a hard failure or a non-finite result, but do not silently return a valid
    # finite answer.
    try:
        q1, q2 = line_line_closest_points(ray1, ray2)
    except Exception:
        return

    assert not np.all(np.isfinite(q1[:3])) or not np.all(np.isfinite(q2[:3]))
