import numpy as np
import pytest

from homog import homog as hg


def make_ray(origin, direction):
    return hg.hray(np.asarray(origin, dtype=float), np.asarray(direction, dtype=float))


def test_hray_converts_3d_origin_and_direction_to_homogeneous_ray():
    ray = hg.hray([1.0, 2.0, 3.0], [0.0, 0.0, 2.0])
    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:, 0], [1.0, 2.0, 3.0, 1.0])
    np.testing.assert_allclose(ray[:, 1], [0.0, 0.0, 1.0, 0.0])


def test_hray_preserves_4d_origin_and_normalizes_direction():
    ray = hg.hray([1.0, 2.0, 3.0, 1.0], [0.0, 3.0, 0.0, 0.0])
    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:, 0], [1.0, 2.0, 3.0, 1.0])
    np.testing.assert_allclose(ray[:, 1], [0.0, 1.0, 0.0, 0.0])


def test_hray_broadcasts_origin_and_direction_arrays():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    directions = np.array([0.0, 0.0, 5.0])
    rays = hg.hray(origins, directions)
    assert rays.shape == (2, 4, 2)
    np.testing.assert_allclose(rays[0, :, 0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(rays[1, :, 0], [1.0, 2.0, 3.0, 1.0])
    np.testing.assert_allclose(rays[:, :, 1], np.array([[0.0, 0.0, 1.0, 0.0],
                                                        [0.0, 0.0, 1.0, 0.0]]))


def test_hray_keeps_ray_direction_unit_length():
    ray = hg.hray([0.0, 0.0, 0.0], [2.0, -2.0, 1.0])
    assert np.isclose(np.linalg.norm(ray[:3, 1]), 1.0)


def test_line_line_distance_identical_intersecting_lines_is_zero():
    r1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = make_ray([2.0, 0.0, 0.0], [3.0, 0.0, 0.0])
    d = hg.line_line_distance(r1, r2)
    assert np.isclose(d, 0.0)


def test_line_line_distance_skew_lines_simple_case():
    r1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = make_ray([0.0, 1.0, 1.0], [0.0, 1.0, 0.0])
    d = hg.line_line_distance(r1, r2)
    assert np.isclose(d, 1.0)


def test_line_line_distance_parallel_lines_uses_perpendicular_separation():
    r1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = make_ray([0.0, 2.0, 3.0], [10.0, 0.0, 0.0])
    d = hg.line_line_distance(r1, r2)
    assert np.isclose(d, np.sqrt(13.0))


def test_line_line_distance_anti_parallel_lines_uses_perpendicular_separation():
    r1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = make_ray([0.0, -4.0, 3.0], [-2.0, 0.0, 0.0])
    d = hg.line_line_distance(r1, r2)
    assert np.isclose(d, 5.0)


def test_line_line_distance_translation_invariance():
    r1 = make_ray([1.0, 2.0, 3.0], [1.0, 0.0, 0.0])
    r2 = make_ray([4.0, 6.0, 8.0], [0.0, 1.0, 0.0])
    d1 = hg.line_line_distance(r1, r2)

    shift = np.array([10.0, -7.0, 2.5])
    r1s = make_ray(np.array([1.0, 2.0, 3.0]) + shift, [1.0, 0.0, 0.0])
    r2s = make_ray(np.array([4.0, 6.0, 8.0]) + shift, [0.0, 1.0, 0.0])
    d2 = hg.line_line_distance(r1s, r2s)

    assert np.isclose(d1, d2)


def test_line_line_closest_points_intersecting_lines_return_same_point():
    r1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = make_ray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    q1, q2 = hg.line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1[:3], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(q2[:3], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(q1, q2, atol=1e-12)


def test_line_line_closest_points_skew_lines_give_expected_points():
    r1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = make_ray([0.0, 1.0, 1.0], [0.0, 1.0, 0.0])
    q1, q2 = hg.line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1[:3], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(q2[:3], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(q1[3], 1.0)
    np.testing.assert_allclose(q2[3], 1.0)
    assert np.isclose(np.linalg.norm(q1[:3] - q2[:3]), 1.0)


def test_line_line_closest_points_outputs_points_on_each_line():
    r1 = make_ray([1.0, 2.0, 3.0], [1.0, 1.0, 0.0])
    r2 = make_ray([4.0, 0.0, 1.0], [0.0, 1.0, 1.0])
    q1, q2 = hg.line_line_closest_points(r1, r2)

    # q1 should lie on line 1: q1 = p1 + t * dir1
    v1 = q1[:3] - r1[:3, 0]
    assert np.isclose(np.cross(v1, r1[:3, 1]) @ np.cross(v1, r1[:3, 1]), 0.0, atol=1e-12)

    # q2 should lie on line 2
    v2 = q2[:3] - r2[:3, 0]
    assert np.isclose(np.cross(v2, r2[:3, 1]) @ np.cross(v2, r2[:3, 1]), 0.0, atol=1e-12)

    # Segment between closest points must be perpendicular to both directions.
    sep = q2[:3] - q1[:3]
    assert np.isclose(np.dot(sep, r1[:3, 1]), 0.0, atol=1e-12)
    assert np.isclose(np.dot(sep, r2[:3, 1]), 0.0, atol=1e-12)


def test_line_line_closest_points_parallel_lines_returns_a_valid_pair():
    r1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = make_ray([0.0, 2.0, 0.0], [2.0, 0.0, 0.0])
    q1, q2 = hg.line_line_closest_points(r1, r2)

    # For parallel lines, the implementation still returns a consistent pair;
    # the separation should equal the line-line distance.
    d = hg.line_line_distance(r1, r2)
    assert np.isclose(np.linalg.norm(q2[:3] - q1[:3]), d)
    assert np.isclose(np.dot(q2[:3] - q1[:3], r1[:3, 1]), 0.0, atol=1e-12)


def test_line_line_distance_and_closest_points_agree_on_distance():
    r1 = make_ray([2.0, -1.0, 0.5], [1.0, 2.0, 0.0])
    r2 = make_ray([-3.0, 4.0, 1.5], [0.0, 1.0, 2.0])
    d = hg.line_line_distance(r1, r2)
    q1, q2 = hg.line_line_closest_points(r1, r2)
    assert np.isclose(np.linalg.norm(q2[:3] - q1[:3]), d, atol=1e-12)
