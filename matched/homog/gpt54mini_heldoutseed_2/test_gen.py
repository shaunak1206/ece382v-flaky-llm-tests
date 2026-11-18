import numpy as np
import pytest

from homog import homog as hg


def make_ray(origin, direction):
    return hg.hray(np.asarray(origin, dtype=float), np.asarray(direction, dtype=float))


def test_hray_from_3d_origin_and_direction_has_expected_shape_and_homogeneous_parts():
    r = make_ray([1.0, 2.0, 3.0], [0.0, 0.0, 5.0])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:3, 0], [1.0, 2.0, 3.0])
    assert r[3, 0] == 1
    np.testing.assert_allclose(r[:3, 1], [0.0, 0.0, 1.0])
    assert r[3, 1] == 0


def test_hray_normalizes_direction_vector():
    r = make_ray([0.0, 0.0, 0.0], [3.0, 4.0, 0.0])
    np.testing.assert_allclose(r[:3, 1], [0.6, 0.8, 0.0])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_preserves_4d_origin_and_direction():
    origin = np.array([1.0, 2.0, 3.0, 1.0])
    direction = np.array([0.0, 0.0, 2.0, 0.0])
    r = hg.hray(origin, direction)
    np.testing.assert_allclose(r[:, 0], origin)
    np.testing.assert_allclose(r[:3, 1], [0.0, 0.0, 1.0])
    assert r[3, 1] == 0


def test_hray_broadcasts_over_multiple_rays():
    origins = np.array([[0.0, 0.0, 0.0],
                        [1.0, 2.0, 3.0]])
    directions = np.array([[1.0, 0.0, 0.0],
                           [0.0, 0.0, 2.0]])
    r = hg.hray(origins, directions)
    assert r.shape == (2, 4, 2)
    np.testing.assert_allclose(r[0, :3, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(r[1, :3, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(r[0, :3, 1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(r[1, :3, 1], [0.0, 0.0, 1.0])


def test_hray_valid_ray_passes_is_valid_rays():
    r = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert hg.is_valid_rays(r)


def test_line_line_distance_intersecting_lines_is_zero():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    dist = hg.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 0.0)


def test_line_line_distance_skew_lines_matches_expected_distance():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    dist = hg.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 1.0)


def test_line_line_distance_parallel_lines_uses_perpendicular_separation():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 2.0, 0.0], [2.0, 0.0, 0.0])
    dist = hg.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 2.0)


def test_line_line_distance_antiparallel_lines_uses_perpendicular_separation():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, -3.0, 0.0], [-5.0, 0.0, 0.0])
    dist = hg.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 3.0)


def test_line_line_distance_broadcasted_input_returns_array():
    ray1 = np.stack([
        make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ], axis=0)
    ray2 = np.stack([
        make_ray([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        make_ray([0.0, 2.0, 0.0], [0.0, 1.0, 0.0]),
    ], axis=0)
    dist = hg.line_line_distance(ray1, ray2)
    assert dist.shape == (2,)
    np.testing.assert_allclose(dist, [1.0, 0.0])


def test_line_line_closest_points_intersecting_lines_return_same_point():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    q1, q2 = hg.line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(q1[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(q2[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(q1, q2)


def test_line_line_closest_points_skew_lines_return_expected_points():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    q1, q2 = hg.line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(q1[:3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(q2[:3], [0.0, 0.0, 1.0])
    assert np.isclose(hg.line_line_distance(ray1, ray2), np.linalg.norm(q2[:3] - q1[:3]))


def test_line_line_closest_points_points_lie_on_their_respective_lines():
    ray1 = make_ray([1.0, 2.0, 3.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    q1, q2 = hg.line_line_closest_points(ray1, ray2)

    t1 = q1[:3] - ray1[:3, 0]
    t2 = q2[:3] - ray2[:3, 0]
    np.testing.assert_allclose(np.cross(t1, ray1[:3, 1]), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.cross(t2, ray2[:3, 1]), [0.0, 0.0, 0.0])


def test_line_line_closest_points_closest_segment_is_perpendicular_to_both_lines():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    q1, q2 = hg.line_line_closest_points(ray1, ray2)
    seg = q2[:3] - q1[:3]
    assert np.isclose(np.dot(seg, ray1[:3, 1]), 0.0)
    assert np.isclose(np.dot(seg, ray2[:3, 1]), 0.0)


def test_line_line_closest_points_broadcasted_input_returns_expected_shapes():
    ray1 = np.stack([
        make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ], axis=0)
    ray2 = np.stack([
        make_ray([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        make_ray([0.0, 2.0, 0.0], [0.0, 1.0, 0.0]),
    ], axis=0)

    q1, q2 = hg.line_line_closest_points(ray1, ray2)
    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)
    np.testing.assert_allclose(q1[0, :3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(q2[0, :3], [0.0, 0.0, 1.0])
