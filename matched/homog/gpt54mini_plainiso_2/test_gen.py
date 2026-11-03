import numpy as np
import pytest

from homog.homog import hray, hpoint, hnormalized, is_valid_rays
from homog.homog import line_line_distance, line_line_closest_points


def test_hray_from_3d_origin_and_direction_produces_valid_ray():
    r = hray([1.0, 2.0, 3.0], [0.0, 0.0, 2.0])
    assert r.shape == (4, 2)
    assert np.allclose(r[:, 0], [1.0, 2.0, 3.0, 1.0])
    assert np.allclose(r[:3, 1], [0.0, 0.0, 1.0])
    assert r[3, 1] == 0.0
    assert is_valid_rays(r)


def test_hray_accepts_4d_origin_and_direction_without_changing_origin():
    origin = np.array([4.0, 5.0, 6.0, 1.0])
    direction = np.array([0.0, 3.0, 0.0, 0.0])
    r = hray(origin, direction)
    assert np.allclose(r[:, 0], origin)
    assert np.allclose(r[:3, 1], [0.0, 1.0, 0.0])
    assert r[3, 0] == 1.0
    assert r[3, 1] == 0.0
    assert is_valid_rays(r)


def test_hray_broadcasts_origin_and_direction():
    origins = np.array([[0.0, 0.0, 0.0],
                        [1.0, 2.0, 3.0]])
    direction = np.array([0.0, 0.0, 5.0])
    r = hray(origins, direction)
    assert r.shape == (2, 4, 2)
    assert np.allclose(r[0, :, 0], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(r[1, :, 0], [1.0, 2.0, 3.0, 1.0])
    assert np.allclose(r[:, :3, 1], [[0.0, 0.0, 1.0],
                                     [0.0, 0.0, 1.0]])
    assert np.all(r[:, 3, 0] == 1.0)
    assert np.all(r[:, 3, 1] == 0.0)
    assert is_valid_rays(r)


def test_hray_direction_is_normalized_even_if_input_is_not_unit_length():
    r = hray([0.0, 0.0, 0.0], [3.0, 4.0, 0.0])
    assert np.allclose(r[:3, 1], [0.6, 0.8, 0.0])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_with_batch_input_returns_expected_shape_and_validity():
    origin = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])
    direction = np.array([[1.0, 0.0, 0.0],
                          [0.0, 2.0, 0.0],
                          [0.0, 0.0, 3.0]])
    r = hray(origin, direction)
    assert r.shape == (3, 4, 2)
    assert is_valid_rays(r)
    assert np.allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1.0)


def test_line_line_distance_intersecting_lines_is_zero():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    d = line_line_distance(ray1, ray2)
    assert np.isclose(d, 0.0)


def test_line_line_distance_skew_lines_matches_expected_value():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 1.0, 1.0], [0.0, 1.0, 0.0])
    d = line_line_distance(ray1, ray2)
    assert np.isclose(d, 1.0)


def test_line_line_distance_perpendicular_offset_lines_matches_expected_value():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 0.0, 2.5], [0.0, 1.0, 0.0])
    d = line_line_distance(ray1, ray2)
    assert np.isclose(d, 2.5)


def test_line_line_distance_parallel_lines_uses_perpendicular_separation():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 2.0, 0.0], [5.0, 0.0, 0.0])
    d = line_line_distance(ray1, ray2)
    assert np.isclose(d, 2.0)


def test_line_line_distance_anti_parallel_lines_uses_perpendicular_separation():
    ray1 = hray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    ray2 = hray([3.0, 0.0, 0.0], [0.0, -2.0, 0.0])
    d = line_line_distance(ray1, ray2)
    assert np.isclose(d, 3.0)


def test_line_line_distance_is_symmetric():
    ray1 = hray([1.0, 2.0, 3.0], [1.0, 1.0, 0.0])
    ray2 = hray([-2.0, 0.5, 1.0], [0.0, 1.0, 1.0])
    d12 = line_line_distance(ray1, ray2)
    d21 = line_line_distance(ray2, ray1)
    assert np.isclose(d12, d21)


def test_line_line_closest_points_intersecting_lines_return_same_point():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    assert q1.shape == (4,)
    assert q2.shape == (4,)
    assert np.allclose(q1, q2)
    assert np.allclose(q1[:3], [0.0, 0.0, 0.0])
    assert q1[3] == 1.0 and q2[3] == 1.0


def test_line_line_closest_points_skew_lines_produce_points_on_each_line():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 1.0, 1.0], [0.0, 1.0, 0.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    assert np.allclose(q1[:3], [0.0, 0.0, 0.0])
    assert np.allclose(q2[:3], [0.0, 0.0, 1.0])
    assert np.isclose(np.linalg.norm(q2[:3] - q1[:3]), 1.0)
    assert np.isclose(line_line_distance(ray1, ray2), np.linalg.norm(q2[:3] - q1[:3]))


def test_line_line_closest_points_are_on_the_input_lines():
    ray1 = hray([1.0, 2.0, 3.0], [1.0, 0.0, 0.0])
    ray2 = hray([4.0, 5.0, 6.0], [0.0, 1.0, 0.0])
    q1, q2 = line_line_closest_points(ray1, ray2)

    t1 = q1[:3] - ray1[:3, 0]
    t2 = q2[:3] - ray2[:3, 0]
    assert np.allclose(np.cross(t1, ray1[:3, 1]), [0.0, 0.0, 0.0])
    assert np.allclose(np.cross(t2, ray2[:3, 1]), [0.0, 0.0, 0.0])


def test_line_line_closest_points_midpoint_property_for_skew_lines():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 1.0, 1.0], [0.0, 1.0, 0.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    midpoint = 0.5 * (q1[:3] + q2[:3])
    assert np.allclose(midpoint, [0.0, 0.0, 0.5])


def test_line_line_closest_points_parallel_lines_raise_or_return_nonfinite():
    ray1 = hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = hray([0.0, 1.0, 0.0], [2.0, 0.0, 0.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    assert q1.shape == (4,)
    assert q2.shape == (4,)
    assert np.any(~np.isfinite(q1[:3])) or np.any(~np.isfinite(q2[:3])) or np.allclose(q1[:3], q2[:3])


def test_hpoint_round_trips_three_vector_and_ray_functions_accept_it():
    p = hpoint([7.0, 8.0, 9.0])
    assert np.allclose(p, [7.0, 8.0, 9.0, 1.0])
    r = hray(p, [1.0, 0.0, 0.0])
    assert is_valid_rays(r)
    assert np.allclose(r[:, 0], p)
