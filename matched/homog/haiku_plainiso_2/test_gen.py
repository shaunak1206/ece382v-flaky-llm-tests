"""Unit tests for hray, line_line_distance and line_line_closest_points."""
import numpy as np
import pytest

from homog.homog import (
    hray,
    is_valid_rays,
    line_line_closest_points,
    line_line_distance,
)


def _ray(origin, direction):
    """One ray as a batch of length 1, shape (1, 4, 2). Direction is normalized."""
    return hray(np.array([origin], dtype=float), np.array([direction], dtype=float))


def _raw_ray(point, axis):
    """One ray as a batch of length 1, with the axis left unnormalized."""
    r = np.zeros((1, 4, 2))
    r[0, :3, 0] = point
    r[0, 3, 0] = 1.0
    r[0, :3, 1] = axis
    return r


# ---------------------------------------------------------------- hray


def test_hray_shape_is_four_by_two():
    r = hray([1.0, 2.0, 3.0], [0.0, 0.0, 1.0])
    assert r.shape == (4, 2)


def test_hray_origin_and_direction_columns():
    r = hray([1.0, 2.0, 3.0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(r[:, 0], [1.0, 2.0, 3.0, 1.0])
    np.testing.assert_allclose(r[:, 1], [0.0, 0.0, 1.0, 0.0])


def test_hray_normalizes_direction():
    r = hray([0.0, 0.0, 0.0], [0.0, 3.0, 4.0])
    np.testing.assert_allclose(r[:, 1], [0.0, 0.6, 0.8, 0.0])


def test_hray_accepts_homogeneous_origin():
    r = hray([1.0, 2.0, 3.0, 1.0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(r[:, 0], [1.0, 2.0, 3.0, 1.0])


def test_hray_result_is_valid_ray():
    r = hray([0.5, -1.0, 2.0], [1.0, 2.0, 2.0])
    assert is_valid_rays(r)


def test_hray_batched_origins_and_directions():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.0, 5.0]])
    dirs = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
    r = hray(origins, dirs)
    assert r.shape == (3, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origins)
    np.testing.assert_allclose(r[:, 3, 0], 1.0)
    np.testing.assert_allclose(
        r[:, :3, 1], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    np.testing.assert_allclose(r[:, 3, 1], 0.0)


def test_hray_rejects_origin_of_wrong_length():
    with pytest.raises(ValueError):
        hray([1.0, 2.0], [0.0, 0.0, 1.0])


# ------------------------------------------------- line_line_distance


def test_distance_skew_axis_aligned_lines():
    d = line_line_distance(_ray([0, 0, 0], [1, 0, 0]), _ray([0, 0, 1], [0, 1, 0]))
    np.testing.assert_allclose(d, [1.0], atol=1e-9)


def test_distance_skew_lines_with_offset():
    d = line_line_distance(_ray([0, 0, 0], [1, 0, 0]), _ray([2, 3, 4], [0, 1, 0]))
    np.testing.assert_allclose(d, [4.0], atol=1e-9)


def test_distance_skew_lines_with_diagonal_direction():
    d = line_line_distance(_ray([0, 0, 0], [1, 0, 0]), _ray([0, 0, 1], [0, 1, 1]))
    np.testing.assert_allclose(d, [1.0 / np.sqrt(2.0)], atol=1e-9)


def test_distance_parallel_lines():
    d = line_line_distance(_ray([0, 0, 0], [0, 0, 1]), _ray([3, 4, 7], [0, 0, 1]))
    np.testing.assert_allclose(d, [5.0], atol=1e-9)


def test_distance_antiparallel_lines():
    d = line_line_distance(_ray([0, 0, 0], [0, 0, 1]), _ray([3, 4, 7], [0, 0, -1]))
    np.testing.assert_allclose(d, [5.0], atol=1e-9)


def test_distance_collinear_lines_are_zero():
    d = line_line_distance(_ray([0, 0, 0], [1, 0, 0]), _ray([5, 0, 0], [1, 0, 0]))
    np.testing.assert_allclose(d, [0.0], atol=1e-9)


def test_distance_intersecting_lines_are_zero():
    d = line_line_distance(_ray([0, 0, 0], [1, 0, 0]), _ray([2, 0, 0], [1, 1, 0]))
    np.testing.assert_allclose(d, [0.0], atol=1e-9)


def test_distance_is_symmetric_in_its_arguments():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([2, 3, 4], [0, 1, 1])
    d12 = line_line_distance(r1, r2)
    d21 = line_line_distance(r2, r1)
    np.testing.assert_allclose(d12, d21, atol=1e-9)
    assert d12[0] > 0


def test_distance_batched_mixed_cases():
    r1 = hray(np.zeros((3, 3)),
              np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
    r2 = hray(np.array([[0.0, 0.0, 1.0], [3.0, 4.0, 7.0], [5.0, 0.0, 0.0]]),
              np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]))
    d = line_line_distance(r1, r2)
    assert d.shape == (3,)
    np.testing.assert_allclose(d, [1.0, 5.0, 0.0], atol=1e-9)


# ---------------------------------------- line_line_closest_points


def test_closest_points_axis_aligned_skew_lines():
    q1, q2 = line_line_closest_points(_ray([0, 0, 0], [1, 0, 0]),
                                      _ray([0, 0, 1], [0, 1, 0]))
    np.testing.assert_allclose(q1, [[0.0, 0.0, 0.0, 1.0]], atol=1e-9)
    np.testing.assert_allclose(q2, [[0.0, 0.0, 1.0, 1.0]], atol=1e-9)


def test_closest_points_offset_skew_lines_known_values():
    q1, q2 = line_line_closest_points(_ray([0, 0, 0], [1, 0, 0]),
                                      _ray([2, 3, 4], [0, 1, 0]))
    np.testing.assert_allclose(q1, [[2.0, 0.0, 0.0, 1.0]], atol=1e-9)
    np.testing.assert_allclose(q2, [[2.0, 0.0, 4.0, 1.0]], atol=1e-9)


def test_closest_points_lie_on_their_lines():
    r1 = _ray([1.0, -2.0, 0.5], [1.0, 2.0, 2.0])
    r2 = _ray([-3.0, 1.0, 4.0], [0.0, 1.0, -1.0])
    q1, q2 = line_line_closest_points(r1, r2)
    p1, a1 = r1[:, :3, 0], r1[:, :3, 1]
    p2, a2 = r2[:, :3, 0], r2[:, :3, 1]
    np.testing.assert_allclose(np.cross(q1[:, :3] - p1, a1), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.cross(q2[:, :3] - p2, a2), 0.0, atol=1e-9)


def test_closest_points_segment_is_orthogonal_to_both_axes():
    r1 = _ray([1.0, -2.0, 0.5], [1.0, 2.0, 2.0])
    r2 = _ray([-3.0, 1.0, 4.0], [0.0, 1.0, -1.0])
    q1, q2 = line_line_closest_points(r1, r2)
    seg = q1[:, :3] - q2[:, :3]
    np.testing.assert_allclose(np.sum(seg * r1[:, :3, 1], axis=-1), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.sum(seg * r2[:, :3, 1], axis=-1), 0.0, atol=1e-9)


def test_closest_points_distance_matches_line_line_distance():
    r1 = _ray([1.0, -2.0, 0.5], [1.0, 2.0, 2.0])
    r2 = _ray([-3.0, 1.0, 4.0], [0.0, 1.0, -1.0])
    q1, q2 = line_line_closest_points(r1, r2)
    gap = np.linalg.norm(q1[:, :3] - q2[:, :3], axis=-1)
    np.testing.assert_allclose(gap, line_line_distance(r1, r2), atol=1e-9)


def test_closest_points_unnormalized_axes_give_same_points():
    q1, q2 = line_line_closest_points(_raw_ray([0, 0, 0], [3.0, 0.0, 0.0]),
                                      _raw_ray([2, 3, 4], [0.0, -0.5, 0.0]))
    np.testing.assert_allclose(q1, [[2.0, 0.0, 0.0, 1.0]], atol=1e-9)
    np.testing.assert_allclose(q2, [[2.0, 0.0, 4.0, 1.0]], atol=1e-9)


def test_closest_points_batched_rows():
    r1 = hray(np.zeros((2, 3)), np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    r2 = hray(np.array([[2.0, 3.0, 4.0], [0.0, 0.0, 1.0]]),
              np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)
    np.testing.assert_allclose(q1, [[2.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
                               atol=1e-9)
    np.testing.assert_allclose(q2, [[2.0, 0.0, 4.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
                               atol=1e-9)
