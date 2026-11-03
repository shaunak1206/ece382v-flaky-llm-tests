import numpy as np
import pytest

from homog.homog import (hray, hdot, hcross, hnorm, is_valid_rays,
                         line_line_distance, line_line_closest_points)


def _rays(origins, directions):
    """Build a batch of rays with shape (N, 4, 2)."""
    return hray(np.array(origins, dtype='f8'), np.array(directions, dtype='f8'))


# ---------------------------------------------------------------- hray


def test_hray_shape_and_values_from_3vectors():
    r = hray([1, 2, 3], [0, 0, 2])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_accepts_4vector_origin():
    r = hray([4, 5, 6, 1], [1, 0, 0])
    np.testing.assert_allclose(r[:, 0], [4, 5, 6, 1])
    np.testing.assert_allclose(r[:, 1], [1, 0, 0, 0])


def test_hray_accepts_4vector_direction_with_zero_w():
    r3 = hray([1, 2, 3], [3, 0, 4])
    r4 = hray([1, 2, 3], [3, 0, 4, 0])
    np.testing.assert_allclose(r3, r4)


def test_hray_direction_is_normalized():
    r = hray([0, 0, 0], [3, 4, 12])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)
    np.testing.assert_allclose(r[:3, 1], np.array([3, 4, 12]) / 13.0)


def test_hray_direction_scale_invariant():
    d = np.array([1.0, -2.0, 0.5])
    r1 = hray([1, 1, 1], d)
    r2 = hray([1, 1, 1], 7.5 * d)
    np.testing.assert_allclose(r1, r2)


def test_hray_homogeneous_components():
    r = hray([9, 8, 7], [1, 1, 1])
    assert r[3, 0] == 1
    assert r[3, 1] == 0


def test_hray_batch_shape():
    rs = np.random.RandomState(1)
    origins = rs.randn(5, 3)
    dirs = rs.randn(5, 3)
    r = hray(origins, dirs)
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origins)
    np.testing.assert_allclose(np.linalg.norm(r[:, :3, 1], axis=-1), np.ones(5))
    np.testing.assert_allclose(r[:, 3, 0], np.ones(5))
    np.testing.assert_allclose(r[:, 3, 1], np.zeros(5))


def test_hray_broadcasts_single_origin_over_many_directions():
    dirs = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
    r = hray([1, 2, 3], dirs)
    assert r.shape == (3, 4, 2)
    for i in range(3):
        np.testing.assert_allclose(r[i, :, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, :3, 1], np.eye(3))


def test_hray_broadcasts_many_origins_over_single_direction():
    origins = np.array([[0.0, 0, 0], [1, 1, 1], [2, 3, 4], [5, 6, 7]])
    r = hray(origins, [0, 0, 1])
    assert r.shape == (4, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origins)
    for i in range(4):
        np.testing.assert_allclose(r[i, :, 1], [0, 0, 1, 0])


def test_hray_output_is_valid_ray_and_float():
    r = hray([1, 2, 3], [2, -1, 4])
    assert is_valid_rays(r)
    assert r.dtype == np.float64


def test_hray_invalid_origin_length_raises():
    with pytest.raises(ValueError):
        hray([1, 2], [0, 0, 1])


# ------------------------------------------------- line_line_distance


def test_distance_skew_axis_aligned_lines():
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[0, 0, 5]], [[0, 1, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [5.0], atol=1e-9)


def test_distance_skew_diagonal_lines():
    r1 = _rays([[0, 0, 0], [0, 0, 0]], [[1, 1, 0], [1, 1, 0]])
    r2 = _rays([[0, 0, 4], [3, -3, 2]], [[1, -1, 0], [1, -1, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [4.0, 2.0],
                               atol=1e-9)


def test_distance_intersecting_lines_is_zero():
    r1 = _rays([[1, 2, 3]], [[1, 0, 0]])
    r2 = _rays([[1, 5, 3]], [[0, 1, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [0.0], atol=1e-9)


def test_distance_parallel_lines_uses_perpendicular_offset():
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[10, 3, 4]], [[1, 0, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [5.0], atol=1e-9)


def test_distance_antiparallel_lines():
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[7, 3, 4]], [[-1, 0, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [5.0], atol=1e-9)


def test_distance_identical_lines_is_zero():
    r1 = _rays([[0, 0, 0]], [[0, 1, 0]])
    r2 = _rays([[0, 6, 0]], [[0, 1, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [0.0], atol=1e-9)


def test_distance_is_symmetric_and_nonnegative():
    rs = np.random.RandomState(7)
    r1 = hray(rs.randn(8, 3), rs.randn(8, 3))
    r2 = hray(rs.randn(8, 3), rs.randn(8, 3))
    d12 = line_line_distance(r1, r2)
    d21 = line_line_distance(r2, r1)
    assert d12.shape == (8,)
    np.testing.assert_allclose(d12, d21, atol=1e-9)
    assert np.all(d12 >= 0)


def test_distance_unchanged_by_sliding_origin_along_own_axis():
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[0, 0, 5]], [[0, 1, 0]])
    r2_slid = _rays([[0, 17, 5]], [[0, 1, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2),
                               line_line_distance(r1, r2_slid), atol=1e-9)


# ------------------------------------------ line_line_closest_points


def test_closest_points_axis_aligned_skew_lines():
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[2, 3, 5]], [[0, 1, 0]])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[2, 0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(q2, [[2, 0, 5, 1]], atol=1e-9)


def test_closest_points_coincide_for_intersecting_lines():
    r1 = _rays([[1, 2, 3]], [[1, 0, 0]])
    r2 = _rays([[1, 5, 3]], [[0, 1, 0]])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[1, 2, 3, 1]], atol=1e-9)
    np.testing.assert_allclose(q2, [[1, 2, 3, 1]], atol=1e-9)


def test_closest_points_output_shape_and_homogeneous_w():
    rs = np.random.RandomState(3)
    r1 = hray(rs.randn(6, 3), rs.randn(6, 3))
    r2 = hray(rs.randn(6, 3), rs.randn(6, 3))
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (6, 4)
    assert q2.shape == (6, 4)
    np.testing.assert_allclose(q1[:, 3], np.ones(6), atol=1e-12)
    np.testing.assert_allclose(q2[:, 3], np.ones(6), atol=1e-12)


def test_closest_points_lie_on_their_lines():
    rs = np.random.RandomState(11)
    r1 = hray(rs.randn(10, 3), rs.randn(10, 3))
    r2 = hray(rs.randn(10, 3), rs.randn(10, 3))
    q1, q2 = line_line_closest_points(r1, r2)
    # (Q - origin) must be parallel to the axis, i.e. cross product is zero.
    off1 = hcross(q1 - r1[:, :, 0], r1[:, :, 1])
    off2 = hcross(q2 - r2[:, :, 0], r2[:, :, 1])
    np.testing.assert_allclose(hnorm(off1), np.zeros(10), atol=1e-8)
    np.testing.assert_allclose(hnorm(off2), np.zeros(10), atol=1e-8)


def test_closest_points_segment_is_perpendicular_to_both_lines():
    rs = np.random.RandomState(5)
    r1 = hray(rs.randn(10, 3), rs.randn(10, 3))
    r2 = hray(rs.randn(10, 3), rs.randn(10, 3))
    q1, q2 = line_line_closest_points(r1, r2)
    seg = q2 - q1
    np.testing.assert_allclose(hdot(seg, r1[:, :, 1]), np.zeros(10), atol=1e-8)
    np.testing.assert_allclose(hdot(seg, r2[:, :, 1]), np.zeros(10), atol=1e-8)


def test_closest_points_separation_matches_line_line_distance():
    rs = np.random.RandomState(21)
    r1 = hray(rs.randn(10, 3), rs.randn(10, 3))
    r2 = hray(rs.randn(10, 3), rs.randn(10, 3))
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(hnorm(q2 - q1), line_line_distance(r1, r2),
                               atol=1e-8)
