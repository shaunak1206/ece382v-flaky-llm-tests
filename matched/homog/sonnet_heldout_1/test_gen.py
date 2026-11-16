import numpy as np
import pytest

from homog.homog import (
    hray,
    line_line_distance,
    line_line_closest_points,
    is_valid_rays,
    rand_ray,
    hdot,
    hcross,
    hnorm,
)


def _ray(origin, direction):
    """Build a batch-of-one ray with shape (1, 4, 2)."""
    return hray(np.array([origin], dtype='f8'), np.array([direction], dtype='f8'))


# ----------------------------- hray -----------------------------


def test_hray_shape_single():
    r = hray([1, 2, 3], [0, 0, 1])
    assert r.shape == (4, 2)


def test_hray_origin_column_and_direction_column():
    r = hray([1, 2, 3], [0, 0, 1])
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_normalizes_direction():
    r = hray([0, 0, 0], [0, 0, 5])
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0])
    r = hray([0, 0, 0], [3, 4, 0])
    np.testing.assert_allclose(r[:3, 1], [0.6, 0.8, 0])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_accepts_four_element_inputs():
    r = hray([1, 2, 3, 1], [0, 2, 0, 0])
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 1, 0, 0])


def test_hray_batch_shape():
    np.random.seed(0)
    origin = np.random.randn(5, 3)
    direction = np.random.randn(5, 3)
    r = hray(origin, direction)
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origin)
    np.testing.assert_allclose(r[:, 3, 0], 1)
    np.testing.assert_allclose(r[:, 3, 1], 0)
    np.testing.assert_allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1)


def test_hray_broadcasts_single_origin_over_many_directions():
    np.random.seed(1)
    direction = np.random.randn(6, 3)
    r = hray([1, 2, 3], direction)
    assert r.shape == (6, 4, 2)
    for i in range(6):
        np.testing.assert_allclose(r[i, :, 0], [1, 2, 3, 1])


def test_hray_result_is_valid_ray():
    np.random.seed(2)
    r = hray(np.random.randn(4, 3), np.random.randn(4, 3))
    assert is_valid_rays(r)


# ------------------------ line_line_distance ------------------------


def test_distance_skew_perpendicular_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 0, 2.5], [0, 1, 0])
    np.testing.assert_allclose(line_line_distance(r1, r2), [2.5])


def test_distance_skew_nonorthogonal_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([5, 7, 2], [1, 1, 0])
    np.testing.assert_allclose(line_line_distance(r1, r2), [2.0])


def test_distance_intersecting_lines_is_zero():
    r1 = _ray([1, 2, 3], [1, 0, 0])
    r2 = _ray([1, 2, 3], [0, 1, 0])
    np.testing.assert_allclose(line_line_distance(r1, r2), [0.0], atol=1e-9)


def test_distance_parallel_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([7, 3, 4], [1, 0, 0])
    np.testing.assert_allclose(line_line_distance(r1, r2), [5.0])


def test_distance_antiparallel_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([-2, 3, 4], [-1, 0, 0])
    np.testing.assert_allclose(line_line_distance(r1, r2), [5.0])


def test_distance_is_symmetric():
    np.random.seed(3)
    r1 = rand_ray(8)
    r2 = rand_ray(8)
    np.testing.assert_allclose(line_line_distance(r1, r2),
                               line_line_distance(r2, r1))


def test_distance_invariant_to_sliding_origin_along_line():
    np.random.seed(4)
    r1 = rand_ray(6)
    r2 = rand_ray(6)
    r2b = r2.copy()
    r2b[:, :3, 0] += 3.7 * r2[:, :3, 1]
    np.testing.assert_allclose(line_line_distance(r1, r2),
                               line_line_distance(r1, r2b), atol=1e-8)


def test_distance_batch_matches_individual():
    origins1 = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype='f8')
    dirs1 = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype='f8')
    origins2 = np.array([[0, 0, 1], [0, 3, 4], [1, 1, 1]], dtype='f8')
    dirs2 = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]], dtype='f8')
    r1 = hray(origins1, dirs1)
    r2 = hray(origins2, dirs2)
    d = line_line_distance(r1, r2)
    assert d.shape == (3, )
    np.testing.assert_allclose(d, [1.0, 5.0, 0.0], atol=1e-9)


# --------------------- line_line_closest_points ---------------------


def test_closest_points_simple_skew():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([1, 0, 2], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[1, 0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(q2, [[1, 0, 2, 1]], atol=1e-9)


def test_closest_points_intersecting_lines_coincide():
    r1 = _ray([1, 2, 3], [1, 0, 0])
    r2 = _ray([4, 2, 3], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[4, 2, 3, 1]], atol=1e-9)
    np.testing.assert_allclose(q2, [[4, 2, 3, 1]], atol=1e-9)


def test_closest_points_output_shapes_and_homogeneous_coordinate():
    np.random.seed(5)
    r1 = rand_ray(7)
    r2 = rand_ray(7)
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (7, 4)
    assert q2.shape == (7, 4)
    np.testing.assert_allclose(q1[:, 3], 1)
    np.testing.assert_allclose(q2[:, 3], 1)


def test_closest_points_lie_on_their_lines():
    np.random.seed(6)
    r1 = rand_ray(10)
    r2 = rand_ray(10)
    q1, q2 = line_line_closest_points(r1, r2)
    pt1, ax1 = r1[:, :, 0], r1[:, :, 1]
    pt2, ax2 = r2[:, :, 0], r2[:, :, 1]
    np.testing.assert_allclose(hnorm(hcross(q1 - pt1, ax1)), 0, atol=1e-8)
    np.testing.assert_allclose(hnorm(hcross(q2 - pt2, ax2)), 0, atol=1e-8)


def test_closest_points_segment_perpendicular_to_both_lines():
    np.random.seed(7)
    r1 = rand_ray(10)
    r2 = rand_ray(10)
    q1, q2 = line_line_closest_points(r1, r2)
    ax1, ax2 = r1[:, :, 1], r2[:, :, 1]
    np.testing.assert_allclose(hdot(q2 - q1, ax1), 0, atol=1e-8)
    np.testing.assert_allclose(hdot(q2 - q1, ax2), 0, atol=1e-8)


def test_closest_points_distance_matches_line_line_distance():
    np.random.seed(8)
    r1 = rand_ray(10)
    r2 = rand_ray(10)
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(hnorm(q2 - q1),
                               line_line_distance(r1, r2),
                               atol=1e-8)


def test_closest_points_swapping_rays_swaps_results():
    np.random.seed(9)
    r1 = rand_ray(5)
    r2 = rand_ray(5)
    a1, a2 = line_line_closest_points(r1, r2)
    b2, b1 = line_line_closest_points(r2, r1)
    np.testing.assert_allclose(a1, b1, atol=1e-8)
    np.testing.assert_allclose(a2, b2, atol=1e-8)


def test_closest_points_parallel_lines_not_finite():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 1, 0], [1, 0, 0])
    with np.errstate(all='ignore'):
        q1, q2 = line_line_closest_points(r1, r2)
    assert not np.all(np.isfinite(q1))
